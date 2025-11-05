"""
Modified from: imaginaire.trainer.ImaginaireTrainer
Refer to the style of ImaginareTrainner, and used for online updating.
"""

import base64
import io
import os
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union
import copy
from abc import ABC, abstractmethod

import numpy as np
import pydantic
from PIL import Image
from fastapi import FastAPI
from omegaconf import OmegaConf

import json
import torch
from torchvision.transforms import transforms

from cosmos_predict2.configs.base.config import Config
from cosmos_predict2.models.video2world_model import Predict2ModelManagerConfig
from cosmos_predict2.models.video2world_expert_model import Predict2Video2WorldExpertModel, Predict2Video2WorldModelConfig
from cosmos_predict2.configs.expert.config import PREDICT2_VIDEO2WORLD_PIPELINE_2B_EXPERT, Video2WorldExpertPipelineConfig
from cosmos_predict2.pipelines.video2world_expert import Video2WorldExpertPipeline

from cosmos_predict2.utils.vis_helpers import save_action_as_image
from cosmos_predict2.configs.expert.config import create_config_from_checkpoint
from cosmos_predict2.utils.ckpt_load_helpers import load_config_from_checkpoint_dir
from cosmos_predict2.connects.protocols import StepRequestFromEvaluator, StepRequestFromPolicy
from cosmos_predict2.connects.utils import (
    get_frames_from_multiview_video, cat_multiview_video_with_zeros,
    get_skipped_indices,
)

from imaginaire.lazy_config import instantiate
from imaginaire.lazy_config import LazyCall as L
from imaginaire.utils import callback, distributed, log, misc
from imaginaire.utils.io import save_image_or_video
from cosmos_predict2.utils.printer import print_batch


class OnlineGPUServiceAPI(ABC):
     # Relative classes: ImaginaireTrainer, robokit.GPUServiceAPI
    def __init__(self):
        self.model = None
        self.config_dict = None
        self.optimizer = None
        self.scheduler = None
        self.grad_scaler = None
        self.dataset_meta = None

        ''' Constant params, should be different across envs '''
        self.max_cache_action = 4 * 5  # will notify evaluator the max length
        self.chunk_action_horizon = self.max_cache_action
        self.chunk_max_obs = 4 * 1 + 1
        self.future_skip_frames = 1  # should be consistent with training dataset

        self.cosmos_root = "/home/geyuan/code/cospred2nvidia"
        self.model_time = "2025-10-15_16-48-08"
        self.iteration = "000034000"
        self.load_ema = True
        self.save_prefix = "tcl"
        self.input_dataset_path = f"/home/geyuan/local_soft/TCL/1009_spoon_pick_place"
        self.input_dataset_indices = (100,)
        self.input_video_frame_index = 0

        self.infer_args = OmegaConf.create({
            "model_size": "2B",
            "dit_path": f"{self.cosmos_root}/checkpoints/cosmos_predict2/debug/cospred2_2b_expert_tcl_{self.model_time}/checkpoints/model/iter_{self.iteration}.pt",
            "input_video": self.input_dataset_path,
            "dataset_index": self.input_dataset_indices[-1],  # not used
            "frame_index": self.input_video_frame_index,  # not used
            "num_obs_frames": self.chunk_max_obs,
            "num_sampling_step": 35,
            "fps": 10,
            "guidance": 0,
            "seed": 0,
            "chunk_size": self.chunk_action_horizon,  # v2
            "load_ema": self.load_ema,
            "text_encoder_path": "",  #chunk_action_horizon PushT doesn't use text encoder
        })

        self.online_hyper_params: Dict[str, Any] = {
            'max_iters': 1000,  # ori: 1000
            'lr': 2 ** (-16),  # ori: 2 ** (-15)
            'f_max': 0.1,  # ori: 0.17
            'weight_decay': 0.1,  # ori: 0.01
            'update_steps_per_feedback': 2,
            'grad_accum_steps': 2,
            'num_sampling_step': self.infer_args.num_sampling_step,  # for record
            'lora_max_layers': 3,
            'lora_single_layer_modules': [
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.out_proj",
                "cross_attn.q_proj",
                "mlp.layer1",
                "mlp.layer2",
            ],
            'online_attention_entropy_layers': -1,
            'min_p_all_actions_as_condition': 1.,
            'steps_p_all_actions_as_condition': 20,
            'freezing_expert': True,
        }

        ''' Dynamic params, will be updated by self '''
        self.called_times = 0

        self.last_obs_frame_last = None
        self.last_out_action = None

        self.online_iteration = 0
        self.online_p_all_actions_as_condition = 1.0  # max:1.0

    def get_model_and_others(self, device: str):
        """ Should be same across different evaluators. """
        if self.model is not None and self.config_dict is not None:
            return {
                "model": self.model,
                "optimizer": self.optimizer,
                "scheduler": self.scheduler,
                "grad_scaler": self.grad_scaler,
                "args": self.infer_args,
                "args.dit_path": self.infer_args.dit_path,
                "config_dict": self.config_dict,
                "dataset_meta": self.dataset_meta,
            }

        args = self.infer_args

        ''' Random seed and cuDNN settings '''
        misc.set_random_seed(seed=args.seed, by_rank=False)  # NOTE: is by_rank=False correct here?
        # Initialize cuDNN.
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        # Floating-point precision settings.
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

        ''' Load original config from the checkpoint file '''
        config_dict, config_path = load_config_from_checkpoint_dir(args.dit_path)
        log.info(f"Loading config from: {config_path}")
        pipe_config, dataset_config = create_config_from_checkpoint(config_dict)
        log.info("Successfully created config from checkpoint file")

        ''' Modify the config based on input args '''
        config_dict: Config
        config_dict.model.config: Predict2Video2WorldModelConfig
        config_dict.model.config.pipe_config: Video2WorldExpertPipelineConfig
        config_dict.model.config.model_manager_config: Predict2ModelManagerConfig

        text_encoder_path = args.text_encoder_path
        hyper_params = self.online_hyper_params

        self.min_p_all_actions_as_condition = hyper_params['min_p_all_actions_as_condition']  # for online update
        self.steps_p_all_actions_as_condition = hyper_params['steps_p_all_actions_as_condition']
        self.delta_p_all_actions_as_condition = (1.0 - self.min_p_all_actions_as_condition) / self.steps_p_all_actions_as_condition

        config_dict.model.config.loss_scale = 1.0  # NOTE: smaller scale during online update
        config_dict.model.config.train_architecture = "lora"  # ori: "base"
        lora_max_layers = hyper_params['lora_max_layers']  # for 2B model
        lora_layers = [str(i) for i in range(lora_max_layers)]
        lora_single_layer_modules = hyper_params['lora_single_layer_modules']
        lora_multi_layer_modules = []
        for layer in lora_layers:
            for module in lora_single_layer_modules:
                lora_multi_layer_modules.append(f"{layer}.{module}")
        config_dict.model.config.lora_target_modules = ",".join(lora_multi_layer_modules)
        # NOTE: try attention entropy loss during online update?
        from omegaconf import open_dict
        with open_dict(config_dict.model.config.pipe_config.net):
            config_dict.model.config.pipe_config.net.online_attention_entropy_layers = \
                hyper_params['online_attention_entropy_layers']
        print(config_dict.model.config.pipe_config.net)

        config_dict.model.config.model_manager_config.dit_path = args.dit_path
        config_dict.model.config.model_manager_config.text_encoder_path = text_encoder_path
        config_dict.optimizer.lr = hyper_params['lr']  # smaller lr for online update
        config_dict.optimizer.weight_decay = hyper_params['weight_decay']
        config_dict.scheduler.warm_up_steps = [0]  # no warm up when online update
        config_dict.scheduler.cycle_lengths = [1000000]
        config_dict.scheduler.verbosity_interval = 10
        config_dict.scheduler.f_start = [1e-6]
        config_dict.scheduler.f_max = [hyper_params['f_max']]  # ori:0.17
        config_dict.scheduler.f_min = config_dict.scheduler.f_max
        config_dict.model.config.pipe_config.p_all_actions_as_condition = 1.0  # NOTE: no action as condition during online update
        log.info("Config has been updated for inference and online update.")

        ''' Create trainable model and move to device '''
        log.info(f"Initializing Video2WorldPipeline with model size: {args.model_size}")
        model: Predict2Video2WorldExpertModel = instantiate(
            config_dict.model,
            load_ema=args.load_ema,
            text_encoder_path=text_encoder_path,
        )  # will call pipeline.from_config()
        model = model.to(device, memory_format=config_dict.trainer.memory_format)
        for module in [model.net, model.pipe.tokenizer]:
            if module is not None:
                module.to(memory_format=config_dict.trainer.memory_format, device=device, dtype=torch.bfloat16)

        ''' Initialize the optimizer, lr_scheduler, and grad_scaler '''
        optimizer, scheduler = model.init_optimizer_scheduler(config_dict.optimizer, config_dict.scheduler)
        grad_scaler = torch.amp.GradScaler("cuda", **config_dict.trainer.grad_scaler_args)
        log.info("Initialized the optimizer and scheduler.")

        # [] Load the model checkpoint and get the starting iteration number.
        # NOTE: if we don't save the model weights, we can skip this step.

        # [] Create a DDP model wrapper.
        # NOTE: DDP or FSDP is not supported yet. We use single GPU for online update.

        # [] Check inference related settings.
        print("[DEBUG] pipe.scheduler:", type(model.pipe.scheduler))
        print("[DEBUG] pipe.scaling:", type(model.pipe.scaling))
        print("[DEBUG] pipe.tokenizer:", type(model.pipe.tokenizer))
        print("[DEBUG] pipe.conditioner:", type(model.pipe.conditioner))

        # [] Initialize for online update
        model.pipe.init_for_online_update(
            max_batches=1, freezing_expert=hyper_params['freezing_expert'],
            future_skip_frames=self.future_skip_frames,
        )

        # [] Keep model and others in memory
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.grad_scaler = grad_scaler
        self.config_dict = config_dict
        self.dataset_meta = self.get_dataset_meta()  # should be after setting config_dict
        return {
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "grad_scaler": grad_scaler,
            "args": args,
            "args.dit_path": args.dit_path,
            "config_dict": config_dict,
            "dataset_meta": self.dataset_meta,
        }

    # called by xxx_api.get("/")
    def read_root(self):
        return {"message": "Hello, World!"}

    # called by xxx_api.get("/init")
    def init_model(self):
        self.get_model_and_others("cuda")
        return {"message": "Initialized.", "max_cache_action": self.max_cache_action}

    # called by xxx_api.get("/reset")
    def reset_model(self):
        # [] Reset random seed
        misc.set_random_seed(seed=self.infer_args.seed, by_rank=True)

        # [] Reset inference params
        self.last_obs_frame_last = None
        self.last_out_action = None

        # [] Reset online update params
        self.online_iteration = 0
        self.online_p_all_actions_as_condition = 1.0  # max:1.0

        # Reset optimizer, scheduler, grad_scaler
        del self.optimizer, self.scheduler, self.grad_scaler
        config_dict = self.config_dict
        optimizer, scheduler = self.model.init_optimizer_scheduler(config_dict.optimizer, config_dict.scheduler)
        grad_scaler = torch.amp.GradScaler("cuda", **config_dict.trainer.grad_scaler_args)
        log.info("Initialized the optimizer and scheduler.")

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.grad_scaler = grad_scaler

        # Reset ema
        self.model.pipe.reset_for_online_update()

        return {"max_cache_action": self.max_cache_action}

    # called by xxx_api.post("/step")
    def model_step(self, step_request: StepRequestFromEvaluator) -> Dict[str, List]:
        agent = self.model
        optimizer = self.optimizer
        scheduler = self.scheduler
        grad_scaler = self.grad_scaler
        args = self.infer_args
        weight_path = args.dit_path
        print(f"[OnlineGPUServiceAPI] Using cached ckpt from: {weight_path}. Model type:", type(agent))

        # [] Parse input observation
        step_data = step_request.decode_to_raw()
        instruction_text = step_data["instruction"]
        stage_flag = step_data["stage_flag"]
        gt_video = step_data["gt_video"]  # (B,V*Ts,H,W,3) uint8, Ts can be larger than v1
        tcp_state = step_data["tcp_state"]  # (B,Ts,D) float32 or None
        n_cameras = agent.pipe.config.net.n_cameras_emb

        # o o o o o o o o o ; o o o o o o o o o |       V*Ts, V=2, multi-view obs returned by evaluator
        # o o o o o o o o o ;                           Ts=9, length of a single view
        #         o o o o o ;                           v1=5, obs before v1 arr not used in the Stage I
        B, VTs, H, W, C = gt_video.shape
        Ts = VTs // n_cameras  # Ts: single-view frame length, Ts >= v1
        v1 = args.num_obs_frames  # v1 set by the agent

        self.last_obs_frame_last = get_frames_from_multiview_video(
            gt_video, sample_n_views=n_cameras, start_idx=Ts - v1, end_idx=Ts
        )  # (B,V*Ts,H,W,3) uint8 cut to (B,V*v1,H,W,3) uint8

        # [] Check if needed to perform online update
        if stage_flag == 0:  # cold start
            pass  # do not update model
        elif self.online_iteration <= self.online_hyper_params['max_iters']:
            assert stage_flag == 1, "stage_flag should be 1 in hot start"
            # State II. Online update the model using GT frames (input + output) + actions (input)
            # a. set model to train mode
            if not agent.training:
                agent.train()
            # b. prepare training data
            online_data_batch = agent.pipe.prepare_data_batch_for_online_update(
                torch.from_numpy(gt_video).permute(0, 4, 1, 2, 3),  # (B,C,V*Ts,H,W) uint8 in [0,255]
            )
            # print_batch("online_data_batch", online_data_batch)
            save_image_or_video(
                (online_data_batch["video"].float() / 255.)[0],  # (B,C,T,H,W) float32 in [0,1]
                f"output/{self.save_prefix}_online_input_{self.iteration}_{self.called_times:03d}.mp4",
                fps=10
            )

            # c. training step
            online_update_steps_per_feedback = self.online_hyper_params['update_steps_per_feedback']
            online_grad_accum_steps = self.online_hyper_params['grad_accum_steps']
            for update_step_idx in range(online_update_steps_per_feedback):
                with distributed.ddp_sync_grad(agent, (self.online_iteration + 1) % online_grad_accum_steps == 0):
                    output_batch, loss = agent.training_step(online_data_batch, self.online_iteration)
                    loss_scaled = grad_scaler.scale(loss / 1)
                    print(f"[DEBUG] online update: iteration={self.online_iteration}, loss={loss.item():.4f}")
                    loss_scaled.backward()

                # Online update iteration and p_all_actions_as_condition
                self.online_iteration += 1
                self.online_p_all_actions_as_condition = max(
                    self.min_p_all_actions_as_condition,
                    self.online_p_all_actions_as_condition - self.delta_p_all_actions_as_condition
                )
                agent.pipe.p_all_actions_as_condition = self.online_p_all_actions_as_condition

                # d. optimizer, scheduler step
                if self.online_iteration % online_grad_accum_steps == 0:
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                    scheduler.step()
                    agent.on_before_zero_grad(optimizer, scheduler, iteration=self.online_iteration)
                    optimizer.zero_grad(set_to_none=True)
                    # e. update ema shadow weights
                    if agent.pipe.dit_ema_bf16 is not None:
                        agent.pipe.online_update_ema_bf16()
                if update_step_idx < online_update_steps_per_feedback - 1:  # prepare next batch
                    online_data_batch = agent.pipe.get_data_batch_for_online_update()
            # f. set model back to eval mode
            if agent.training:
                agent.eval()
            # g. print hyper params
            if self.online_iteration % 10 == 0:
                print('[DEBUG] hyper params:', self.online_hyper_params)

        # [] Inference step
        assert Ts >= v1, f"Only support Ts>=v1 in cold start, got Ts={Ts}, v1={v1}"
        # Stage I. Frames -> Actions
        in_cond = get_frames_from_multiview_video(
            gt_video, sample_n_views=n_cameras, start_idx=Ts - v1, end_idx=Ts
        )  # (B,V*Ts,H,W,3) cut to (B,V*v1,H,W,3) uint8
        in_frames = cat_multiview_video_with_zeros(
            in_cond, sample_n_views=n_cameras, zero_length=self.max_cache_action // self.future_skip_frames,
        )  # (B,V*(v1+a),H,W,3) uint8
        in_actions = np.zeros((B, self.max_cache_action, agent.pipe.config.net.action_dof),
                              dtype=np.float32)  # (B,a,2) float32, in [-1,1]
        # Norm input robot states
        in_robot_states = self.norm_robot_states(tcp_state)[:, -v1:]  # [0,512] -> [-1,1]
        print("[DEBUG] online_api: in_frames:", in_frames.shape, in_frames.min(), in_frames.max(),
              "in_actions:", in_actions.shape, in_actions.min(), in_actions.max(),
              "in_robot_states:", in_robot_states.shape, in_robot_states.min(), in_robot_states.max())
        save_image_or_video(
            (torch.from_numpy(gt_video).float().permute(0, 4, 1, 2, 3) / 255.)[0],  # (B,C,T,H,W) float32 in [0,1]
            f"output/{self.save_prefix}_env_{self.iteration}_{self.called_times:03d}.mp4",
            fps=10
        )

        out_video, out_action = agent.pipe(
            in_frames,  # (B,v1+a,H,W,3) uint8
            in_actions,  # (B,a,D) float32
            in_robot_states,  # (B,v1,D) float32
            prompt=instruction_text,
            num_conditional_frames=v1,  # ori:1
            num_conditional_actions=0,
            n_views=n_cameras,
            guidance=args.guidance,
            seed=args.seed,
            num_sampling_step=args.num_sampling_step,
            fps=args.fps,
            use_ema=True,  # use ema dit
        )  # out_action:(B,chunk_size,2) float32 in [-1,1]; out_video:(B,C,T,H,W) float32 in [-1,1]
        save_image_or_video(
            out_video,
            f"output/{self.save_prefix}_expert_pred_{self.iteration}_{self.called_times:03d}.mp4",
            fps=10
        )
        save_action_as_image(
            out_action[0, :, :3].cpu().numpy(),
            save_path=f"output/{self.save_prefix}_out_action_{self.iteration}_{self.called_times:03d}.png"
        )

        # Denorm action
        out_action = self.denorm_action(out_action)

        print(f"[DEBUG] {self.save_prefix}_api: out_action:", out_action.shape, out_action.min(), out_action.max(),
              "out_video:", out_video.shape, out_video.min(), out_video.max())
        self.last_out_action = out_action.copy()
        self.called_times += 1

        request_to_evaluator = StepRequestFromPolicy.encode_from_raw(action=out_action)
        return request_to_evaluator.model_dump(mode="json")

    @abstractmethod
    def get_dataset_meta(self) -> Dict[str, Any]: pass

    @abstractmethod
    def norm_robot_states(self, robot_states_B_T_D: Union[np.ndarray, torch.Tensor]) -> np.ndarray: pass

    @abstractmethod
    def denorm_action(self, action_B_T_D: Union[np.ndarray, torch.Tensor]) -> np.ndarray: pass
