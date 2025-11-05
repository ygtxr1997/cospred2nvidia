import os
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple, Dict, Any, Union
from abc import abstractmethod, ABC

import numpy as np
from fastapi import FastAPI
from omegaconf import OmegaConf

import torch
from torchvision.transforms import transforms

from robokit.connects.protocols import StepRequestFromEvaluator, StepRequestFromPolicy
from robokit.data_manager.utils_multiview import get_frames_from_multiview_video, cat_multiview_video_with_zeros

from cosmos_predict2.configs.base.config import Config
from cosmos_predict2.models.video2world_model import Predict2ModelManagerConfig
from cosmos_predict2.models.video2world_expert_model import Predict2Video2WorldExpertModel, Predict2Video2WorldModelConfig
from cosmos_predict2.configs.expert.config import PREDICT2_VIDEO2WORLD_PIPELINE_2B_EXPERT, Video2WorldExpertPipelineConfig
from cosmos_predict2.pipelines.video2world_expert import Video2WorldExpertPipeline
from cosmos_predict2.data.action_conditioned.tcl_dataset import TCLImageDataset

from cosmos_predict2.utils.vis_helpers import save_action_as_image
from cosmos_predict2.configs.expert.config import create_config_from_checkpoint
from cosmos_predict2.utils.ckpt_load_helpers import load_config_from_checkpoint_dir

from imaginaire.lazy_config import instantiate
from imaginaire.lazy_config import LazyCall as L
from imaginaire.utils import callback, distributed, log, misc
from imaginaire.utils.io import save_image_or_video
from cosmos_predict2.utils.printer import print_batch

""" How to use me?
On `dongxu-g7.cs.hku.hk`, assuming your port is `6060`, then run:
>$ CUDA_VISIBLE_DEVICES={GPU_INDEX} uvicorn gpu_service:gpu_app --port '6060'
Example:
cd ~/code/cospred2nvidia/
CUDA_VISIBLE_DEVICES=6 uvicorn video2world_tcl_api:gpu_app --port 6070
"""
class EmptyGPUServiceAPI:
    def __init__(self):
        super().__init__()
        self.model = None
        self.config = None

        self.max_cache_action = 4 * 5  # will notify evaluator the max length
        self.chunk_action_horizon = self.max_cache_action
        self.chunk_max_obs = 4 * 1 + 1

        self.called_times = 0
        self.last_obs_frame_last = None
        self.last_out_action = None

        self.cosmos_root = "/home/geyuan/code/cospred2nvidia"
        self.model_time = "2025-10-15_16-48-08"
        self.iteration = "000034000"
        self.load_ema = True
        self.input_dataset_path = f"/home/geyuan/local_soft/TCL/1009_spoon_pick_place"
        self.input_dataset_indices = (100,)
        self.input_video_frame_index = 0

        self.online_iteration = 0
        self.max_online_iteration = 1000
        self.online_hyper_params = {
            'lr': 2 ** (-16),  # ori: 2 ** (-15)
            'f_max': 0.1,  # ori: 0.17
            'weight_decay': 0.1,  # ori: 0.01
        }

        self.model_and_others = self.get_model_and_others("cuda")  # pre-load the model to speed up the first request

    def get_model_and_others(self, device: str) -> Dict[str, Any]:
        args = OmegaConf.create({
            "model_size": "2B",
            "dit_path": f"{self.cosmos_root}/checkpoints/cosmos_predict2/debug/cospred2_2b_expert_tcl_{self.model_time}/checkpoints/model/iter_{self.iteration}.pt",
            "input_video": self.input_dataset_path,
            "dataset_index": self.input_dataset_indices[-1],  # not used
            "frame_index": self.input_video_frame_index,  # not used
            "num_obs_frames": self.chunk_max_obs,
            "guidance": 0,
            "seed": 0,
            "chunk_size": self.chunk_action_horizon,  # v2
            "load_ema": self.load_ema,
        })

        # [] Random seed and cuDNN settings.
        misc.set_random_seed(seed=args.seed, by_rank=True)
        # Initialize cuDNN.
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        # Floating-point precision settings.
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

        # [] Load original config from the checkpoint file
        config_dict, config_path = load_config_from_checkpoint_dir(args.dit_path)
        log.info(f"Loading config from: {config_path}")
        pipe_config, dataset_config = create_config_from_checkpoint(config_dict)
        log.info("Successfully created config from checkpoint file")

        # [] Modify the config based on input args
        # config_dict.keys: 'model', 'optimizer', 'scheduler', 'dataloader_train', 'dataloader_val', 'job',
        #   'trainer', 'model_parallel', 'checkpoint', 'defaults', '_is_frozen'
        config_dict: Config
        config_dict.model.config: Predict2Video2WorldModelConfig
        config_dict.model.config.pipe_config: Video2WorldExpertPipelineConfig
        config_dict.model.config.model_manager_config: Predict2ModelManagerConfig

        text_encoder_path = ""  # PushT doesn't use text encoder
        hyper_params = self.online_hyper_params

        config_dict.model.config.train_architecture = "lora"  # ori: "base"
        config_dict.model.config.model_manager_config.dit_path = args.dit_path
        config_dict.model.config.model_manager_config.text_encoder_path = text_encoder_path  # PushT doesn't use text encoder
        config_dict.optimizer.lr = hyper_params['lr']  # smaller lr for online update
        config_dict.optimizer.weight_decay = hyper_params['weight_decay']
        config_dict.scheduler.warm_up_steps = [0]  # no warm up when online update
        config_dict.scheduler.cycle_lengths = [1000]
        config_dict.scheduler.verbosity_interval = 10
        config_dict.scheduler.f_start = [1e-6]
        config_dict.scheduler.f_max = [hyper_params['f_max']]  # ori:0.17
        config_dict.scheduler.f_min = config_dict.scheduler.f_max
        config_dict.model.config.pipe_config.p_all_actions_as_condition = 1.  # no action as condition during online update
        log.info("Config has been updated for inference and online update.")

        # [] Create trainable model and move to device
        log.info(f"Initializing Video2WorldPipeline with model size: {args.model_size}")
        model: Predict2Video2WorldExpertModel = instantiate(
            config_dict.model,
            load_ema=args.load_ema,
        )  # will call pipeline.from_config()
        model = model.to(device, memory_format=config_dict.trainer.memory_format)
        for module in [model.net, model.pipe.tokenizer]:
            if module is not None:
                module.to(memory_format=config_dict.trainer.memory_format, device=device, dtype=torch.bfloat16)

        # [] Initialize the optimizer, lr_scheduler, and grad_scaler.
        optimizer, scheduler = model.init_optimizer_scheduler(config_dict.optimizer, config_dict.scheduler)
        grad_scaler = torch.amp.GradScaler("cuda", **config_dict.trainer.grad_scaler_args)
        log.info("Initialized the optimizer and scheduler.")
        print(type(model))

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
        model.pipe.init_for_online_update(max_batches=1)

        return {
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "grad_scaler": grad_scaler,
            "args": args,
            "args.dit_path": args.dit_path,
        }
        # Return `model` instead of `pipe` to support online update

    def init_model(self):
        print("init model")

        input_path = self.input_dataset_path
        data_parent = os.path.dirname(os.path.abspath(input_path))
        data_task_name = os.path.basename(os.path.abspath(input_path))

        self.tcl_dataset = TCLImageDataset(
            shape_meta={
                "action": {
                    "shape": [7]
                },
                "obs": {
                    "image": {
                        "shape": [3, 128, 160],
                        "type": "rgb"
                    },
                    "gripper": {  # additional
                        "shape": [3, 128, 160],
                        "type": "rgb"
                    },
                    "joint_state": {  # additional, 7}
                        "shape": [6],
                    },
                }
            },
            horizon=self.max_cache_action,  # action length
            max_train_episodes=90,  # not used
            pad_before=self.chunk_max_obs - 1,  # m_obs-1
            pad_after=1,
            seed=42,
            val_ratio=0.0,  # ori:0.02

            data_root=input_path,
            h5_path=f"{data_parent}/hdf5/{data_task_name}_240p.h5",
            use_h5=True,
            transform_color_jitter=False,

            # language related
            language_emb_model="t5xxl",  # ori: "clip", or: "t5xxl"

            # multi-view related
            camera_keys=[
                "image",
                "gripper",
            ],
            p_camera_drop=0.,  # ori:0
            switch_camera_view=True,  # [Warning] only when data collection makes mistake
        )

    def reset_model(self):
        print("reset model")

    def predict_action(self, obs_dict: Dict[str, Any]) -> Union[np.ndarray, Dict[str, Any]]:
        return {"action_pred": torch.zeros((1, 20, 7), dtype=torch.float32)}

    def denorm_action(self, action: np.ndarray) -> np.ndarray:
        action = self.tcl_dataset.denorm_action(
            action,
            norm_type=self.tcl_dataset.norm_action_type,
            meta_data=self.tcl_dataset.dataset_meta_dict,
        )
        return action


gpu_app = FastAPI()
gpu_service_api = EmptyGPUServiceAPI()

@lru_cache()
def get_agent(device: str):
    model_and_others = gpu_service_api.model_and_others
    return model_and_others


@gpu_app.get("/")
def read_root():
    return {"message": "Hello, World!"}


@gpu_app.get("/init")
def model_init():
    gpu_service_api.init_model()
    return {"message": "Initialized.", "max_cache_action": gpu_service_api.max_cache_action}

@gpu_app.get("/reset")
def model_reset():
    gpu_service_api.reset_model()
    return {"message": "Done", "max_cache_action": gpu_service_api.max_cache_action}


@gpu_app.post("/step")
def model_step(step_request: StepRequestFromEvaluator):
    agent_and_others = get_agent("cuda")
    agent = agent_and_others["model"]
    optimizer = agent_and_others["optimizer"]
    scheduler = agent_and_others["scheduler"]
    grad_scaler = agent_and_others["grad_scaler"]
    args = agent_and_others["args"]
    weight_path = agent_and_others["args.dit_path"]
    # agent, args, weight_path = get_agent("cuda")  # agent is a `Predict2Video2WorldExpertModel` object
    print("[video2world_tcl_api] Using cached ckpt from: None. Model type:", type(agent), weight_path)

    # parse input observation
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

    gpu_service_api.last_obs_frame_last = get_frames_from_multiview_video(
        gt_video, sample_n_views=n_cameras, start_idx=Ts - v1, end_idx=Ts
    )  # (B,V*Ts,H,W,3) uint8 cut to (B,V*v1,H,W,3) uint8

    if stage_flag == 0:  # cold start
        pass  # do not update model
    elif gpu_service_api.online_iteration < gpu_service_api.max_online_iteration:  # hot start
        assert stage_flag == 1, "stage_flag should be 1 in hot start"
        # State II. Online update the model using GT frames (input + output) + actions (input)
        # a. set model to train mode
        if not agent.training:
            agent.train()
        # b. prepare training data
        online_data_batch = agent.pipe.get_data_batch_for_online_update(
            torch.from_numpy(gt_video).permute(0, 4, 1, 2, 3),  # (B,C,V*Ts,H,W) uint8 in [0,255]
        )
        # print_batch("online_data_batch", online_data_batch)
        save_image_or_video(
            (online_data_batch["video"].float() / 255.)[0],  # (B,C,T,H,W) float32 in [0,1]
            f"output/tcl_online_input_{gpu_service_api.iteration}_{gpu_service_api.called_times:03d}.mp4",
            fps=10
        )

        # c. training step
        online_update_steps_per_feedback = 2
        config_grad_accum_steps = 2
        online_iteration = gpu_service_api.online_iteration

        for _ in range(online_update_steps_per_feedback):
            with distributed.ddp_sync_grad(agent, (online_iteration + 1) % config_grad_accum_steps == 0):
                output_batch, loss = agent.training_step(online_data_batch, online_iteration)
                loss_scaled = grad_scaler.scale(loss / 1)
                print("[DEBUG] online update: iteration, loss:", online_iteration, loss.item())
                loss_scaled.backward()
            gpu_service_api.online_iteration += 1
            # d. optimizer, scheduler step
            if gpu_service_api.online_iteration % config_grad_accum_steps == 0:
                grad_scaler.step(optimizer)
                grad_scaler.update()
                scheduler.step()
                agent.on_before_zero_grad(optimizer, scheduler, iteration=gpu_service_api.online_iteration)
                optimizer.zero_grad(set_to_none=True)
                # e. update ema shadow weights
                if agent.pipe.dit_ema_bf16 is not None:
                    agent.pipe.online_update_ema_bf16()
        # f. set model back to eval mode
        if agent.training:
            agent.eval()
        # g. print hyper params
        if gpu_service_api.online_iteration % 10 == 0:
            print('[DEBUG] hyper params:', gpu_service_api.online_hyper_params)

    if True or stage_flag == 0:  # cold start
        assert Ts >= v1, f"Only support Ts>=v1 in cold start, got Ts={Ts}, v1={v1}"

        # Stage I. Frames -> Actions
        in_cond = get_frames_from_multiview_video(
            gt_video, sample_n_views=n_cameras, start_idx=Ts - v1, end_idx=Ts
        )  # (B,V*Ts,H,W,3) cut to (B,V*v1,H,W,3) uint8
        in_frames = cat_multiview_video_with_zeros(
            in_cond, sample_n_views=n_cameras, zero_length=gpu_service_api.max_cache_action
        )  # (B,V*(v1+a),H,W,3) uint8
        in_actions = np.zeros((B, gpu_service_api.max_cache_action, agent.pipe.config.net.action_dof),
                              dtype=np.float32)  # (B,a,2) float32, in [-1,1]
        # in_robot_states = dataset.norm_agent_pos(torch.from_numpy(tcp_state[:, -v1:])).numpy()
        in_robot_states = (tcp_state[:, -v1:] / 256.) - 1.  # [0,512] -> [-1,1]
        print("[DEBUG] expert_api: in_frames:", in_frames.shape, in_frames.min(), in_frames.max(),
              "in_actions:", in_actions.shape, in_actions.min(), in_actions.max(),
              "in_robot_states:", None if in_robot_states is None else in_robot_states.shape, )
        save_image_or_video(
            (torch.from_numpy(gt_video).float().permute(0, 4, 1, 2, 3) / 255.)[0],  # (B,C,T,H,W) float32 in [0,1]
            f"output/tcl_env_{gpu_service_api.iteration}_{gpu_service_api.called_times:03d}.mp4",
            fps=10
        )
        out_video, out_action = agent.pipe(
            in_frames,  # (B,v1+a,H,W,3) uint8
            in_actions,  # (B,a,D) float32
            in_robot_states,  # (B,v1,D) float32
            num_conditional_frames=v1,  # ori:1
            num_conditional_actions=0,
            n_views=n_cameras,
            guidance=args.guidance,
            seed=args.seed,
            fps=30,  # hard code fps for tcl
            use_ema=True,  # use ema dit
        )  # out_action:(B,chunk_size,2) float32 in [-1,1]; out_video:(B,C,T,H,W) float32 in [-1,1]
        save_image_or_video(
            out_video,
            f"output/tcl_expert_pred_{gpu_service_api.iteration}_{gpu_service_api.called_times:03d}.mp4",
            fps=10
        )
        save_action_as_image(
            out_action[0, :, :2].cpu().numpy(),
            save_path=f"output/tcl_out_action_{gpu_service_api.iteration}_{gpu_service_api.called_times:03d}.png"
        )

        # denorm action
        out_action = torch.clamp(out_action, min=-1., max=1.).cpu().numpy()
        out_action = gpu_service_api.denorm_action(out_action)  # in [act_min, act_max]

    print("[DEBUG] tcl_api: out_action:", out_action.shape, out_action.min(), out_action.max(),
          "out_video:", out_video.shape, out_video.min(), out_video.max())
    gpu_service_api.last_out_action = out_action.copy()
    gpu_service_api.called_times += 1

    # 4. Send action request
    request_to_evaluator = StepRequestFromPolicy.encode_from_raw(action=out_action)
    return request_to_evaluator.model_dump(mode="json")


if __name__ == "__main__":
    # DEBUG
    import numpy as np

    agent = get_agent("cuda")

    zero_rgb = np.zeros((2, 480, 848, 3), dtype=np.uint8)  # (T,H,W,C)

    debug_request = StepRequestFromEvaluator.create_zero_request()
    pred_action = model_step(debug_request)
    print(pred_action)
