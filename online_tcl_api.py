from contextlib import asynccontextmanager
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from omegaconf import OmegaConf
import numpy as np
from fastapi import FastAPI, Request

import torch

from cosmos_predict2.online.online_trainer import OnlineGPUServiceAPI, StepRequestFromEvaluator
from cosmos_predict2.online.online_trainer import (
    distributed,
    get_frames_from_multiview_video,
    cat_multiview_video_with_zeros,
    get_skipped_indices,
    save_image_or_video,
    save_action_as_image,
    StepRequestFromPolicy
)
from cosmos_predict2.data.action_conditioned.tcl_dataset import TCLImageDataset


""" How to use me?
conda activate cosmos-predict2
cd ~/code/cospred2nvidia/
export PYTHONPATH=~/code/cospred2nvidia/
CUDA_VISIBLE_DEVICES=3 uvicorn online_tcl_api:gpu_app --port 6070
"""
class OnlineTCLAPI(OnlineGPUServiceAPI):
    def __init__(self):
        self.load_model_during_init = False
        super(OnlineTCLAPI, self).__init__()

        ''' Constant params, should be different across envs '''
        self.max_cache_action = 4 * 8  # will notify evaluator the max length
        self.chunk_action_horizon = self.max_cache_action
        self.chunk_max_obs = 4 * 0 + 1
        self.future_skip_frames = 8  # should be consistent with training dataset

        self.env_name = "sweep"
        if self.env_name == "sweep":
            self.model_time = "2025-11-04_16-25-46"
            self.iteration = "000040000"
        else:
            raise NotImplementedError(f"env_name={self.env_name} not implemented.")

        self.cosmos_root = "/home/geyuan/code/cospred2nvidia"
        self.load_ema = True
        self.save_prefix = self.env_name
        self.input_dataset_path = f"{self.cosmos_root}/datasets/pusht/pusht_256_val.zarr"
        self.input_dataset_indices = (300, 100, 0)
        self.input_video_frame_index = 0

        self.infer_args = OmegaConf.create({
            "model_size": "2B",
            "dit_path": f"{self.cosmos_root}/checkpoints/cosmos_predict2/debug/cospred2_2b_force_tcl_{self.model_time}/checkpoints/model/iter_{self.iteration}.pt",
            "input_video": self.input_dataset_path,
            "dataset_index": self.input_dataset_indices[-1],  # not used
            "frame_index": self.input_video_frame_index,  # not used
            "num_obs_frames": self.chunk_max_obs,  # v1
            "num_sampling_step": 10,  # ori:10
            "fps": 20,  # ori:10
            "guidance": 0,
            "seed": 0,
            "chunk_size": self.chunk_action_horizon,  # v2
            "load_ema": self.load_ema,
            "text_encoder_path": "checkpoints/google-t5/t5-11b",  # tcl needs a text encoder
        })

        if self.env_name == "sweep":
            self.online_hyper_params.update({
                'max_iters': 10000,  # ori: 1000
                'lr': 1e-5,  # ori: 2 ** (-16)
                'f_max': 0.1,  # ori: 0.17
                'weight_decay': 0.01,  # ori: 0.01
                'update_steps_per_feedback': 2,
                'grad_accum_steps': 1,
                'num_sampling_step': self.infer_args.num_sampling_step,  # for record
                'lora_max_layers': 3,
                'lora_single_layer_modules': [
                    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj",
                    "mlp.layer1", "mlp.layer2",
                ],
                'online_attention_entropy_layers': -1,
            })
        else:
            raise NotImplementedError(f"Domain shift={self.env_name} not implemented.")

        ''' Get model and config at initialization? '''
        if self.load_model_during_init:
            self.get_model_and_others("cuda")

    # called by xxx_api.post("/step")
    # NOTE: since having force data, we need to override this function
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
        tcp_state = step_data["tcp_state"]  # (B,Ts,D+6) float32 or None, NOTE: includes force data
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
            online_data_batch = agent.pipe.get_data_batch_for_online_update(
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
            for _ in range(online_update_steps_per_feedback):
                with distributed.ddp_sync_grad(agent, (self.online_iteration + 1) % online_grad_accum_steps == 0):
                    output_batch, loss = agent.training_step(online_data_batch, self.online_iteration)
                    loss_scaled = grad_scaler.scale(loss / 1)
                    print(f"[DEBUG] online update: iteration={self.online_iteration}, loss={loss.item():.4f}")
                    loss_scaled.backward()
                self.online_iteration += 1
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
        )  # (B,V*Ts,H,W,3) cut to (B,V*v1,H,W,3) uint8, in_cond can be continued without skipping
        in_frames = cat_multiview_video_with_zeros(
            in_cond, sample_n_views=n_cameras, zero_length=self.max_cache_action // self.future_skip_frames
        )  # (B,V*(v1+a/k),H,W,3) uint8
        in_actions = np.zeros((B, self.max_cache_action, agent.pipe.config.net.action_dof),
                              dtype=np.float32)  # (B,a,2) float32, in [-1,1]
        # Norm input robot states
        in_robot_states, in_forces = self.norm_robot_states(tcp_state)  # (B,Ts,D) in [-1,1]
        in_robot_states = in_robot_states[:, -v1:]  # cut to (B,v1,D)
        in_forces = np.concatenate(
            (in_forces[:, -v1:],
             np.zeros((B, self.max_cache_action // self.future_skip_frames, 6), dtype=np.float32)),
            axis=1
        )  # (B,v1+a/k,6) float32 in [-1,1], pad zeros after v1 for forces input
        print("[DEBUG] online_api: in_frames:", in_frames.shape, in_frames.min(), in_frames.max(),
              "in_actions:", in_actions.shape, in_actions.min(), in_actions.max(),
              "in_robot_states:", in_robot_states.shape, in_robot_states.min(), in_robot_states.max(),
              "in_forces:", in_forces.shape, in_forces.min(), in_forces.max())
        save_image_or_video(
            (torch.from_numpy(gt_video).float().permute(0, 4, 1, 2, 3) / 255.)[0],  # (B,C,T,H,W) float32 in [0,1]
            f"output/{self.save_prefix}_env_{self.iteration}_{self.called_times:03d}.mp4",
            fps=10
        )

        out_video, out_action, out_force = agent.pipe(
            in_frames,  # (B,v1+v2,H,W,3), v2=a/k, uint8
            in_actions,  # (B,a,D) float32
            in_forces,  # (B,v1+v2,6) float32
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
        save_action_as_image(
            out_force[0, :, :3].cpu().numpy(),
            save_path=f"output/{self.save_prefix}_out_force_{self.iteration}_{self.called_times:03d}.png"
        )

        # Denorm action
        out_action = self.denorm_action(out_action)

        print(f"[DEBUG] {self.save_prefix}_api: out_action:", out_action.shape, out_action.min(), out_action.max(),
              "out_video:", out_video.shape, out_video.min(), out_video.max(),
              "out_force:", out_force.shape, out_force.min(), out_force.max(),)
        self.last_out_action = out_action.copy()
        self.called_times += 1

        request_to_evaluator = StepRequestFromPolicy.encode_from_raw(action=out_action)
        return request_to_evaluator.model_dump(mode="json")

    def get_dataset_meta(self):
        ## After run this, you can access self.dataset_meta
        # 1. Get dataset path
        # 2. Get meta from json
        import os, json
        meta_json_path = os.path.abspath(os.path.join(
            self.infer_args.dit_path, "../../../", "statistics.json"
        ))
        with open(meta_json_path, 'r') as json_file:
            statistics = json.load(json_file)
        self.dataset_meta = statistics["stats"]  # keys:`rel_actions`, `robot_obs`, `force_torque`
        self.norm_action_type = str(self.config_dict.dataloader_train.sampler.dataset.norm_action_type)
        self.norm_force_type = self.config_dict.dataloader_train.sampler.dataset.get("norm_force_type", "minmax")
        print("[DEBUG] online_tcl_api: norm action type:", self.norm_action_type,
              "norm force type:", self.norm_force_type)
        return self.dataset_meta

    def norm_robot_states(self, robot_states_B_T_D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        force_B_T_D = robot_states_B_T_D[:, :, -6:].astype(np.float32)  # (B,T,6)
        tcp_pose_B_T_D = robot_states_B_T_D[:, :, :6].astype(np.float32)  # (B,T,6)
        tcp_pose_B_T_D = TCLImageDataset.norm_state_or_force(
            tcp_pose_B_T_D, norm_type="mean", meta_data=self.dataset_meta["robot_obs"]
        )
        force_B_T_D = TCLImageDataset.norm_state_or_force(
            force_B_T_D, self.norm_force_type, meta_data=self.dataset_meta["force_torque"]
        )
        return tcp_pose_B_T_D, force_B_T_D

    def denorm_action(self, action_B_T_D: torch.Tensor) -> np.ndarray:
        # out_action = torch.clamp(action_B_T_D, min=-1., max=1.), CAUTION:only minmax needs to be clamped
        out_action = action_B_T_D
        out_action = TCLImageDataset.denorm_action(
            out_action.cpu().numpy(), norm_type=self.norm_action_type, meta_data=self.dataset_meta["rel_actions"]
        )
        print("[DEBUG] online_tcl_api: "
              "out_action[:, :, :6]:", out_action[:, :, :6].min(), out_action[:, :, :6].max(),
              "out_action[:, :, 6:]:", out_action[:, :, 6:].min(), out_action[:, :, 6:].max())
        return out_action


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动：创建并加载一次（驻留于该进程/GPU）
    api = OnlineTCLAPI()                   # 你已在 __init__ 里做了 self.get_model_and_others("cuda")
    app.state.online_api = api
    app.state.api_lock = Lock()              # 可选：串行化有状态更新
    try:
        yield
    finally:
        # 关闭：释放资源（可选）
        del app.state.online_api
        torch.cuda.empty_cache()

gpu_app = FastAPI(lifespan=lifespan)

@gpu_app.get("/")
def read_root(request: Request):
    return request.app.state.online_api.read_root()

@gpu_app.get("/init")
def model_init(request: Request):
    return request.app.state.online_api.init_model()

@gpu_app.get("/reset")
def model_reset(request: Request):
    return request.app.state.online_api.reset_model()

@gpu_app.post("/step")
def model_step(request_data: StepRequestFromEvaluator, request: Request):
    # 如果 online 学习/更新是有状态的，建议加锁避免并发写同一模型
    with request.app.state.api_lock:
        return request.app.state.online_api.model_step(request_data)


if __name__ == '__main__':
    api = OnlineTCLAPI()
    api.get_model_and_others("cuda")
