from contextlib import asynccontextmanager
from threading import Lock
from omegaconf import OmegaConf
import numpy as np
from fastapi import FastAPI, Request

import torch

from cosmos_predict2.online.online_trainer import OnlineGPUServiceAPI, StepRequestFromEvaluator


""" How to use me?
conda activate cosmos-predict2
cd ~/code/cospred2nvidia/
export PYTHONPATH=~/code/cospred2nvidia/
CUDA_VISIBLE_DEVICES=7 uvicorn online_pusht_api:gpu_app --port 6067
"""
class OnlinePushTAPI(OnlineGPUServiceAPI):
    def __init__(self):
        self.load_model_during_init = False
        super(OnlinePushTAPI, self).__init__()

        ''' Constant params, should be different across envs '''
        self.cosmos_root = "/home/geyuan/code/cospred2nvidia"
        self.model_time = "2025-10-10_12-52-51"
        self.iteration = "000020000"
        self.load_ema = True
        self.save_prefix = "pusht"
        self.input_dataset_path = f"{self.cosmos_root}/datasets/pusht/pusht_256_val.zarr"
        self.input_dataset_indices = (300, 100, 0)
        self.input_video_frame_index = 0

        self.infer_args = OmegaConf.create({
            "model_size": "2B",
            "dit_path": f"{self.cosmos_root}/checkpoints/cosmos_predict2/debug/cospred2_2b_expert_pusht_{self.model_time}/checkpoints/model/iter_{self.iteration}.pt",
            "input_video": self.input_dataset_path,
            "dataset_index": self.input_dataset_indices[-1],  # not used
            "frame_index": self.input_video_frame_index,  # not used
            "num_obs_frames": self.chunk_max_obs,
            "num_sampling_step": 10,  # ori:35
            "fps": 10,
            "guidance": 0,
            "seed": 0,  # ori:0, set None: just for visualization
            "chunk_size": self.chunk_action_horizon,  # v2
            "load_ema": self.load_ema,
            "text_encoder_path": "",  # PushT doesn't use text encoder
        })

        self.domain_shift = "light"  # `rainbow`, `light`, `none`
        if self.domain_shift == "light":
            self.online_hyper_params.update({
                'max_iters': 10000,  # ori: 1000
                'lr': 5e-6,  # ori: 1e-4
                'f_max': 0.1,  # ori: 0.17
                'weight_decay': 0.01,  # ori: 0.01
                'update_steps_per_feedback': 2,
                'grad_accum_steps': 1,
                'num_sampling_step': self.infer_args.num_sampling_step,  # for record
                'lora_max_layers': 4,  # now:12
                'lora_single_layer_modules': [
                    "self_attn.q_proj",
                    "self_attn.k_proj", "self_attn.v_proj",
                    "self_attn.out_proj",
                    "mlp.layer1", "mlp.layer2",
                    # NOTE: unfreeze expert layers
                    "self_attn.ex_q_proj",
                    # "self_attn.ex_k_proj", "self_attn.ex_v_proj",
                    # "self_attn.ex_out_proj",
                    # "ex_mlp.layer1", "ex_mlp.layer2",
                ],
                'online_attention_entropy_layers': 4,  # 99:all layers, -1:none layers
                'min_p_all_actions_as_condition': 0.5,  # ori: 1.0
                'freezing_expert': True,  # ori: True
            })
        elif self.domain_shift == "rainbow":
            self.online_hyper_params = {
                'max_iters': 10000,  # ori: 1000
                'lr': 3e-5,  # ori: 2 ** (-16)
                'f_max': 0.1,  # ori: 0.17
                'weight_decay': 0.01,  # ori: 0.01
                'update_steps_per_feedback': 2,
                'grad_accum_steps': 1,
                'num_sampling_step': self.infer_args.num_sampling_step,  # for record
                'lora_max_layers': 3,
                'lora_single_layer_modules': [
                    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj",
                    "mlp.layer1", "mlp.layer2",
                    # NOTE: unfreeze expert layers?
                    # "self_attn.ex_q_proj", "self_attn.ex_k_proj", "self_attn.ex_v_proj", "self_attn.ex_out_proj",
                    # "ex_mlp.layer1", "ex_mlp.layer2",
                ]
            }
        elif self.domain_shift == "none":
            self.online_hyper_params = {
                'max_iters': 10000,  # ori: 1000
                'lr': 1e-5,  # ori: 2 ** (-16)
                'f_max': 0.1,  # ori: 0.17
                'weight_decay': 0.1,  # ori: 0.01
                'update_steps_per_feedback': 2,
                'grad_accum_steps': 1,
                'num_sampling_step': self.infer_args.num_sampling_step,  # for record
                'lora_max_layers': 2,
                'lora_single_layer_modules': [
                    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj",
                    "mlp.layer1", "mlp.layer2",
                ]
            }
        else:
            raise NotImplementedError(f"Domain shift={self.domain_shift} not implemented.")

        ''' Get model and config at initialization? '''
        if self.load_model_during_init:
            self.get_model_and_others("cuda")

    def get_dataset_meta(self):
        # 1. Get dataset path
        # 2. Get meta from json
        # Note: pusht has no meta for now, so we return an empty dict
        return {}

    def norm_robot_states(self, robot_states_B_T_D: np.ndarray) -> np.ndarray:
        in_robot_states = (robot_states_B_T_D / 256.) - 1.  # [0,512] -> [-1,1]
        return in_robot_states

    def denorm_action(self, action_B_T_D: torch.Tensor):
        out_action = torch.clamp(action_B_T_D, min=-1., max=1.)
        out_action = (out_action * 256. + 256.).cpu().numpy()  # in [0,512]
        return out_action


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动：创建并加载一次（驻留于该进程/GPU）
    api = OnlinePushTAPI()                   # 你已在 __init__ 里做了 self.get_model_and_others("cuda")
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
    api = OnlinePushTAPI()
    api.get_model_and_others("cuda")
