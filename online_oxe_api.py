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
CUDA_VISIBLE_DEVICES=3 uvicorn online_oxe_api:gpu_app --port 20070
"""
class OnlineOXEAPI(OnlineGPUServiceAPI):
    def __init__(self):
        self.load_model_during_init = False
        super(OnlineOXEAPI, self).__init__()

        ''' Constant params, should be different across envs '''
        self.max_cache_action = 4 * 3  # will notify evaluator the max length
        self.chunk_action_horizon = self.max_cache_action
        self.chunk_max_obs = 4 * 0 + 1

        self.env_name = "fractal"
        if self.env_name == "fractal":
            self.model_time = "2025-10-28_15-04-37"
            self.iteration = "000060000"
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
            "dit_path": f"{self.cosmos_root}/checkpoints/cosmos_predict2/debug/cospred2_2b_expert_oxe_{self.model_time}/checkpoints/model/iter_{self.iteration}.pt",
            "input_video": self.input_dataset_path,
            "dataset_index": self.input_dataset_indices[-1],  # not used
            "frame_index": self.input_video_frame_index,  # not used
            "num_obs_frames": self.chunk_max_obs,  # v1
            "num_sampling_step": 15,  # ori:35
            "fps": 10,
            "guidance": 0,
            "seed": 0,
            "chunk_size": self.chunk_action_horizon,  # v2
            "load_ema": self.load_ema,
            "text_encoder_path": "checkpoints/google-t5/t5-11b",  # oxe needs a text encoder
        })

        if self.env_name == "fractal":
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
            })
        else:
            raise NotImplementedError(f"Domain shift={self.env_name} not implemented.")

        ''' Get model and config at initialization? '''
        if self.load_model_during_init:
            self.get_model_and_others("cuda")

    def get_dataset_meta(self):
        ## After run this, you can access self.dataset_meta
        # 1. Get dataset path
        # 2. Get meta from json
        if self.env_name == "fractal":
            return {
                "meta_p01": [-0.22453528, -0.14820013, -0.23158971, -0.35179949, -0.41930113, -0.43643461,  0.],
                "meta_p99": [0.17824687, 0.1493838 , 0.21842355, 0.5892666 , 0.35272657, 0.44796681, 1.],
            }
        elif self.env_name == "bridge":
            return {
                "meta_p01": [-0.02853955, -0.04143204, -0.02597738, -0.08020887, -0.0921306 , -0.20548619,  0.],
                "meta_p99": [0.02812228, 0.04063032, 0.03994889, 0.08121916, 0.07724379, 0.2021405 , 1.],
            }
        else:
            raise NotImplementedError(f"env_name={self.env_name} not implemented.")

    def norm_robot_states(self, robot_states_B_T_D: np.ndarray) -> np.ndarray:
        in_robot_states = robot_states_B_T_D  # NOTE: do nothing?
        return in_robot_states

    def denorm_action(self, action_B_T_D: torch.Tensor) -> np.ndarray:
        meta_p01 = self.dataset_meta["meta_p01"]
        meta_p99 = self.dataset_meta["meta_p99"]
        meta_p01 = torch.Tensor(meta_p01).to(action_B_T_D.device)
        meta_p99 = torch.Tensor(meta_p99).to(action_B_T_D.device)
        # out_action = torch.clamp(action_B_T_D, min=-1., max=1.)  # CAUTION: clamp should be banned for mean_std actions
        # NOTE: SimplerEnv will handle the action denormalization internally
        # out_action = (out_action + 1.0) / 2.0 * (meta_p99 - meta_p01) + meta_p01
        out_action = action_B_T_D
        return out_action.cpu().numpy()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动：创建并加载一次（驻留于该进程/GPU）
    api = OnlineOXEAPI()                   # 你已在 __init__ 里做了 self.get_model_and_others("cuda")
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
    api = OnlineOXEAPI()
    api.get_model_and_others("cuda")
