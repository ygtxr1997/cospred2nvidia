import base64
import io
import os
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import copy

import numpy as np
import pydantic
from PIL import Image
from fastapi import FastAPI
from omegaconf import OmegaConf

import json
import torch
from torchvision.transforms import transforms

from cosmos_predict2.configs.expert.config import PREDICT2_VIDEO2WORLD_PIPELINE_2B_EXPERT
from cosmos_predict2.pipelines.video2world_expert import Video2WorldExpertPipeline
from cosmos_predict2.data.action_conditioned.pusht_dataset import PushTImageDataset
from imaginaire.utils import distributed, log, misc
from imaginaire.utils.io import save_image_or_video


""" How to use me?
export PYTHONPATH=~/code/cospred2nvidia/
CUDA_VISIBLE_DEVICES=6 uvicorn video2world_expert_api:gpu_app --port 6060
"""
gpu_app = FastAPI()
max_cache_action = 12

LAST_OBS_FRAME_LAST = None
LAST_OUT_ACTION = None

COSMOS_ROOT="/home/geyuan/code/cospred2nvidia"
MODEL_TIME = "2025-09-09_22-33-33"
ITERATION = "000040000"
#INPUT_DATASET_PATH="./datasets/pusht/pusht_cchi_v7_replay.zarr"
#INPUT_DATASET_PATH="./datasets/pusht/pusht_orange_random_v2.zarr"
INPUT_DATASET_PATH = f"{COSMOS_ROOT}/datasets/pusht/pusht_256_val.zarr"
INPUT_DATASET_INDICES = (300, 100, 0)
INPUT_VIDEO_FRAME_INDEX = 0


class StepRequestFromEvaluator(pydantic.BaseModel):
    instruction: str
    stage_flag: int  # 0:cold start, 1:hot start

    gt_video: List[List[str]]  # len1=B, len2=v2, each str is base64 of (H,W,rgb)
    tcp_state: Optional[List[List[float]]] = None # len1=B, len2=2, float32, not used

    def decode_to_raw(self) -> Dict[str, Any]:
        def base64_to_image_H_W_C(img_base64: str) -> np.ndarray:
            img_byte = base64.b64decode(img_base64)
            img_pil = Image.open(io.BytesIO(img_byte), formats=["JPEG"])
            img_np = np.array(img_pil)  # (H,W,3) uint8
            return img_np

        def base64_to_video_T_H_W_C(vid_base64: List[str]) -> np.ndarray:
            return np.stack([base64_to_image_H_W_C(img_b64) for img_b64 in vid_base64], axis=0)

        gt_video_B_T_H_W_C = np.stack([base64_to_video_T_H_W_C(vid_b64) for vid_b64 in self.gt_video], axis=0)  # (B,v2,H,W,3) uint8
        tcp_state_B_D = None if self.tcp_state is None else np.array(self.tcp_state, dtype=np.float32)  # (B,2) float32
        return {
            "instruction": self.instruction,
            "stage_flag": self.stage_flag,
            "gt_video": gt_video_B_T_H_W_C,
            "tcp_state": tcp_state_B_D,
        }

    @classmethod
    def encode_from_raw(cls,
                        instruction: str,
                        stage_flag: int,
                        gt_video: np.ndarray,  # (B,v1+v2,H,W,3) uint8
                        tcp_state: Optional[np.ndarray] = None,  # (B,2) float32
                        ) -> 'StepRequestFromEvaluator':
        instance = cls(
            instruction=instruction,
            stage_flag=stage_flag,
            gt_video=cls.video_np_to_base64(gt_video),
            tcp_state=None if tcp_state is None else tcp_state.tolist()
        )
        return instance

    @staticmethod
    def video_np_to_base64(video: np.ndarray) -> List[List[str]]:
        assert video.ndim in [4, 5], f"video should be (B,v1+v2,H,W,3) or (v1+v2,H,W,3), got {video.shape}"
        assert video.dtype == np.uint8

        def image_to_base64(img_H_W_C) -> str:
            image_pil = Image.fromarray(img_H_W_C)
            image_bytes = io.BytesIO()
            image_pil.save(image_bytes, format="JPEG")
            image_bytes = image_bytes.getvalue()
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            return image_base64

        def video_to_base64(vid_T_H_W_C) -> List[str]:
            return [image_to_base64(img) for img in vid_T_H_W_C]

        if video.ndim == 4:  # No batch-dim
            video_base64 = [video_to_base64(video)]  # add batch-dim
        elif video.ndim == 5:
            video_base64 = [video_to_base64(vid) for vid in video]
        else:
            raise NotImplementedError

        return video_base64


class StepRequestFromPolicy(pydantic.BaseModel):
    action: List[List[List[float]]]  # len1=B, len2=a1, len3=2
    max_cache_action: int = None

    def decode_to_raw(self) -> Dict[str, Any]:
        return {
            "action": np.array(self.action, dtype=np.float32)  # (B,a1,2) float32
        }

    @classmethod
    def encode_from_raw(cls, action: np.ndarray) -> 'StepRequestFromPolicy':
        if action.ndim == 2:  # (T,2)
            action = action[np.newaxis, ...]  # add batch dim
        assert action.ndim == 3, f"action should be (B,a1,2), got {action.shape}"
        assert action.dtype in [np.float32, np.float64], f"action should be float32 or float64, got {action.dtype}"
        instance = cls(action=action.tolist())
        return instance


@lru_cache()
def get_agent(device: str):
    ## Op1. Debug model, sleep only
    # model = DebugModel(sleep_duration=100)
    ## Op2. Replay model, load action data and sleep
    # model = ReplayModel(sleep_duration=25,
    #                     replay_root="/home/geyuan/datasets/TCL/collected_data")

    args = OmegaConf.create({
        "model_size": "2B",
        "dit_path": f"{COSMOS_ROOT}/checkpoints/cosmos_predict2/debug/predict2_video2world_2b_expert_training_{MODEL_TIME}/checkpoints/model/iter_{ITERATION}.pt",
        "input_video": INPUT_DATASET_PATH,
        "dataset_index": INPUT_DATASET_INDICES[-1],  # not used
        "frame_index": INPUT_VIDEO_FRAME_INDEX,  # not used
        "num_obs_frames": 5,
        "guidance": 0,
        "seed": 0,
        "chunk_size": 12,  # a1+a2
    })

    config = PREDICT2_VIDEO2WORLD_PIPELINE_2B_EXPERT
    dit_path = args.dit_path
    text_encoder_path = ""

    misc.set_random_seed(seed=args.seed, by_rank=True)
    # Initialize cuDNN.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    # Floating-point precision settings.
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Load models
    log.info(f"Initializing Video2WorldPipeline with model size: {args.model_size}")
    pipe = Video2WorldExpertPipeline.from_config(
        config=config,
        dit_path=dit_path,
        text_encoder_path=text_encoder_path,
        device=device,
        torch_dtype=torch.bfloat16,
        # load_ema_to_reg=args.load_ema,
        # load_prompt_refiner=True,
    )

    return pipe, args, dit_path

    import hydra

    # 1. Load hydra config
    train_dir = f"/home/geyuan/code/dp23rss_fork/data/outputs/{log_time}_train_diffusion_transformer_hybrid_pusht_images"
    train_dir = train_dir.replace('-', '/')
    hydra_config_path = os.path.join(train_dir, ".hydra/config.yaml")
    hydra_config = OmegaConf.load(hydra_config_path)
    model = hydra.utils.instantiate(hydra_config.policy)
    print(type(model))

    # 2. Load weights
    weight_paths = os.listdir(os.path.join(train_dir, "checkpoints"))
    weight_paths = list(filter(lambda x: x.endswith(".ckpt"), weight_paths))
    weight_paths.sort()
    print(weight_paths)
    weight_path = os.path.join(train_dir, "checkpoints", weight_paths[w_idx])
    weight = torch.load(weight_path, map_location="cpu", weights_only=False)['state_dicts']
    weight = weight['model']
    # for k, v in weight.items():
    #     print(k, v.shape)

    model.load_state_dict(weight)
    model = model.to(device).eval()
    print(f"[get_agent] model loaded from: {weight_path}")

    # 3. Other settings
    model.infer_frame_idx = 0

    return model, hydra_config, weight_path


@gpu_app.get("/")
def read_root():
    return {"message": "Hello, World!"}


@gpu_app.get("/init")
def model_init():
    return {"message": "Initialized.", "max_cache_action": max_cache_action}


@gpu_app.get("/reset")
def model_reset():
    agent, image_shape, _ = get_agent("cuda")
    # agent.reset()
    return {"max_cache_action": max_cache_action}


@gpu_app.post("/step")
def model_step(step_request: StepRequestFromEvaluator) -> Dict:
    agent, args, weight_path = get_agent("cuda")  # shape:[C,H,W]
    print("[video2world_expert_api] Using cached ckpt from: None. Model type:", type(agent), weight_path)

    # parse input observation
    step_data = step_request.decode_to_raw()
    instruction_text = step_data["instruction"]
    stage_flag = step_data["stage_flag"]
    gt_video = step_data["gt_video"]  # (B,v2,H,W,3) uint8
    tcp_state = step_data["tcp_state"]  # (B,2) float32 or None

    B, v2, H, W, C = gt_video.shape
    v1 = args.num_obs_frames

    global LAST_OBS_FRAME_LAST, LAST_OUT_ACTION
    LAST_OBS_FRAME_LAST = gt_video[:, -v1:]  # (B,1,H,W,3) uint8

    if stage_flag == 0:  # cold start
        # out_action = np.array([[[384, 128]] * max_cache_action] * B)  # [B,T,2], default action
        # out_action = np.random.randn(B, max_cache_action, 2) * 40. + 256.  # add noise
        assert v2 >= v1, f"Only support v2>=v1 in cold start, got v2={v2}, v1={v1}"

        # Stage I. Frames -> Actions
        in_frames = np.concatenate((gt_video[:, -v1:],
                                    np.zeros((B, max_cache_action, H, W, C))), axis=1).astype(np.uint8)  # (B,v1+a,H,W,3) uint8
        in_actions = np.zeros((B, max_cache_action, 2), dtype=np.float32)  # (B,a,2) float32, in [-1,1]
        print("[DEBUG] expert_api: in_frames:", in_frames.shape, in_frames.min(), in_frames.max(),
              "in_actions:", in_actions.shape, in_actions.min(), in_actions.max())
        save_image_or_video(
            (torch.from_numpy(gt_video).float().permute(0, 4, 1, 2, 3) / 255.)[0],  # (B,C,T,H,W) float32 in [0,1]
            f"output/pusht_expert_env_{ITERATION}.mp4",
            fps=5
        )
        out_video, out_action = agent(
            in_frames,  # (B,v1,H,W,3) uint8
            in_actions,  # (B,0,2) float32
            num_conditional_frames=v1,  # ori:1
            num_conditional_actions=0,
            guidance=args.guidance,
            seed=args.seed,
        )  # out_action:(B,chunk_size,2) float32 in [-1,1]; out_video:(B,C,T,H,W) float32 in [-1,1]
        save_image_or_video(
            out_video,
            f"output/pusht_expert_pred_{ITERATION}.mp4",
            fps=5
        )


        # # Stage II. Frames+Actions -> Frames
        # in_frames = in_frames
        # in_actions = out_action.cpu().numpy()  # (B,a1,2) float32, in [-1,1]
        # out_video, _ = agent(
        #     in_frames,  # (B,v1,H,W,3) uint8
        #     in_actions,  # (B,a1,2) float32
        #     num_conditional_frames=v1,  # ori:1
        #     num_conditional_actions=max_cache_action,  # a1
        #     guidance=args.guidance,
        #     seed=args.seed,
        # )  # out_video:(B,chunk_size,H,W,3) float32 in [-1,1]
        # save_image_or_video(
        #     out_video,
        #     f"output/pusht_expert_pred2_{ITERATION}.mp4",
        #     fps=5
        # )

        out_action = torch.clamp(out_action, min=-1., max=1.)
        out_action = (out_action * 256. + 256.).cpu().numpy()  # in [0,512]

    else:  # hot start
        assert LAST_OBS_FRAME_LAST is not None and LAST_OUT_ACTION is not None, "last_out_action and last_obs_final should not be None in hot start"
        assert v2 == max_cache_action, f"Only support v2=1 (cold start) or v2=max_cache_action (hot start), got v2={v2}"
        assert LAST_OUT_ACTION.shape[1] == max_cache_action, f"Only support last_out_action.shape[1]=max_cache_action, got {LAST_OUT_ACTION.shape}"
        B, a1, Da = LAST_OUT_ACTION.shape
        noised_action = np.zeros((B, args.chunk_size - a1, Da), dtype=np.float32)

        in_frames = np.concatenate([LAST_OBS_FRAME_LAST, gt_video], axis=1)  # (B,v1+v2,H,W,3) uint8
        in_actions = np.concatenate([LAST_OUT_ACTION, noised_action], axis=1, dtype=np.float32)  # (B,a1+a2,2) float32, in [0,512]
        in_actions = in_actions / 512.0  # in [0,1]
        out_video, out_action = agent(
            in_frames,  # (B,v1+v2,H,W,3) uint8
            in_actions,  # (B,a1,2) float32
            num_conditional_frames=1 + max_cache_action,  # ori:1
            action_cond_len=max_cache_action,  # a1
            guidance=args.guidance,
            seed=args.seed,
        )
        out_action = out_action[:, a1:, :]  # (B,a2,2) float32
        out_action = (out_action * 512.0).cpu().numpy()  # in [0,512]
        assert out_action.shape[:-1] == (B, max_cache_action), f"Only support out_action.shape=(B,{max_cache_action},2), got {out_action.shape}"

        # save the generated video for debug
        save_fps = 5  # use lower fps for clearer visualization

        # save the generated video
        output_path = f"output/pusht_expert_env_{ITERATION}.mp4"
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        log.info(f"Saving generated video to: {output_path}")
        save_image_or_video(out_video, output_path, fps=save_fps)
        log.success(f"Successfully saved video to: {output_path}")

        # save the evaluator feedback video
        feedback_video = gt_video[0]  # (B,v2,H,W,3) -> (v2,H,W,3)
        feedback_video = torch.from_numpy(feedback_video).permute(3, 0, 1, 2).float() / 255.
        output_path = f"output/pusht_expert_env_{ITERATION}_feedback.mp4"
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        log.info(f"Saving feedback video to: {output_path}")
        save_image_or_video(feedback_video, output_path, fps=save_fps)
        log.success(f"Successfully saved video to: {output_path}")

    print("[DEBUG] expert_api: out_action:", out_action.shape, out_action.min(), out_action.max(),
          "out_video:", out_video.shape, out_video.min(), out_video.max())
    LAST_OUT_ACTION = out_action.copy()

    # out_action = [[[256.]*2] * max_cache_action] * 25  # [B,T,2]
    request_to_evaluator = StepRequestFromPolicy(action=out_action)
    return request_to_evaluator.model_dump(mode="json")


if __name__ == "__main__":
    import time
    agent = get_agent("cuda")

    zero_rgb = np.zeros((2, 480, 848, 3), dtype=np.uint8)  # (T,H,W,C)

    debug_request = StepRequestWithObservation(
        primary_rgb=ServiceConnector.img_np_to_base64(zero_rgb),
        gripper_rgb=ServiceConnector.img_np_to_base64(zero_rgb),
        instruction="none",
        joint_state=[[0.] * 6] * 2,
    )
    pred_action = model_step(debug_request)
