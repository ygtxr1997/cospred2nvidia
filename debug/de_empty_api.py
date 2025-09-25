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
max_cache_action = 24

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
    return None, None, None


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

    out_action = [[[256.]*10] * max_cache_action] * 4  # [B,T,10]
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
