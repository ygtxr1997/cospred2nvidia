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

from cosmos_predict2.utils.vis_helpers import save_action_as_image
from cosmos_predict2.configs.expert.config import create_config_from_checkpoint
from cosmos_predict2.utils.ckpt_load_helpers import load_config_from_checkpoint_dir
from cosmos_predict2.connects.protocols import StepRequestFromEvaluator, StepRequestFromPolicy
from cosmos_predict2.connects.utils import get_frames_from_multiview_video, cat_multiview_video_with_zeros


from imaginaire.utils import distributed, log, misc
from imaginaire.utils.io import save_image_or_video


""" How to use me?
export PYTHONPATH=~/code/cospred2nvidia/
CUDA_VISIBLE_DEVICES=6 uvicorn video2world_pusht_api:gpu_app --port 6060
"""
gpu_app = FastAPI()
max_cache_action = 4*5  # will notify evaluator the max length
chunk_action_horizon = max_cache_action
chunk_max_obs = 4*1 + 1

CALLED_TIMES = 0
LAST_OBS_FRAME_LAST = None
LAST_OUT_ACTION = None

COSMOS_ROOT="/home/geyuan/code/cospred2nvidia"
MODEL_TIME = "2025-10-10_12-52-51"
ITERATION = "000020000"
LOAD_EMA = True
INPUT_DATASET_PATH = f"{COSMOS_ROOT}/datasets/pusht/pusht_256_val.zarr"
INPUT_DATASET_INDICES = (300, 100, 0)
INPUT_VIDEO_FRAME_INDEX = 0


@lru_cache()
def get_agent(device: str):
    args = OmegaConf.create({
        "model_size": "2B",
        "dit_path": f"{COSMOS_ROOT}/checkpoints/cosmos_predict2/debug/cospred2_2b_expert_pusht_{MODEL_TIME}/checkpoints/model/iter_{ITERATION}.pt",
        "input_video": INPUT_DATASET_PATH,
        "dataset_index": INPUT_DATASET_INDICES[-1],  # not used
        "frame_index": INPUT_VIDEO_FRAME_INDEX,  # not used
        "num_obs_frames": chunk_max_obs,
        "guidance": 0,
        "seed": 0,
        "chunk_size": chunk_action_horizon,  # v2
        "load_ema": LOAD_EMA,
    })

    config_dict, config_path = load_config_from_checkpoint_dir(args.dit_path)
    log.info(f"Loading config from: {config_path}")
    pipe_config, dataset_config = create_config_from_checkpoint(config_dict)
    log.info("Successfully created config from checkpoint file")

    dit_path = args.dit_path
    text_encoder_path = ""  # PushT doesn't use text encoder

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
        config=pipe_config,
        dit_path=dit_path,
        text_encoder_path=text_encoder_path,
        device=device,
        torch_dtype=torch.bfloat16,
        load_ema_to_reg=args.load_ema,
        # load_prompt_refiner=True,
    )

    return pipe, args, dit_path


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
    print("[video2world_pusht_api] Using cached ckpt from: None. Model type:", type(agent), weight_path)

    # parse input observation
    step_data = step_request.decode_to_raw()
    instruction_text = step_data["instruction"]
    stage_flag = step_data["stage_flag"]
    gt_video = step_data["gt_video"]  # (B,V*Ts,H,W,3) uint8, Ts can be larger than v1
    tcp_state = step_data["tcp_state"]  # (B,Ts,D) float32 or None
    n_cameras = agent.config.net.n_cameras_emb

    # o o o o o o o o o ; o o o o o o o o o |       V*Ts, V=2, multi-view obs returned by evaluator
    # o o o o o o o o o ;                           Ts=9, length of a single view
    #         o o o o o ;                           v1=5, obs before v1 arr not used in the Stage I
    B, VTs, H, W, C = gt_video.shape
    Ts = VTs // n_cameras  # Ts: single-view frame length, Ts >= v1
    v1 = args.num_obs_frames  # v1 set by the agent

    global LAST_OBS_FRAME_LAST, LAST_OUT_ACTION, CALLED_TIMES
    LAST_OBS_FRAME_LAST = get_frames_from_multiview_video(
        gt_video, sample_n_views=n_cameras, start_idx=Ts - v1, end_idx=Ts
    )  # (B,V*Ts,H,W,3) uint8 cut to (B,V*v1,H,W,3) uint8

    if stage_flag == 0:  # cold start
        assert Ts >= v1, f"Only support Ts>=v1 in cold start, got Ts={Ts}, v1={v1}"

        # Stage I. Frames -> Actions
        in_cond = get_frames_from_multiview_video(
            gt_video, sample_n_views=n_cameras, start_idx=Ts - v1, end_idx=Ts
        )  # (B,V*Ts,H,W,3) cut to (B,V*v1,H,W,3) uint8
        in_frames = cat_multiview_video_with_zeros(
            in_cond, sample_n_views=n_cameras, zero_length=max_cache_action
        )  # (B,V*(v1+a),H,W,3) uint8
        in_actions = np.zeros((B, max_cache_action, agent.config.net.action_dof), dtype=np.float32)  # (B,a,2) float32, in [-1,1]
        # in_robot_states = dataset.norm_agent_pos(torch.from_numpy(tcp_state[:, -v1:])).numpy()
        in_robot_states = (tcp_state[:, -v1:] / 256.) - 1.  # [0,512] -> [-1,1]
        print("[DEBUG] expert_api: in_frames:", in_frames.shape, in_frames.min(), in_frames.max(),
              "in_actions:", in_actions.shape, in_actions.min(), in_actions.max(),
              "in_robot_states:", None if in_robot_states is None else in_robot_states.shape, )
        save_image_or_video(
            (torch.from_numpy(gt_video).float().permute(0, 4, 1, 2, 3) / 255.)[0],  # (B,C,T,H,W) float32 in [0,1]
            f"output/pusht_env_{ITERATION}_{CALLED_TIMES:03d}.mp4",
            fps=10
        )
        out_video, out_action = agent(
            in_frames,  # (B,v1+a,H,W,3) uint8
            in_actions,  # (B,a,D) float32
            in_robot_states,  # (B,v1,D) float32
            num_conditional_frames=v1,  # ori:1
            num_conditional_actions=0,
            n_views=n_cameras,
            guidance=args.guidance,
            seed=args.seed,
            fps=10,  # hard code fps for pusht
        )  # out_action:(B,chunk_size,2) float32 in [-1,1]; out_video:(B,C,T,H,W) float32 in [-1,1]
        save_image_or_video(
            out_video,
            f"output/pusht_expert_pred_{ITERATION}.mp4",
            fps=10
        )
        save_action_as_image(
            out_action[0, :, :2].cpu().numpy(),
            save_path=f"output/pusht_out_action_{ITERATION}_{CALLED_TIMES:03d}.png"
        )

        # denorm action
        out_action = torch.clamp(out_action, min=-1., max=1.)
        out_action = (out_action * 256. + 256.).cpu().numpy()  # in [0,512]


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

    else:  # hot start
        raise NotImplementedError
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
        out_action = (out_action * 512.0).cpu().numpy()  # in [0,512], for pusht
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

    print("[DEBUG] pusht_api: out_action:", out_action.shape, out_action.min(), out_action.max(),
          "out_video:", out_video.shape, out_video.min(), out_video.max())
    LAST_OUT_ACTION = out_action.copy()
    CALLED_TIMES += 1

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
