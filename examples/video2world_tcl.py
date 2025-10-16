# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
import yaml
from pathlib import Path

import attrs
import mediapy as mp
import numpy as np

# Set TOKENIZERS_PARALLELISM environment variable to avoid deadlocks with multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import einops
from megatron.core import parallel_state

# from cosmos_predict2.configs.action_conditioned.config import PREDICT2_VIDEO2WORLD_PIPELINE_2B_ACTION_CONDITIONED
# from cosmos_predict2.pipelines.video2world_action import Video2WorldActionConditionedPipeline
from cosmos_predict2.configs.expert.config import (
    PREDICT2_VIDEO2WORLD_PIPELINE_2B_EXPERT,
    create_config_from_checkpoint,
)
from cosmos_predict2.configs.expert.experiment.exp_libero import cospred2_2b_expert_libero
from cosmos_predict2.pipelines.video2world_expert import Video2WorldExpertPipeline
# from cosmos_predict2.data.action_conditioned.pusht_dataset import PushTImageDataset
from cosmos_predict2.data.action_conditioned.tcl_dataset import TCLImageDataset
from cosmos_predict2.utils.ckpt_load_helpers import load_config_from_checkpoint_dir
from cosmos_predict2.connects.utils import get_frames_from_multiview_video, cat_multiview_video_with_zeros
from cosmos_predict2.utils.vis_helpers import save_action_as_image
from imaginaire.utils import distributed, log, misc
from imaginaire.utils.io import save_image_or_video


def get_action_sequence(val_dataset, dataset_index=0):
    data = val_dataset[dataset_index]
    action = data["action"]  # (T,2), in [-1,1]
    agent_pos = data["agent_pos"]  # (T,8), in [-1,1]
    video = data["video"].permute(1, 2, 3, 0)  # (3,V*T,256,256)->(V*T,H,W,C), in [0,255]
    t5_text_embeddings = data["t5_text_embeddings"]  # (512,1024), in bf16
    return action.numpy(), agent_pos.numpy(), video.numpy(), t5_text_embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video-to-World Generation with Cosmos Predict2")
    parser.add_argument(
        "--model_size",
        choices=["2B"],
        default="2B",
        help="Size of the model to use for video-to-world generation",
    )
    parser.add_argument(
        "--dit_path",
        type=str,
        default="",
        help="Custom path to the DiT model checkpoint for post-trained models.",
    )
    parser.add_argument(
        "--load_ema",
        action="store_true",
        help="Use EMA weights for generation.",
    )

    parser.add_argument(
        "--dataset_index",
        type=int,
        default=0,
        help="Index of the dataset.",
    )
    parser.add_argument(
        "--frame_index",
        type=int,
        default=0,
        help="Index of the frame to use for conditioning (0 for first frame, 1 for second frame, etc.)",
    )

    parser.add_argument(
        "--input_video",
        type=str,
        default="assets/video2world/input0.jpg",
        help="Path to input image or video for conditioning (include file extension)",
    )
    parser.add_argument(
        "--input_annotation",
        type=str,
        default="assets/video2world/input0.jpg",
        help="Path to input image or video for conditioning (include file extension)",
    )
    parser.add_argument(
        "--num_conditional_frames",
        type=int,
        default=1,
        choices=[1],
        help="Number of frames to condition on (1 for single frame, 5 for multi-frame conditioning) (Not used)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=24,
        help="Chunk size",
    )
    parser.add_argument("--autoregressive", action="store_true", help="Use autoregressive mode")
    parser.add_argument("--guidance", type=float, default=7, help="Guidance value")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument(
        "--save_path",
        type=str,
        default="output/generated_video.mp4",
        help="Path to save the generated video (include file extension)",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for context parallel inference (should be a divisor of the total frames)",
    )
    parser.add_argument("--disable_guardrail", action="store_true", help="Disable guardrail checks on prompts")
    parser.add_argument(
        "--disable_prompt_refiner", action="store_true", help="Disable prompt refiner that enhances short prompts"
    )
    return parser.parse_args()


def setup_pipeline(args: argparse.Namespace):
    log.info(f"Using model size: {args.model_size}")
    if args.model_size == "2B":
        config_dict, config_path = load_config_from_checkpoint_dir(args.dit_path)
        log.info(f"Loading config from: {config_path}")
        config, config_dataset = create_config_from_checkpoint(config_dict)
        log.info("Successfully created config from checkpoint file")
        print(config)
        # config = PREDICT2_VIDEO2WORLD_PIPELINE_2B_EXPERT
        # exp_config = cospred2_2b_expert_libero['model']['config']['pipe_config']  # load from exp config
        dit_path = "checkpoints/nvidia/Cosmos-Predict2-2B-Sample-Action-Conditioned/model-480p-4fps.pth"
    else:
        raise ValueError("Invalid model size. Choose either '2B' or '14B'.")
    if hasattr(args, "dit_path") and args.dit_path:
        dit_path = args.dit_path

    # text_encoder_path = "checkpoints/google-t5/t5-11b"
    text_encoder_path = ""

    misc.set_random_seed(seed=args.seed, by_rank=True)
    # Initialize cuDNN.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    # Floating-point precision settings.
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize distributed environment for multi-GPU inference
    if hasattr(args, "num_gpus") and args.num_gpus > 1:
        log.info(f"Initializing distributed environment with {args.num_gpus} GPUs for context parallelism")
        distributed.init()
        parallel_state.initialize_model_parallel(context_parallel_size=args.num_gpus)
        log.info(f"Context parallel group initialized with {args.num_gpus} GPUs")

    # Disable guardrail if requested
    if args.disable_guardrail:
        log.warning("Guardrail checks are disabled")
        config.guardrail_config.enabled = False

    # Disable prompt refiner if requested
    if args.disable_prompt_refiner:
        log.warning("Prompt refiner is disabled")
        config.prompt_refiner_config.enabled = False

    # Load models
    log.info(f"Initializing Video2WorldPipeline with model size: {args.model_size}")
    pipe = Video2WorldExpertPipeline.from_config(
        config=config,
        dit_path=dit_path,
        text_encoder_path=text_encoder_path,
        device="cuda",
        torch_dtype=torch.bfloat16,
        load_ema_to_reg=args.load_ema,
        load_prompt_refiner=False,
    )

    return pipe


def read_first_frame(val_dataset, dataset_index=0, frame_index=0):
    data = val_dataset[dataset_index]
    sample_n_views = int(data["sample_n_views"])
    video = data["video"]  # (3,V*T,H,W), in [0,255]
    print("[DEBUG] Video shape:", video.shape)

    video_C_V_T_H_W = einops.rearrange(video, "c (v t) h w -> c v t h w", v=sample_n_views)
    first_frame_C_V_H_W = video_C_V_T_H_W[:, :, frame_index]  # (3,V,H,W)
    # video = video.permute(1, 2, 3, 0).cpu().numpy()  # (T,96,96,3), in [0,255]
    first_frame_V_H_W_C = first_frame_C_V_H_W.permute(1, 2, 3, 0).cpu().numpy()  # (V,H,W,C), in [0,255]
    return first_frame_V_H_W_C  # Return first frame as numpy array


def process_single_generation(
    pipe, input_path, output_path, guidance, seed, chunk_size, autoregressive,
    dataset_index=0, frame_index=0,
):
    # chunk_action_max_len = 13 - 1  # a1=v1+v2-1, v1=1, v2=12
    chunk_action_horizon = chunk_size  # 24, horizon=a1+a2
    chunk_max_obs = 4 * 1 + 1
    # pusht_val_dataset = PushTImageDataset(
    #     zarr_path=input_path,  #"./datasets/pusht/pusht_orange_random_v2.zarr",
    #     max_obs=chunk_max_obs,  # v1 in [1,max_obs]
    #     max_act_out=chunk_action_horizon * 6,  # can be much longer since we can do autoregressive generation (a1+a2)
    # )
    max_act_out = chunk_action_horizon * 15

    data_parent = os.path.dirname(os.path.abspath(input_path))
    data_task_name = os.path.basename(os.path.abspath(input_path))

    tcl_dataset = TCLImageDataset(
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
        horizon=max_act_out,  # action length
        max_train_episodes=90,  # not used
        pad_before=chunk_max_obs - 1,  # m_obs-1
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
    print("[DEBUG] TCL dataset length:", len(tcl_dataset))
    actions, agent_pos, frames, t5_embeddings = get_action_sequence(tcl_dataset, dataset_index=dataset_index)
    # actions: (T,2), in [-1,1]
    # agent_pos: (T,8), in [-1,1]
    # frames: (V*T,H,W,C), in [0,255]
    first_frame = read_first_frame(tcl_dataset, dataset_index=dataset_index, frame_index=frame_index)
    # first_frame: (V,H,W,C), in [0,255]
    print("[DEBUG] single_generation gt. actions.shape:", actions.shape, "agent_pos.shape:", agent_pos.shape,
          "frames.shape:", frames.shape, "first_frame.shape:", first_frame.shape,
          "t5_embeddings.shape:", t5_embeddings.shape)
    '''
    actions.shape: (77, 2) 
    frames.shape: (77, 256, 256, 3) 
    first_frame.shape: (256, 256, 3)
    '''

    log.info(f"Running Video2WorldPipeline\ninput: {input_path}")

    sample_n_views = first_frame.shape[0]
    if autoregressive:
        log.info("Using autoregressive mode")
        video_chunks = []
        video_chunks2 = []
        for i in range(0, len(actions), chunk_size):
            frame_start = i
            frame_end = i + chunk_max_obs + chunk_size  # load all frames (including obs and gt)
            action_start = i  # + chunk_max_obs - 1
            action_end = action_start + chunk_size

            print("[DEBUG] Autoregressive chunk:", i, f"video:[{frame_start},{frame_end}), action:[{action_start},{action_end})", )
            if actions[action_start : action_end].shape[0] < chunk_size:
                log.info("Reached end of actions")
                break

            # v, H, W, C = frames[frame_start : frame_start + chunk_max_obs].shape
            # in_frames = np.concatenate(
            #     (frames[frame_start : frame_start + chunk_max_obs],
            #      np.zeros((chunk_size, H, W, C))), axis=0).astype(np.uint8)  # zero out gt frames
            v_cond_start = frame_start
            v_cond_end = v_cond_start + chunk_max_obs
            cond_frames = get_frames_from_multiview_video(
                frames[None, :], sample_n_views, start_idx=v_cond_start, end_idx=v_cond_end
            )[0]  # (V*chunk_max_obs,H,W,C)
            in_frames = cat_multiview_video_with_zeros(
                cond_frames[None, :], sample_n_views, zero_length=chunk_size
            )[0].astype(np.uint8) # (V*(chunk_max_obs+chunk_size),H,W,C), zero out gt frames
            print("[DEBUG] in_frames.shape:", in_frames.shape)

            save_image_or_video(
                (torch.from_numpy(in_frames).float().permute(3, 0, 1, 2) / 255.),  # (C,T,H,W) float32 in [0,1]
                f"output/in_video_view01_{i:02d}.mp4",
                fps=30
            )

            video, out_action = pipe(
                # first_frame,
                in_frames,  # T=v1+v2
                actions[action_start: action_end] * 0.,  # H=v2, zero out gt actions
                agent_pos[v_cond_start: v_cond_end],  # v1, robot states, as observation
                # np.zeros_like(actions[action_start : action_end]),  # chunk_size=(v1+v2)+a2, zero out gt actions
                prompt=t5_embeddings,  # (512,1024), in bf16
                num_conditional_frames=chunk_max_obs,
                num_conditional_actions=0, #chunk_size,  # use all actions as condition (chunk_size) or predict all actions (0)
                n_views=sample_n_views,
                guidance=guidance,
                seed=i,
            )  # video:(B,C,V*T,H,W), in [-1,1], action:(B,a1+a2,2), in [-1,1]
            # first_frame = ((video[0, :, -1].permute(1, 2, 0).cpu().numpy() / 2 + 0.5).clip(0, 1) * 255).astype(np.uint8)

            with torch.no_grad():
                # out_action = normalizer['action'].unnormalize(out_action)
                # vis_in_action = normalizer['action'].unnormalize(
                #     torch.from_numpy(actions[action_start : action_end]).unsqueeze(0)
                # ).numpy()
                out_action = out_action
                vis_in_action = torch.from_numpy(actions[action_start : action_end])

                out_action = tcl_dataset.denorm_action(out_action.cpu(),
                                                       norm_type=tcl_dataset.norm_action_type,
                                                       meta_data=tcl_dataset.dataset_meta_dict
                                                       )
                vis_in_action = tcl_dataset.denorm_action(vis_in_action.cpu(),
                                                          norm_type=tcl_dataset.norm_action_type,
                                                          meta_data=tcl_dataset.dataset_meta_dict
                                                          )

            print("[DEBUG] autoregressive video.shape:", video.shape, "out_action.shape:", out_action.shape)
            # video = video[:, :, :-1]  # (1,3,13-1,256,256)
            chunk_video_len_per_view = chunk_max_obs + chunk_size  # v1+v2
            save_image_or_video(video[0, :, :chunk_video_len_per_view], save_path=f"output/out_video_view0_{i:02d}.mp4", fps=4)
            save_image_or_video(video[0, :, chunk_video_len_per_view: chunk_video_len_per_view*2], save_path=f"output/out_video_view1_{i:02d}.mp4", fps=4)
            save_action_as_image(out_action[0, :, :3].cpu().numpy(), save_path=f"output/out_action_{i:02d}.png")
            save_action_as_image(vis_in_action[:, :3].cpu().numpy(), save_path=f"output/in_action_{i:02d}.png")
            video_chunks.append(video[:, :, :chunk_video_len_per_view])  # (B,3,v1+v2,H,W)
            video_chunks2.append(video[:, :, chunk_video_len_per_view:])  # (B,3,v1+v2,H,W)
        video = torch.cat([video_chunks[0]] + [chunk[:, :, chunk_max_obs:] for chunk in video_chunks[1:]], dim=2)
        video2 = torch.cat([video_chunks2[0]] + [chunk[:, :, chunk_max_obs:] for chunk in video_chunks2[1:]], dim=2)
        print("[DEBUG] Final concatenated video.shape:", video.shape,)
    else:
        video, _ = pipe(
            first_frame,
            actions[:chunk_size],
            num_conditional_frames=1,
            guidance=guidance,
            seed=seed,
        )
        video2 = video  # placeholder

    if video is not None:
        save_fps = 30  # use lower fps for clearer visualization

        # save the generated video
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        view0_output_path = output_path.replace(".mp4", "view0.mp4")
        log.info(f"Saving generated video to: {view0_output_path}")
        save_image_or_video(video, view0_output_path, fps=save_fps)
        log.success(f"Successfully saved video to: {view0_output_path}")

        save_image_or_video(video2, output_path.replace(".mp4", "view1.mp4"), fps=save_fps)

        # save the ground truth video for comparison
        gt_output_path = output_path.replace(".mp4", "_gt.mp4")
        save_image_or_video(torch.from_numpy(frames).permute(3, 0, 1, 2) / 255., gt_output_path, fps=save_fps)
        log.success(f"Successfully saved ground truth video to: {gt_output_path}")
        return True
    return False


def generate_video(args: argparse.Namespace, pipe: Video2WorldExpertPipeline) -> None:
    process_single_generation(
        pipe=pipe,
        input_path=args.input_video,
        output_path=args.save_path,
        guidance=args.guidance,
        seed=args.seed,
        chunk_size=args.chunk_size,
        autoregressive=args.autoregressive,
        dataset_index=args.dataset_index,
        frame_index=args.frame_index,
    )
    return


def cleanup_distributed():
    """Clean up the distributed environment if initialized."""
    if parallel_state.is_initialized():
        parallel_state.destroy_model_parallel()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    try:
        pipe = setup_pipeline(args)
        generate_video(args, pipe)
    finally:
        # Make sure to clean up the distributed environment
        cleanup_distributed()
