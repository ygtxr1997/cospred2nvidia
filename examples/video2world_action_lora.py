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

import mediapy as mp
import numpy as np

# Set TOKENIZERS_PARALLELISM environment variable to avoid deadlocks with multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from megatron.core import parallel_state

from cosmos_predict2.configs.action_conditioned.config import PREDICT2_VIDEO2WORLD_PIPELINE_2B_ACTION_CONDITIONED
from cosmos_predict2.pipelines.video2world_action import Video2WorldActionConditionedPipeline
from imaginaire.utils import distributed, log, misc
from imaginaire.utils.io import save_image_or_video


def get_action_sequence(annotation_path):
    with open(annotation_path, "r") as file:
        data = json.load(file)

    # rescale the action to the original scale
    action_ee = np.array(data["action"])[:, :6] * 20
    gripper = np.array(data["continuous_gripper_state"])[1:, None]

    # concatenate the end-effector displacement and gripper width
    action = np.concatenate([action_ee, gripper], axis=1)
    return action


def add_lora_to_model(
    model,
    lora_rank=16,
    lora_alpha=16,
    lora_target_modules="q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2",
    init_lora_weights=True,
):
    """
    Add LoRA to a model using PEFT library.
    Args:
        model: The model to add LoRA to
        lora_rank: Rank of the LoRA adaptation
        lora_alpha: Alpha parameter for LoRA
        lora_target_modules: Comma-separated list of target modules
        init_lora_weights: Whether to initialize LoRA weights
    """
    from peft import LoraConfig, inject_adapter_in_model

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        init_lora_weights=init_lora_weights,
        target_modules=lora_target_modules.split(","),
    )
    model = inject_adapter_in_model(lora_config, model)
    # Upcast LoRA parameters to fp32 for better stability
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.to(torch.float32)
    return model


def setup_lora_pipeline(config, dit_path, text_encoder_path, args):
    """
    Set up a pipeline with LoRA support.
    This function creates the pipeline, adds LoRA, then loads the checkpoint.
    """
    import numpy as np

    from cosmos_predict2.auxiliary.cosmos_reason1 import CosmosReason1
    from cosmos_predict2.auxiliary.text_encoder import CosmosT5TextEncoder
    from cosmos_predict2.models.utils import init_weights_on_device, load_state_dict
    from cosmos_predict2.module.denoiser_scaling import RectifiedFlowScaling
    from cosmos_predict2.schedulers.rectified_flow_scheduler import RectifiedFlowAB2Scheduler
    from imaginaire.lazy_config import instantiate
    from imaginaire.utils.ema import FastEmaModelUpdater

    # Create a pipe
    pipe = Video2WorldActionConditionedPipeline(device="cuda", torch_dtype=torch.bfloat16)
    pipe.config = config
    pipe.precision = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[config.precision]
    pipe.tensor_kwargs = {"device": "cuda", "dtype": pipe.precision}
    log.warning(f"precision {pipe.precision}")
    # 1. set data keys and data information
    pipe.sigma_data = config.sigma_data
    pipe.setup_data_key()
    # 2. setup up diffusion processing and scaling~(pre-condition)
    pipe.scheduler = RectifiedFlowAB2Scheduler(
        sigma_min=config.timestamps.t_min,
        sigma_max=config.timestamps.t_max,
        order=config.timestamps.order,
        t_scaling_factor=config.rectified_flow_t_scaling_factor,
    )
    pipe.scaling = RectifiedFlowScaling(
        pipe.sigma_data, config.rectified_flow_t_scaling_factor, config.rectified_flow_loss_weight_uniform
    )
    # 3. Set up tokenizer
    pipe.tokenizer = instantiate(config.tokenizer)
    assert (
        pipe.tokenizer.latent_ch == pipe.config.state_ch
    ), f"latent_ch {pipe.tokenizer.latent_ch} != state_shape {pipe.config.state_ch}"
    # 4. Load text encoder
    if text_encoder_path:
        # inference
        pipe.text_encoder = CosmosT5TextEncoder(device="cuda", cache_dir=text_encoder_path)
        pipe.text_encoder.to("cuda")
    else:
        # training
        pipe.text_encoder = None
    # 5. Initialize conditioner
    pipe.conditioner = instantiate(config.conditioner)
    assert (
        sum(p.numel() for p in pipe.conditioner.parameters() if p.requires_grad) == 0
    ), "conditioner should not have learnable parameters"
    # Load prompt refiner
    # pipe.prompt_refiner = CosmosReason1(
    #     checkpoint_dir=config.prompt_refiner_config.checkpoint_dir,
    #     offload_model_to_cpu=config.prompt_refiner_config.offload_model_to_cpu,
    #     enabled=config.prompt_refiner_config.enabled,
    # )
    if config.guardrail_config.enabled:
        from cosmos_predict2.auxiliary.guardrail.common import presets as guardrail_presets

        pipe.text_guardrail_runner = guardrail_presets.create_text_guardrail_runner(
            config.guardrail_config.checkpoint_dir, config.guardrail_config.offload_model_to_cpu
        )
        pipe.video_guardrail_runner = guardrail_presets.create_video_guardrail_runner(
            config.guardrail_config.checkpoint_dir, config.guardrail_config.offload_model_to_cpu
        )
    else:
        pipe.text_guardrail_runner = None
        pipe.video_guardrail_runner = None
    # 6. Set up DiT WITHOUT loading checkpoint first
    log.info("Initializing DiT model...")
    with init_weights_on_device():
        dit_config = config.net
        pipe.dit = instantiate(dit_config).eval()  # inference
    # 7. Add LoRA to the DiT model BEFORE loading checkpoint
    log.info("Adding LoRA to the DiT model...")
    log.info(
        f"LoRA parameters: rank={args.lora_rank}, alpha={args.lora_alpha}, target_modules={args.lora_target_modules}"
    )
    pipe.dit = add_lora_to_model(
        pipe.dit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_target_modules=args.lora_target_modules,
        init_lora_weights=args.init_lora_weights,
    )
    # 8. Handle EMA model if enabled
    if config.ema.enabled:
        log.info("Setting up EMA model...")
        pipe.dit_ema = instantiate(dit_config).eval()
        pipe.dit_ema.requires_grad_(False)
        # Add LoRA to EMA model
        log.info("Adding LoRA to the EMA DiT model...")
        pipe.dit_ema = add_lora_to_model(
            pipe.dit_ema,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_target_modules=args.lora_target_modules,
            init_lora_weights=args.init_lora_weights,
        )
        pipe.dit_ema_worker = FastEmaModelUpdater()  # default when not using FSDP
        s = config.ema.rate
        pipe.ema_exp_coefficient = np.roots([1, 7, 16 - s**-2, 12 - s**-2]).real.max()
        # copying is only necessary when starting the training at iteration 0.
        # Actual state_dict should be loaded after the pipe is created.
        pipe.dit_ema_worker.copy_to(src_model=pipe.dit, tgt_model=pipe.dit_ema)
    # 9. NOW load the LoRA checkpoint with strict=False
    if dit_path:
        log.info(f"Loading LoRA checkpoint from {dit_path}")
        state_dict = load_state_dict(dit_path)
        # Split state dict for regular and EMA models
        state_dict_dit_regular = dict()
        state_dict_dit_ema = dict()
        for k, v in state_dict.items():
            if k.startswith("net."):
                state_dict_dit_regular[k[4:]] = v
            elif k.startswith("net_ema."):
                state_dict_dit_ema[k[4:]] = v
        # Load regular model with strict=False to allow LoRA weights
        log.info("Loading regular DiT model weights...")
        missing_keys = pipe.dit.load_state_dict(state_dict_dit_regular, strict=False, assign=True)
        if missing_keys.missing_keys:
            log.warning(f"Missing keys in regular model: {missing_keys.missing_keys}")
        if missing_keys.unexpected_keys:
            log.warning(f"Unexpected keys in regular model: {missing_keys.unexpected_keys}")
        # Load EMA model if enabled
        if config.ema.enabled and state_dict_dit_ema:
            log.info("Loading EMA DiT model weights...")
            missing_keys_ema = pipe.dit_ema.load_state_dict(state_dict_dit_ema, strict=False, assign=True)
            if missing_keys_ema.missing_keys:
                log.warning(f"Missing keys in EMA model: {missing_keys_ema.missing_keys}")
            if missing_keys_ema.unexpected_keys:
                log.warning(f"Unexpected keys in EMA model: {missing_keys_ema.unexpected_keys}")
        del state_dict, state_dict_dit_regular, state_dict_dit_ema
        log.success(f"Successfully loaded LoRA checkpoint from {dit_path}")
    else:
        log.warning("No checkpoint path provided, using random weights")
    # 10. Move models to device
    pipe.dit = pipe.dit.to(device="cuda", dtype=torch.bfloat16)
    if config.ema.enabled:
        pipe.dit_ema = pipe.dit_ema.to(device="cuda", dtype=torch.bfloat16)
    torch.cuda.empty_cache()
    # 11. Set up training states
    if parallel_state.is_initialized():
        pipe.data_parallel_size = parallel_state.get_data_parallel_world_size()
    else:
        pipe.data_parallel_size = 1
    # Print parameter counts
    total_params = sum(p.numel() for p in pipe.dit.parameters())
    trainable_params = sum(p.numel() for p in pipe.dit.parameters() if p.requires_grad)
    log.info(f"Total parameters: {total_params:,}")
    log.info(f"Trainable LoRA parameters: {trainable_params:,}")
    log.info(f"LoRA parameter ratio: {trainable_params/total_params*100:.2f}%")
    return pipe


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
        "--frame_index",
        type=int,
        default=0,
        help="Index of the frame to use for conditioning (0 for first frame, 1 for second frame, etc.)",
    )

    # LoRA-specific arguments
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Enable LoRA inference mode",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,
        help="Rank of the LoRA adaptation",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="Alpha parameter for LoRA",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2",
        help="Comma-separated list of target modules for LoRA",
    )
    parser.add_argument(
        "--init_lora_weights",
        action="store_true",
        default=True,
        help="Whether to initialize LoRA weights",
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
        help="Number of frames to condition on (1 for single frame, 5 for multi-frame conditioning)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=12,
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
        config = PREDICT2_VIDEO2WORLD_PIPELINE_2B_ACTION_CONDITIONED
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

    # Load models - for LoRA, we need to handle this differently
    log.info(f"Initializing Video2WorldPipeline with model size: {args.model_size}")
    if args.use_lora:
        # For LoRA inference, we need to add LoRA before loading the checkpoint
        log.info("LoRA inference mode detected - using custom pipeline loading")
        pipe = setup_lora_pipeline(config, dit_path, text_encoder_path, args)
    else:
        pipe = Video2WorldActionConditionedPipeline.from_config(
            config=config,
            dit_path=dit_path,
            text_encoder_path=text_encoder_path,
            device="cuda",
            torch_dtype=torch.bfloat16,
            # load_ema_to_reg=args.load_ema,
            load_prompt_refiner=True,
        )

    return pipe


def read_first_frame(video_path, frame_index=0):
    video = mp.read_video(video_path)  # Returns (T, H, W, C) numpy array
    print("[DEBUG] Video shape:", video.shape)
    return video[frame_index]  # Return first frame as numpy array


def process_single_generation(
    pipe, input_path, input_annotation, output_path, guidance, seed, chunk_size, autoregressive,
    frame_index=0,
):
    actions = get_action_sequence(input_annotation)
    first_frame = read_first_frame(input_path, frame_index=frame_index)

    log.info(f"Running Video2WorldPipeline\ninput: {input_path}")

    if autoregressive:
        log.info("Using autoregressive mode")
        video_chunks = []
        for i in range(0, len(actions), chunk_size):
            if actions[i : i + chunk_size].shape[0] < chunk_size:
                log.info("Reached end of actions")
                break
            video = pipe(
                first_frame,
                actions[i : i + chunk_size],
                num_conditional_frames=1,
                guidance=guidance,
                seed=i,
            )
            first_frame = ((video[0, :, -1].permute(1, 2, 0).cpu().numpy() / 2 + 0.5).clip(0, 1) * 255).astype(np.uint8)
            video_chunks.append(video)
        video = torch.cat([video_chunks[0]] + [chunk[:, :, :-1] for chunk in video_chunks[1:]], dim=2)
    else:
        video = pipe(
            first_frame,
            actions[:chunk_size],
            num_conditional_frames=1,
            guidance=guidance,
            seed=seed,
        )

    if video is not None:
        # save the generated video
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        log.info(f"Saving generated video to: {output_path}")
        save_image_or_video(video, output_path, fps=4)
        log.success(f"Successfully saved video to: {output_path}")
        return True
    return False


def generate_video(args: argparse.Namespace, pipe: Video2WorldActionConditionedPipeline) -> None:
    process_single_generation(
        pipe=pipe,
        input_path=args.input_video,
        input_annotation=args.input_annotation,
        output_path=args.save_path,
        guidance=args.guidance,
        seed=args.seed,
        chunk_size=args.chunk_size,
        autoregressive=args.autoregressive,
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
