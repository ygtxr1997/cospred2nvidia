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

import attrs

from cosmos_predict2.conditioner import ActionConditioner, BooleanFlag, ReMapkey, TextAttr
from cosmos_predict2.configs.base.config_video2world import (
    ConditioningStrategy,
    CosmosGuardrailConfig,
    CosmosReason1Config,
    Video2WorldPipelineConfig,
)
from cosmos_predict2.configs.base.config_text2image import CosmosGuardrailConfig, SolverTimestampConfig
from cosmos_predict2.configs.base.defaults.ema import EMAConfig
from cosmos_predict2.models.text2image_dit import SACConfig
# from cosmos_predict2.models.video2world_action_dit import ActionConditionedMinimalV1LVGDiT
from cosmos_predict2.models.video2world_expert_dit import ExpertMinimalV1LVGDiT
from cosmos_predict2.tokenizers.tokenizer import TokenizerInterface
from imaginaire.config import make_freezable
from imaginaire.lazy_config import LazyCall as L
from imaginaire.lazy_config import LazyDict


@make_freezable
@attrs.define(slots=False)
class Video2WorldExpertPipelineConfig(Video2WorldPipelineConfig):
    # action related
    input_action_key: str = "action"
    input_agent_pos_key: str = "agent_pos"
    max_obs: int = 5
    max_act_out: int = 12
    p_all_actions_as_condition: float = 0.5


# Modified: action_conditioned/config.py PREDICT2_VIDEO2WORLD_NET_2B_ACTION_CONDITIONED
PREDICT2_VIDEO2WORLD_NET_2B_EXPERT = L(ExpertMinimalV1LVGDiT)(
    max_img_h=240,
    max_img_w=240,
    max_frames=128,
    in_channels=16,
    out_channels=16,
    patch_spatial=2,
    patch_temporal=1,
    concat_padding_mask=True,
    # attention settings
    model_channels=2048,
    num_blocks=28,
    num_heads=16,
    atten_backend="minimal_a2a",
    # positional embedding settings
    pos_emb_cls="rope3d",
    pos_emb_learnable=True,
    pos_emb_interpolation="crop",
    use_adaln_lora=True,
    adaln_lora_dim=256,
    rope_h_extrapolation_ratio=3.0,
    rope_w_extrapolation_ratio=3.0,
    rope_t_extrapolation_ratio=1.0,
    extra_per_block_abs_pos_emb=False,
    rope_enable_fps_modulation=False,
    sac_config=L(SACConfig)(
        every_n_blocks=1,
        mode="predict2_2b_720",
    ),
    # NOTE: add action dimension
    action_dim=2 * 12,  # (act_dim * horizon), ori:7*12
    # NOTE: add expert params
    action_dof=2,  # pusht: 2-DoF; robotic arm: 7-DoF;
    ex_num_latent_frames=12,  # (B,T,1,W,D), T=num_latent_frames, can be different from video, latent and ori action are consistent, good?
    ex_num_tokens_per_latent_frame=1,  # W=num_tokens_per_frame, embeddings repeat times
    ex_dim=512,  # expert feature dimension, D=dim
    ex_num_heads=16,
    ex_mlp_ratio=4.0,
    ex_adaln_lora_dim=128,
)

# Modified: action_conditioned/config.py PREDICT2_VIDEO2WORLD_PIPELINE_2B_ACTION_CONDITIONED
PREDICT2_VIDEO2WORLD_PIPELINE_2B_EXPERT = Video2WorldExpertPipelineConfig(
    adjust_video_noise=True,
    conditioner=L(ActionConditioner)(
        fps=L(ReMapkey)(
            dropout_rate=0.0,
            dtype=None,
            input_key="fps",
            output_key="fps",
        ),
        padding_mask=L(ReMapkey)(
            dropout_rate=0.0,
            dtype=None,
            input_key="padding_mask",
            output_key="padding_mask",
        ),
        text=L(TextAttr)(
            dropout_rate=0.2,
            input_key=["t5_text_embeddings"],
        ),
        use_video_condition=L(BooleanFlag)(
            dropout_rate=0.0,
            input_key="fps",
            output_key="use_video_condition",
        ),
        ## NOTE: unlike action_conditioned, expert takes action as additional input instead of crossattn_emb
        # NOTE: add additional action as condition
        action=L(ReMapkey)(
            input_key="action",
            output_key="action",
            dropout_rate=0.0,
            dtype=None,
        ),
    ),
    conditioning_strategy=str(ConditioningStrategy.FRAME_REPLACE),
    min_num_conditional_frames=1,
    max_num_conditional_frames=1,
    net=PREDICT2_VIDEO2WORLD_NET_2B_EXPERT,  # modified
    precision="bfloat16",
    rectified_flow_t_scaling_factor=1.0,
    rectified_flow_loss_weight_uniform=True,
    resize_online=True,
    resolution="720",  # ori:720
    ema=L(EMAConfig)(enabled=False),  # defaults to inference
    sigma_conditional=0.0001,
    sigma_data=1.0,
    state_ch=16,
    state_t=5,  # 5=2+3, ori:4
    text_encoder_class="T5",
    tokenizer=L(TokenizerInterface)(
        chunk_duration=81,
        load_mean_std=False,
        name="tokenizer",
        vae_pth="checkpoints/nvidia/Cosmos-Predict2-2B-Video2World/tokenizer/tokenizer.pth",
    ),
    # disable prompt refiner and guardrail for action conditional
    prompt_refiner_config=CosmosReason1Config(
        checkpoint_dir="checkpoints/nvidia/Cosmos-Reason1-7B",
        offload_model_to_cpu=True,
        enabled=False,
    ),
    guardrail_config=CosmosGuardrailConfig(
        checkpoint_dir="checkpoints/",
        offload_model_to_cpu=True,
        enabled=False,
    ),
    # action related
    max_obs=5,
    max_act_out=12,
    p_all_actions_as_condition=0.3,
)
