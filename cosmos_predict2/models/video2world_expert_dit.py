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

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange

from torchvision import transforms

from cosmos_predict2.conditioner import DataType
from cosmos_predict2.models.video2world_dit import MinimalV1LVGDiT
from cosmos_predict2.models.text2image_dit import VideoRopePosition3DEmb, LearnablePosEmbAxis
from imaginaire.utils.graph import create_cuda_graph
from imaginaire.utils import log


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.activation = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# Modified: models/video2world_action_dit.py ActionConditionedMinimalV1LVGDiT
class ExpertMinimalV1LVGDiT(MinimalV1LVGDiT):
    def __init__(self, *args, **kwargs):
        assert "action_dim" in kwargs, "action_dim must be provided"
        action_dim = kwargs["action_dim"]
        del kwargs["action_dim"]
        super().__init__(*args, **kwargs)

        out_features_dim = 4 * self.model_channels  # (T*D)
        self.action_embedder_B_D = Mlp(
            in_features=action_dim,
            hidden_features=self.model_channels,
            out_features=out_features_dim,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0,
        )
        # NOTE: action is no more taken as the crossattn_emb
        # self.action_embedder_B_3D = Mlp(
        #     in_features=action_dim,
        #     hidden_features=self.model_channels * 4,
        #     out_features=self.model_channels * 3,
        #     act_layer=lambda: nn.GELU(approximate="tanh"),
        #     drop=0,
        # )

    def forward(
        self,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        condition_video_input_mask_B_C_T_H_W: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        data_type: Optional[DataType] = DataType.VIDEO,
        use_cuda_graphs: bool = False,
        action: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor | List[torch.Tensor] | Tuple[torch.Tensor, List[torch.Tensor]]:
        del kwargs

        if data_type == DataType.VIDEO:
            x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, condition_video_input_mask_B_C_T_H_W.type_as(x_B_C_T_H_W)], dim=1)
        else:
            B, _, T, H, W = x_B_C_T_H_W.shape
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, torch.zeros((B, 1, T, H, W), dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device)], dim=1
            )
        # NOTE: project action to action embedding, action:(B,horizon,act_dim)
        assert action is not None, "action must be provided"
        B, C, T, _, _ = x_B_C_T_H_W.shape  # (B,16+1,4,32,32)
        action = rearrange(action, "b t d -> b 1 (t d)")
        action_emb_B_1_TWD = self.action_embedder_B_D(action)  # ->(B,1,T*D)
        action_emb_B_T_1_1_D = rearrange(
            action_emb_B_1_TWD, "b 1 (t d) -> b t 1 1 d",
            t=T,
        )
        # action_emb_B_3D = self.action_embedder_B_3D(action)

        assert isinstance(
            data_type, DataType
        ), f"Expected DataType, got {type(data_type)}. We need discuss this flag later."
        assert not (self.training and use_cuda_graphs), "CUDA Graphs are supported only for inference"
        x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = self.prepare_embedded_sequence(
            x_B_C_T_H_W,
            action_emb_B_T_1_1_D,  # NOTE: add action embedding as additional input
            fps=fps,
            padding_mask=padding_mask,
        )  # NOTE: from now on, x is (B,T,H+1,W,D)
        '''
        x_B_T_H_W_D.shape: torch.Size([12, 4, 17, 16, 2048]) 
        rope_emb_L_1_1_D.shape: torch.Size([1088, 1, 1, 128]) 
        crossattn_emb.shape: torch.Size([12, 512, 1024])
        '''

        if timesteps_B_T.ndim == 1:
            timesteps_B_T = timesteps_B_T.unsqueeze(1)
        t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)

        #### NOTE: original NVIDIA action conditioned implementation
        # # NOTE: add action embedding to the timestep embedding and adaln_lora
        # t_embedding_B_T_D = t_embedding_B_T_D + action_emb_B_D
        # adaln_lora_B_T_3D = adaln_lora_B_T_3D + action_emb_B_3D
        #### END

        t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

        # for logging purpose
        affline_scale_log_info = {}
        affline_scale_log_info["t_embedding_B_T_D"] = t_embedding_B_T_D.detach()
        self.affline_scale_log_info = affline_scale_log_info
        self.affline_emb = t_embedding_B_T_D
        self.crossattn_emb = crossattn_emb

        if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
            assert (
                x_B_T_H_W_D.shape == extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape
            ), f"{x_B_T_H_W_D.shape} != {extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape}"

        if use_cuda_graphs:
            shapes_key = create_cuda_graph(
                self.cuda_graphs,
                self.blocks,
                x_B_T_H_W_D,
                t_embedding_B_T_D,
                crossattn_emb,
                rope_emb_L_1_1_D,
                adaln_lora_B_T_3D,
                extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
            )
            blocks = self.cuda_graphs[shapes_key]
        else:
            blocks = self.blocks

        block_kwargs = {
            "rope_emb_L_1_1_D": rope_emb_L_1_1_D,
            "adaln_lora_B_T_3D": adaln_lora_B_T_3D,
            "extra_per_block_pos_emb": extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
        }
        for block in blocks:
            x_B_T_H_W_D = block(
                x_B_T_H_W_D,
                t_embedding_B_T_D,
                crossattn_emb,
                **block_kwargs,
            )

        # NOTE: split video and action tokens
        B, T, H, W, D = x_B_T_H_W_D.shape
        x_B_T_H_W_D, _action_emb = torch.split(
            x_B_T_H_W_D,
            [H - action_emb_B_T_1_1_D.shape[2], action_emb_B_T_1_1_D.shape[2]],
            dim=2
        )  # (B,T,H,W,D), (B,T,1,W,D)

        x_B_T_H_W_O = self.final_layer(x_B_T_H_W_D, t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
        x_B_C_Tt_Hp_Wp = self.unpatchify(x_B_T_H_W_O)
        return x_B_C_Tt_Hp_Wp

    def prepare_embedded_sequence(
        self,
        x_B_C_T_H_W: torch.Tensor,
        action_emb_B_T_1_1_D: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Prepares an embedded sequence tensor by applying positional embeddings and handling padding masks.

        Args:
            x_B_C_T_H_W (torch.Tensor): video
            action_emb_B_T_1_1_D (torch.Tensor): action embedding
            fps (Optional[torch.Tensor]): Frames per second tensor to be used for positional embedding when required.
                                    If None, a default value (`self.base_fps`) will be used.
            padding_mask (Optional[torch.Tensor]): current it is not used

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - A tensor of shape (B, T, H, W, D) with the embedded sequence.
                - An optional positional embedding tensor, returned only if the positional embedding class
                (`self.pos_emb_cls`) includes 'rope'. Otherwise, None.

        Notes:
            - If `self.concat_padding_mask` is True, a padding mask channel is concatenated to the input tensor.
            - The method of applying positional embeddings depends on the value of `self.pos_emb_cls`.
            - If 'rope' is in `self.pos_emb_cls` (case insensitive), the positional embeddings are generated using
                the `self.pos_embedder` with the shape [T, H, W].
            - If "fps_aware" is in `self.pos_emb_cls`, the positional embeddings are generated using the
            `self.pos_embedder` with the fps tensor.
            - Otherwise, the positional embeddings are generated without considering fps.
        """
        if self.concat_padding_mask:
            padding_mask = transforms.functional.resize(
                padding_mask, list(x_B_C_T_H_W.shape[-2:]), interpolation=transforms.InterpolationMode.NEAREST
            )
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, padding_mask.unsqueeze(1).repeat(1, 1, x_B_C_T_H_W.shape[2], 1, 1)], dim=1
            )
        x_B_T_H_W_D = self.x_embedder(x_B_C_T_H_W)  # ->(B,4,16,16,D)
        B, T, H, W, D = x_B_T_H_W_D.shape
        action_emb_B_T_1_W_D = action_emb_B_T_1_1_D.repeat(1, 1, 1, W, 1)  # ->(B,T,1,W,D)
        assert action_emb_B_T_1_W_D.shape == (B, T, 1, W, D), f"{action_emb_B_T_1_W_D.shape} != {(B, T, 1, W, D)}"

        if self.extra_per_block_abs_pos_emb:
            extra_pos_emb = self.extra_pos_embedder(x_B_T_H_W_D, fps=fps)
        else:
            extra_pos_emb = None

        if "rope" in self.pos_emb_cls.lower():
            assert hasattr(self, "expert_pos_embedder"), "pos_embedder not built"
            # NOTE: use expert_pos_embedder to process action embedding
            video_pos_emb_THW_1_1_D = self.pos_embedder(x_B_T_H_W_D, fps=fps)
            action_pos_emb_T1W_1_1_D = self.expert_pos_embedder(action_emb_B_T_1_W_D, fps=fps)

            def _cat_rope_emb(_video_emb_THW_1_1_D, _action_emb_T1W_1_1_D):
                # reshape for convenience using rearrange
                video_T_H_W_D = rearrange(_video_emb_THW_1_1_D, "(t h w) 1 1 d -> t h w d", t=T, h=H, w=W)
                action_T_1_W_D = rearrange(_action_emb_T1W_1_1_D, "(t w) 1 1 d -> t 1 w d", t=T, w=W)

                # cat at H-dim
                combined_T_H1_W_D = torch.cat([video_T_H_W_D, action_T_1_W_D], dim=1)

                # reshape back using rearrange
                return rearrange(combined_T_H1_W_D, "t h w d -> (t h w) 1 1 d")

            x_B_T_H_W_D = torch.cat([x_B_T_H_W_D, action_emb_B_T_1_W_D], dim=2)  # (B,T,H+1,W,D)
            cat_rope_emb_THW_1_1_D = _cat_rope_emb(video_pos_emb_THW_1_1_D, action_pos_emb_T1W_1_1_D)  # (T*(H+1)*W,1,1,D)

            return x_B_T_H_W_D, cat_rope_emb_THW_1_1_D, extra_pos_emb
            # return x_B_T_H_W_D, self.pos_embedder(x_B_T_H_W_D, fps=fps), extra_pos_emb
        x_B_T_H_W_D = x_B_T_H_W_D + self.pos_embedder(x_B_T_H_W_D)  # [B, T, H, W, D]

        return x_B_T_H_W_D, None, extra_pos_emb

    def build_pos_embed(self) -> None:
        if self.pos_emb_cls == "rope3d":
            cls_type = VideoRopePosition3DEmb
        else:
            raise ValueError(f"Unknown pos_emb_cls {self.pos_emb_cls}")

        log.debug(f"Building positional embedding with {self.pos_emb_cls} class, impl {cls_type}")
        kwargs = dict(
            model_channels=self.model_channels,
            len_h=self.max_img_h // self.patch_spatial,
            len_w=self.max_img_w // self.patch_spatial,
            len_t=self.max_frames // self.patch_temporal,
            max_fps=self.max_fps,
            min_fps=self.min_fps,
            is_learnable=self.pos_emb_learnable,
            interpolation=self.pos_emb_interpolation,
            head_dim=self.model_channels // self.num_heads,
            h_extrapolation_ratio=self.rope_h_extrapolation_ratio,
            w_extrapolation_ratio=self.rope_w_extrapolation_ratio,
            t_extrapolation_ratio=self.rope_t_extrapolation_ratio,
            enable_fps_modulation=self.rope_enable_fps_modulation,
        )
        self.pos_embedder = cls_type(
            **kwargs,  # type: ignore
        )

        # NOTE: add expert pos_embedder
        expert_kwargs = kwargs.copy()  # keep most settings the same
        expert_kwargs["len_h"] = 1
        self.expert_pos_embedder = cls_type(
            **expert_kwargs,  # type: ignore
        )

        if self.extra_per_block_abs_pos_emb:
            kwargs["h_extrapolation_ratio"] = self.extra_h_extrapolation_ratio
            kwargs["w_extrapolation_ratio"] = self.extra_w_extrapolation_ratio
            kwargs["t_extrapolation_ratio"] = self.extra_t_extrapolation_ratio
            self.extra_pos_embedder = LearnablePosEmbAxis(
                **kwargs,  # type: ignore
            )
