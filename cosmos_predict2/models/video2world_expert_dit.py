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
from typing import Mapping, Sequence, Union, Callable
import math

import torch
import torch.nn as nn
import transformer_engine as te
from einops import rearrange

from torchvision import transforms

from cosmos_predict2.conditioner import DataType
from cosmos_predict2.models.video2world_dit import MinimalV1LVGDiT
from cosmos_predict2.models.text2image_dit import VideoRopePosition3DEmb, LearnablePosEmbAxis
from cosmos_predict2.models.text2image_dit import (Block, Attention, VideoSize, GPT2FeedForward,
                                                   Timesteps, TimestepEmbedding)
from cosmos_predict2.models.text2image_dit import apply_rotary_pos_emb, NattenA2AAttnOp, NeighborhoodAttention
from imaginaire.utils.graph import create_cuda_graph
from imaginaire.utils import log


class AttentionWExpert(Attention):
    def __init__(
            self,
            # Parent class parameters
            query_dim: int,
            context_dim: Optional[int] = None,
            n_heads: int = 8,
            head_dim: int = 64,
            dropout: float = 0.0,
            qkv_format: str = "bshd",
            backend: str = "transformer_engine",
            natten_params: Optional[Mapping] = None,
            # Expert-specific parameters
            ex_query_dim: int = 512,
            ex_n_heads: int = 8,
            ex_head_dim: int = 64,
            ex_dropout: float = 0.0,
    ) -> None:
        super(AttentionWExpert, self).__init__(query_dim, context_dim, n_heads, head_dim, dropout,
                                               qkv_format, backend, natten_params)
        self.ex_dropout = ex_dropout

        is_self_attn = context_dim is None
        if is_self_attn:  # `ex_inner_dim` == `inner_dim`
            self.ex_n_heads = n_heads
            self.ex_head_dim = head_dim
            ex_context_dim = ex_query_dim
        else:
            self.ex_n_heads = ex_n_heads
            self.ex_head_dim = ex_head_dim
            ex_context_dim = context_dim
        ex_inner_dim = self.ex_head_dim * self.ex_n_heads

        self.ex_q_proj = nn.Linear(ex_query_dim, ex_inner_dim, bias=False)
        self.ex_q_norm = te.pytorch.RMSNorm(self.ex_head_dim, eps=1e-6)

        self.ex_k_proj = nn.Linear(ex_context_dim, ex_inner_dim, bias=False)
        self.ex_k_norm = te.pytorch.RMSNorm(self.ex_head_dim, eps=1e-6)

        self.ex_v_proj = nn.Linear(ex_context_dim, ex_inner_dim, bias=False)
        self.ex_v_norm = nn.Identity()

        self.ex_out_proj = nn.Linear(ex_inner_dim, ex_query_dim, bias=False)
        self.ex_output_dropout = nn.Dropout(ex_dropout) if ex_dropout > 1e-4 else nn.Identity()

        self._ex_query_dim = ex_query_dim
        self._ex_context_dim = ex_context_dim
        self._ex_inner_dim = ex_inner_dim

        self.init_weights()

    def init_weights(self) -> None:
        # Parent class init weights
        super().init_weights()

        # Expert init weights
        if hasattr(self, "ex_q_proj"):
            std = 1.0 / math.sqrt(self._ex_query_dim)
            torch.nn.init.trunc_normal_(self.ex_q_proj.weight, std=std, a=-3 * std, b=3 * std)
            std = 1.0 / math.sqrt(self._ex_context_dim)
            torch.nn.init.trunc_normal_(self.ex_k_proj.weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.trunc_normal_(self.ex_v_proj.weight, std=std, a=-3 * std, b=3 * std)

            std = 1.0 / math.sqrt(self._ex_inner_dim)
            torch.nn.init.trunc_normal_(self.ex_out_proj.weight, std=std, a=-3 * std, b=3 * std)

            for layer in self.ex_q_norm, self.ex_k_norm, self.ex_v_norm:
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()

    def compute_qkv(
            self,
            x: torch.Tensor,
            context: Optional[torch.Tensor] = None,
            rope_emb: Optional[torch.Tensor] = None,
            # Expert-specific
            ex_input: Optional[torch.Tensor] = None,
            ex_rope_emb: Optional[torch.Tensor] = None,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        # Branch [1]: Video
        q = self.q_proj(x)
        is_self_attn = context is None
        context = x if is_self_attn else context
        k = self.k_proj(context)
        v = self.v_proj(context)
        q, k, v = map(
            lambda t: rearrange(t, "b ... (h d) -> b ... h d", h=self.n_heads, d=self.head_dim),
            (q, k, v),
        )

        # Branch [2]: Expert
        q_ex = self.ex_q_proj(ex_input)
        context_ex = ex_input if is_self_attn else context
        k_ex = self.ex_k_proj(context_ex)
        v_ex = self.ex_v_proj(context_ex)
        q_ex, k_ex, v_ex = map(
            lambda t: rearrange(t, "b ... (h d) -> b ... h d", h=self.ex_n_heads, d=self.ex_head_dim),
            (q_ex, k_ex, v_ex),
        )

        def apply_norm_and_rotary_pos_emb(
            q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
            _q_norm: Callable, _k_norm: Callable, _v_norm: Callable,
            rope_emb: Optional[torch.Tensor]
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            q = _q_norm(q)
            k = _k_norm(k)
            v = _v_norm(v)
            if self.is_selfattn and rope_emb is not None:  # only apply to self-attention!
                q = apply_rotary_pos_emb(q, rope_emb, tensor_format=self.qkv_format, fused=True)
                k = apply_rotary_pos_emb(k, rope_emb, tensor_format=self.qkv_format, fused=True)
            return q, k, v

        # Branch [1]: Video
        q, k, v = apply_norm_and_rotary_pos_emb(q, k, v, self.q_norm, self.k_norm, self.v_norm, rope_emb)
        # Branch [2]: Expert
        q_ex, k_ex, v_ex = apply_norm_and_rotary_pos_emb(q_ex, k_ex, v_ex,
                                                         self.ex_q_norm, self.ex_k_norm, self.ex_v_norm, ex_rope_emb)

        return (q, k, v), (q_ex, k_ex, v_ex)

    def compute_attention(
            self,  q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
            video_size: Optional[VideoSize] = None,
            q_ex: Optional[torch.Tensor] = None,
            k_ex: Optional[torch.Tensor] = None,
            v_ex: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        additional_args = {}
        if isinstance(self.attn_op, (NattenA2AAttnOp, NeighborhoodAttention)):
            additional_args["video_size"] = video_size

        # Check shapes
        if self.is_selfattn and q_ex is not None and k_ex is not None and v_ex is not None:
            # (B,num_tokens,h,d)
            assert (q.shape[0] == q_ex.shape[0] and q.shape[-2:] == q_ex.shape[-2:]
                    and k.shape[0] == k_ex.shape[0] and k.shape[-2:] == k_ex.shape[-2:]
                    and v.shape[0] == v_ex.shape[0] and v.shape[-2:] == v_ex.shape[-2:]), \
                f"SA shape: {q.shape}, {k.shape}, {v.shape} != {q_ex.shape}, {k_ex.shape}, {v_ex.shape}"
            # Branch [1] and [2]: Video and Expert
            ori_len, expert_len = q.shape[1], q_ex.shape[1]
            q = torch.cat((q, q_ex), dim=1)
            k = torch.cat((k, k_ex), dim=1)
            v = torch.cat((v, v_ex), dim=1)
            result = self.attn_op(q, k, v, **additional_args)  # (B,S1+S2,H,D)
            result_ori, result_expert = torch.split(result, [ori_len, expert_len], dim=1)
        elif not self.is_selfattn and q_ex is not None and k_ex is not None and v_ex is not None:
            assert q.shape[-2:] == k.shape[-2:], f"CA shape: {q.shape} != {k.shape}, {v.shape}"
            assert q_ex.shape[-2:] == k_ex.shape[-2:], f"CA shape: {q_ex.shape} != {k_ex.shape}, {v_ex.shape}"
            result_ori = self.attn_op(q, k, v, **additional_args)  # (B,S1,H,D)
            result_expert = self.attn_op(q_ex, k_ex, v_ex, **additional_args)  # (B,S2,H,D)
        else:
            assert q_ex is None and k_ex is None and v_ex is None, "q_ex, k_ex, v_ex should be all None or not None"
            result_ori = self.attn_op(q, k, v, **additional_args)  # (B,S1,H,D)
            result_expert = None

        # Branch [1]: Video
        result_ori = self.output_dropout(self.output_proj(result_ori))
        # Branch [2]: Expert
        result_expert = self.ex_output_dropout(
            self.ex_out_proj(result_expert)) if result_expert is not None else None
        return result_ori, result_expert

    def forward(
            self,
            x: torch.Tensor,
            context: Optional[torch.Tensor] = None,
            rope_emb: Optional[torch.Tensor] = None,
            video_size: Optional[VideoSize] = None,
            # Expert-specific
            ex_input: Optional[torch.Tensor] = None,
            ex_rope_emb: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # In Self-Attn: context will be set as [x, ex_input]
        # In Cross-Attn: context will be set as [context (e.g. text embeddings)]
        (q, k, v), (q_ex, k_ex, v_ex) = self.compute_qkv(x, context, rope_emb=rope_emb,
                                                         ex_input=ex_input, ex_rope_emb=ex_rope_emb)
        attn_ori, attn_expert = self.compute_attention(q, k, v, video_size=video_size,
                                                       q_ex=q_ex, k_ex=k_ex, v_ex=v_ex)
        return attn_ori, attn_expert


class BlockWExpert(Block):
    def __init__(
            self,
            # Parent class parameters
            x_dim: int,
            context_dim: int,
            num_heads: int,
            mlp_ratio: float = 4.0,
            use_adaln_lora: bool = False,
            adaln_lora_dim: int = 256,
            self_attention_backend: str = "transformer_engine",
            cross_attention_backend: str = "transformer_engine",
            natten_params: Optional[Mapping] = None,
            # Expert-specific parameters
            ex_dim: int = 256,  # expert_dim can be different from x_dim
            ex_num_heads: int = 16,
            ex_mlp_ratio: float = 4.0,
            ex_adaln_lora_dim: int = 64,
    ):
        super().__init__(x_dim, context_dim, num_heads, mlp_ratio, use_adaln_lora,
                         adaln_lora_dim, self_attention_backend, cross_attention_backend, natten_params)
        # Replace original self/cross-attention with AttentionWExpert
        self.self_attn = AttentionWExpert(
            x_dim,
            None,
            num_heads,
            x_dim // num_heads,
            qkv_format="bshd",
            backend=self_attention_backend,
            natten_params=natten_params,
            # Expert-specific
            ex_query_dim=ex_dim,
            ex_n_heads=ex_num_heads,
            ex_head_dim=ex_dim // ex_num_heads,
        )

        self.cross_attn = AttentionWExpert(
            x_dim,
            context_dim=context_dim,
            n_heads=num_heads,
            head_dim=x_dim // num_heads,
            qkv_format="bshd",
            backend=cross_attention_backend,
            # Expert-specific
            ex_query_dim=ex_dim,
            ex_n_heads=ex_num_heads,
            ex_head_dim=ex_dim // ex_num_heads,
        )

        # Other expert-specific modules: LayerNorm, adaln, MLP
        self.ex_dim = ex_dim
        self.ex_num_heads = ex_num_heads
        self.ex_mlp_ratio = ex_mlp_ratio

        self.ex_layer_norm_self_attn = nn.LayerNorm(ex_dim, elementwise_affine=False, eps=1e-6)  # no params
        self.ex_layer_norm_cross_attn = nn.LayerNorm(ex_dim, elementwise_affine=False, eps=1e-6)  # no params
        self.ex_layer_norm_mlp = nn.LayerNorm(ex_dim, elementwise_affine=False, eps=1e-6)

        self.ex_mlp = GPT2FeedForward(ex_dim, int(ex_dim * ex_mlp_ratio))

        if self.use_adaln_lora:
            self.ex_adaln_modulation_self_attn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(ex_dim, ex_adaln_lora_dim, bias=False),  # expert shares the same t_embedding with ori
                nn.Linear(ex_adaln_lora_dim, 3 * ex_dim, bias=False),
            )
            self.ex_adaln_modulation_cross_attn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(ex_dim, ex_adaln_lora_dim, bias=False),
                nn.Linear(ex_adaln_lora_dim, 3 * ex_dim, bias=False),
            )
            self.ex_adaln_modulation_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(ex_dim, ex_adaln_lora_dim, bias=False),
                nn.Linear(ex_adaln_lora_dim, 3 * ex_dim, bias=False),
            )
        else:
            self.ex_adaln_modulation_self_attn = nn.Sequential(nn.SiLU(), nn.Linear(ex_dim, 3 * ex_dim, bias=False))
            self.ex_adaln_modulation_cross_attn = nn.Sequential(nn.SiLU(), nn.Linear(ex_dim, 3 * ex_dim, bias=False))
            self.ex_adaln_modulation_mlp = nn.Sequential(nn.SiLU(), nn.Linear(ex_dim, 3 * ex_dim, bias=False))

    def reset_parameters(self) -> None:
        super().reset_parameters()

        self.ex_layer_norm_self_attn.reset_parameters()
        self.ex_layer_norm_cross_attn.reset_parameters()
        self.ex_layer_norm_mlp.reset_parameters()

        if self.use_adaln_lora:
            std = 1.0 / math.sqrt(self.ex_dim)
            torch.nn.init.trunc_normal_(self.ex_adaln_modulation_self_attn[1].weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.trunc_normal_(self.ex_adaln_modulation_cross_attn[1].weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.trunc_normal_(self.ex_adaln_modulation_mlp[1].weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.zeros_(self.ex_adaln_modulation_self_attn[2].weight)
            torch.nn.init.zeros_(self.ex_adaln_modulation_cross_attn[2].weight)
            torch.nn.init.zeros_(self.ex_adaln_modulation_mlp[2].weight)
        else:
            torch.nn.init.zeros_(self.ex_adaln_modulation_self_attn[1].weight)
            torch.nn.init.zeros_(self.ex_adaln_modulation_cross_attn[1].weight)
            torch.nn.init.zeros_(self.ex_adaln_modulation_mlp[1].weight)

    def init_weights(self) -> None:
        self.reset_parameters()  # reset expert LayerNorm and adaln
        self.self_attn.init_weights()  # AttentionWExpert
        self.cross_attn.init_weights()  # AttentionWExpert
        self.mlp.init_weights()
        self.ex_mlp.init_weights()  # init expert MLP

    def forward(
            self,
            x_B_T_H_W_D: torch.Tensor,  # (B,4,16,16,2048)
            emb_B_T_D: torch.Tensor,
            crossattn_emb: torch.Tensor,
            rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
            adaln_lora_B_T_3D: Optional[torch.Tensor] = None,
            extra_per_block_pos_emb: Optional[torch.Tensor] = None,
            # Expert-specific
            ex_B_T_1_W_D: Optional[torch.Tensor] = None,  # (B,4,1,8,512)
            ex_rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
            ex_t_embedding_B_T_D: Optional[torch.Tensor] = None,
            ex_adaln_lora_B_T_3D: Optional[torch.Tensor] = None,
    ):
        #### <<<< Copied from parent class <<<< ####
        if extra_per_block_pos_emb is not None:
            x_B_T_H_W_D = x_B_T_H_W_D + extra_per_block_pos_emb

        if self.use_adaln_lora:
            shift_self_attn_B_T_D, scale_self_attn_B_T_D, gate_self_attn_B_T_D = (
                    self.adaln_modulation_self_attn(emb_B_T_D) + adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
            shift_cross_attn_B_T_D, scale_cross_attn_B_T_D, gate_cross_attn_B_T_D = (
                    self.adaln_modulation_cross_attn(emb_B_T_D) + adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
            shift_mlp_B_T_D, scale_mlp_B_T_D, gate_mlp_B_T_D = (
                    self.adaln_modulation_mlp(emb_B_T_D) + adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
        else:
            shift_self_attn_B_T_D, scale_self_attn_B_T_D, gate_self_attn_B_T_D = self.adaln_modulation_self_attn(
                emb_B_T_D
            ).chunk(3, dim=-1)
            shift_cross_attn_B_T_D, scale_cross_attn_B_T_D, gate_cross_attn_B_T_D = self.adaln_modulation_cross_attn(
                emb_B_T_D
            ).chunk(3, dim=-1)
            shift_mlp_B_T_D, scale_mlp_B_T_D, gate_mlp_B_T_D = self.adaln_modulation_mlp(
                emb_B_T_D).chunk(3, dim=-1)

        # Reshape tensors from (B, T, D) to (B, T, 1, 1, D) for broadcasting
        shift_self_attn_B_T_1_1_D = rearrange(shift_self_attn_B_T_D, "b t d -> b t 1 1 d")
        scale_self_attn_B_T_1_1_D = rearrange(scale_self_attn_B_T_D, "b t d -> b t 1 1 d")
        gate_self_attn_B_T_1_1_D = rearrange(gate_self_attn_B_T_D, "b t d -> b t 1 1 d")

        shift_cross_attn_B_T_1_1_D = rearrange(shift_cross_attn_B_T_D, "b t d -> b t 1 1 d")
        scale_cross_attn_B_T_1_1_D = rearrange(scale_cross_attn_B_T_D, "b t d -> b t 1 1 d")
        gate_cross_attn_B_T_1_1_D = rearrange(gate_cross_attn_B_T_D, "b t d -> b t 1 1 d")

        shift_mlp_B_T_1_1_D = rearrange(shift_mlp_B_T_D, "b t d -> b t 1 1 d")
        scale_mlp_B_T_1_1_D = rearrange(scale_mlp_B_T_D, "b t d -> b t 1 1 d")
        gate_mlp_B_T_1_1_D = rearrange(gate_mlp_B_T_D, "b t d -> b t 1 1 d")
        #### >>>>>>>>> End >>>>>>>>>> ####


        #### <<<< Branch [2]: Expert adaln <<<< ####
        if self.use_adaln_lora:
            ex_shift_self_attn_B_T_D, ex_scale_self_attn_B_T_D, ex_gate_self_attn_B_T_D = (
                    self.ex_adaln_modulation_self_attn(ex_t_embedding_B_T_D) + ex_adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
            ex_shift_cross_attn_B_T_D, ex_scale_cross_attn_B_T_D, ex_gate_cross_attn_B_T_D = (
                    self.ex_adaln_modulation_cross_attn(ex_t_embedding_B_T_D) + ex_adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
            ex_shift_mlp_B_T_D, ex_scale_mlp_B_T_D, ex_gate_mlp_B_T_D = (
                    self.ex_adaln_modulation_mlp(ex_t_embedding_B_T_D) + ex_adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
        else:
            ex_shift_self_attn_B_T_D, ex_scale_self_attn_B_T_D, ex_gate_self_attn_B_T_D = (
                self.ex_adaln_modulation_self_attn(
                ex_t_embedding_B_T_D
            ).chunk(3, dim=-1))
            ex_shift_cross_attn_B_T_D, ex_scale_cross_attn_B_T_D, ex_gate_cross_attn_B_T_D = (
                self.ex_adaln_modulation_cross_attn(
                ex_t_embedding_B_T_D
            ).chunk(3, dim=-1))
            ex_shift_mlp_B_T_D, ex_scale_mlp_B_T_D, ex_gate_mlp_B_T_D = (
                self.ex_adaln_modulation_mlp(ex_t_embedding_B_T_D).chunk(3, dim=-1))

        # Reshape tensors from (B, T, D) to (B, T, 1, 1, D) for broadcasting
        ex_shift_self_attn_B_T_1_1_D = rearrange(ex_shift_self_attn_B_T_D, "b t d -> b t 1 1 d")
        ex_scale_self_attn_B_T_1_1_D = rearrange(ex_scale_self_attn_B_T_D, "b t d -> b t 1 1 d")
        ex_gate_self_attn_B_T_1_1_D = rearrange(ex_gate_self_attn_B_T_D, "b t d -> b t 1 1 d")

        ex_shift_cross_attn_B_T_1_1_D = rearrange(ex_shift_cross_attn_B_T_D, "b t d -> b t 1 1 d")
        ex_scale_cross_attn_B_T_1_1_D = rearrange(ex_scale_cross_attn_B_T_D, "b t d -> b t 1 1 d")
        ex_gate_cross_attn_B_T_1_1_D = rearrange(ex_gate_cross_attn_B_T_D, "b t d -> b t 1 1 d")

        ex_shift_mlp_B_T_1_1_D = rearrange(ex_shift_mlp_B_T_D, "b t d -> b t 1 1 d")
        ex_scale_mlp_B_T_1_1_D = rearrange(ex_scale_mlp_B_T_D, "b t d -> b t 1 1 d")
        ex_gate_mlp_B_T_1_1_D = rearrange(ex_gate_mlp_B_T_D, "b t d -> b t 1 1 d")

        B, T_ex, H_ex, W_ex, D_ex = ex_B_T_1_W_D.shape
        B, T,    H,    W,    D    = x_B_T_H_W_D.shape
        #### >>>>>>>>> End >>>>>>>>>> ####


        #### <<<< Copied from parent class <<<< ####
        # (1) Self-Attention
        def _fn(_x_B_T_H_W_D, _norm_layer, _scale_B_T_1_1_D, _shift_B_T_1_1_D):
            return _norm_layer(_x_B_T_H_W_D) * (1 + _scale_B_T_1_1_D) + _shift_B_T_1_1_D

        # (1.1) LayerNorm + AdaLN
        normalized_x_B_T_H_W_D = _fn(
            x_B_T_H_W_D,
            self.layer_norm_self_attn,
            scale_self_attn_B_T_1_1_D,
            shift_self_attn_B_T_1_1_D,
        )
        #### >>>>>>>>> End >>>>>>>>>> ####


        #### <<<< Branch [2]: Expert Self-Attention, Cross-Attention <<<< ####
        # [2](1.1) LayerNorm + AdaLN
        ex_normalized_B_T_H_W_D = _fn(
            ex_B_T_1_W_D,
            self.ex_layer_norm_self_attn,
            ex_scale_self_attn_B_T_1_1_D,
            ex_shift_self_attn_B_T_1_1_D,
        )

        video_size = VideoSize(T=T, H=H, W=W)

        if self.cp_size is not None and self.cp_size > 1:
            video_size = VideoSize(T=T * self.cp_size, H=H, W=W)

        # [2](1.2) Self-Attention + RoPE + AdaLN
        result_B_S_D, ex_result_B_S_D = self.self_attn.forward(
            rearrange(normalized_x_B_T_H_W_D, "b t h w d -> b (t h w) d"),
            None,
            rope_emb=rope_emb_L_1_1_D,
            video_size=video_size,
            # Expert-specific
            ex_input=rearrange(ex_normalized_B_T_H_W_D, "b t 1 w d -> b (t 1 w) d"),
            ex_rope_emb=ex_rope_emb_L_1_1_D,
        )
        result_B_T_H_W_D = rearrange(result_B_S_D, "b (t h w) d -> b t h w d", t=T, h=H, w=W)
        ex_result_B_T_1_W_D = rearrange(ex_result_B_S_D, "b (t 1 w) d -> b t 1 w d", t=T_ex, w=W_ex)

        x_B_T_H_W_D = x_B_T_H_W_D + gate_self_attn_B_T_1_1_D * result_B_T_H_W_D
        ex_B_T_1_W_D = ex_B_T_1_W_D + ex_gate_self_attn_B_T_1_1_D * ex_result_B_T_1_W_D

        # [2](2) Cross-Attention
        def _x_fn(
            _x_B_T_H_W_D: torch.Tensor,
            layer_norm_cross_attn: Callable,
            _scale_cross_attn_B_T_1_1_D: torch.Tensor,
            _shift_cross_attn_B_T_1_1_D: torch.Tensor,
            # Expert-specific
            _ex_B_T_1_W_D: torch.Tensor,
            _ex_layer_norm_cross_attn: Callable,
            _ex_scale_cross_attn_B_T_1_1_D: torch.Tensor,
            _ex_shift_cross_attn_B_T_1_1_D: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            _normalized_x_B_T_H_W_D = _fn(
                _x_B_T_H_W_D, layer_norm_cross_attn, _scale_cross_attn_B_T_1_1_D, _shift_cross_attn_B_T_1_1_D
            )
            _normalized_ex_B_T_1_W_D = _fn(
                _ex_B_T_1_W_D, _ex_layer_norm_cross_attn, _ex_scale_cross_attn_B_T_1_1_D, _ex_shift_cross_attn_B_T_1_1_D
            )

            _result_B_S_D, _ex_result_B_S_D = self.cross_attn.forward(
                rearrange(_normalized_x_B_T_H_W_D, "b t h w d -> b (t h w) d"),
                crossattn_emb,
                rope_emb=rope_emb_L_1_1_D,
                # Expert-specific
                ex_input=rearrange(_normalized_ex_B_T_1_W_D, "b t 1 w d -> b (t 1 w) d"),
                ex_rope_emb=ex_rope_emb_L_1_1_D,
            )
            _result_B_T_H_W_D = rearrange(_result_B_S_D, "b (t h w) d -> b t h w d", t=T, h=H, w=W)
            _ex_result_B_T_1_W_D = rearrange(_ex_result_B_S_D, "b (t 1 w) d -> b t 1 w d", t=T_ex, w=W_ex)
            return _result_B_T_H_W_D, _ex_result_B_T_1_W_D

        # [2](2.1) LayerNorm + AdaLN
        # [2](2.2) Cross-Attention + AdaLN
        result_B_T_H_W_D, ex_result_B_T_1_W_D = _x_fn(
            x_B_T_H_W_D,
            self.layer_norm_cross_attn,
            scale_cross_attn_B_T_1_1_D,
            shift_cross_attn_B_T_1_1_D,
            # Expert-specific
            ex_B_T_1_W_D,
            self.ex_layer_norm_cross_attn,
            ex_scale_cross_attn_B_T_1_1_D,
            ex_shift_cross_attn_B_T_1_1_D,
        )
        x_B_T_H_W_D = result_B_T_H_W_D * gate_cross_attn_B_T_1_1_D + x_B_T_H_W_D
        ex_B_T_1_W_D = ex_result_B_T_1_W_D * ex_gate_cross_attn_B_T_1_1_D + ex_B_T_1_W_D
        #### >>>>>>>>> End >>>>>>>>>> ####


        #### <<<< Copied from parent class <<<< ####
        # (3) MLP
        # (3.1) LayerNorm + AdaLN
        normalized_x_B_T_H_W_D = _fn(
            x_B_T_H_W_D,
            self.layer_norm_mlp,
            scale_mlp_B_T_1_1_D,
            shift_mlp_B_T_1_1_D,
        )
        # (3.2) MLP + AdaLN
        result_B_T_H_W_D = self.mlp(normalized_x_B_T_H_W_D)
        x_B_T_H_W_D = x_B_T_H_W_D + gate_mlp_B_T_1_1_D * result_B_T_H_W_D
        #### >>>>>>>>> End >>>>>>>>>> ####


        #### <<<< Branch [2]: Expert MLP <<<< ####
        # [2](3) MLP
        # [2](3.1) LayerNorm + AdaLN
        ex_normalized_x_B_T_1_W_D = _fn(
            ex_B_T_1_W_D,
            self.ex_layer_norm_mlp,
            ex_scale_mlp_B_T_1_1_D,
            ex_shift_mlp_B_T_1_1_D,
        )
        # [2](3.2) MLP + AdaLN
        ex_result_B_T_1_W_D = self.ex_mlp(ex_normalized_x_B_T_1_W_D)
        ex_B_T_1_W_D = ex_B_T_1_W_D + ex_gate_mlp_B_T_1_1_D * ex_result_B_T_1_W_D
        return x_B_T_H_W_D, ex_B_T_1_W_D  # original and expert outputs
        #### >>>>>>>>> End >>>>>>>>>> ####


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

    def init_weights(self) -> None:
        # Initialize weights
        std = 1.0 / math.sqrt(self.fc1.in_features)
        torch.nn.init.trunc_normal_(self.fc1.weight, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.zeros_(self.fc1.bias)

        std = 1.0 / math.sqrt(self.fc2.in_features)
        torch.nn.init.trunc_normal_(self.fc2.weight, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.zeros_(self.fc2.bias)


# Modified: models/video2world_action_dit.py ActionConditionedMinimalV1LVGDiT
class ExpertMinimalV1LVGDiT(MinimalV1LVGDiT):
    def __init__(self, *args, **kwargs):
        assert "action_dim" in kwargs, "action_dim must be provided"
        action_dim = kwargs["action_dim"]
        del kwargs["action_dim"]

        # Default parameters
        if kwargs.get("mlp_ratio") is None:
            kwargs["mlp_ratio"] = 4.0
        if kwargs.get("crossattn_emb_channels") is None:
            kwargs["crossattn_emb_channels"] = 1024

        # Expert-specific parameters
        assert "ex_dim" in kwargs, "ex_dim must be provided"
        action_dof = kwargs["action_dof"]
        ex_num_latent_frames = int(kwargs["ex_num_latent_frames"])
        ex_num_tokens_per_latent_frame = int(kwargs["ex_num_tokens_per_latent_frame"])
        ex_dim = kwargs["ex_dim"]
        ex_num_heads = kwargs["ex_num_heads"]
        ex_mlp_ratio = kwargs["ex_mlp_ratio"]
        ex_adaln_lora_dim = kwargs["ex_adaln_lora_dim"]
        del kwargs["action_dof"], kwargs['ex_num_latent_frames'], kwargs["ex_num_tokens_per_latent_frame"], (
            kwargs)["ex_dim"], (kwargs)["ex_num_heads"], kwargs["ex_mlp_ratio"], kwargs["ex_adaln_lora_dim"]

        super().__init__(*args, **kwargs)

        # Add action encoder
        self.action_t_embedder = nn.Sequential(
            Timesteps(ex_dim),
            TimestepEmbedding(ex_dim, ex_dim, use_adaln_lora=kwargs["use_adaln_lora"]),
        )
        self.action_embedder_B_D = Mlp(
            in_features=action_dim,
            hidden_features=ex_dim * 4,
            out_features=ex_dim * ex_num_latent_frames,
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

        # Replace Blocks with BlockWExpert
        num_blocks = len(self.blocks)
        self.blocks = nn.ModuleList(
            [
                BlockWExpert(
                    x_dim=self.model_channels,
                    context_dim=kwargs["crossattn_emb_channels"],
                    num_heads=kwargs["num_heads"],
                    mlp_ratio=kwargs["mlp_ratio"],
                    use_adaln_lora=kwargs["use_adaln_lora"],
                    adaln_lora_dim=kwargs["adaln_lora_dim"],
                    self_attention_backend=kwargs["atten_backend"]
                    if kwargs.get("natten_parameters") is None or kwargs["natten_parameters"] is None
                    or kwargs["natten_parameters"][i] is None else "natten",
                    cross_attention_backend=kwargs["atten_backend"],
                    natten_params=None if kwargs.get("natten_parameters") is None
                    or kwargs["natten_parameters"] is None else kwargs["natten_parameters"][i],
                    # Expert-specific
                    ex_dim=ex_dim,
                    ex_num_heads=ex_num_heads,
                    ex_mlp_ratio=ex_mlp_ratio,
                    ex_adaln_lora_dim=ex_adaln_lora_dim,
                )
                for i in range(num_blocks)
            ]
        )

        # Add action decoder
        self.action_decoder_B_D = Mlp(
            in_features=ex_dim * 4 * ex_num_tokens_per_latent_frame,
            hidden_features=action_dim * 4,  # can be simpler
            out_features=action_dim,  # action_dim = horizon * action_dof
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0,
        )
        self.action_dof = action_dof
        self.ex_num_tokens_per_latent_frame = ex_num_tokens_per_latent_frame
        self.action_reshape = lambda act_B_TD: rearrange(
            act_B_TD, "b (t d) -> b t d", d=action_dof)

        # Initialize weights once again to include new modules
        self.init_weights()
        self.count_parameters()

    def init_weights(self) -> None:
        super().init_weights()  # including blocks
        if hasattr(self, "action_decoder_B_D"):
            self.action_t_embedder[1].init_weights()
            self.action_embedder_B_D.init_weights()
            self.action_decoder_B_D.init_weights()

    def count_parameters(self) -> int:
        total_params = 0
        trainable_params = 0

        expert_params = 0
        base_params = 0
        action_embedder_params = 0
        action_decoder_params = 0

        for name, param in self.named_parameters():
            param_count = param.numel()
            total_params += param_count

            if param.requires_grad:
                trainable_params += param_count

            if 'ex_' in name or 'expert' in name or 'action' in name:
                expert_params += param.numel()
                if 'action_embedder_B_D' in name:
                    action_embedder_params += param.numel()
                if 'action_decoder_B_D' in name:
                    action_decoder_params += param.numel()
            else:
                base_params += param.numel()

        count_result = {
            'total_parameters (M)': total_params / 1_000_000,
            'trainable_parameters (M)': trainable_params / 1_000_000,
            'non_trainable_parameters (M)': (total_params - trainable_params) / 1_000_000,
            'expert_parameters (M)': expert_params / 1_000_000,
            'action_embedder_parameters (M)': action_embedder_params / 1_000_000,
            'action_decoder_parameters (M)': action_decoder_params / 1_000_000,
            'base_parameters (M)': base_params / 1_000_000,
            'total_size_mb': total_params * 4 / (1024 * 1024),  # 假设float32
        }
        print("[ExpertMinimalV1LVGDiT] Parameter Count:")
        for k, v in count_result.items():
            print(f"{k}: {v:.2f}")
        return total_params

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
        (x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
         action_emb_B_T_1_W_D, action_rope_emb_L_1_1_D) = (
            self.prepare_embedded_sequence(
            x_B_C_T_H_W,
            action_emb_B_T_1_1_D,  # NOTE: add action embedding as additional input
            fps=fps,
            padding_mask=padding_mask,
        ))
        '''
        x_B_T_H_W_D.shape: torch.Size([12, 4, 17, 16, 2048]) 
        rope_emb_L_1_1_D.shape: torch.Size([1088, 1, 1, 128]) 
        crossattn_emb.shape: torch.Size([12, 512, 1024])
        '''

        if timesteps_B_T.ndim == 1:
            timesteps_B_T = timesteps_B_T.unsqueeze(1)
        t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)
        action_t_embedding_B_T_D, action_adaln_lora_B_T_3D = self.action_t_embedder(timesteps_B_T)

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
            "ex_rope_emb_L_1_1_D": action_rope_emb_L_1_1_D,
            "ex_t_embedding_B_T_D": action_t_embedding_B_T_D,
            "ex_adaln_lora_B_T_3D": action_adaln_lora_B_T_3D,
        }  # fixed for all blocks
        for block in blocks:
            x_B_T_H_W_D, action_emb_B_T_1_W_D = block(
                x_B_T_H_W_D,
                t_embedding_B_T_D,
                crossattn_emb,
                ex_B_T_1_W_D=action_emb_B_T_1_W_D,  # (B,4,1,8,512)
                **block_kwargs,
            )

        # NOTE: reformat video and action tokens
        x_B_T_H_W_O = self.final_layer(x_B_T_H_W_D, t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
        x_B_C_Tt_Hp_Wp = self.unpatchify(x_B_T_H_W_O)

        action_B_HorizonDof = self.action_decoder_B_D(
            rearrange(action_emb_B_T_1_W_D, "b t 1 w d -> b (t w d)")
        )
        action_B_Horizon_Dof = self.action_reshape(action_B_HorizonDof)  # (B, horizon, dof)
        return x_B_C_Tt_Hp_Wp

    def prepare_embedded_sequence(
        self,
        x_B_C_T_H_W: torch.Tensor,
        action_emb_B_T_1_1_D: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor],
            Optional[torch.Tensor]]:
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
        action_emb_B_T_1_W_D = action_emb_B_T_1_1_D.repeat(1, 1, 1, self.ex_num_tokens_per_latent_frame, 1)  # ->(B,T,1,W,D)

        if self.extra_per_block_abs_pos_emb:
            extra_pos_emb = self.extra_pos_embedder(x_B_T_H_W_D, fps=fps)
        else:
            extra_pos_emb = None

        if "rope" in self.pos_emb_cls.lower():
            assert hasattr(self, "expert_pos_embedder"), "pos_embedder not built"
            # NOTE: use expert_pos_embedder to process action embedding
            video_pos_emb_THW_1_1_D = self.pos_embedder(x_B_T_H_W_D, fps=fps)
            action_pos_emb_T1W_1_1_D = self.expert_pos_embedder(action_emb_B_T_1_W_D, fps=fps)

            ## No need to cat rope embedding anymore, since we process them respectively in BlockWExpert
            # def _cat_rope_emb(_video_emb_THW_1_1_D, _action_emb_T1W_1_1_D):
            #     # reshape for convenience using rearrange
            #     video_T_H_W_D = rearrange(_video_emb_THW_1_1_D, "(t h w) 1 1 d -> t h w d", t=T, h=H, w=W)
            #     action_T_1_W_D = rearrange(_action_emb_T1W_1_1_D, "(t w) 1 1 d -> t 1 w d", t=T, w=W)
            #
            #     # cat at H-dim
            #     combined_T_H1_W_D = torch.cat([video_T_H_W_D, action_T_1_W_D], dim=1)
            #
            #     # reshape back using rearrange
            #     return rearrange(combined_T_H1_W_D, "t h w d -> (t h w) 1 1 d")

            # x_B_T_H_W_D = torch.cat([x_B_T_H_W_D, action_emb_B_T_1_W_D], dim=2)  # (B,T,H+1,W,D)
            # cat_rope_emb_THW_1_1_D = _cat_rope_emb(video_pos_emb_THW_1_1_D, action_pos_emb_T1W_1_1_D)  # (T*(H+1)*W,1,1,D)

            return x_B_T_H_W_D, video_pos_emb_THW_1_1_D, extra_pos_emb, action_emb_B_T_1_W_D, action_pos_emb_T1W_1_1_D
            # return x_B_T_H_W_D, self.pos_embedder(x_B_T_H_W_D, fps=fps), extra_pos_emb
        x_B_T_H_W_D = x_B_T_H_W_D + self.pos_embedder(x_B_T_H_W_D)  # [B, T, H, W, D]

        return x_B_T_H_W_D, None, extra_pos_emb, None, None

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
