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

import os

from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_predict2.data.action_conditioned.tcl_dataset import TCLImageDataset, TCLMergeDataset
from imaginaire.lazy_config import LazyCall as L


n_v_cond, n_v_out = 4 * 0 + 1, 4 * 12  # 0+1+48=49
future_skip = 6
n_a_out = n_v_out
n_v_skip_out = n_v_out // future_skip  # 48/6=8
n_latent_v_cond, n_latent_v_out = ((n_v_cond - 1) // 4 + 1), n_v_skip_out // 4  # 1, 48/6/4=2
horizon = n_v_cond + n_v_out  # without frame skip, 49
pad_before = n_v_cond - 1
tcl_train_dataset = L(TCLMergeDataset)(
    shape_meta={
        "action": {
            "shape": [7]
        },
        "obs": {
            "image": {
                "shape": [3, 16*10, 16*15],  # now:16*12,16*17 ;ori:112=16*7,160=16*10
                "type": "rgb"
            },
            "gripper": {  # additional
                "shape": [3, 16*10, 16*15],
                "type": "rgb"
            },
            "joint_state": {  # additional, 7}
                "shape": [6],  # will be padded with 2 zeros -> (8,)
            },
            "force": {  # additional
                "shape": [6],
            }
        }
    },
    norm_action_type="mean",  # ori: "minmax", or "mean"
    horizon=n_a_out,  # action length
    max_train_episodes=90,  # not used
    pad_before=pad_before,  # m_obs-1
    pad_after=1,
    seed=42,
    val_ratio=0.0,  # ori:0.02

    data_roots=[
        "/home/geyuan/datasets/TCL/1024_sweep_bean/",
        "/home/geyuan/datasets/TCL/1024_eggs_pick_place/",
        "/home/geyuan/datasets/TCL/1024_pour_water/",
        "/home/geyuan/datasets/TCL/1024_wipe_white_board/",
    ],
    h5_paths=[
        "/home/geyuan/datasets/TCL/hdf5/1024_sweep_bean_240p.h5",
        "/home/geyuan/datasets/TCL/hdf5/1024_eggs_pick_place_240p.h5",
        "/home/geyuan/datasets/TCL/hdf5/1024_pour_water_240p.h5",
        "/home/geyuan/datasets/TCL/hdf5/1024_wipe_white_board_240p.h5",
    ],
    use_h5=True,
    transform_color_jitter=False,

    # language related
    language_emb_model="t5xxl",  # ori: "clip", or: "t5xxl"

    # multi-view related
    camera_keys=[
        "image",
        "gripper",
    ],
    p_camera_drop=0.2,  # ori:0.2
    switch_camera_view=False,  # [Warning] only when data collection makes mistake
    future_frame_skip=future_skip,  # ori:1
)

tcl_val_dataset = tcl_train_dataset


def get_sampler(dataset):
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=3,  # ori:0
    )


tcl_train_dataloader = L(DataLoader)(
    dataset=tcl_train_dataset,
    sampler=L(get_sampler)(dataset=tcl_train_dataset),
    batch_size=5,  # ori: 1
    drop_last=True,
    num_workers=8,
    pin_memory=True,
)

tcl_val_dataloader = L(DataLoader)(
    dataset=tcl_val_dataset,
    sampler=L(get_sampler)(dataset=tcl_val_dataset),
    batch_size=1,
    drop_last=True,
)
