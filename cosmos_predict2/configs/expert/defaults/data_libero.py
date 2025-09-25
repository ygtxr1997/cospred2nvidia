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

from cosmos_predict2.data.action_conditioned.libero_dataset import LiberoReplayImageDataset
from imaginaire.lazy_config import LazyCall as L


libero_train_dataset = L(LiberoReplayImageDataset)(
    shape_meta={
        "image_resolution": 128,
        "action": {
            "shape": [10]
        },
        "obs": {
            "agentview_rgb": {
                "shape": [3, 128, 128],
                "type": "rgb"
            },
            "language": {
                "shape": [15],
            }
        }
    },
    dataset_path="/home/geyuan/datasets/LIBERO_uva25rss/libero_10",
    horizon=32,  # max length of each clip
    pad_before=4 * 2,  # m_obs-1
    pad_after=1,
    n_obs_steps=4 * 2 + 1,  # not used
    abs_action=True,
    rotation_rep="rotation_6d",
    use_cache=True,
    seed=42,
    val_ratio=0.02,
    language_emb_model="clip",
    data_aug=True,
    normalizer_type="all",  # not used
)

libero_val_dataset = libero_train_dataset


def get_sampler(dataset):
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=3,  # ori:0
    )


libero_train_dataloader = L(DataLoader)(
    dataset=libero_train_dataset,
    sampler=L(get_sampler)(dataset=libero_train_dataset),
    batch_size=4,  # ori: 1
    drop_last=True,
    num_workers=8,
    pin_memory=True,
)

libero_val_dataloader = L(DataLoader)(
    dataset=libero_val_dataset,
    sampler=L(get_sampler)(dataset=libero_val_dataset),
    batch_size=1,
    drop_last=True,
)


# def register_training_and_val_data_expert():
#     cs = ConfigStore.instance()
#
#     # for local dataset
#     cs.store(
#         group="dataloader_train",
#         package="dataloader_train",
#         name="libero_train",
#         node=libero_train_dataloader,
#     )
#     cs.store(
#         group="dataloader_val",
#         package="dataloader_val",
#         name="libero_val",
#         node=libero_val_dataloader,
#     )
