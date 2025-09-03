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

from cosmos_predict2.data.action_conditioned.action_conditioned_dataset import ActionConditionedDataset
from cosmos_predict2.data.action_conditioned.pusht_dataset import PushTImageDataset
from imaginaire.lazy_config import LazyCall as L

base_path = "./datasets/bridge/"
train_annotation_path = os.path.join(base_path, "annotation/train")
val_annotation_path = os.path.join(base_path, "annotation/val")
test_annotation_path = os.path.join(base_path, "annotation/test")


# bridge_train_dataset = L(ActionConditionedDataset)(
#     train_annotation_path=train_annotation_path,
#     val_annotation_path=val_annotation_path,
#     test_annotation_path=test_annotation_path,
#     video_path=base_path,
#     sequence_interval=1,
#     num_frames=13,
#     cam_ids=[0],
#     accumulate_action=False,
#     video_size=[480, 640],
#     val_start_frame_interval=1,
#     mode="train",
# )
#
# bridge_val_dataset = L(ActionConditionedDataset)(
#     train_annotation_path=train_annotation_path,
#     val_annotation_path=val_annotation_path,
#     test_annotation_path=test_annotation_path,
#     video_path=base_path,
#     sequence_interval=1,
#     num_frames=13,
#     cam_ids=[0],
#     accumulate_action=False,
#     video_size=[480, 640],
#     val_start_frame_interval=1,
#     mode="val",
# )


pusht_train_dataset = L(PushTImageDataset)(
    # zarr_path="./datasets/pusht/pusht_cchi_v7_replay.zarr",
zarr_path="./datasets/pusht/pusht_256.zarr",
    horizon=12,
    pad_before=0,
    pad_after=8,
)

pusht_val_dataset = L(PushTImageDataset)(
    zarr_path="./datasets/pusht/pusht_orange_random_v2.zarr",
    horizon=12,
    pad_before=0,
    pad_after=8,
)


def get_sampler(dataset):
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=0,
    )


bridge_train_dataloader = L(DataLoader)(
    dataset=pusht_train_dataset,
    sampler=L(get_sampler)(dataset=pusht_train_dataset),
    batch_size=4,  # ori: 1
    drop_last=True,
    num_workers=8,
    pin_memory=True,
)

bridge_val_dataloader = L(DataLoader)(
    dataset=pusht_val_dataset,
    sampler=L(get_sampler)(dataset=pusht_val_dataset),
    batch_size=1,
    drop_last=True,
)


def register_training_and_val_data_expert():
    cs = ConfigStore.instance()

    # for local dataset
    cs.store(
        group="dataloader_train",
        package="dataloader_train",
        name="pusht_train",
        node=bridge_train_dataloader,
    )
    cs.store(
        group="dataloader_val",
        package="dataloader_val",
        name="pusht_val",
        node=bridge_val_dataloader,
    )
