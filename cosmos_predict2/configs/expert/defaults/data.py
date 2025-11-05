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

# from cosmos_predict2.data.action_conditioned.action_conditioned_dataset import ActionConditionedDataset
# from cosmos_predict2.data.action_conditioned.pusht_dataset import PushTImageDataset
# from cosmos_predict2.data.action_conditioned.libero_dataset import LiberoReplayImageDataset
# from cosmos_predict2.data.action_conditioned.uha_dataset import OxeUhaDataModule
from cosmos_predict2.configs.expert.defaults.data_pusht import (
    pusht_train_dataloader, pusht_val_dataloader
)
from cosmos_predict2.configs.expert.defaults.data_libero import (
    libero_train_dataloader, libero_val_dataloader
)
from cosmos_predict2.configs.expert.defaults.data_uha import (
    oxe_uha_train_dataloader, oxe_uha_val_dataloader
)
from cosmos_predict2.configs.expert.defaults.data_tcl import (
    tcl_train_dataloader, tcl_val_dataloader
)
from cosmos_predict2.configs.expert.defaults.data_calvin import (
    calvin_train_dataloader, calvin_val_dataloader
)
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
#
#
# def get_sampler(dataset):
#     return DistributedSampler(
#         dataset,
#         num_replicas=parallel_state.get_data_parallel_world_size(),
#         rank=parallel_state.get_data_parallel_rank(),
#         shuffle=True,
#         seed=3,  # ori:0
#     )


# NOTE: OXE UHA dataset, different from above using torch.DataLoader with torch.Dataset as input,
# use the datamodule which creates the dataloader inside.

def register_training_and_val_data_expert():
    cs = ConfigStore.instance()

    # for local dataset
    cs.store(
        group="dataloader_train",
        package="dataloader_train",
        name="pusht_train",
        node=pusht_train_dataloader,
    )
    cs.store(
        group="dataloader_val",
        package="dataloader_val",
        name="pusht_val",
        node=pusht_val_dataloader,
    )

    cs.store(
        group="dataloader_train",
        package="dataloader_train",
        name="libero_train",
        node=libero_train_dataloader,
    )
    cs.store(
        group="dataloader_val",
        package="dataloader_val",
        name="libero_val",
        node=libero_val_dataloader,
    )

    cs.store(
        group="dataloader_train",
        package="dataloader_train",
        name="oxe_train",
        node=oxe_uha_train_dataloader,
    )
    cs.store(
        group="dataloader_val",
        package="dataloader_val",
        name="oxe_val",
        node=oxe_uha_val_dataloader,
    )

    cs.store(
        group="dataloader_train",
        package="dataloader_train",
        name="tcl_train",
        node=tcl_train_dataloader,
    )
    cs.store(
        group="dataloader_val",
        package="dataloader_val",
        name="tcl_val",
        node=tcl_val_dataloader,
    )

    cs.store(
        group="dataloader_train",
        package="dataloader_train",
        name="calvin_train",
        node=calvin_train_dataloader,
    )
    cs.store(
        group="dataloader_val",
        package="dataloader_val",
        name="calvin_val",
        node=calvin_val_dataloader,
    )
