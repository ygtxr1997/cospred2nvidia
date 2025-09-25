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

from hydra.core.config_store import ConfigStore

cs = ConfigStore.instance()

"""
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train  \
    --config=cosmos_predict2/configs/base/config.py  \
    -- experiment="cospred2_2b_expert_libero"
"""
cospred2_2b_expert_libero = dict(
    defaults=[
        {"override /model": "predict2_v2w_2b_expert_fsdp"},  # modified
        {"override /optimizer": "fusedadamw"},
        {"override /scheduler": "lambdalinear"},
        {"override /ckpt_type": "standard"},
        {"override /dataloader_train": "libero_train"},  # modified
        "_self_",
    ],
    model=dict(
        config=dict(
            fsdp_shard_size=-1,
            # train_architecture="lora",
            pipe_config=dict(
                net=dict(
                    action_dim=10*24,  # (act_dim * horizon)
                    action_dof=10,  # libero: 3 xyz + 6 rot + 1 gripper
                    ex_num_latent_frames=24,
                ),
                state_t=2+1+6,  # raw:8+1+24=33,
                max_obs=8+1,
                max_act_out=24,
                p_all_actions_as_condition=0.3,
            ),
            # model_manager_config=dict(
            #     dit_path="checkpoints/cosmos_predict2/debug/cospred2_2b_expert_libero_2025-09-22_11-09-29/checkpoints/model/iter_000020000.pt",
            # )
        )
    ),
    job=dict(group="debug", name="cospred2_2b_expert_libero_${now:%Y-%m-%d}_${now:%H-%M-%S}"),
    model_parallel=dict(
        context_parallel_size=1,
    ),
    dataloader_train=dict(
        batch_size=28,  # ori:24
        num_workers=6,
    ),
    trainer=dict(
        distributed_parallelism="fsdp",
        max_iter=40000,
        callbacks=dict(
            # iter_speed=dict(hit_thres=10),
            device_monitor=dict(every_n=2000),
        )
    ),
    checkpoint=dict(
        save_iter=2000,
    ),
    optimizer=dict(
        lr=1e-4,  # or:1e-4,
    ),
)


for _item in [
    # predict2_video2world_2b
    # predict2_video2world_2b_action_conditioned_training,
    cospred2_2b_expert_libero
]:
    # Get the experiment name from the global variable, e.g. exp01_wan_lora -> experiment_name = "exp01_wan_lora"
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]

    cs.store(
        group="experiment",
        package="_global_",
        name=experiment_name,
        node=_item,
    )
