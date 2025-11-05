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
CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc_per_node=4 --master_port=12341 -m scripts.train  \
    --config=cosmos_predict2/configs/base/config.py  \
    -- experiment="cospred2_2b_expert_tcl"
"""
data_name = "tcl"
data_name_to_robot_states_dim = {
    "fractal": 8,  # 7 joint + 1 gripper
    "bridge": 7,   # 6 joint + 1 gripper
    "tcl": 6,      # 6 tcp   + 2 blank
}

n_v_cond, n_v_out = 4 * 0 + 1, 4 * 6  # 4+1+20=25
n_a_out = n_v_out
n_latent_v_cond, n_latent_v_out = 1 * 0 + 1, 1 * 6  # 1+1+5=7
horizon = n_v_cond + n_v_out # 25
pad_before = n_v_cond - 1
cospred2_2b_expert_tcl = dict(
    defaults=[
        {"override /model": "predict2_v2w_2b_expert_fsdp"},  # modified
        {"override /optimizer": "fusedadamw"},
        {"override /scheduler": "lambdalinear"},
        {"override /ckpt_type": "standard"},
        {"override /dataloader_train": "tcl_train"},  # modified
        "_self_",
    ],
    model=dict(
        config=dict(
            fsdp_shard_size=-1,
            # train_architecture="lora",
            pipe_config=dict(
                ema=dict(enabled=True),  # ema is usually better
                net=dict(
                    action_dim=7*n_a_out,  # (act_dim * horizon)
                    action_dof=7,  # tcl: 3 xyz + 3 rot + 1 gripper
                    ex_num_latent_frames=n_a_out,
                    ex_dim=1024,
                    ex_adaln_lora_dim=256,
                    extra_robot_states_dim=data_name_to_robot_states_dim[data_name]*n_v_cond,  # (D*T), (joint + gripper) * max_obs, fractal:8, bridge:7
                    state_t=n_latent_v_cond + n_latent_v_out,  # same as pipeline
                    n_cameras_emb=2,  # ori:2
                    concat_view_embedding=False,  # NVIDIA doesn't provide the 7-view ckpt
                ),
                state_t=n_latent_v_cond + n_latent_v_out,  # raw:8+1+24=33,
                max_obs=n_v_cond,
                max_act_out=n_a_out,
                p_all_actions_as_condition=0.3,
            ),
            # model_manager_config=dict(
            #     dit_path="checkpoints/cosmos_predict2/debug/cospred2_2b_expert_tcl_2025-10-15_16-48-08/checkpoints/model/iter_000006000.pt",
            # )
        )
    ),
    job=dict(group="debug", name="cospred2_2b_expert_tcl_${now:%Y-%m-%d}_${now:%H-%M-%S}"),
    model_parallel=dict(
        context_parallel_size=1,
    ),
    dataloader_train=dict(
        batch_size=16,  # now:16, ori:20
        num_workers=8,
    ),
    trainer=dict(
        distributed_parallelism="fsdp",
        max_iter=200000,
        callbacks=dict(
            # iter_speed=dict(hit_thres=10),
            device_monitor=dict(every_n=2000),
            manual_gc=dict(every_n=5),
        )
    ),
    checkpoint=dict(
        save_iter=2000,
    ),
    optimizer=dict(
        lr=1e-4,  # or:1e-4,
    ),
    scheduler=dict(  # better
        cycle_lengths=[100_000, 100_000],  # ori: [20_000, 20_000]
        warm_up_steps=[2_000, 0],
        f_start=[0.01, 0.01],
        f_max=[1.0, 1.0],
        f_min=[1.0, 0.1],
        verbosity_interval=2_000
    ),
)


for _item in [
    # predict2_video2world_2b
    # predict2_video2world_2b_action_conditioned_training,
    cospred2_2b_expert_tcl,
]:
    # Get the experiment name from the global variable, e.g. exp01_wan_lora -> experiment_name = "exp01_wan_lora"
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]

    cs.store(
        group="experiment",
        package="_global_",
        name=experiment_name,
        node=_item,
    )
