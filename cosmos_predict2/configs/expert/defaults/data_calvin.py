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
import omegaconf.dictconfig
import os

from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler
import torchvision

from cosmos_predict2.data.action_conditioned.calvin_dataset import ClearDataset, calvin_utils
from imaginaire.lazy_config import LazyCall as L
from imaginaire.lazy_config import instantiate


def create_lazy_dict(**kwargs):
    return {k: v for k, v in kwargs.items()}


obs_image_shape = (16 * 12, 16 * 12)  # (height, width)
calvin_transforms_yaml = {
    "train": {
        "rgb_static": [
            {
                "_target_": "torchvision.transforms.Resize",
                "size": obs_image_shape,
                "antialias": True
            },
            {
                "_target_": "cosmos_predict2.data.action_conditioned.calvin_utils.RandomShiftsAug",
                "pad": 10
            },
            {
                "_target_": "cosmos_predict2.data.action_conditioned.calvin_utils.ScaleImageTensor"
            },
            # {
            #     "_target_": "torchvision.transforms.Normalize",
            #     "mean": [0.48145466, 0.4578275, 0.40821073],
            #     "std": [0.26862954, 0.26130258, 0.27577711]
            # }
        ],
        "gen_static": [
            {
                "_target_": "torchvision.transforms.Resize",
                "size": obs_image_shape,
                "antialias": True
            },
            {
                "_target_": "cosmos_predict2.data.action_conditioned.calvin_utils.ScaleImageTensor"
            },
            # {
            #     "_target_": "torchvision.transforms.Normalize",
            #     "mean": [0.48145466, 0.4578275, 0.40821073],
            #     "std": [0.26862954, 0.26130258, 0.27577711]
            # }
        ],
        "rgb_gripper": [
            {
                "_target_": "torchvision.transforms.Resize",
                "size": obs_image_shape,
                "antialias": True
            },
            {
                "_target_": "cosmos_predict2.data.action_conditioned.calvin_utils.RandomShiftsAug",
                "pad": 4
            },
            {
                "_target_": "cosmos_predict2.data.action_conditioned.calvin_utils.ScaleImageTensor"
            },
            # {
            #     "_target_": "torchvision.transforms.Normalize",
            #     "mean": [0.48145466, 0.4578275, 0.40821073],
            #     "std": [0.26862954, 0.26130258, 0.27577711]
            # }
        ],
        "gen_gripper": [
            {
                "_target_": "torchvision.transforms.Resize",
                "size": obs_image_shape,
                "antialias": True
            },
            {
                "_target_": "cosmos_predict2.data.action_conditioned.calvin_utils.ScaleImageTensor"
            },
            # {
            #     "_target_": "torchvision.transforms.Normalize",
            #     "mean": [0.48145466, 0.4578275, 0.40821073],
            #     "std": [0.26862954, 0.26130258, 0.27577711]
            # }
        ],
        "depth_static": [
            {
                "_target_": "torchvision.transforms.Resize",
                "size": 200
            },
            {
                "_target_": "cosmos_predict2.data.action_conditioned.calvin_utils.AddDepthNoise",
                "shape": [1000.0],
                "rate": [1000.0]
            },
            {
                "_target_": "cosmos_predict2.data.action_conditioned.calvin_utils.AddGaussianNoise",
                "mean": [0.0],
                "std": [0.01]
            }
        ],
        "depth_gripper": [
            {
                "_target_": "torchvision.transforms.Resize",
                "size": 84
            },
            {
                "_target_": "cosmos_predict2.data.action_conditioned.calvin_utils.AddGaussianNoise",
                "mean": [0.0],
                "std": [0.01]
            }
        ],
        "rgb_tactile": [
            {
                "_target_": "torchvision.transforms.Resize",
                "size": 70
            },
            {
                "_target_": "torchvision.transforms.RandomCrop",
                "size": 64
            },
            {
                "_target_": "cosmos_predict2.data.action_conditioned.calvin_utils.ScaleImageTensor"
            },
            {
                "_target_": "torchvision.transforms.Normalize",
                "mean": [0.5],
                "std": [0.5]
            },
            {
                "_target_": "cosmos_predict2.data.action_conditioned.calvin_utils.AddGaussianNoise",
                "mean": [0.0],
                "std": [0.01]
            }
        ],
        "depth_tactile": [
            {
                "_target_": "torchvision.transforms.Resize",
                "size": 64,
                "antialias": True
            },
            {
                "_target_": "torchvision.transforms.Normalize",
                "mean": [0.1],
                "std": [0.2]
            },
            {
                "_target_": "cosmos_predict2.data.action_conditioned.calvin_utils.AddGaussianNoise",
                "mean": [0.0],
                "std": [0.01]
            }
        ],
        "robot_obs": [
            {
                "_target_": "cosmos_predict2.data.action_conditioned.calvin_utils.NormalizeVector"
            },
            {
                "_target_": "cosmos_predict2.data.action_conditioned.calvin_utils.AddGaussianNoise",
                "mean": [0.0],
                "std": [0.01]
            }
        ],
        "scene_obs": [
            {
                "_target_": "cosmos_predict2.data.action_conditioned.calvin_utils.NormalizeVector"
            },
            {
                "_target_": "cosmos_predict2.data.action_conditioned.calvin_utils.AddGaussianNoise",
                "mean": [0.0],
                "std": [0.01]
            }
        ],
        "language": [
            {
                "_target_": "cosmos_predict2.data.action_conditioned.calvin_utils.AddGaussianNoise",
                "mean": [0.0],
                "std": [0.01]
            }
        ]
    },
    "val": {
        "rgb_static": [
            {
                "_target_": "torchvision.transforms.Resize",
                "size": obs_image_shape,
                "antialias": True
            },
            {
                "_target_": "cosmos_predict2.data.action_conditioned.calvin_utils.ScaleImageTensor"
            },
            # {
            #     "_target_": "torchvision.transforms.Normalize",
            #     "mean": [0.48145466, 0.4578275, 0.40821073],
            #     "std": [0.26862954, 0.26130258, 0.27577711]
            # }
        ],
        "rgb_gripper": [
            {
                "_target_": "torchvision.transforms.Resize",
                "size": obs_image_shape,
                "antialias": True
            },
            {
                "_target_": "cosmos_predict2.data.action_conditioned.calvin_utils.ScaleImageTensor"
            },
            # {
            #     "_target_": "torchvision.transforms.Normalize",
            #     "mean": [0.48145466, 0.4578275, 0.40821073],
            #     "std": [0.26862954, 0.26130258, 0.27577711]
            # }
        ],
        "gen_static": [
            {
                "_target_": "torchvision.transforms.Resize",
                "size": obs_image_shape,
                "antialias": True
            },
            {
                "_target_": "cosmos_predict2.data.action_conditioned.calvin_utils.ScaleImageTensor"
            },
            # {
            #     "_target_": "torchvision.transforms.Normalize",
            #     "mean": [0.48145466, 0.4578275, 0.40821073],
            #     "std": [0.26862954, 0.26130258, 0.27577711]
            # }
        ],
        "gen_gripper": [
            {
                "_target_": "torchvision.transforms.Resize",
                "size": obs_image_shape,
                "antialias": True
            },
            {
                "_target_": "cosmos_predict2.data.action_conditioned.calvin_utils.ScaleImageTensor"
            },
            # {
            #     "_target_": "torchvision.transforms.Normalize",
            #     "mean": [0.48145466, 0.4578275, 0.40821073],
            #     "std": [0.26862954, 0.26130258, 0.27577711]
            # }
        ],
        "depth_static": [
            {
                "_target_": "torchvision.transforms.Resize",
                "size": 200
            }
        ],
        "depth_gripper": [
            {
                "_target_": "torchvision.transforms.Resize",
                "size": 84
            }
        ],
        "rgb_tactile": [
            {
                "_target_": "torchvision.transforms.Resize",
                "size": 70
            },
            {
                "_target_": "torchvision.transforms.RandomCrop",
                "size": 64
            },
            {
                "_target_": "cosmos_predict2.data.action_conditioned.calvin_utils.ScaleImageTensor"
            },
            {
                "_target_": "torchvision.transforms.Normalize",
                "mean": [0.5],
                "std": [0.5]
            }
        ],
        "depth_tactile": [
            {
                "_target_": "torchvision.transforms.Resize",
                "size": 64
            },
            {
                "_target_": "torchvision.transforms.Normalize",
                "mean": [0.1],
                "std": [0.2]
            }
        ],
        "robot_obs": [
            {
                "_target_": "cosmos_predict2.data.action_conditioned.calvin_utils.NormalizeVector"
            }
        ],
        "scene_obs": [
            {
                "_target_": "cosmos_predict2.data.action_conditioned.calvin_utils.NormalizeVector"
            }
        ]
    }
}

root_data_dir = "/home/geyuan/code/mdt24rss_fork/dataset/task_D_D"
# root_data_dir = "/home/geyuan/code/mdt24rss_fork/dataset/task_ABC_D"
training_dir = os.path.join(root_data_dir, "training")
val_dir = os.path.join(root_data_dir, "validation")

transforms_modified_by_stat = L(calvin_utils.load_dataset_statistics)(
    train_dataset_dir=training_dir,
    val_dataset_dir=val_dir,
    transforms=calvin_transforms_yaml,  # input should be a non-instantiated dict
)


def get_runnable_transforms(_transforms: omegaconf.dictconfig.DictConfig, _split: str = "train"):
    _train_transforms = {}
    _val_transforms = {}
    for cam in _transforms.train:
        cam_transforms = []
        for transform in _transforms.train[cam]:
            if transform._target_ == "torchvision.transforms.ColorJitter":
                instantiated_transform = torchvision.transforms.ColorJitter(
                    brightness=transform.brightness,
                    contrast=tuple(transform.contrast),
                    saturation=tuple(transform.saturation),
                )
            else:
                instantiated_transform = instantiate(transform)
            cam_transforms.append(instantiated_transform)
        _train_transforms[cam] = cam_transforms
    _val_transforms = {
        cam: [instantiate(transform) for transform in _transforms.val[cam]] for cam in _transforms.val
    }
    return dict(
        train={key: torchvision.transforms.Compose(transforms=val) for key, val in _train_transforms.items()},
        val={key: torchvision.transforms.Compose(transforms=val) for key, val in _val_transforms.items()},
    )[_split]


runnable_train_transforms = L(get_runnable_transforms)(
    _transforms=transforms_modified_by_stat, _split="train"
)
runnable_val_transforms = L(get_runnable_transforms)(
    _transforms=transforms_modified_by_stat, _split="val"
)


n_v_cond, n_v_out = 4 * 0 + 1, 4 * 3  # 0+1+12=13
future_skip = 3
n_a_out = n_v_out
n_v_skip_out = n_v_out // future_skip  # 48/6=8
n_latent_v_cond, n_latent_v_out = ((n_v_cond - 1) // 4 + 1), n_v_skip_out // 4  # 1, 12/3/4=1
horizon = n_v_cond + n_v_out  # without frame skip, 13
pad_before = n_v_cond - 1

calvin_train_dataset = L(ClearDataset)(
    ## BaseDataset ##
    key="lang",
    datasets_dir=training_dir,
    obs_space=dict(
        rgb_obs=['rgb_static', 'rgb_gripper'],  #, 'gen_static', 'gen_gripper'],
        depth_obs=[],
        state_obs=['robot_obs'],
        actions=['rel_actions'],
        language=['language'],
    ),
    proprio_state=dict(
        n_state_obs=8,
        keep_indices=[[0, 7], [14, 15]],
        robot_orientation_idx=[3, 6],
        normalize=True,
        normalize_robot_orientation=True,
    ),
    batch_size=16,
    pad=False,
    lang_folder="lang_clip_resnet50",
    num_workers=None,
    transforms=runnable_train_transforms,
    window_sampling_strategy="geometric",
    min_window_size=-1,  # ori:3
    max_window_size=-1,  # ori:3
    ## DiskDataset ##
    skip_frames=1,
    save_format="npz",
    ## ExtendedDiskDataset ##
    action_seq_len=n_a_out,  # modified
    obs_seq_len=n_v_cond,  # modified
    img_gen_frame_diff=-2,  # ori:3
    ## Extracted speed-up ##
    use_extracted_rel_actions=True,
    ## CosmosPredict2 related ##
    future_frame_skip=future_skip,  # added
    language_emb_model="t5xxl",  # added
    obs_image_shape=obs_image_shape,  # added
    zero_robot_state=True,  # added
    ## Others ##
    max_len=None,
    chose_ratio=1,
)

calvin_val_dataset = calvin_train_dataset


def get_sampler(dataset):
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=4,  # ori:0
    )


calvin_train_dataloader = L(DataLoader)(
    dataset=calvin_train_dataset,
    sampler=L(get_sampler)(dataset=calvin_train_dataset),
    batch_size=5,  # ori: 1
    drop_last=True,
    num_workers=8,
    pin_memory=True,
)

calvin_val_dataloader = L(DataLoader)(
    dataset=calvin_val_dataset,
    sampler=L(get_sampler)(dataset=calvin_val_dataset),
    batch_size=1,
    drop_last=True,
)


