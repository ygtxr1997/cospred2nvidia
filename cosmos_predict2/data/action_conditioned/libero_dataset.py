from typing import Dict, List, Tuple, Union
import concurrent.futures
import multiprocessing
import zarr
import os
import shutil
import copy
import glob
from filelock import FileLock

import h5py  # 3.14.0
from tqdm import tqdm
from threadpoolctl import threadpool_limits  # 3.6.0

import torch
import numpy as np
from transformers import AutoTokenizer
import torchvision.transforms as transforms

# from unified_video_action.common.pytorch_util import dict_apply
# from unified_video_action.dataset.base_dataset import BaseImageDataset, LinearNormalizer
# from unified_video_action.model.common.normalizer import (
#     LinearNormalizer,
#     SingleFieldLinearNormalizer,
# )
# from unified_video_action.model.common.rotation_transformer import RotationTransformer
# from unified_video_action.codecs.imagecodecs_numcodecs import register_codecs, Jpeg2k
# from unified_video_action.common.replay_buffer import ReplayBuffer
# from unified_video_action.common.sampler import SequenceSampler, get_val_mask
# from unified_video_action.common.normalize_util import (
#     robomimic_abs_action_only_normalizer_from_stat,
#     robomimic_abs_action_only_dual_arm_normalizer_from_stat,
#     get_range_normalizer_from_stat,
#     get_image_range_normalizer,
#     get_identity_normalizer_from_stat,
#     array_to_stats,
# )
from .libero_utils import (
    dict_apply,
    LinearNormalizer,
    SingleFieldLinearNormalizer,
    RotationTransformer,
    register_codecs,
    Jpeg2k,
    ReplayBuffer,
    SequenceSampler,
    get_val_mask,
    robomimic_abs_action_only_normalizer_from_stat,
    robomimic_abs_action_only_dual_arm_normalizer_from_stat,
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats,
)

register_codecs()



class BaseImageDataset(torch.utils.data.Dataset):
    def get_validation_dataset(self) -> "BaseImageDataset":
        # return an empty dataset by default
        return BaseImageDataset()

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        raise NotImplementedError()

    def get_all_actions(self) -> torch.Tensor:
        raise NotImplementedError()

    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        output:
            obs:
                key: T, *
            action: T, Da
        """
        raise NotImplementedError()


class LiberoReplayImageDataset(BaseImageDataset):
    VIEW_CHOICES = ["agentview_rgb", "eye_in_hand_rgb",]
    CAMERA_TO_VIEW_ID = {
        "agentview_rgb": 0,
        "eye_in_hand_rgb": 1,
    }
    def __init__(
            self,
            shape_meta: dict,
            dataset_path: str,
            horizon=1,
            pad_before=0,
            pad_after=0,
            n_obs_steps=None,
            abs_action=False,
            rotation_rep="rotation_6d",  # ignored when abs_action=False
            use_legacy_normalizer=False,
            use_cache=False,
            seed=42,
            val_ratio=0.0,
            language_emb_model=None,
            data_aug=False,
            normalizer_type=None,
            # extra args for cache building,
            cache_zarr_path: str = None,
            # multi-view related
            camera_keys: Union[List[str], Tuple[str]] = ("agentview_rgb",),  # each in `agentview_rgb`, `eye_in_hand_rgb`
            p_camera_drop: float = 0.0,
    ):
        """

        Args:
            shape_meta: {
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
            }
            dataset_path: "data/libero_10"
            horizon: 32
            pad_before: 7
            pad_after: 1
            n_obs_steps: 16
            abs_action: True
            rotation_rep: "rotation_6d"
            use_legacy_normalizer:
            use_cache: True
            seed: 42
            val_ratio: 0.02
            language_emb_model: "clip"
            data_aug: True
            normalizer_type: all  # not used
        """
        for camera_key in camera_keys:
            assert camera_key in self.VIEW_CHOICES, f"camera_key must be in {self.VIEW_CHOICES}"

        rotation_transformer = RotationTransformer(
            from_rep="axis_angle", to_rep=rotation_rep
        )

        replay_buffer = None
        if use_cache:

            if language_emb_model == "clip":
                cache_zarr_path = cache_zarr_path or dataset_path + "_clip.zarr.zip"
                assert cache_zarr_path.endswith("_clip.zarr.zip"), \
                    "cache path must end with _clip.zarr.zip for clip model"
            elif language_emb_model == "t5xxl":
                cache_zarr_path = cache_zarr_path or dataset_path + "_clip_t5xxl.zarr.zip"
                assert cache_zarr_path.endswith("_clip_t5xxl.zarr.zip"), \
                    "cache path must end with _clip_t5xxl.zarr.zip for t5xxl model"
            else:
                raise NotImplementedError(f"Language model {language_emb_model} not implemented")

            cache_lock_path = cache_zarr_path + ".lock"
            print("Acquiring lock on cache.")
            print("Cache path:", cache_zarr_path)

            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    # cache does not exist, create a new one
                    try:
                        print("Cache does not exist. Creating!")

                        replay_buffer = _convert_robomimic_to_replay(
                            store=zarr.MemoryStore(),
                            shape_meta=shape_meta,
                            dataset_path=dataset_path,
                            abs_action=abs_action,
                            rotation_transformer=rotation_transformer,
                            language_emb_model=language_emb_model,
                        )
                        print("Saving cache to disk.")
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(store=zip_store)
                    except Exception as e:
                        shutil.rmtree(cache_zarr_path)
                        raise e
                else:
                    print("Loading cached ReplayBuffer from Disk.")
                    with zarr.ZipStore(cache_zarr_path, mode="r") as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=zarr.MemoryStore()
                        )
                    print("Loaded!")
        else:
            replay_buffer = _convert_robomimic_to_replay(
                store=zarr.MemoryStore(),
                shape_meta=shape_meta,
                dataset_path=dataset_path,
                abs_action=abs_action,
                rotation_transformer=rotation_transformer,
                language_emb_model=language_emb_model,
            )

        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            type = attr.get("type", "low_dim")
            if type == "rgb":
                rgb_keys.append(key)
            elif type == "low_dim":
                lowdim_keys.append(key)

        self.data_aug = data_aug

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed
        )
        train_mask = ~val_mask

        sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )

        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.abs_action = abs_action
        self.n_obs_steps = n_obs_steps
        self.train_mask = train_mask
        self.horizon = horizon
        self.state_t = 1 + (horizon - 1) // 4  # 33->9, 17->5
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.use_legacy_normalizer = use_legacy_normalizer

        stat = array_to_stats(self.replay_buffer["action"])
        self.meta_action_mean = torch.from_numpy(stat["mean"])
        self.meta_action_std = torch.from_numpy(stat["std"])
        self.meta_action_min = torch.from_numpy(stat["min"])
        self.meta_action_max = torch.from_numpy(stat["max"])
        print("[LiberoDataset] action: "
              "\nmean:", ",\t".join(f"{x:.4f}" for x in self.meta_action_mean),
              "\nstd:", ",\t".join(f"{x:.4f}" for x in self.meta_action_std),
              "\nmin:", ",\t".join(f"{x:.4f}" for x in self.meta_action_min),
              "\nmax:", ",\t".join(f"{x:.4f}" for x in self.meta_action_max),)
        '''
        mean: -0.0410,	0.0349,	0.8389,	0.3086,	0.6320,	-0.0298,	0.6294,	-0.2721,	0.1521,	-0.1065 
        std: 0.1048,	0.1465,	0.2572,	0.5414,	0.4351,	0.1485,	0.4493,	0.4955,	0.2436,	0.9941 
        min: -0.5226,	-0.3341,	0.4075,	-1.0000,	-0.7127,	-0.6373,	-0.8550,	-1.0000,	-0.6161,	-1.0000 
        max: 0.2064,	0.3886,	1.3320,	1.0000,	1.0000,	0.6735,	1.0000,	0.9999,	0.9838,	1.0000
        '''
        self.has_joint_states = "joint_states" in self.replay_buffer
        if self.has_joint_states:
            joint_states_stat = array_to_stats(self.replay_buffer["joint_states"])
            # print("[LiberoDataset] joint_states: "
            #       "\nmean:", ",\t".join(f"{x:.4f}" for x in joint_states_stat["mean"]),
            #       "\nstd:", ",\t".join(f"{x:.4f}" for x in joint_states_stat["std"]),
            #       "\nmin:", ",\t".join(f"{x:.4f}" for x in joint_states_stat["min"]),
            #       "\nmax:", ",\t".join(f"{x:.4f}" for x in joint_states_stat["max"]),)
            self.meta_joint_mean = torch.from_numpy(joint_states_stat["mean"])
            self.meta_joint_std = torch.from_numpy(joint_states_stat["std"])
            '''
            [LiberoDataset] joint_states: 
            mean: -0.0092,	0.3708,	0.0358,	-2.0189,	0.1920,	2.3045,	1.1149 
            std: 0.1234,	0.3207,	0.1815,	0.4222,	0.3866,	0.3434,	0.7641 
            min: -0.6023,	-0.7310,	-0.4286,	-3.0736,	-1.2465,	0.9248,	-2.0455 
            max: 0.5995,	1.6696,	0.9495,	-0.0662,	2.9000,	3.7689,	2.8993
            '''

            # If joint_states is available, we here assume `ee_states` and `gripper_states` are also available
            ee_states_stat = array_to_stats(self.replay_buffer["ee_states"])
            self.meta_ee_states_mean = torch.from_numpy(ee_states_stat["mean"])
            self.meta_ee_states_std = torch.from_numpy(ee_states_stat["std"])

            gripper_states_stat = array_to_stats(self.replay_buffer["gripper_states"])
            self.meta_gripper_states_mean = torch.from_numpy(gripper_states_stat["mean"])
            self.meta_gripper_states_std = torch.from_numpy(gripper_states_stat["std"])
            # print("[LiberoDataset] ee_states: "
            #       "\nmean:", ",\t".join(f"{x:.4f}" for x in ee_states_stat["mean"]),
            #       "\nstd:", ",\t".join(f"{x:.4f}" for x in ee_states_stat["std"]),
            #       "\nmin:", ",\t".join(f"{x:.4f}" for x in ee_states_stat["min"]),
            #       "\nmax:", ",\t".join(f"{x:.4f}" for x in ee_states_stat["max"]),)
            # print("[LiberoDataset] gripper_states: "
            #       "\nmean:", ",\t".join(f"{x:.4f}" for x in gripper_states_stat["mean"]),
            #       "\nstd:", ",\t".join(f"{x:.4f}" for x in gripper_states_stat["std"]),
            #       "\nmin:", ",\t".join(f"{x:.4f}" for x in gripper_states_stat["min"]),
            #       "\nmax:", ",\t".join(f"{x:.4f}" for x in gripper_states_stat["max"]),)
            '''
            [LiberoDataset] ee_states: 
            mean: -0.0416,	0.0326,	0.8413,	2.8851,	-0.6716,	-0.1957 
            std: 0.1045,	0.1442,	0.2571,	0.3585,	1.2720,	0.3817 
            min: -0.4813,	-0.3363,	0.4456,	1.0537,	-3.6578,	-2.0025 
            max: 0.2067,	0.3914,	1.3317,	3.8439,	3.6063,	1.3686
            [LiberoDataset] gripper_states: 
            mean: 0.0284,	-0.0287 
            std: 0.0133,	0.0132 
            min: -0.0020,	-0.0412 
            max: 0.0428,	0.0014
            '''
            print(f"[LiberoDataset] Found joint_states({self.replay_buffer['joint_states'].shape}), "
                  f"ee_states({self.replay_buffer['ee_states'].shape}), "
                  f"gripper_states({self.replay_buffer['gripper_states'].shape}) in the replay buffer. ")

        self.language_emb_model = language_emb_model
        if "t5xxl" in self.language_emb_model:
            self.t5_meta = {
                "unique_embeddings": self.replay_buffer.meta["t5_unique_embeddings"],
                "unique_valid_lengths": self.replay_buffer.meta["t5_valid_lengths"],
                "unique_texts": self.replay_buffer.meta["t5_unique_texts"],
            }
            print(f"[LiberoDataset] using t5xxl model for language embedding. "
                  f"found {len(self.t5_meta['unique_embeddings'])} unique texts.")

        self.camera_keys = camera_keys
        self.p_camera_drop = p_camera_drop
        print("[LiberoDataset] camera_keys:", self.camera_keys, ", p_camera_drop:", self.p_camera_drop)

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # action
        stat = array_to_stats(self.replay_buffer["action"])  # (N,D) -> {'min':(D,), ..., 'std':(D,)}
        if self.abs_action:
            if stat["mean"].shape[-1] > 10:
                # dual arm
                this_normalizer = (
                    robomimic_abs_action_only_dual_arm_normalizer_from_stat(stat)
                )
            else:
                this_normalizer = robomimic_abs_action_only_normalizer_from_stat(stat)

            if self.use_legacy_normalizer:
                this_normalizer = normalizer_from_stat(stat)
        else:
            # already normalized
            this_normalizer = get_identity_normalizer_from_stat(stat)
        normalizer["action"] = this_normalizer

        # obs
        for key in self.lowdim_keys:
            stat = array_to_stats(self.replay_buffer[key])

            if key.endswith("pos"):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith("quat"):
                # quaternion is in [-1,1] already
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith("qpos"):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith("language"):
                continue  ## skip
            else:
                raise RuntimeError("unsupported")
            normalizer[key] = this_normalizer

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"])

    def norm_action(self, action: torch.Tensor) -> torch.Tensor:
        return (action - self.meta_action_mean) / self.meta_action_std

    def denorm_action(self, action: torch.Tensor) -> torch.Tensor:
        raw_action = action * self.meta_action_std + self.meta_action_mean  # gripper: [-1,1]

        raw_action = torch.clamp(
            raw_action,
            min=self.meta_action_min,
            max=self.meta_action_max
        )

        # binarize gripper
        raw_action[..., -1] = (raw_action[..., -1] > 0).float() * 2. - 1.  # to {-1,1}
        return raw_action

    def norm_agent_pos(self, agent_pos: torch.Tensor) -> torch.Tensor:
        assert self.has_joint_states, "joint_states not available in the dataset"
        assert agent_pos.shape[-1] == 8, f"agent_pos should have 8 dims, got {agent_pos.shape[-1]}"
        joint_states = agent_pos[..., :7]  # (T,7)
        gripper_states = agent_pos[..., 7:]  # (T,1)
        normed_gripper_states = (gripper_states - self.meta_gripper_states_mean[0]) / self.meta_gripper_states_std[0]
        normed_joint_states = (joint_states - self.meta_joint_mean) / self.meta_joint_std
        normed_agent_pos = torch.cat([normed_joint_states, normed_gripper_states], dim=-1)
        return normed_agent_pos

    def denorm_agent_pos(self, agent_pos: torch.Tensor) -> torch.Tensor:
        assert self.has_joint_states, "joint_states not available in the dataset"
        assert agent_pos.shape[-1] == 8, f"agent_pos should have 8 dims, got {agent_pos.shape[-1]}"
        joint_states = agent_pos[..., :7]  # (T,7)
        gripper_states = agent_pos[..., 7:]  # (T,1)
        denormed_gripper_states = gripper_states * self.meta_gripper_states_std[0] + self.meta_gripper_states_mean[0]
        denormed_joint_states = joint_states * self.meta_joint_std + self.meta_joint_mean
        denormed_agent_pos = torch.cat([denormed_joint_states, denormed_gripper_states], dim=-1)
        return denormed_agent_pos

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)
        '''
        replay_sample: Dict,keys=dict_keys(['action', 'agentview_rgb', 'ee_states', 'eye_in_hand_rgb', 'gripper_states', 'joint_states', 'language'])
        action,<class 'numpy.ndarray'>,shape=(33, 10),min=-1.0000,max=0.9993
        agentview_rgb,<class 'numpy.ndarray'>,shape=(33, 128, 128, 3),min=0.0000,max=251.0000
        ee_states,<class 'numpy.ndarray'>,shape=(33, 6),min=-0.3174,max=3.2354
        eye_in_hand_rgb,<class 'numpy.ndarray'>,shape=(33, 128, 128, 3),min=0.0000,max=197.0000
        gripper_states,<class 'numpy.ndarray'>,shape=(33, 2),min=-0.0396,max=0.0397
        joint_states,<class 'numpy.ndarray'>,shape=(33, 7),min=-2.4466,max=2.3973
        language,<class 'numpy.ndarray'>,shape=(33, 2, 30),min=0.0000,max=49407.0000
        '''

        obs_dict = dict()
        for key in self.rgb_keys:
            obs_dict[key] = np.moveaxis(data[key], -1, 1).astype(np.float32) / 255.0
            obs_dict[key] = np.rot90(obs_dict[key], k=2, axes=(2, 3)).copy()
            obs_dict[key] = np.flip(obs_dict[key], axis=3).copy()
            del data[key]
        for key in self.lowdim_keys:
            obs_dict[key] = data[key].astype(np.float32)
            del data[key]

        if self.data_aug:
            # (T,H,W,C)
            image_tensors = []
            T, H, W, C = -1, -1, -1, -1
            for camera_key in self.camera_keys:
                assert camera_key in obs_dict, f"camera_key {camera_key} not in obs_dict {obs_dict.keys()}"
                # combine static and gripper view for consistent augmentation
                image_tensor = torch.tensor(obs_dict[camera_key], dtype=torch.float32)  # (T,C,H,W)
                if len(image_tensors) == 0:
                    T, H, W, C = image_tensor.shape
                else:  # check shape
                    assert image_tensor.shape == (T, H, W, C), \
                        f"image shape mismatch: {image_tensor.shape} vs {(T, H, W, C)}"
                image_tensors.append(image_tensor)

            # image_tensor = torch.tensor(obs_dict["agentview_rgb"], dtype=torch.float32)
            image_tensor = torch.cat(image_tensors, dim=0)
            video_seed = torch.randint(0, 10000, (1,)).item()

            def consistent_augmentations(frame):
                # Set the random seed for each frame to ensure consistent augmentation
                torch.manual_seed(video_seed)
                augmentation = transforms.Compose(
                    [
                        transforms.ColorJitter(
                            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
                        ),  # Color jitter
                    ]
                )
                return augmentation(frame)

            augmented_images = torch.stack(
                [consistent_augmentations(frame) for frame in image_tensor]
            )  # NOTE: is this fast enough?
            augmented_images = augmented_images.chunk(len(self.camera_keys), dim=0)
            drop_mask: np.ndarray = np.random.rand(len(self.camera_keys)) < self.p_camera_drop
            if drop_mask.all():
                drop_mask[np.random.randint(len(drop_mask))] = False  # ensure at least one view is kept
            for idx, camera_key in enumerate(self.camera_keys):
                if drop_mask[idx]:
                    obs_dict[camera_key] = np.zeros((T, H, W, C), dtype=np.float32)
                else:
                    obs_dict[camera_key] = augmented_images[idx].numpy()
                assert obs_dict[camera_key].shape == (T, H, W, C), \
                    f"after augmentation, {camera_key} image shape mismatch: {obs_dict[camera_key].shape}"

        torch_data = {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "action": torch.from_numpy(data["action"].astype(np.float32)),
        }

        normed_actions = self.norm_action(torch_data["action"])
        if self.has_joint_states:
            normed_gripper_states = (torch_data["obs"]["gripper_states"] - self.meta_gripper_states_mean) / self.meta_gripper_states_std
            normed_ee_states = (torch_data["obs"]["ee_states"] - self.meta_ee_states_mean) / self.meta_ee_states_std
            normed_joint_states = (torch_data["obs"]["joint_states"] - self.meta_joint_mean) / self.meta_joint_std
        else:
            normed_gripper_states = torch.zeros((self.horizon, 2), dtype=torch.float32)
            normed_ee_states = torch.zeros((self.horizon, 6), dtype=torch.float32)
            normed_joint_states = torch.zeros((self.horizon, 7), dtype=torch.float32)

        ''' Get t5 embeddings '''
        if "t5" in self.language_emb_model:
            assert "t5_text_indices" in data, "t5_text_indices not found in data"
            t5_text_indices = data["t5_text_indices"]  # (T,)
            t5_text_embeddings = torch.from_numpy(
                self.t5_meta["unique_embeddings"][t5_text_indices][0]
            ).to(torch.bfloat16)  # (T[0],512,1024), only 1 unique text in a clip
        else:
            t5_text_embeddings = torch.zeros(512, 1024, dtype=torch.bfloat16)

        # Check the view choice
        ret_n_views = len(self.camera_keys)
        view_indices_selection = [self.CAMERA_TO_VIEW_ID[camera_key] for camera_key in self.camera_keys]
        view_indices_t = torch.tensor(view_indices_selection).repeat_interleave(self.horizon)
        latent_view_indices_t = torch.tensor(view_indices_selection).repeat_interleave(self.state_t)
        n_video_tensors = []
        for camera_key in self.camera_keys:
            one_video_tensor = (torch_data['obs'][camera_key].permute(1, 0, 2, 3) * 255.).to(torch.uint8)  # (C,T,H,W)
            n_video_tensors.append(one_video_tensor)
        ret_video = torch.cat(n_video_tensors, dim=1)  # (C,T*ret_n_views,H,W)

        # Get agent pos
        ret_agent_pos = torch.cat((normed_joint_states, normed_gripper_states[:, :1]), dim=1)  # (T,7+1)

        ''' Remap keys to match the cosmos-predict2 output format '''
        remapped_data = {
            "action": normed_actions,  # (horizon,10), normalized by (x-mean)/std ~[-1,1]
            "video": ret_video,  # (3,T,128,128), [0,255] torch.uint8
            "agent_pos": ret_agent_pos,  # (T,7+1), [-1,1]
            "annotation_file": "None",
            "__key__": "None",
            "t5_text_embeddings": t5_text_embeddings,
            "t5_text_mask": torch.ones(512, dtype=torch.int64),  # although embeddings have zero vectors, mask is all 1
            "fps": 20,  # ori:10
            "image_size": torch.tensor([
                128, 128, 128, 128
            ]),
            "num_frames": self.horizon,  # v_cond (+v_out)
            "padding_mask": torch.zeros(1, 128, 128),  # (T,H,W) not used; cond mask is set in conditioner
            # "num_conditional_frames": n_v_cond,  # different across in a single batch
            # "num_conditional_actions": n_a_cond,
            # Multi-view related
            "sample_n_views": ret_n_views,
            "view_indices": view_indices_t,
            "latent_view_indices_B_T": latent_view_indices_t,  # here is (T,), but will be (B,T) in DataLoader
        }
        return remapped_data


def _convert_actions(raw_actions, abs_action, rotation_transformer):
    actions = raw_actions
    if abs_action:
        is_dual_arm = False
        if raw_actions.shape[-1] == 14:
            # dual arm
            raw_actions = raw_actions.reshape(-1, 2, 7)
            is_dual_arm = True

        pos = raw_actions[..., :3]
        rot = raw_actions[..., 3:6]
        gripper = raw_actions[..., 6:]
        rot = rotation_transformer.forward(rot)
        raw_actions = np.concatenate([pos, rot, gripper], axis=-1).astype(np.float32)

        if is_dual_arm:
            raw_actions = raw_actions.reshape(-1, 20)
        actions = raw_actions
    return actions


def _convert_robomimic_to_replay(
        store,
        shape_meta,
        dataset_path,
        abs_action,
        rotation_transformer,
        n_workers=None,
        max_inflight_tasks=None,
        language_emb_model=None,
):
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    # parse shape_meta
    rgb_keys = list()
    lowdim_keys = list()
    # construct compressors and chunks
    obs_shape_meta = shape_meta["obs"]
    for key, attr in obs_shape_meta.items():
        shape = attr["shape"]
        type = attr.get("type", "low_dim")
        if type == "rgb":
            rgb_keys.append(key)
        elif type == "low_dim":
            lowdim_keys.append(key)

    root = zarr.group(store)
    data_group = root.require_group("data", overwrite=True)
    meta_group = root.require_group("meta", overwrite=True)

    file_handles = []  # Store file handles if you need to keep them open
    demos_all = {}
    language_all = {}
    count = 0

    if language_emb_model == "clip":
        tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    else:
        raise NotImplementedError(f"Language model {language_emb_model} not implemented")

    dataset_paths = glob.glob(dataset_path + "/*.hdf5")
    print("dataset paths:", "\n".join(dataset_paths))
    '''
    uva text: [
    'living room scene 2 put both the cream cheese box and the butter in the basket'
    'living room scene 6 put the white mug on the plate and put the chocolate pudding to the right of the plate'
    'kitchen scene 8 put both moka pots on the stove'
    'kitchen scene 3 turn on the stove and put the moka pot on it'
    'kitchen scene 6 put the yellow and white mug in the microwave and close it'
    'living room scene 1 put both the alphabet soup and the cream cheese box in the basket'
    'kitchen scene 4 put the black bowl in the bottom drawer of the cabinet and close it'
    'living room scene 5 put the white mug on the left plate and put the yellow and white mug on the right plate'
    'living room scene 2 put both the alphabet soup and the tomato sauce in the basket'
    'study scene 1 pick up the book and place it in the back compartment of the caddy'
    ]
    dataset paths: [
    'KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_demo.hdf5', 
    'STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_demo.hdf5', 
    'KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_demo.hdf5', 
    'LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_demo.hdf5', 
    'KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5', 
    'LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_demo.hdf5', 
    'LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_demo.hdf5', 
    'LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_demo.hdf5', 
    'KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo.hdf5', 
    'LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_demo.hdf5']
    '''
    uva_order = {
        "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_demo.hdf5": 0,
        "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_demo.hdf5": 1,
        "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_demo.hdf5": 2,
        "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5": 3,
        "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_demo.hdf5": 4,
        "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_demo.hdf5": 5,
        "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo.hdf5": 6,
        "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_demo.hdf5": 7,
        "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_demo.hdf5": 8,
        "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_demo.hdf5": 9,
    }
    dataset_paths = sorted(dataset_paths, key=lambda x: uva_order[x.split("/")[-1]])
    print("[DEBUG] use uva order for dataset paths:", "\n".join(dataset_paths))

    for dataset_path_each in dataset_paths:
        language_goal = " ".join(dataset_path_each.split("/")[-1][:-10].split("_"))
        print(f"Loading {dataset_path_each}")
        file = h5py.File(
            dataset_path_each, "r"
        )  # Open the file without closing it immediately
        file_handles.append(
            file
        )  # Keep track of the file handle to avoid it being closed
        demos = file["data"]

        for i in range(len(demos)):
            demo = demos[f"demo_{i}"]
            demos_all[f"demo_{count}"] = demo
            language_all[f"demo_{count}"] = language_goal
            count += 1
    print("Total demos:", count)

    seq_max_len = 30

    if language_emb_model == "clip":
        language_all_tokens = [
            tokenizer(
                language_all[f"demo_{i}"],
                padding="max_length",
                max_length=seq_max_len,
                return_tensors="pt",
            )
            for i in range(len(language_all))
        ]
        language_input_ids = [
            item.input_ids.unsqueeze(1) for item in language_all_tokens
        ]
        language_attention_mask = [
            item.attention_mask.unsqueeze(1) for item in language_all_tokens
        ]
    else:
        raise NotImplementedError(f"Language model {language_emb_model} not implemented")

    demos = demos_all
    episode_ends = list()
    prev_end = 0
    for i in range(len(demos)):
        demo = demos[f"demo_{i}"]
        episode_length = demo["actions"].shape[0]
        episode_end = prev_end + episode_length
        prev_end = episode_end
        episode_ends.append(episode_end)
    n_steps = episode_ends[-1]
    episode_starts = [0] + episode_ends[:-1]
    _ = meta_group.array(
        "episode_ends", episode_ends, dtype=np.int64, compressor=None, overwrite=True
    )

    # save lowdim data
    for key in tqdm(lowdim_keys + ["action"], desc="Loading lowdim data"):
        data_key = "obs/" + key
        if key == "action":
            data_key = "actions"
            this_language_data = list()
        if key == "language":
            continue
        this_data = list()
        for i in range(len(demos)):
            demo = demos[f"demo_{i}"]
            this_data.append(demo[data_key][:].astype(np.float32))

            if key == "action":
                if language_emb_model == "clip":
                    language_tokens = torch.cat(
                        [language_input_ids[i], language_attention_mask[i]], dim=1
                    )
                    this_language_data.append(
                        language_tokens.repeat(this_data[-1].shape[0], 1, 1)
                    )
                else:
                    raise NotImplementedError(f"Language model {language_emb_model} not implemented")

        this_data = np.concatenate(this_data, axis=0)

        if key == "action":
            this_data = _convert_actions(
                raw_actions=this_data,
                abs_action=abs_action,
                rotation_transformer=rotation_transformer,
            )

            assert this_data.shape == (n_steps,) + tuple(shape_meta["action"]["shape"])

            this_language_data = np.concatenate(this_language_data, axis=0)
            if language_emb_model == "clip":
                assert this_language_data.shape == (n_steps,) + tuple([2, seq_max_len])
            else:
                raise NotImplementedError(f"Language model {language_emb_model} not implemented")
        else:
            assert this_data.shape == (n_steps,) + tuple(
                shape_meta["obs"][key]["shape"]
            )
        _ = data_group.array(
            name=key,
            data=this_data,
            shape=this_data.shape,
            chunks=this_data.shape,
            compressor=None,
            dtype=this_data.dtype,
        )

        if key == "action":
            _ = data_group.array(
                name="language",
                data=this_language_data,
                shape=this_language_data.shape,
                chunks=this_language_data.shape,
                compressor=None,
                dtype=this_language_data.dtype,
            )

    def img_copy(zarr_arr, zarr_idx, hdf5_arr, hdf5_idx):
        try:
            zarr_arr[zarr_idx] = hdf5_arr[hdf5_idx]
            # make sure we can successfully decode
            _ = zarr_arr[zarr_idx]
            return True
        except Exception as e:
            return False

    with tqdm(
            total=n_steps * len(rgb_keys), desc="Loading image data", mininterval=1.0
    ) as pbar:
        # one chunk per thread, therefore no synchronization needed
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = set()
            for key in rgb_keys:
                data_key = "obs/" + key
                shape = tuple(shape_meta["obs"][key]["shape"])
                c, h, w = shape
                this_compressor = Jpeg2k(level=50)
                img_arr = data_group.require_dataset(
                    name=key,
                    shape=(n_steps, h, w, c),
                    chunks=(1, h, w, c),
                    compressor=this_compressor,
                    dtype=np.uint8,
                )

                for episode_idx in range(len(demos)):
                    demo = demos[f"demo_{episode_idx}"]
                    hdf5_arr = demo["obs"][key]
                    for hdf5_idx in range(hdf5_arr.shape[0]):
                        if len(futures) >= max_inflight_tasks:
                            # limit number of inflight tasks
                            completed, futures = concurrent.futures.wait(
                                futures, return_when=concurrent.futures.FIRST_COMPLETED
                            )
                            for f in completed:
                                if not f.result():
                                    raise RuntimeError("Failed to encode image!")
                            pbar.update(len(completed))

                        zarr_idx = episode_starts[episode_idx] + hdf5_idx
                        futures.add(
                            executor.submit(
                                img_copy, img_arr, zarr_idx, hdf5_arr, hdf5_idx
                            )
                        )
            completed, futures = concurrent.futures.wait(futures)
            for f in completed:
                if not f.result():
                    raise RuntimeError("Failed to encode image!")
            pbar.update(len(completed))

    # Ensure you close all files when you're done with them
    for file in file_handles:
        file.close()

    replay_buffer = ReplayBuffer(root)
    return replay_buffer


def normalizer_from_stat(stat):
    max_abs = np.maximum(stat["max"].max(), np.abs(stat["min"]).max())
    scale = np.full_like(stat["max"], fill_value=1 / max_abs)
    offset = np.zeros_like(stat["max"])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale, offset=offset, input_stats_dict=stat
    )
