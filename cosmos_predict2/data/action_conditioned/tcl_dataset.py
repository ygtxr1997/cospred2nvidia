import copy
import os
from typing import Union, List, Tuple, Dict
import numpy as np
import torch
import torch.utils.data
from torchvision.transforms import transforms

from robokit.data.tcl_datasets import TCLDataset, TCLDatasetHDF5


class TCLImageDataset(torch.utils.data.Dataset):
    VIEW_CHOICES = ["image", "gripper"]
    CAMERA_TO_VIEW_ID = {
        "image": 0,
        "gripper": 1,
    }
    def __init__(self,
                 # RoboKit Dataset
                 data_root: str,
                 # Data sequence
                 horizon: int,
                 pad_before: int,
                 pad_after: int,
                 # Data format
                 shape_meta: dict,
                 norm_action_type: str = "minmax",
                 # MDT related
                 batch_size: int = 64,
                 num_workers: int = 8,
                 key: str = "lang",
                 chose_ratio: float = 1.,
                 img_gen_frame_diff: int = 3,
                 # Others
                 seed: int = 42,
                 val_ratio: float = 0.01,
                 split: str = "train",
                 remap_index: np.ndarray = None,  # only for val_set
                 max_train_episodes: int = 90,
                 transform_color_jitter: bool = True,
                 # RoboKit Dataset
                 h5_path: str = None,
                 use_h5: bool = False,
                 statistics_path: str = None,  # if None, loading `statistics.json' from data root
                 # Language related
                 language_emb_model: str = '',
                 # Multi-view related
                 camera_keys: Union[List[str], Tuple[str]] = ("image",),
                 p_camera_drop: float = 0.0,
                 switch_camera_view: bool = False,  # [Warning] only when data collection makes mistake
                 ):
        for camera_key in camera_keys:
            assert camera_key in self.VIEW_CHOICES, f"camera_key must be in {self.VIEW_CHOICES}"

        # RoboKit Dataset
        self.data_root = data_root
        self.shape_meta = shape_meta
        self.load_keys = ["rel_actions", "primary_rgb", "gripper_rgb", "robot_obs", "language_text"]
        self.h5_path = h5_path
        self.use_h5 = use_h5
        if not use_h5:
            self.tcl_dataset = TCLDataset(data_root, use_extracted=True, load_keys=self.load_keys)
        else:
            assert os.path.exists(h5_path), f"h5_path: ${h5_path} not exists"
            self.tcl_dataset = TCLDatasetHDF5(
                data_root, h5_path,
                use_extracted=True, load_keys=self.load_keys)
        if statistics_path is None:
            statistics_path = os.path.join(data_root, "statistics.json")
        self.data_meta = self.tcl_dataset.load_statistics_from_json(statistics_path)
        self.norm_action_type = norm_action_type
        assert self.norm_action_type in ["minmax", "mean", "identity"], "norm type must be minmax, mean, or identity"
        self.all_rel_actions = self.tcl_dataset.extracted_data["rel_actions"]
        self.dataset_min = np.array(self.data_meta["min"])
        self.dataset_max = np.array(self.data_meta["max"])
        self.dataset_mean = np.array(self.data_meta["mean"])
        self.dataset_std = np.array(self.data_meta["std"])
        self.dataset_total_len = self.data_meta["total_len"]
        self.dataset_meta_dict = {
            'min': self.dataset_min, 'max': self.dataset_max, 'mean': self.dataset_mean, 'std': self.dataset_std,
        }
        if self.norm_action_type == "mean":
            eps = 1e-6  # 或者其他合适的小值
            assert np.all(self.dataset_std > eps), \
                f"Some std values too small: {self.dataset_std}, min_std={self.dataset_std.min()}"

        self.tasks = self.tcl_dataset.tasks
        self.task_lengths = self.tcl_dataset.task_lengths
        self.ep_fns = self.tcl_dataset.ep_fns
        self.map_index_to_task_id = self.tcl_dataset.map_index_to_task_id

        # MDT related
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.chose_ratio = chose_ratio
        self.img_gen_frame_diff = img_gen_frame_diff

        # Others
        self.seed = seed
        self.val_ratio = val_ratio
        self.split = split
        self.max_train_episodes = max_train_episodes

        # Language and Multi-view related
        self.language_emb_model = language_emb_model
        if "t5xxl" in self.language_emb_model:
            self.language_embedding_key = "language_embedding"
            self.language_embedding_subdir = "lang_emb_t5xxl"  # default subdir
            language_embedding_file_path = os.path.join(
                self.data_root, "../", self.language_embedding_subdir, "all/t5_embeddings.npz")
            language_embedding_file_path = os.path.abspath(language_embedding_file_path)
            print("[DEBUG] Loading language embeddings from:", language_embedding_file_path)
            lang_emb_data = np.load(language_embedding_file_path, allow_pickle=True)
            self.lang_text_to_emb = lang_emb_data['text_to_embedding_map'].item()

        self.camera_keys = camera_keys
        self.p_camera_drop = p_camera_drop
        self.switch_camera_view = switch_camera_view

        # 创建重映射索引
        if self.split == "train":
            self.index_all = np.arange(len(self.tcl_dataset))  # split train and val
            np.random.shuffle(self.index_all)
            val_size = int(len(self.index_all) * val_ratio)
            self.index_val = self.index_all[:val_size]
            self.index_train = self.index_all[val_size:]
            self.remap_index = self.index_train  # local to global
        elif self.split == "val":
            self.remap_index = remap_index  # provided by the train_set
        else:
            raise NotImplementedError("split not implemented")

        # Sampling a data sequence
        max_obs = pad_before + 1
        max_act_out = horizon
        assert max_obs >= 1 and max_obs % 4 == 1, "max_obs-1 must be non-negative and multiple of 4 to match 4x downsampled image size"
        assert max_act_out % 4 == 0, "max_act_out must be positive and multiple of 4 to match 4x downsampled image size"
        self.max_obs = max_obs
        self.max_act_out = max_act_out
        self.max_seq_len = self.max_obs + max_act_out

        self.horizon = horizon
        self.state_t = 1 + (self.max_seq_len - 1) // 4  # 33->9, 17->5
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.task_prefix_lengths = [0 for _ in range(len(self.task_lengths))]
        for i in range(1, len(self.task_prefix_lengths)):  # 3 means [0+1+2]
            self.task_prefix_lengths[i] = self.task_prefix_lengths[i - 1] + self.task_lengths[i - 1]

        # Data format and preprocessing
        self.obs_image_shape = shape_meta["obs"]["image"]["shape"]  # [3, H, W]
        self.obs_gripper_shape = shape_meta["obs"]["gripper"]["shape"] if "gripper" in shape_meta["obs"] else None
        self.joint_state_shape = shape_meta["obs"]["joint_state"]["shape"]
        self.action_shape = shape_meta["action"]["shape"]  # [7,]
        obs_image_wh_ratio = float(self.obs_image_shape[2]) / float(self.obs_image_shape[1])  # wh 4:3=16:12=12:9
        transform_list = [
            transforms.ToPILImage(),  # wh 16:9
            transforms.Resize(self.obs_image_shape[1:]),
        ]
        self.out_resize = self.obs_image_shape[1:]
        if transform_color_jitter:
            # DP used
            # transforms.RandomResizedCrop(size=self.obs_image_shape[1:], scale=(0.68, 0.82),  # 12/16=0.75
            #                              ratio=(0.9 * obs_image_wh_ratio, 1.1 * obs_image_wh_ratio)),  # not using this would be better?
            # transform_list.append(transforms.ColorJitter(brightness=0.05,
            #                                              contrast=0.05,
            #                                              saturation=0.05,
            #                                              hue=0.05))
            transform_list.append(transforms.ColorJitter(brightness=0.4,
                                                         contrast=0.4,
                                                         saturation=0.4,
                                                         hue=0.15))
        transform_list.append(transforms.ToTensor())
        self.obs_image_transform = transforms.Compose(transform_list)  # Similar augmentation params with OCTO

        gripper_transform_list = copy.deepcopy(transform_list)
        gripper_transform_list[1] = transforms.Resize(self.obs_gripper_shape[1:])
        self.obs_gripper_transform = transforms.Compose(gripper_transform_list)

        gen_transform_list = copy.deepcopy(gripper_transform_list)
        gen_transform_list[1] = transforms.Resize((112, 112))
        self.gen_transform = transforms.Compose(gen_transform_list)

        print(f"[TCLImageDataset] dataset loaded, split={self.split}, val_ratio={self.val_ratio}, len={len(self)}; "
              f"meta_total_len={self.dataset_total_len}, norm_type={self.norm_action_type}, "
              f"action_min={self.dataset_min}, action_max={self.dataset_max}, "
              f"action_mean={self.dataset_mean}, action_std={self.dataset_std}")

    @staticmethod
    def save_meta_as_json(data_meta: dict, save_path: str):
        import json
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            print("[Warning] making dir {}".format(save_dir))
            os.makedirs(save_dir, exist_ok=True)
        with open(save_path, "w") as fp:
            json.dump(data_meta, fp, indent=4)

    def get_validation_dataset(self):
        return self.create_val_dataset(self)

    @classmethod
    def create_val_dataset(cls, instance: 'TCLImageDataset'):
        val_set = cls(
            data_root=instance.data_root,
            horizon=instance.horizon,
            pad_before=instance.pad_before,
            pad_after=instance.pad_after,
            shape_meta=instance.shape_meta,
            norm_action_type=instance.norm_action_type,
            seed=instance.seed,
            val_ratio=instance.val_ratio,
            split='val',
            remap_index=instance.index_val,
            max_train_episodes=instance.max_train_episodes,
            use_h5=False,  # no need to use h5
            batch_size=16,
            language_emb_model=instance.language_emb_model,
            camera_keys=instance.camera_keys,
            p_camera_drop=instance.p_camera_drop,
        )
        val_set.tcl_dataset.total_length = min(6000, val_set.tcl_dataset.total_length)
        return val_set

    def __len__(self):
        return len(self.remap_index)

    def __getitem__(self, abs_idx):
        abs_idx = self.remap_index[abs_idx]  # convert relative index to global absolute index

        abs_idx = abs_idx % self.__len__()
        obs_data = self._get_obs_data(abs_idx)
        act_data = self._get_act_data(abs_idx)

        item_data = {
            "robot_obs": obs_data["joint_state"],  # (T,8)
            "rgb_obs": {
                "rgb_static": obs_data['image'],
                "rgb_gripper": obs_data['gripper'],  # (T,C,H,W), in [-1,1]
                "gen_static": obs_data['gen_primary'],  # (1,C,H,W)
                "gen_gripper": obs_data['gen_gripper'],
            },
            "depth_obs": {},
            "actions": act_data,  # (T,7)
            "state_info": {
                "scene_obs": np.zeros((1, 24)),
                "robot_obs": np.zeros((1, 15))
            },
            "lang": {},
            "lang_text": obs_data['language'],  # str
            "idx": abs_idx,
            "future_frame_diff": self.img_gen_frame_diff,
        }

        ''' Get t5 embeddings '''
        if "t5" in self.language_emb_model:
            lang_text = obs_data['language']
            assert lang_text in self.lang_text_to_emb, f"{lang_text} not found in self.lang_text_to_emb"
            t5_text_embeddings = torch.from_numpy(
                self.lang_text_to_emb[lang_text]
            ).to(torch.bfloat16)  # (512,1024)
        else:
            t5_text_embeddings = torch.zeros(512, 1024, dtype=torch.bfloat16)

        # Camera drop
        T, C, H, W = obs_data['image'].shape
        drop_mask: np.ndarray = np.random.rand(len(self.camera_keys)) < self.p_camera_drop
        if drop_mask.all():
            drop_mask[np.random.randint(len(drop_mask))] = False  # ensure at least one view is kept
        for idx, camera_key in enumerate(self.camera_keys):
            if drop_mask[idx]:
                obs_data[camera_key] = torch.zeros((T, H, W, C), dtype=torch.float32)
            else:
                obs_data[camera_key] = obs_data[camera_key].permute(0, 2, 3, 1)  # (T,C,H,W)->(T,H,W,C)
            assert obs_data[camera_key].shape == (T, H, W, C), \
                f"after augmentation, {camera_key} image shape mismatch: {obs_data[camera_key].shape}"

        # Check the view choice
        ret_n_views = len(self.camera_keys)
        view_indices_selection = [self.CAMERA_TO_VIEW_ID[camera_key] for camera_key in self.camera_keys]
        view_indices_t = torch.tensor(view_indices_selection).repeat_interleave(self.horizon)
        latent_view_indices_t = torch.tensor(view_indices_selection).repeat_interleave(self.state_t)
        n_video_tensors = []
        for camera_key in self.camera_keys:
            one_video_tensor = (obs_data[camera_key].permute(3, 0, 1, 2) * 127.5 + 127.5).to(torch.uint8)  # (C,T,H,W)
            n_video_tensors.append(one_video_tensor)
        ret_video = torch.cat(n_video_tensors, dim=1)  # (C,T*ret_n_views,H,W)

        ret_action = torch.from_numpy(act_data).to(torch.float32)
        ret_agent_pos = obs_data["joint_state"]  # without normalization

        ''' Remap keys to match the cosmos-predict2 output format '''
        remapped_data = {
            "action": ret_action,  # (horizon,7), normalized by (x-mean)/std ~[-1,1]
            "video": ret_video,  # (3,T,128,160), [0,255] torch.uint8
            "agent_pos": ret_agent_pos,  # (T,7+1), [-1,1]
            "annotation_file": "None",
            "__key__": "None",
            "lang_text": obs_data['language'],  # str
            "t5_text_embeddings": t5_text_embeddings,
            "t5_text_mask": torch.ones(512, dtype=torch.int64),  # although embeddings have zero vectors, mask is all 1
            "fps": 30,  # ori:10
            "image_size": torch.tensor([
                128, 160, 240, 320
            ]),
            "num_frames": self.horizon,  # v_cond (+v_out)
            "padding_mask": torch.zeros(1, 128, 160),  # (T,H,W) not used; cond mask is set in conditioner
            # Multi-view related
            "sample_n_views": ret_n_views,
            "view_indices": view_indices_t,
            "latent_view_indices_B_T": latent_view_indices_t,  # here is (T,), but will be (B,T) in DataLoader
        }
        return remapped_data

    def _abs_idx_to_rel_idx(self, abs_idx: int):
        task_id = self.map_index_to_task_id[abs_idx]
        task_len = self.task_lengths[task_id]
        task_prefix_len = self.task_prefix_lengths[task_id]
        rel_idx = abs_idx - task_prefix_len
        assert 0 <= rel_idx < task_len
        return rel_idx, task_id

    def _rel_idx_to_abs_idx(self, rel_idx: int, task_id: int):
        task_prefix_len = self.task_prefix_lengths[task_id]
        task_len = self.task_lengths[task_id]
        if rel_idx < 0:
            abs_idx = None
        elif rel_idx >= task_len:
            abs_idx = None
        else:
            abs_idx = rel_idx + task_prefix_len
            assert task_prefix_len <= abs_idx < task_prefix_len + task_len
        return abs_idx

    def _get_abs_obs_indices(self, abs_now_idx: int):
        # [start_idx, end_idx)
        rel_now_idx, task_id = self._abs_idx_to_rel_idx(abs_now_idx)
        start_idx = rel_now_idx - self.pad_before
        end_idx = rel_now_idx + 1 + self.max_act_out  # NOTE: we need to load all frames for cosmos, ori:rel_now_idx + 1
        rel_indices = list(range(start_idx, end_idx))
        return [self._rel_idx_to_abs_idx(rel_id, task_id) for rel_id in rel_indices]

    def _get_abs_gen_index(self, abs_now_idx: int):
        # [start_idx, end_idx)
        rel_now_idx, task_id = self._abs_idx_to_rel_idx(abs_now_idx)
        gen_idx = rel_now_idx + self.img_gen_frame_diff
        abs_gen_idx = self._rel_idx_to_abs_idx(gen_idx, task_id)
        return abs_gen_idx if abs_gen_idx is not None else abs_now_idx

    def _get_abs_act_indices(self, abs_now_idx: int):
        # [start_idx, end_idx)
        rel_now_idx, task_id = self._abs_idx_to_rel_idx(abs_now_idx)
        start_idx = rel_now_idx
        end_idx = rel_now_idx + self.horizon
        rel_indices = list(range(start_idx, end_idx))
        return [self._rel_idx_to_abs_idx(rel_id, task_id) for rel_id in rel_indices]

    def _get_obs_data(self, abs_idx: int):
        task_id = self.map_index_to_task_id[abs_idx]
        abs_obs_indices = self._get_abs_obs_indices(abs_idx)
        # print(f"[DEBUG] _get_obs_data: abs_idx={abs_idx}, task_id={task_id}, "
        #       f"task_1st_ep={self.task_prefix_lengths[task_id]}, "
        #       f"task_last_ep={self.task_prefix_lengths[task_id] + self.task_lengths[task_id]}"
        #       )
        # print(f"[DEBUG] _get_obs_data: abs_obs_indices={abs_obs_indices} ")
        abs_gen_index = self._get_abs_gen_index(abs_idx)
        obs_keys = self.shape_meta["obs"].keys()
        obs_data = {
            k: [] for k in obs_keys
        }
        language = ""
        for idx in abs_obs_indices:
            if idx is None:
                c, h, w = self.obs_image_shape
                zero_rgb = torch.ones((c, h, w)).to(torch.float32) * -1  # all -1
                primary_rgb = zero_rgb
                gripper_rgb = zero_rgb
                tcp_pose = torch.zeros((8,)).to(torch.float32)
            else:
                sample_dict = self.tcl_dataset.__getitem__(idx)
                if not self.switch_camera_view:  # commonly used
                    primary_rgb = sample_dict['primary_rgb']  # (H,W,C)
                    gripper_rgb = sample_dict['gripper_rgb']  # (H,W,C)
                else:
                    primary_rgb = sample_dict['gripper_rgb']  # (H,W,C)
                    gripper_rgb = sample_dict['primary_rgb']  # (H,W,C)
                tcp_pose = joint_state = sample_dict['robot_obs'][:6]  # (6,)
                language = sample_dict['language_text']  # string
                # Preprocess
                primary_rgb = self.obs_image_transform(primary_rgb)  # (C,H,W), in [0, 1]
                primary_rgb = primary_rgb * 2. - 1.  # in [-1, 1]
                tcp_pose = torch.from_numpy(np.concatenate([tcp_pose, np.zeros(2)])).to(torch.float32)  # (8,)
                if "gripper" in obs_keys:
                    gripper_rgb = self.obs_gripper_transform(gripper_rgb)
                    gripper_rgb = gripper_rgb * 2. - 1.
            obs_data["image"].append(primary_rgb)
            obs_data["joint_state"].append(tcp_pose)
            if "gripper" in obs_keys:
                obs_data["gripper"].append(gripper_rgb)
        obs_data = {k: torch.stack(v) for k, v in obs_data.items()}
        # obs_data["image"] = torch.stack(obs_data["image"])  # should be (T,C,H,W)
        # obs_data["joint_state"] = torch.stack(obs_data["joint_state"])  # (T,6)
        obs_data['language'] = language

        # Get goal data for generation
        goal_dict = self.tcl_dataset.__getitem__(abs_gen_index)
        gen_primary = self.gen_transform(goal_dict['primary_rgb'])  # (C,H,W,), in [0,1]
        gen_gripper = self.gen_transform(goal_dict['gripper_rgb'])
        gen_primary = gen_primary * 2. - 1.
        gen_gripper = gen_gripper * 2. - 1.
        obs_data['gen_primary'] = gen_primary[None, :, :, :]
        obs_data['gen_gripper'] = gen_gripper[None, :, :, :]

        return obs_data

    def _get_act_data(self, abs_idx: int):
        task_id = self.map_index_to_task_id[abs_idx]
        abs_act_indices = self._get_abs_act_indices(abs_idx)
        # print(f"[DEBUG] _get_act_data: abs_idx={abs_idx}, task_id={task_id}, "
        #       f"task_1st_ep={self.task_prefix_lengths[task_id]}, "
        #       f"task_last_ep={self.task_prefix_lengths[task_id] + self.task_lengths[task_id]}"
        #       )
        # print(f"[DEBUG] _get_obs_data: abs_obs_indices={abs_act_indices} ")
        act_data = []
        for idx in abs_act_indices:
            if idx is None:
                zero_act = np.zeros(self.action_shape).astype(np.float32)
                act_data.append(zero_act)
            else:
                rel_action = self.all_rel_actions[idx]  # (7,)
                act_data.append(rel_action)
        act_data = np.stack(act_data)  # (T,7), in [act_min, act_max]

        act_data = self.norm_action(act_data, self.norm_action_type, self.dataset_meta_dict)

        return act_data

    @staticmethod
    def norm_action(action_data, norm_type: str, meta_data: dict):
        # Consider different norm types
        if norm_type == "minmax":
            dataset_min = meta_data['min']
            dataset_max = meta_data['max']
            action_data = (action_data - dataset_min) / (dataset_max - dataset_min)  # norm here, in [0,1]
            action_data = action_data * 2. - 1.  # in [-1,1]
        elif norm_type == "mean":
            dataset_min = meta_data['min']
            dataset_max = meta_data['max']
            dataset_mean = meta_data['mean']
            dataset_std = meta_data['std']
            # Split into pose (first 6 dims) and gripper (last dim)
            pose_data = action_data[..., :-1]  # (T, 6)
            gripper_data = action_data[..., -1:]  # (T, 1)

            # Normalize pose with mean/std
            pose_normalized = (pose_data - dataset_mean[:-1]) / dataset_std[:-1]

            # Normalize gripper with minmax
            gripper_normalized = (gripper_data - dataset_min[-1:]) / (
                    dataset_max[-1:] - dataset_min[-1:])
            gripper_normalized = gripper_normalized * 2. - 1.  # to [-1,1]

            # Concatenate back
            action_data = np.concatenate([pose_normalized, gripper_normalized], axis=-1)
        else:
            assert norm_type == "identity"
            action_data = action_data
        return action_data

    @staticmethod
    def denorm_action(action_data, norm_type: str, meta_data: dict):
        if norm_type == "minmax":
            dataset_min = meta_data['min']
            dataset_max = meta_data['max']
            action_data = (action_data + 1.) / 2.  # [-1,1] to [0,1]
            action_data = action_data * (dataset_max - dataset_min) + dataset_min  # to original scale
        elif norm_type == "mean":
            dataset_min = meta_data['min']
            dataset_max = meta_data['max']
            dataset_mean = meta_data['mean']
            dataset_std = meta_data['std']
            # Split into pose (first 6 dims) and gripper (last dim)
            pose_data = action_data[..., :-1]  # (T, 6)
            gripper_data = action_data[..., -1:]  # (T, 1)

            # Denormalize pose with mean/std
            pose_denormalized = pose_data * dataset_std[:-1] + dataset_mean[:-1]

            # Denormalize gripper with minmax
            gripper_denormalized = (gripper_data + 1.) / 2.  # to [0,1]
            gripper_denormalized = gripper_denormalized * (dataset_max[-1:] - dataset_min[-1:]) + dataset_min[-1:]

            # Concatenate back
            action_data = np.concatenate([pose_denormalized, gripper_denormalized], axis=-1)
        else:
            assert norm_type == "identity"
            action_data = action_data
        return action_data


class TCLMergeDataset(torch.utils.data.Dataset):
    def __init__(self,
                 # RoboKit Dataset
                 data_roots: list,  # List of data root paths, Difference (1)
                 # Data sequence
                 horizon: int,
                 pad_before: int,
                 pad_after: int,
                 # Data format
                 shape_meta: dict,
                 norm_action_type: str = "minmax",
                 # MDT related
                 batch_size: int = 64,
                 num_workers: int = 8,
                 key: str = "lang",
                 chose_ratio: float = 1.,
                 img_gen_frame_diff: int = 3,
                 # Others
                 seed: int = 42,
                 val_ratio: float = 0.01,
                 split: str = "train",
                 val_sets: list = None,  # Pre-created validation datasets for val split, Difference (2)
                 max_train_episodes: int = 90,
                 transform_color_jitter: bool = True,
                 # RoboKit Dataset
                 h5_paths: list = None,  # Difference (3)
                 use_h5: bool = False,
                 statistics_path: str = None,
                 **kwargs
                 ):
        self.data_roots = data_roots
        self.h5_paths = h5_paths

        self.seed = seed
        self.val_ratio = val_ratio
        self.split = split
        self.norm_action_type = norm_action_type

        # Create individual TCLImageDatasets with identity normalization
        if split == "train":
            # Create datasets from data_roots
            self.datasets = []
            self.dataset_lengths = []
            self.val_datasets = []  # Store validation datasets

            for data_idx, data_root in enumerate(self.data_roots):
                dataset = TCLImageDataset(
                    data_root=data_root,  # different across sub-datasets
                    horizon=horizon,
                    pad_before=pad_before,
                    pad_after=pad_after,
                    shape_meta=shape_meta,
                    norm_action_type="identity",  # Use identity first, we'll handle norm later
                    seed=seed,
                    val_ratio=val_ratio,
                    split=split,
                    h5_path=h5_paths[data_idx],  # different across sub-datasets
                    use_h5=use_h5,
                    max_train_episodes=max_train_episodes,
                    transform_color_jitter=transform_color_jitter,
                    **kwargs
                )
                self.datasets.append(dataset)
                self.dataset_lengths.append(len(dataset))

                # Create validation dataset if this is a train split
                val_dataset = dataset.get_validation_dataset()
                self.val_datasets.append(val_dataset)
        elif split == "val":
            assert val_sets is not None, "val_sets must not be None for val split"
            # Use pre-created validation datasets
            self.datasets = val_sets
            self.dataset_lengths = [len(d) for d in val_sets]
            self.val_datasets = []  # Empty for val split
        else:
            raise NotImplementedError("split type not supported")

        # Merge metadata for action normalization
        self.statistics_path = statistics_path
        self._merge_metadata()

        # Validate norm_action_type
        assert self.norm_action_type in ["minmax", "mean", "identity"], "norm type must be minmax, mean, or identity"
        if self.norm_action_type == "mean":
            eps = 1e-6
            assert np.all(self.merged_std > eps), \
                f"Some std values too small: {self.merged_std}, min_std={self.merged_std.min()}"

        # Copy other attributes from first dataset for compatibility
        first_dataset = self.datasets[0]
        self.horizon = first_dataset.horizon
        self.pad_before = first_dataset.pad_before
        self.pad_after = first_dataset.pad_after
        self.shape_meta = first_dataset.shape_meta
        self.img_gen_frame_diff = first_dataset.img_gen_frame_diff
        self.action_shape = first_dataset.action_shape

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.chose_ratio = chose_ratio
        self.img_gen_frame_diff = img_gen_frame_diff

        print(f"[TCLMergeDataset] datasets loaded from {len(data_roots)} roots, "
              f"split={self.split}, val_ratio={self.val_ratio}, len={len(self)}; "
              f"meta_total_len={self.merged_total_len}, norm_type={self.norm_action_type}, "
              f"action_min={self.merged_min}, action_max={self.merged_max}, "
              f"action_mean={self.merged_mean}, action_std={self.merged_std}")

    def _merge_metadata(self):
        """Merge metadata from all datasets"""
        if self.statistics_path is not None:
            import json
            print("[TCLMergeDataset] loading dataset statistics from:", self.statistics_path)
            with open(self.statistics_path, 'r') as json_file:
                statistics = json.load(json_file)
            self.merged_min = np.array(statistics["min"])
            self.merged_max = np.array(statistics["max"])
            self.merged_mean = np.array(statistics["mean"])
            self.merged_std = np.array(statistics["std"])
            self.merged_total_len = statistics["total_len"]
        else:  # Calculate merged statistics
            all_mins = []
            all_maxs = []
            all_means = []
            all_stds = []
            all_total_lens = []

            for dataset in self.datasets:
                all_mins.append(dataset.dataset_min)
                all_maxs.append(dataset.dataset_max)
                all_means.append(dataset.dataset_mean)
                all_stds.append(dataset.dataset_std)
                all_total_lens.append(dataset.dataset_total_len)

            # Calculate merged statistics
            self.merged_min = np.min(all_mins, axis=0)
            self.merged_max = np.max(all_maxs, axis=0)
            self.merged_total_len = sum(all_total_lens)

            # Calculate weighted mean and std
            total_samples = sum(all_total_lens)
            weighted_mean = np.zeros_like(all_means[0])
            for mean, length in zip(all_means, all_total_lens):
                weighted_mean += mean * length / total_samples
            self.merged_mean = weighted_mean

            # Calculate merged std using formula: var = E[X^2] - (E[X])^2
            weighted_var = np.zeros_like(all_stds[0])
            for mean, std, length in zip(all_means, all_stds, all_total_lens):
                var = std ** 2
                second_moment = var + mean ** 2
                weighted_var += second_moment * length / total_samples
            merged_var = weighted_var - self.merged_mean ** 2
            self.merged_std = np.sqrt(merged_var)

        self.merged_meta_dict = {
            'min': self.merged_min, 'max': self.merged_max, 'mean': self.merged_mean, 'std': self.merged_std,
        }

    def save_meta(self, save_json_path: str):
        meta_statistics = {
            "min": self.merged_min.tolist(),
            "max": self.merged_max.tolist(),
            "mean": self.merged_mean.tolist(),
            "std": self.merged_std.tolist(),
            "total_len": int(self.merged_total_len),
        }
        TCLImageDataset.save_meta_as_json(meta_statistics, save_json_path)
        print(f"[TCLMergeDataset] Meta data saved to: {save_json_path}")

    def get_validation_dataset(self):
        return self.create_val_dataset(self)

    @classmethod
    def create_val_dataset(cls, instance: 'TCLMergeDataset'):
        """Create validation dataset using pre-created validation sets"""
        if not hasattr(instance, 'val_datasets') or not instance.val_datasets:
            raise ValueError("No validation datasets available. Make sure this is a train dataset.")

        val_set = cls(
            data_roots=instance.data_roots,  # Keep for compatibility
            horizon=instance.horizon,
            pad_before=instance.pad_before,
            pad_after=instance.pad_after,
            shape_meta=instance.shape_meta,
            norm_action_type=instance.norm_action_type,  # Use the same norm type for val_set
            seed=instance.seed,
            val_ratio=instance.val_ratio,
            split='val',
            val_sets=instance.val_datasets,  # Pass pre-created validation datasets
            transform_color_jitter=False,  # No aug for val_set
        )
        return val_set

    def __len__(self):
        return sum(self.dataset_lengths)

    def __getitem__(self, idx):
        # Find which dataset this index belongs to
        current_idx = idx
        for i, dataset in enumerate(self.datasets):
            if current_idx < len(dataset):
                item_data = dataset.__getitem__(current_idx)

                # Update the idx field to reflect the global index
                item_data["idx"] = idx

                # Apply merged normalization to actions
                item_data["actions"] = TCLImageDataset.norm_action(
                    item_data["actions"], self.norm_action_type, meta_data=self.merged_meta_dict
                )

                return item_data
            current_idx -= len(dataset)

        raise IndexError(f"Index {idx} out of range")
