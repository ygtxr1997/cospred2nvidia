import copy
import os
from typing import Union, List, Tuple, Dict
import numpy as np
import torch
import torch.utils.data
from torchvision.transforms import transforms

from robokit.datasets.tcl_datasets import TCLDataset, TCLDatasetHDF5
from robokit.debug_utils.printer import beautiful_print


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
                 horizon: int,  # action horizon
                 pad_before: int,  # obs length - 1
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
                 verbose: bool = True,
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
                 future_frame_skip: int = 1,  # to reduce future frame prediction difficulty, default 1 means no skip
                 # NOTE: current force is also sampled like future frame skip
                 ):
        for camera_key in camera_keys:
            assert camera_key in self.VIEW_CHOICES, f"camera_key must be in {self.VIEW_CHOICES}"

        # RoboKit Dataset
        self.data_root = data_root
        self.shape_meta = shape_meta
        self.load_keys = ["rel_actions", "primary_rgb", "gripper_rgb", "robot_obs", "language_text", "force_torque"]
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
        self.data_meta = self.tcl_dataset.load_meta_from_json(statistics_path)
        self.norm_action_type = norm_action_type
        assert self.norm_action_type in ["minmax", "mean", "identity"], "norm type must be minmax, mean, or identity"
        self.all_rel_actions = self.tcl_dataset.extracted_data["rel_actions"]
        self.dataset_stats = self.data_meta["stats"]  # key: `rel_actions`, `robot_obs`, `force_torque`
        self.dataset_total_len = self.data_meta["total_len"]

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
        self.np_random = np.random.RandomState(self.seed)

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
        self.future_frame_skip = future_frame_skip

        # 创建重映射索引
        if self.split == "train":
            self.index_all = np.arange(len(self.tcl_dataset))  # split train and val
            # np.random.shuffle(self.index_all)
            self.np_random.shuffle(self.index_all)
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
        self.max_seq_len = self.max_obs + max_act_out  # without frames skipping
        self.max_pred_frames = max_act_out // self.future_frame_skip
        assert self.max_pred_frames * self.future_frame_skip == max_act_out, \
            "max_act_out must be divisible by future_frame_skip"
        assert self.max_pred_frames % 4 == 0, "max_pred_frames must be multiple of 4 to match 4x downsampled image size"

        self.horizon = horizon  # not used
        self.state_t = (1 + (self.max_obs - 1) // 4) + (self.max_pred_frames // 4)  # for multi-view latent state_t
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.task_prefix_lengths = [0 for _ in range(len(self.task_lengths))]
        for i in range(1, len(self.task_prefix_lengths)):  # 3 means [0+1+2]
            self.task_prefix_lengths[i] = self.task_prefix_lengths[i - 1] + self.task_lengths[i - 1]

        # Data format and preprocessing
        self.obs_image_shape = shape_meta["obs"]["image"]["shape"]  # [3, H, W]
        self.obs_gripper_shape = shape_meta["obs"]["gripper"]["shape"] if "gripper" in shape_meta["obs"] else None
        self.obs_force_shape = shape_meta["obs"]["force"]["shape"] if "force" in shape_meta["obs"] else None
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
              f"meta_total_len={self.dataset_total_len}, norm_type={self.norm_action_type}.")
        if verbose:
            beautiful_print(self.dataset_stats)

    def save_meta(self, save_path: str, force_path: bool = True):
        self.tcl_dataset.save_meta_as_json(save_path, meta_statistics=self.data_meta, force_path=force_path)

    @staticmethod
    def load_meta(json_path: str):
        import json
        with open(json_path, 'r') as json_file:
            statistics = json.load(json_file)
        return statistics

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
            future_frame_skip=instance.future_frame_skip,
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

        # Get forces
        T, C, H, W = obs_data['image'].shape
        if "force" in self.shape_meta["obs"]:
            ret_force = obs_data["force"]  # (T,6)
        else:
            ret_force = torch.zeros(T, 6, dtype=torch.float32)

        # Camera drop
        drop_mask: np.ndarray = self.np_random.rand(len(self.camera_keys)) < self.p_camera_drop
        if drop_mask.all():
            drop_mask[self.np_random.randint(len(drop_mask))] = False  # ensure at least one view is kept
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
        view_indices_t = torch.tensor(view_indices_selection).repeat_interleave(self.max_obs + self.max_pred_frames)
        latent_view_indices_t = torch.tensor(view_indices_selection).repeat_interleave(self.state_t)
        n_video_tensors = []
        for camera_key in self.camera_keys:
            one_video_tensor = (obs_data[camera_key].permute(3, 0, 1, 2) * 127.5 + 127.5).to(torch.uint8)  # (C,T,H,W)
            n_video_tensors.append(one_video_tensor)
        ret_video = torch.cat(n_video_tensors, dim=1)  # (C,T*ret_n_views,H,W)

        ret_action = torch.from_numpy(act_data).to(torch.float32)
        ret_agent_pos = obs_data["joint_state"]  # normalized tcp pose

        ''' Remap keys to match the cosmos-predict2 output format '''
        remapped_data = {
            "action": ret_action,  # (v2,7), normalized by (x-mean)/std ~[-1,1]
            "video": ret_video,  # (3,v1+v2,114,160), [0,255] torch.uint8
            "agent_pos": ret_agent_pos,  # (v1+v2,6), [-1,1]
            "force": ret_force,  # (v1+v2,6), normalized
            "annotation_file": "None",
            "__key__": "None",
            "lang_text": obs_data['language'],  # str
            "t5_text_embeddings": t5_text_embeddings,
            "t5_text_mask": torch.ones(512, dtype=torch.int64),  # although embeddings have zero vectors, mask is all 1
            "fps": 20,  # ori:30
            "image_size": torch.tensor([
                self.obs_image_shape[1], self.obs_image_shape[2], 240, 320
            ]),
            "num_frames": self.max_obs + self.max_pred_frames,  # v_cond (+v_out), with frame skipping
            "padding_mask": torch.zeros(1, self.obs_image_shape[1], self.obs_image_shape[2]),  # (T,H,W) not used; cond mask is set in conditioner
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
        end_idx = rel_now_idx + 1 + self.max_act_out

        # NOTE: we need to load all frames for cosmos
        # 1. Observation history (1,2,3,4)
        obs_indices = list(range(start_idx, rel_now_idx))
        # 2. Now (5,) + Future frames for generation (8,11,14) (skip=2 dropped: 6,7;9,10;12,13)
        future_indices = list(range(rel_now_idx, end_idx))  # H=8
        future_indices = future_indices[::self.future_frame_skip]  # skip frames

        # rel_indices = list(range(start_idx, end_idx))
        rel_indices = obs_indices + future_indices
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
        end_idx = rel_now_idx + self.max_act_out
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
                tcp_pose = torch.zeros((6,)).to(torch.float32)
                force_torque = torch.zeros((6,)).to(torch.float32)
            else:
                sample_dict = self.tcl_dataset.__getitem__(idx)
                if not self.switch_camera_view:  # commonly used
                    primary_rgb = sample_dict['primary_rgb']  # (H,W,C)
                    gripper_rgb = sample_dict['gripper_rgb']  # (H,W,C)
                else:
                    primary_rgb = sample_dict['gripper_rgb']  # (H,W,C)
                    gripper_rgb = sample_dict['primary_rgb']  # (H,W,C)
                robot_obs = self.norm_state_or_force(
                    sample_dict['robot_obs'], self.norm_action_type, self.dataset_stats["robot_obs"])
                tcp_pose = joint_state = robot_obs[:6]  # (6,), NOTE: we use TCP Pose as the robot state
                language = sample_dict['language_text']  # string
                # Preprocess
                primary_rgb = self.obs_image_transform(primary_rgb)  # (C,H,W), in [0, 1]
                primary_rgb = primary_rgb * 2. - 1.  # in [-1, 1]
                tcp_pose = torch.from_numpy(tcp_pose).to(torch.float32)  # (6,), no padding
                # tcp_pose = torch.from_numpy(np.concatenate([tcp_pose, np.zeros(2)])).to(torch.float32)  # (8,), padding
                if "gripper" in obs_keys:
                    gripper_rgb = self.obs_gripper_transform(gripper_rgb)
                    gripper_rgb = gripper_rgb * 2. - 1.
                if "force" in obs_keys:
                    force_torque = sample_dict['force_torque']  # (6,)
                    force_torque = self.norm_state_or_force(
                        force_torque, self.norm_action_type, self.dataset_stats["force_torque"])
                    force_torque = torch.from_numpy(force_torque).to(torch.float32)
            obs_data["image"].append(primary_rgb)
            obs_data["joint_state"].append(tcp_pose)
            if "gripper" in obs_keys:
                obs_data["gripper"].append(gripper_rgb)
            if "force" in obs_keys:
                obs_data["force"].append(force_torque)
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

        act_data = self.norm_action(act_data, self.norm_action_type, self.dataset_stats["rel_actions"])

        return act_data

    #### Meta Data Example
    # {
    #     "total_len": 28368,
    #     "stats": {
    #         "rel_actions": {
    #             "min": [-0.0984, -0.0922, -0.0914, -0.7853, -0.7321, -0.6968, 0.0],
    #             "max": [0.0937, 0.0945, 0.0961, 1.0228, 0.8995, 2.0322, 1.0],
    #             "mean": [-0.0003, 0.0008, -0.0006, 0.0045, 0.0007, 0.0031, 0.3404],
    #             "std": [0.0321, 0.0259, 0.0255, 0.055, 0.0512, 0.0711, 0.4738],
    #             "n_components": 7,
    #             "total_count": 28368
    #         },
    #         "robot_obs": {
    #             "min": [-0.2971, 0.2079, 0.2589, -3.1404, -0.3506, -0.8783, 0.0588, -0.9609, -0.857, 0.3227, -0.7636,
    #                     0.5171, 1.1618, 0.0],
    #             "max": [0.3126, 0.6141, 0.5443, 3.1412, 0.3682, 1.2539, 0.8471, 0.7843, 0.7682, 2.493, 1.2881, 1.9347,
    #                     4.3481, 1.0],
    #             "mean": [0.0012, 0.387, 0.3935, -2.8221, 0.004, 0.0988, 0.5998, -0.0284, 0.0189, 1.741, 0.0629, 1.19,
    #                      2.988, 0.3404],
    #             "std": [0.1323, 0.0762, 0.0415, 0.3487, 0.1367, 0.3918, 0.3346, 0.358, 0.2716, 0.2991, 0.2894, 0.23,
    #                     0.6303, 0.4738],
    #             "n_components": 14,
    #             "total_count": 28368
    #         },
    #         "force_torque": {
    #             "min": [-45.76, -28.6, -121.34, -1.087, -2.037, -0.079],
    #             "max": [-28.45, -6.86, -102.84, 0.632, 2.615, 0.395],
    #             "mean": [-35.941, -17.3023, -107.9096, -0.2612, 0.9989, 0.1379],
    #             "std": [1.7621, 1.9017, 1.4521, 0.1223, 0.219, 0.0179],
    #             "n_components": 6,
    #             "total_count": 28368
    #         }
    #     }
    # }
    @staticmethod
    def norm_action(action_data: np.ndarray, norm_type: str, meta_data: dict):
        # Consider different norm types
        if norm_type == "minmax":
            dataset_min = np.array(meta_data['min'])
            dataset_max = np.array(meta_data['max'])
            action_data = (action_data - dataset_min) / (dataset_max - dataset_min)  # norm here, in [0,1]
            action_data = action_data * 2. - 1.  # in [-1,1]
        elif norm_type == "mean":
            dataset_min = np.array(meta_data['min'])
            dataset_max = np.array(meta_data['max'])
            dataset_mean = np.array(meta_data['mean'])
            dataset_std = np.array(meta_data['std'])
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
    def denorm_action(action_data: np.ndarray, norm_type: str, meta_data: dict):
        if norm_type == "minmax":
            dataset_min = np.array(meta_data['min'])
            dataset_max = (meta_data['max'])
            action_data = (action_data + 1.) / 2.  # [-1,1] to [0,1]
            action_data = action_data * (dataset_max - dataset_min) + dataset_min  # to original scale
        elif norm_type == "mean":
            dataset_min = np.array(meta_data['min'])
            dataset_max = np.array(meta_data['max'])
            dataset_mean = np.array(meta_data['mean'])
            dataset_std = np.array(meta_data['std'])
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

    @staticmethod
    def norm_state_or_force(in_data: np.ndarray, norm_type: str, meta_data: dict):
        """
        `robot_obs`: (...,14)
            tcp pos (3), tcp ori (3), gripper width (1), joint_states (6) in rad, gripper_action (1)
        `force_torque`: (...,6)
        """
        D = in_data.shape[-1]
        if norm_type == "minmax":
            dataset_min = np.array(meta_data['min'])[:D]
            dataset_max = np.array(meta_data['max'])[:D]
            out_data = (in_data - dataset_min) / (dataset_max - dataset_min)  # norm here, in [0,1]
            out_data = out_data * 2. - 1.  # in [-1,1]
        elif norm_type == "mean":
            dataset_mean = np.array(meta_data['mean'])[:D]
            dataset_std = np.array(meta_data['std'])[:D]
            out_data = (in_data - dataset_mean) / dataset_std
        else:
            assert norm_type == "identity"
            out_data = in_data
        return out_data

    @staticmethod
    def denorm_state_or_force(in_data: np.ndarray, norm_type: str, meta_data: dict):
        D = in_data.shape[-1]
        if norm_type == "minmax":
            dataset_min = np.array(meta_data['min'][:D])
            dataset_max = np.array(meta_data['max'][:D])
            out_data = (in_data + 1.) / 2.  # [-1,1] to [0,1]
            out_data = out_data * (dataset_max - dataset_min) + dataset_min  # to original scale
        elif norm_type == "mean":
            dataset_mean = np.array(meta_data['mean'][:D])
            dataset_std = np.array(meta_data['std'][:D])
            out_data = in_data * dataset_std + dataset_mean
        else:
            assert norm_type == "identity"
            out_data = in_data
        return out_data


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
                 # Language related
                 language_emb_model: str = "t5xxl",
                 # Multi-view related
                 camera_keys: Union[List[str], Tuple[str]] = ("image",),
                 p_camera_drop: float = 0.0,
                 future_frame_skip: int = 1,  # skip=1 means no frame will be dropped
                 **kwargs
                 ):
        self.data_roots = data_roots
        self.h5_paths = h5_paths

        self.seed = seed
        self.val_ratio = val_ratio
        self.split = split
        self.norm_action_type = norm_action_type
        self.language_emb_model = language_emb_model
        self.camera_keys = camera_keys
        self.p_camera_drop = p_camera_drop
        self.future_frame_skip = future_frame_skip

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
                    norm_action_type="identity",  # NOTE: Use identity first, we'll handle norm later
                    seed=seed,
                    val_ratio=val_ratio,
                    split=split,
                    h5_path=h5_paths[data_idx],  # different across sub-datasets
                    use_h5=use_h5,
                    max_train_episodes=max_train_episodes,
                    transform_color_jitter=transform_color_jitter,
                    language_emb_model=language_emb_model,
                    camera_keys=camera_keys,
                    p_camera_drop=p_camera_drop,
                    future_frame_skip=future_frame_skip,
                    verbose=False,
                    **kwargs
                )
                self.datasets.append(dataset)
                self.dataset_lengths.append(len(dataset))

                # Create validation dataset if this is a train split
                if val_ratio > 0.0:
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

        # For compatibility, create necessary attributes
        self.dataset_stats = self.merged_stats  # for compatibility

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
              f"meta_total_len={self.merged_total_len}, norm_type={self.norm_action_type}")
        beautiful_print(self.merged_stats)

    def _merge_metadata(self):
        """Merge metadata from all datasets"""
        if self.statistics_path is not None:
            import json
            print("[TCLMergeDataset] loading dataset statistics from:", self.statistics_path)
            with open(self.statistics_path, 'r') as json_file:
                statistics = json.load(json_file)
            self.merged_stats = statistics["stats"]  # key: `rel_actions`, `robot_obs`, `force_torque`
            self.merged_total_len = statistics["total_len"]
        else:  # Calculate merged statistics
            all_stats = {}
            all_total_lens = []

            for dataset in self.datasets:
                for data_key, stats in dataset.dataset_stats.items():
                    if data_key not in all_stats:
                        all_stats[data_key] = {
                            'mins': [],
                            'maxs': [],
                            'means': [],
                            'stds': [],
                            'total_counts': []
                        }
                    all_stats[data_key]['mins'].append(stats['min'])
                    all_stats[data_key]['maxs'].append(stats['max'])
                    all_stats[data_key]['means'].append(stats['mean'])
                    all_stats[data_key]['stds'].append(stats['std'])
                    all_stats[data_key]['total_counts'].append(stats['total_count'])

                all_total_lens.append(dataset.dataset_total_len)

            # Convert lists to numpy arrays for easier computation
            for data_key, stats in all_stats.items():
                all_stats[data_key]['mins'] = np.array(stats['mins'])  # shape: (num_datasets, D)
                all_stats[data_key]['maxs'] = np.array(stats['maxs'])
                all_stats[data_key]['means'] = np.array(stats['means'])
                all_stats[data_key]['stds'] = np.array(stats['stds'])
                all_stats[data_key]['total_counts'] = np.array(stats['total_counts'])  # shape: (num_datasets,)

            # Prepare stats for each data_key: `rel_actions`, `robot_obs`, `force_torque`
            self.merged_total_len = sum(all_total_lens)  # shape: ()
            self.merged_stats = {}
            for data_key, stats in all_stats.items():
                # Calculate merged statistics
                merged_min = np.min(stats['mins'], axis=0)  # shape: (D,)
                merged_max = np.max(stats['maxs'], axis=0)

                total_samples = sum(stats['total_counts'])
                # Weighted mean
                weighted_mean = np.zeros_like(stats['means'][0])
                for mean, length in zip(stats['means'], stats['total_counts']):
                    weighted_mean += mean * length / total_samples

                # Weighted std using formula: var = E[X^2] - (E[X])^2
                weighted_var = np.zeros_like(stats['stds'][0])
                for mean, std, length in zip(stats['means'], stats['stds'], stats['total_counts']):
                    var = std ** 2
                    second_moment = var + mean ** 2
                    weighted_var += second_moment * length / total_samples
                merged_var = weighted_var - weighted_mean ** 2
                merged_std = np.sqrt(merged_var)

                self.merged_stats[data_key] = {
                    'min': merged_min,
                    'max': merged_max,
                    'mean': weighted_mean,
                    'std': merged_std,
                }

    def save_meta(self, save_path: str, force_path: bool = True):
        import copy, json
        dumpable_dict = copy.deepcopy(self.merged_stats)
        for data_key, stats in dumpable_dict.items():
            for stat_key, value in stats.items():
                dumpable_dict[data_key][stat_key] = value.tolist()
        meta_statistics = {
            "total_len": int(self.merged_total_len),
            "stats": dumpable_dict,
        }
        with open(save_path, 'w') as fp:
            json.dump(meta_statistics, fp, indent=4)
        print(f"[TCLMergeDataset] Meta data saved to: {save_path}")

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
            language_emb_model=instance.language_emb_model,
            camera_keys=instance.camera_keys,
            p_camera_drop=instance.p_camera_drop,
        )
        return val_set

    def __len__(self):
        return sum(self.dataset_lengths)

    def __getitem__(self, idx):
        # Find which dataset this index belongs to
        current_idx = idx
        for i, dataset in enumerate(self.datasets):
            if current_idx < len(dataset):
                dataset: TCLImageDataset
                item_data = dataset.__getitem__(current_idx)

                # Update the idx field to reflect the global index
                item_data["idx"] = idx

                # Apply merged normalization to actions.
                # This can only be done when the action_nom_type is set as `identity` in each sub-dataset
                item_data["action"] = TCLImageDataset.norm_action(
                    item_data["action"], self.norm_action_type, meta_data=self.merged_stats["rel_actions"]
                )
                item_data["agent_pos"] = TCLImageDataset.norm_state_or_force(
                    item_data["agent_pos"], self.norm_action_type, meta_data=self.merged_stats["robot_obs"]
                )
                item_data["force"] = TCLImageDataset.norm_state_or_force(
                    item_data["force"], self.norm_action_type, meta_data=self.merged_stats["force_torque"]
                )

                return item_data
            current_idx -= len(dataset)

        raise IndexError(f"Index {idx} out of range")

    # For compatible
    @staticmethod
    def denorm_action(action_data: np.ndarray, norm_type: str, meta_data: dict):
        return TCLImageDataset.denorm_action(action_data, norm_type, meta_data)

    @staticmethod
    def denorm_state_or_force(in_data: np.ndarray, norm_type: str, meta_data: dict):
        return TCLImageDataset.denorm_state_or_force(in_data, norm_type, meta_data)
