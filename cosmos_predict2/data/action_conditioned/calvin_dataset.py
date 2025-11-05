import logging
import os.path
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
from itertools import chain
import random

import pickle
import numpy as np
from omegaconf import DictConfig
# import pyhash
import torch
from torch.utils.data import Dataset

from cosmos_predict2.data.action_conditioned import calvin_utils
from .calvin_utils import (
    get_state_info_dict,
    process_actions,
    process_depth,
    process_language,
    process_rgb,
    process_state,
    lookup_naming_pattern,
)


def load_pkl(filename: Path) -> Dict[str, np.ndarray]:
    with open(filename, "rb") as f:
        return pickle.load(f)


def load_npz(filename: Path) -> Dict[str, np.ndarray]:
    return np.load(filename.as_posix())


def get_validation_window_size(idx: int, min_window_size: int, max_window_size: int) -> int:
    """
    In validation step, use hash function instead of random sampling for consistent window sizes across epochs.

    Args:
        idx: Sequence index.
        min_window_size: Minimum window size.
        max_window_size: Maximum window size.

    Returns:
        Window size computed with hash function.
    """
    window_range = max_window_size - min_window_size + 1
    return min_window_size + hasher(str(idx)) % window_range


class ClearDataset(Dataset):
    """
    A CLEAR version of disk_dataset.ExtendedDiskDataset, removing the multi-inheriting relationship.

    Args:
        datasets_dir: Path of folder containing episode files (string must contain 'validation' or 'training').
        obs_space: DictConfig of observation space.
        proprio_state: DictConfig with shape of prioprioceptive state.
        key: 'vis' or 'lang'.
        lang_folder: Name of the subdirectory of the dataset containing the language annotations.
        num_workers: Number of dataloading workers for this dataset.
        transforms: Dict with pytorch data transforms.
        batch_size: Batch size.
        min_window_size: Minimum window length of loaded sequences.
        max_window_size: Maximum window length of loaded sequences.
        pad: If True, repeat last frame such that all sequences have length 'max_window_size'.
        aux_lang_loss_window: How many sliding windows to consider for auxiliary language losses, counted from the end
            of an annotated language episode.
        skip_frames: Skip this amount of windows for language dataset.
        save_format: File format in datasets_dir (pkl or npz).
        pretrain: Set to True when pretraining.
    """

    def __init__(
            self,
            ## BaseDataset ##
            datasets_dir: Path,
            obs_space: DictConfig,
            proprio_state: DictConfig,
            key: str,
            lang_folder: str,
            num_workers: int,
            transforms: Dict = {},
            batch_size: int = 32,
            min_window_size: int = 16,
            max_window_size: int = 32,
            pad: bool = True,
            aux_lang_loss_window: int = 1,
            window_sampling_strategy: str = 'random',
            geometric_p_value: float = 0.1,
            ## DiskDataset ##
            skip_frames: int = 1,
            save_format: str = "npz",
            pretrain: bool = False,
            ## ExtendedDiskDataset ##
            obs_seq_len: int = -1,  # used:1
            action_seq_len: int = -1,  # used:10
            future_range: int = -1,  # used:29, just for compatibility, set -1 for not used
            img_gen_frame_diff: int = 3,  # used:3; -1 means random, -2 means no use
            ## Extracted speed-up ##
            use_extracted_rel_actions: bool = False,
            extracted_dir: str = 'extracted/',
            ## Cosmos-Predict2 related ##
            future_frame_skip: int = 1,  # to reduce future frame prediction difficulty, default 1 means no skip
            language_emb_model: str = "t5xxl",
            obs_image_shape: Tuple[int, int] = (224, 224),
            zero_robot_state: bool = False,
            ## Others ##
            max_len: int = None,
            chose_ratio: float = 1.,
    ):
        self.observation_space = obs_space
        self.proprio_state = proprio_state
        self.transforms = transforms
        self.with_lang = key == "lang"
        self.relative_actions = "rel_actions" in self.observation_space["actions"]
        assert window_sampling_strategy in ('random', 'geometric')
        self.window_sampling_strategy = window_sampling_strategy
        self.geometric_p_value = geometric_p_value  # only needed for geomtric sampling
        self.pad = pad
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.min_window_size = min_window_size  # default:21
        self.max_window_size = max_window_size  # default:50
        self.abs_datasets_dir = Path(datasets_dir)
        self.lang_folder = lang_folder  # if self.with_lang else None
        self.aux_lang_loss_window = aux_lang_loss_window
        assert "validation" in self.abs_datasets_dir.as_posix() or "training" in self.abs_datasets_dir.as_posix()
        self.validation = "validation" in self.abs_datasets_dir.as_posix()
        assert self.abs_datasets_dir.is_dir()
        print(f"[DEBUG] loading dataset at {self.abs_datasets_dir}")

        ## DiskDataset ##
        self.save_format = save_format
        if self.save_format == "pkl":
            self.load_file = load_pkl
        elif self.save_format == "npz":
            self.load_file = load_npz
        else:
            raise NotImplementedError
        self.pretrain = pretrain
        self.skip_frames = skip_frames

        # episode_lookup: dataset_order -> ep_fn_idx, no repeat in ep_fn_idx
        if self.with_lang:
            self.episode_lookup, self.lang_lookup, self.lang_ann, self.lang_text = self._build_file_indices_lang(
                self.abs_datasets_dir)
        else:
            self.episode_lookup = self._build_file_indices(self.abs_datasets_dir)

        if max_len is not None:
            self.episode_lookup: np.ndarray = self.episode_lookup[:max_len]

        self.lookup_len = len(self.episode_lookup)
        self.chose_ratio = chose_ratio
        if chose_ratio < 1.:
            self.chose_len = int(self.lookup_len * chose_ratio)  # accessible ep length
            self.chose_indices = np.random.choice(self.lookup_len, self.chose_len, replace=False)

        self.naming_pattern, self.n_digits = lookup_naming_pattern(self.abs_datasets_dir, self.save_format)

        ## ExtendedDiskDataset ##
        self.obs_seq_len = obs_seq_len
        self.action_seq_len = action_seq_len
        self.future_range = future_range  # Number of steps into the future to sample goals
        self.ep_start_end_ids = np.load(self.abs_datasets_dir / "ep_start_end_ids.npy")  # Load sequence boundaries
        self.img_gen_frame_diff = img_gen_frame_diff
        self.random_frame_diff = False if (img_gen_frame_diff > -1 or img_gen_frame_diff == -2) else True

        # Using extracted npy to reduce bandwidth of data loading
        self.use_extracted_rel_actions = use_extracted_rel_actions
        if use_extracted_rel_actions:
            self.extracted_dir = extracted_dir
            if not os.path.exists(extracted_dir):  # maybe a relative path
                self.extracted_dir = os.path.join(self.abs_datasets_dir, "extracted")  # convert to abs path
                assert os.path.exists(self.extracted_dir), "extracted dir not found!"
            with open(os.path.join(self.extracted_dir, "ep_npz_names.list"), "r") as f:
                self.extracted_ep_npz_names = [int(x.strip()) for x in f.readlines()]
                self.extracted_ep_npz_name_to_npy_idx = {self.extracted_ep_npz_names[i]: i
                                                         for i in range(len(self.extracted_ep_npz_names))}
                # key: int, original episode fn's index; value: int, extracted npy's inner index
            self.extracted_ep_rel_actions: np.ndarray = np.load(os.path.join(self.extracted_dir, "ep_rel_actions.npy"))
            print(f"[DEBUG] Extracted files loaded from {self.extracted_dir}")

        # Cosmos-Predict2 related
        self.language_emb_model = language_emb_model
        self.future_frame_skip = future_frame_skip
        self.obs_image_shape = obs_image_shape
        self.zero_robot_state = zero_robot_state
        self.language_emb_model = language_emb_model
        if "t5xxl" in self.language_emb_model:
            self.language_embedding_key = "language_embedding"
            self.language_embedding_subdir = "lang_emb_t5xxl"  # default subdir
            language_embedding_file_path = os.path.join(
                self.abs_datasets_dir, "../", self.language_embedding_subdir, "t5_embeddings.npz")
            language_embedding_file_path = os.path.abspath(language_embedding_file_path)
            print("[DEBUG] Loading language embeddings from:", language_embedding_file_path)
            lang_emb_data = np.load(language_embedding_file_path, allow_pickle=True)
            self.lang_text_to_emb = lang_emb_data['text_to_embedding_map'].item()

        # Sampling a data sequence
        max_obs = obs_seq_len
        max_act_out = action_seq_len
        assert max_obs >= 1 and max_obs % 4 == 1, "max_obs-1 must be non-negative and multiple of 4 to match 4x downsampled image size"
        assert max_act_out % 4 == 0, "max_act_out must be positive and multiple of 4 to match 4x downsampled image size"
        self.max_obs = max_obs
        self.max_act_out = max_act_out
        self.max_seq_len = self.max_obs + max_act_out  # without frames skipping
        self.max_pred_frames = max_act_out // self.future_frame_skip
        assert self.max_pred_frames * self.future_frame_skip == max_act_out, \
            "max_act_out must be divisible by future_frame_skip"
        assert self.max_pred_frames % 4 == 0, "max_pred_frames must be multiple of 4 to match 4x downsampled image size"
        self.state_t = (1 + (self.max_obs - 1) // 4) + (self.max_pred_frames // 4)  # for multi-view latent state_t

        self.debug_print = False

    def __getitem__(self, idx: Union[int, Tuple[int, int]]) -> Dict:
        """
        Get sequence of dataset.

        Args:
            idx: Index of the sequence.

        Returns:
            Loaded sequence.

        Examples:
        CALVIN@3: Dict, keys=['robot_obs', 'rgb_obs', 'depth_obs', 'actions', 'state_info', 'use_for_aux_lang_loss', 'lang', 'lang_text', 'idx']
        --robot_obs, <class 'torch.Tensor'>, shape=torch.Size([8, 1, 8]), min=-3.1405, max=3.1020
        --rgb_obs: Dict, keys=['rgb_static', 'rgb_gripper']
        ----rgb_static, <class 'torch.Tensor'>, shape=torch.Size([8, 1, 3, 200, 200]), min=0.0000, max=255.0000
        ----rgb_gripper, <class 'torch.Tensor'>, shape=torch.Size([8, 1, 3, 84, 84]), min=0.0000, max=255.0000
        --depth_obs: Dict, keys=[]
        --actions, <class 'torch.Tensor'>, shape=torch.Size([8, 48, 7]), min=-1.0000, max=1.0000
        --state_info: Dict, keys=['robot_obs', 'scene_obs']
        ----robot_obs, <class 'torch.Tensor'>, shape=torch.Size([8, 1, 15]), min=-3.1405, max=3.1020
        ----scene_obs, <class 'torch.Tensor'>, shape=torch.Size([8, 1, 24]), min=-3.0652, max=3.1416
        --use_for_aux_lang_loss, <class 'torch.Tensor'>, shape=torch.Size([8]), min=0.0000, max=0.0000
        --lang, <class 'torch.Tensor'>, shape=torch.Size([8, 1024]), min=-11.4766, max=2.0703
        --lang_text: List, len=8, elem_type=<class 'str'>
        ----[0]: <class 'str'>, len=46, value='push down the button to turn off the led light'
        --idx, <class 'torch.Tensor'>, shape=torch.Size([8]), min=21367.0000, max=311646.0000

        Remapped:
        CALVIN@3: Dict, keys=['action', 'video', 'agent_pos', 'annotation_file', '__key__', 'lang_text', 't5_text_embeddings', 't5_text_mask', 'fps', 'image_size', 'num_frames', 'padding_mask', 'sample_n_views', 'view_indices', 'latent_view_indices_B_T']
        --action, <class 'torch.Tensor'>, shape=torch.Size([8, 48, 7]), min=-1.0000, max=1.0000
        --video, <class 'torch.Tensor'>, shape=torch.Size([8, 3, 48, 224, 224]), min=0.0000, max=255.0000
        --agent_pos, <class 'torch.Tensor'>, shape=torch.Size([8, 24, 8]), min=-2.0607, max=3.4952
        --annotation_file: List, len=8, elem_type=<class 'str'>
        ----[0]: <class 'str'>, len=4, value='None'
        --__key__: List, len=8, elem_type=<class 'str'>
        ----[0]: <class 'str'>, len=4, value='None'
        --lang_text: List, len=8, elem_type=<class 'str'>
        ----[0]: <class 'str'>, len=36, value='in the slider pick up the pink block'
        --t5_text_embeddings, <class 'torch.Tensor'>, shape=torch.Size([8, 512, 1024]), min=0.0000, max=0.0000
        --t5_text_mask, <class 'torch.Tensor'>, shape=torch.Size([8, 512]), min=1.0000, max=1.0000
        --fps, <class 'torch.Tensor'>, shape=torch.Size([8]), min=10.0000, max=10.0000
        --image_size, <class 'torch.Tensor'>, shape=torch.Size([8, 4]), min=224.0000, max=256.0000
        --num_frames, <class 'torch.Tensor'>, shape=torch.Size([8]), min=25.0000, max=25.0000
        --padding_mask, <class 'torch.Tensor'>, shape=torch.Size([8, 1, 224, 224]), min=0.0000, max=0.0000
        --sample_n_views, <class 'torch.Tensor'>, shape=torch.Size([8]), min=2.0000, max=2.0000
        --view_indices, <class 'torch.Tensor'>, shape=torch.Size([8, 48]), min=0.0000, max=1.0000
        --latent_view_indices_B_T, <class 'torch.Tensor'>, shape=torch.Size([8, 14]), min=0.0000, max=1.0000
        """
        if isinstance(idx, int):
            if self.chose_ratio < 1.:
                idx = self.chose_indices[idx]
            # When max_ws_size and min_ws_size are equal, avoid unnecessary padding
            # acts like Constant dataset. Currently, used for language data
            if self.min_window_size == self.max_window_size:
                window_size = self.max_window_size
            elif self.min_window_size < self.max_window_size:
                window_size = self._get_window_size(idx)
            else:
                print(f"[ERROR] min_window_size {self.min_window_size} > max_window_size {self.max_window_size}")
                raise ValueError
        else:
            idx, window_size = idx
            if self.chose_ratio < 1.:
                idx = self.chose_indices[idx]

        sequence = self._get_sequences(idx, window_size)
        if self.pad:  # used:False
            pad_size = self._get_pad_size(sequence)
            sequence = self._pad_sequence(sequence, pad_size)
        return sequence

    def _get_sequences(self, idx: int, window_size: int) -> Dict:
        """
        Load sequence of length window_size.

        Args:
            idx: Index of starting frame.
            window_size: Goal frames window skip size. -1: do not load goal frames.

        Returns:
            dict: Dictionary of tensors of loaded sequence with different input modalities and actions.
        """

        episode = self._load_episode(idx, window_size)

        '''
        before: Dict,keys=dict_keys(['rel_actions', 'language', 'rgb_gripper', 'scene_obs', 'rgb_static', 'language_text', 'robot_obs', 'gen_static', 'gen_gripper', 'future_frame_diff'])
        rel_actions,<class 'numpy.ndarray'>,shape=(10, 7)                                                                                                                                                                                                                                                                            
        language,<class 'numpy.ndarray'>,shape=(1024,)                                                                                                                                                                                                                                                                               
        rgb_gripper,<class 'numpy.ndarray'>,shape=(2, 84, 84, 3)
        scene_obs,<class 'numpy.ndarray'>,shape=(2, 24)         
        rgb_static,<class 'numpy.ndarray'>,shape=(2, 200, 200, 3)
        language_text:<class 'str'>,len=30                       
        robot_obs,<class 'numpy.ndarray'>,shape=(2, 15)                                                                                                                                                                                                                                                                              
        gen_static,<class 'numpy.ndarray'>,shape=(200, 200, 3)
        gen_gripper,<class 'numpy.ndarray'>,shape=(84, 84, 3) 
        future_frame_diff,<class 'numpy.ndarray'>,shape=(), 3  

        after process: Dict,keys=dict_keys(['rel_actions', 'language', 'rgb_gripper', 'scene_obs', 'rgb_static', 'language_text', 'robot_obs', 'gen_static', 'gen_gripper', 'future_frame_diff'])
        rel_actions,<class 'numpy.ndarray'>,shape=(10, 7)                                                                                                                                                                                                                                                                            
        language,<class 'numpy.ndarray'>,shape=(1024,)        
        rgb_gripper,<class 'numpy.ndarray'>,shape=(2, 84, 84, 3) 
        scene_obs,<class 'numpy.ndarray'>,shape=(2, 24)       
        rgb_static,<class 'numpy.ndarray'>,shape=(2, 200, 200, 3)                                                                                                                                                                                                                                                                    
        language_text:<class 'str'>,len=37                   
        robot_obs,<class 'numpy.ndarray'>,shape=(2, 15)         
        gen_static,<class 'numpy.ndarray'>,shape=(200, 200, 3)                                                                                                                                                                                                                                                                       
        gen_gripper,<class 'numpy.ndarray'>,shape=(84, 84, 3)                                                                                                         
        future_frame_diff,<class 'numpy.ndarray'>,shape=(), 3   
        '''

        seq_state_obs = process_state(episode, self.observation_space, self.transforms, self.proprio_state)
        seq_rgb_obs = process_rgb(episode, self.observation_space, self.transforms)
        seq_depth_obs = process_depth(episode, self.observation_space, self.transforms)
        seq_acts = process_actions(episode, self.observation_space, self.transforms)
        info = get_state_info_dict(episode)
        seq_lang = process_language(episode, self.transforms, self.with_lang)
        info = self._add_language_info(info, idx)
        seq_dict = {**seq_state_obs, **seq_rgb_obs, **seq_depth_obs, **seq_acts, **info, **seq_lang}  # type:ignore
        seq_dict["idx"] = idx  # type:ignore
        if hasattr(episode, "future_frame_diff"):
            seq_dict['future_frame_diff'] = episode['future_frame_diff']

        '''
        seq_dict: Dict,keys=dict_keys(['robot_obs', 'rgb_obs', 'depth_obs', 'actions', 'state_info', 'lang', 'idx', 'future_frame_diff'])                                                                                                                                                                                    
        robot_obs,<class 'torch.Tensor'>,shape=torch.Size([2, 8])                                                                                                                                                                                                                                                                    
        rgb_obs: Dict,keys=dict_keys(['rgb_static', 'rgb_gripper', 'gen_static', 'gen_gripper'])                                                                      
        -rgb_static,<class 'torch.Tensor'>,shape=torch.Size([2, 3, 224, 224])                                                                                                                                                                                                                                                        
        -rgb_gripper,<class 'torch.Tensor'>,shape=torch.Size([2, 3, 84, 84])                                                                                                                                                                                                                                                         
        -gen_static,<class 'torch.Tensor'>,shape=torch.Size([1, 3, 112, 112])                                                                                                                                                                                                                                                        
        -gen_gripper,<class 'torch.Tensor'>,shape=torch.Size([1, 3, 112, 112])                                                                                                                                                                                                                                                       
        depth_obs: Dict,keys=dict_keys([])                                                                                                                                                                                                                                                                                           
        actions,<class 'torch.Tensor'>,shape=torch.Size([10, 7])                                                                                                                                                                                                                                                                     
        state_info: Dict,keys=dict_keys(['robot_obs', 'scene_obs'])                                                                                                                                                                                                                                                                  
        -robot_obs,<class 'torch.Tensor'>,shape=torch.Size([2, 15])                                                                                                                                                                                                                                                                  
        -scene_obs,<class 'torch.Tensor'>,shape=torch.Size([2, 24])                                                                                                   
        lang,<class 'torch.Tensor'>,shape=torch.Size([0])                                                                                                                                                                                                                                                                            
        idx:<class 'int'>,2563                                                                                                                                        
        future_frame_diff,<class 'numpy.ndarray'>,shape=(), 3   
        '''

        '''
        VIS: Dict,keys=dict_keys(['robot_obs', 'rgb_obs', 'depth_obs', 'actions', 'state_info', 'lang', 'idx', 'future_frame_diff'])
        robot_obs,<class 'torch.Tensor'>,shape=torch.Size([2, 8])
        rgb_obs:Dict,keys=dict_keys(['rgb_static', 'rgb_gripper', 'gen_static', 'gen_gripper'])
        -rgb_static,<class 'torch.Tensor'>,shape=torch.Size([2, 3, 224, 224])
        -rgb_gripper,<class 'torch.Tensor'>,shape=torch.Size([2, 3, 84, 84])
        -gen_static,<class 'torch.Tensor'>,shape=torch.Size([1, 3, 112, 112])
        -gen_gripper,<class 'torch.Tensor'>,shape=torch.Size([1, 3, 112, 112])
        depth_obs:Dict,keys=dict_keys([])
        actions,<class 'torch.Tensor'>,shape=torch.Size([10, 7])
        state_info:Dict,keys=dict_keys(['robot_obs', 'scene_obs'])
        -robot_obs,<class 'torch.Tensor'>,shape=torch.Size([2, 15])
        -scene_obs,<class 'torch.Tensor'>,shape=torch.Size([2, 24])
        lang,<class 'torch.Tensor'>,shape=torch.Size([0])
        idx:<class 'int'>,0
        future_frame_diff,<class 'numpy.ndarray'>,shape=(), 3

        Lang batch: Dict,keys=dict_keys(['robot_obs', 'rgb_obs', 'depth_obs', 'actions', 'state_info', 'use_for_aux_lang_loss', 'lang', 'lang_text', 'idx', 'future_frame_diff'])
        robot_obs,<class 'torch.Tensor'>,shape=torch.Size([2, 8])
        rgb_obs: Dict,keys=dict_keys(['rgb_static', 'rgb_gripper', 'gen_static', 'gen_gripper'])
        -rgb_static,<class 'torch.Tensor'>,shape=torch.Size([2, 3, 224, 224])
        -rgb_gripper,<class 'torch.Tensor'>,shape=torch.Size([2, 3, 84, 84])
        -gen_static,<class 'torch.Tensor'>,shape=torch.Size([1, 3, 112, 112])
        -gen_gripper,<class 'torch.Tensor'>,shape=torch.Size([1, 3, 112, 112])
        depth_obs: Dict,keys=dict_keys([])
        actions,<class 'torch.Tensor'>,shape=torch.Size([10, 7])
        state_info: Dict,keys=dict_keys(['robot_obs', 'scene_obs'])
        -robot_obs,<class 'torch.Tensor'>,shape=torch.Size([2, 15])
        -scene_obs,<class 'torch.Tensor'>,shape=torch.Size([2, 24])
        use_for_aux_lang_loss:<class 'bool'>,False
        lang,<class 'torch.Tensor'>,shape=torch.Size([1024])
        lang_text:<class 'str'>,len=30
        idx:<class 'int'>,0
        future_frame_diff,<class 'numpy.ndarray'>,shape=(), 3
        '''

        # from robokit.debug_utils.printer import print_batch
        # print_batch("seq_dict", seq_dict)
        '''
        seq_dict: Dict, keys=['robot_obs', 'rgb_obs', 'depth_obs', 'actions', 'state_info', 'use_for_aux_lang_loss', 'lang', 'lang_text', 'idx']
        --robot_obs, <class 'torch.Tensor'>, shape=torch.Size([1, 8]), min=-1.2970, max=0.7006, dtype=torch.float32
        --rgb_obs: Dict, keys=['rgb_static', 'rgb_gripper']
        ----rgb_static, <class 'torch.Tensor'>, shape=torch.Size([1, 3, 224, 224]), min=-1.7923, max=2.1459, dtype=torch.float32
        ----rgb_gripper, <class 'torch.Tensor'>, shape=torch.Size([1, 3, 224, 224]), min=-1.7923, max=2.1459, dtype=torch.float32
        --depth_obs: Dict, keys=[]
        --actions, <class 'torch.Tensor'>, shape=torch.Size([48, 7]), min=-1.0000, max=1.0000, dtype=torch.float32
        --state_info: Dict, keys=['robot_obs', 'scene_obs']
        ----robot_obs, <class 'torch.Tensor'>, shape=torch.Size([1, 15]), min=-2.5893, max=3.0891, dtype=torch.float64
        ----scene_obs, <class 'torch.Tensor'>, shape=torch.Size([1, 24]), min=-0.3597, max=3.1274, dtype=torch.float64
        --use_for_aux_lang_loss: <class 'bool'>, value=False
        --lang, <class 'torch.Tensor'>, shape=torch.Size([1024]), min=-10.5948, max=2.4233, dtype=torch.float32
        --lang_text: <class 'str'>, len=21, value='lift the red block up'
        --idx: <class 'int'>, value=145541
        '''

        ''' Reorder the data for cosmos-predict2 '''
        # Get t5 embeddings
        if "t5" in self.language_emb_model:
            lang_text = seq_dict['lang_text']
            assert lang_text in self.lang_text_to_emb, f"{lang_text} not found in self.lang_text_to_emb"
            t5_text_embeddings = torch.from_numpy(
                self.lang_text_to_emb[lang_text]
            ).to(torch.bfloat16)  # (512,1024)
        else:
            t5_text_embeddings = torch.zeros(512, 1024, dtype=torch.bfloat16)

        # Check the view choice
        ret_n_views = 2  # hard coded as 2 for convenience
        view_indices_selection = [0, 1]
        view_indices_t = torch.tensor(view_indices_selection).repeat_interleave(self.max_obs + self.max_pred_frames)
        latent_view_indices_t = torch.tensor(view_indices_selection).repeat_interleave(self.state_t)
        n_video_tensors = []
        for camera_key in ["rgb_static", "rgb_gripper"]:
            one_video_tensor = (seq_dict["rgb_obs"][camera_key].permute(1, 0, 2, 3) * 255.).to(torch.uint8)  # (C,T,H,W)
            n_video_tensors.append(one_video_tensor)
        ret_video = torch.cat(n_video_tensors, dim=1)  # (C,ret_n_views*T,H,W)
        ret_agent_pos = seq_dict["robot_obs"]  # (T,8)
        if self.zero_robot_state:
            ret_agent_pos = 0. * ret_agent_pos

        ''' Remap keys to match the cosmos-predict2 output format '''
        remapped_data = {
            "action": seq_dict["actions"],  # (a,7), in [-1,1]
            "video": ret_video,  # (3,v1+a/k,224,224), [0,255] torch.uint8
            "agent_pos": ret_agent_pos,  # (v1,8), in [-3,3]
            "annotation_file": "None",
            "__key__": "None",
            "lang_text": seq_dict['lang_text'],  # str
            "t5_text_embeddings": t5_text_embeddings,
            "t5_text_mask": torch.ones(512, dtype=torch.int64),  # although embeddings have zero vectors, mask is all 1
            "fps": 10,  # ori:20
            "image_size": torch.tensor([
                self.obs_image_shape[0], self.obs_image_shape[1], 256, 256
            ]),
            "num_frames": self.max_obs + self.max_pred_frames,  # v_cond (+v_out), with frame skipping
            "padding_mask": torch.zeros(1, self.obs_image_shape[0], self.obs_image_shape[1]),
            # (T,H,W) not used; cond mask is set in conditioner
            # Multi-view related
            "sample_n_views": ret_n_views,
            "view_indices": view_indices_t,
            "latent_view_indices_B_T": latent_view_indices_t,  # here is (T,), but will be (B,T) in DataLoader
        }
        return remapped_data

    def _get_window_size(self, idx: int) -> int:
        """
        Sample a window size taking into account the episode limits.

        Args:
            idx: Index of the sequence to load.

        Returns:
            Window size.
        """
        window_diff = self.max_window_size - self.min_window_size
        if len(self.episode_lookup) <= idx + window_diff:
            # last episode when required right >= len(lookup)
            max_window = self.min_window_size + len(self.episode_lookup) - idx - 1
        elif self.episode_lookup[idx + window_diff] != self.episode_lookup[idx] + window_diff:
            # less than max_episode steps until next episode
            steps_to_next_episode = int(
                np.nonzero(
                    self.episode_lookup[idx: idx + window_diff + 1]
                    - (self.episode_lookup[idx] + np.arange(window_diff + 1))
                )[0][0]
            )
            max_window = min(self.max_window_size, (self.min_window_size + steps_to_next_episode - 1))
        else:
            max_window = self.max_window_size

        if self.validation:
            # in validation step, repeat the window sizes for each epoch.
            return get_validation_window_size(idx, self.min_window_size, max_window)
        else:
            if self.window_sampling_strategy == 'geometric':  # used
                p = self.geometric_p_value  # Choose a suitable value for p
                while True:
                    sampled_window_size = 1 + np.random.geometric(p)  # E(G(p))=10
                    if self.min_window_size <= sampled_window_size <= max_window:  # if in [21,50], avg~=28.98
                        return sampled_window_size
            else:  # 'random'
                return np.random.randint(self.min_window_size, max_window + 1)

    def __len__(self) -> int:
        """
        Returns:
            Size of the dataset.
        """
        return len(self.episode_lookup)

    def _get_pad_size(self, sequence: Dict) -> int:
        """
        Determine how many frames to append to end of the sequence

        Args:
            sequence: Loaded sequence.

        Returns:
            Number of frames to pad.
        """
        return self.max_window_size - len(sequence["actions"])

    def _pad_sequence(self, seq: Dict, pad_size: int) -> Dict:
        """
        Pad a sequence by repeating the last frame.

        Args:
            seq: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded sequence.
        """
        seq.update({"robot_obs": self._pad_with_repetition(seq["robot_obs"], pad_size)})
        seq.update({"rgb_obs": {k: self._pad_with_repetition(v, pad_size) for k, v in seq["rgb_obs"].items()}})
        seq.update({"depth_obs": {k: self._pad_with_repetition(v, pad_size) for k, v in seq["depth_obs"].items()}})
        #  NOTE: find better way of distinguishing rk and play action spaces
        if not self.relative_actions:
            # repeat action for world coordinates action space
            seq.update({"actions": self._pad_with_repetition(seq["actions"], pad_size)})
        else:
            # for relative actions zero pad all but the last action dims and repeat last action dim (gripper action)
            seq_acts = torch.cat(
                [
                    self._pad_with_zeros(seq["actions"][..., :-1], pad_size),
                    self._pad_with_repetition(seq["actions"][..., -1:], pad_size),
                ],
                dim=-1,
            )
            seq.update({"actions": seq_acts})
        seq.update({"state_info": {k: self._pad_with_repetition(v, pad_size) for k, v in seq["state_info"].items()}})
        return seq

    @staticmethod
    def _pad_with_repetition(input_tensor: torch.Tensor, pad_size: int) -> torch.Tensor:
        """
        Pad a sequence Tensor by repeating last element pad_size times.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        last_repeated = torch.repeat_interleave(torch.unsqueeze(input_tensor[-1], dim=0), repeats=pad_size, dim=0)
        padded = torch.vstack((input_tensor, last_repeated))
        return padded

    @staticmethod
    def _pad_with_zeros(input_tensor: torch.Tensor, pad_size: int) -> torch.Tensor:
        """
        Pad a Tensor with zeros.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        zeros_repeated = torch.repeat_interleave(
            torch.unsqueeze(torch.zeros(input_tensor.shape[-1]), dim=0), repeats=pad_size, dim=0
        )
        padded = torch.vstack((input_tensor, zeros_repeated))
        return padded

    def _add_language_info(self, info: Dict, idx: int) -> Dict:
        """
        If dataset contains language, add info to determine if this sequence will be used for the auxiliary losses.

        Args:
            info: Info dictionary.
            idx: Sequence index.

        Returns:
            Info dictionary with updated information.
        """
        if not self.with_lang:
            return info
        use_for_aux_lang_loss = (
                idx + self.aux_lang_loss_window >= len(self.lang_lookup)
                or self.lang_lookup[idx] < self.lang_lookup[idx + self.aux_lang_loss_window]
        )
        info["use_for_aux_lang_loss"] = use_for_aux_lang_loss
        return info

    #################### End ####################

    ## Copied from disk_dataset.DiskDataset ##
    def _get_episode_name(self, file_idx: int) -> Path:
        """
        Convert file idx to file path.

        Args:
            file_idx: index of starting frame.

        Returns:
            Path to file.
        """
        return Path(f"{self.naming_pattern[0]}{file_idx:0{self.n_digits}d}{self.naming_pattern[1]}")

    def _build_file_indices_lang(self, abs_datasets_dir: Path) -> Tuple[np.ndarray, List, np.ndarray]:
        """
        This method builds the mapping from index to file_name used for loading the episodes of the language dataset.

        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.

        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
            lang_lookup: Mapping from training example to index of language instruction.
            lang_ann: Language embeddings.
        """
        assert abs_datasets_dir.is_dir()

        episode_lookup = []

        try:
            print("trying to load lang data from: ", abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy")
            lang_data = np.load(abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy", allow_pickle=True).item()
        except Exception:
            print("Exception, trying to load lang data from: ", abs_datasets_dir / "auto_lang_ann.npy")
            lang_data = np.load(abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy", allow_pickle=True).item()

        ep_start_end_ids = lang_data["info"]["indx"]  # each of (second - first) are <= 64, vis:[5124],lang:[1011]
        print('[DEBUG] LANG ep_start_end_ids:', len(ep_start_end_ids), ep_start_end_ids[0], ep_start_end_ids[1],
              ep_start_end_ids[2])
        lang_ann = lang_data["language"]["emb"]  # length total number of annotations
        lang_text = lang_data["language"]["ann"]  # length total number of annotations
        lang_lookup = []
        for i, (start_idx, end_idx) in enumerate(ep_start_end_ids):
            if self.pretrain:
                start_idx = max(start_idx, end_idx + 1 - self.min_window_size - self.aux_lang_loss_window)
            assert end_idx >= self.max_window_size
            cnt = 0
            for idx in range(start_idx, end_idx + 1 - self.min_window_size):
                if cnt % self.skip_frames == 0:
                    lang_lookup.append(i)
                    episode_lookup.append(idx)
                cnt += 1

        print('[DEBUG] LANG episode_lookup len=', len(episode_lookup))
        return np.array(episode_lookup), lang_lookup, lang_ann, lang_text

    #################### End ####################

    ## Copied from disk_dataset.ExtendedDiskDataset ##
    def find_sequence_boundaries(self, idx: int) -> Tuple[int, int]:
        for start_idx, end_idx in self.ep_start_end_ids:
            if start_idx <= idx < end_idx:
                return start_idx, end_idx
        raise ValueError(f"Index {idx} does not belong to any sequence.")

    def _load_episode(self, idx: int, window_size: int) -> Dict[str, np.ndarray]:
        """
        Load consecutive frames saved as individual files on disk and combine to episode dict.

        Args:
            idx: Index of first frame. First frame of observations.
            window_size: Goal frames window skip size. -1: do not load goal frames.

        Returns:
            episode: Dict of numpy arrays containing the episode where keys are the names of modalities.
        """
        start_idx = self.episode_lookup[idx]  # episode_XXX.npz
        end_idx = start_idx + self.action_seq_len + self.obs_seq_len - 1  # episode_XXX.npz
        keys = list(chain(*self.observation_space.values()))
        keys.remove("language")
        keys.append("scene_obs")
        # keys:['rgb_static', 'rgb_gripper', 'gen_static', 'gen_gripper', 'robot_obs', 'rel_actions', 'scene_obs']

        # Modify the episode dict to only include the specified sequence lengths
        if self.random_frame_diff:
            img_gen_frame_diff = random.randint(0, self.action_seq_len - 1)  # -1 means random
        else:
            img_gen_frame_diff = self.img_gen_frame_diff  # used:3; -2 means no use
        gen_img_idx = start_idx + self.obs_seq_len + img_gen_frame_diff - 1  # episode_XXX.npz

        if not self.use_extracted_rel_actions:
            # Op1. original reading actions from episode_xxx.npz one-by-one
            episodes = [self.load_file(self._get_episode_name(file_idx)) for file_idx in range(start_idx, end_idx)]
        else:
            # Op2. reading actions from a single ep_rel_actions.npy file
            # NOTE: ori MDT: frames_indices = range(start_idx, start_idx + self.obs_seq_len)
            frames_pad_before_indices = np.arange(start_idx, start_idx + self.obs_seq_len - 1)
            frames_now_future_skipped_indices = np.arange(
                start_idx + self.obs_seq_len - 1,
                start_idx + self.obs_seq_len + self.action_seq_len
            )[::self.future_frame_skip]
            assert frames_now_future_skipped_indices[-1] == end_idx, \
                f"frames_now_future_skipped_indices last idx {frames_now_future_skipped_indices[-1]} != end_idx {end_idx}"
            frames_skipped_indices = np.concatenate([frames_pad_before_indices, frames_now_future_skipped_indices])
            episodes = [self.load_file(self._get_episode_name(file_idx)) for file_idx in
                        frames_skipped_indices]
            gen_img_episode = self.load_file(self._get_episode_name(gen_img_idx)) if img_gen_frame_diff >= 0 else None
            ex_indices = [self.extracted_ep_npz_name_to_npy_idx[file_idx] for file_idx in range(start_idx, end_idx)]
            ex_actions = self.extracted_ep_rel_actions[ex_indices, :]

        episode = {}
        ''' 
        1. Read from [start,end]
        rel_actions, rgb_static, rgb_gripper, robot_obs, scene_obs 
            episode[rel_actions]: (10, 7), [0:10,...]
            episode[rgb_static]: (1, 200, 200, 3), [:1,...]
            episode[rgb_gripper]: (1, 84, 84, 3), [:1,...]
            episode[robot_obs]: (1, 15), [:1,...]
            episode[scene_obs]: (1, 24), [:1,...]
            gen_img_static: (200, 200, 3), [3,...]
            gen_img_gripper: (84, 84, 3), [3,...]
        '''
        for ep_idx in range(len(episodes)):
            episodes[ep_idx] = ep = dict(episodes[ep_idx])
            # try to fix some missing keys
            if 'rel_actions' not in ep:
                episodes[ep_idx]['rel_actions'] = ep['rel_action']
            if 'scene_obs' not in ep:
                episodes[ep_idx]['scene_obs'] = np.zeros((1, 24))
            if 'robot_obs' not in ep:
                episodes[ep_idx]['robot_obs'] = np.zeros((1, 15))

        # from robokit.debug_utils.printer import print_batch
        # print_batch("[DEBUG] episode", episodes)

        for key in keys:
            if 'gen' in key:
                continue

            stacked_data = np.stack([ep[key] for ep in episodes])  # len=${act_seq_len},eg.10
            if not self.use_extracted_rel_actions:
                raise NotImplementedError
                # Op1. original reading actions from episode_xxx.npz one-by-one
                if key == "rel_actions" or key == 'actions':
                    episode[key] = stacked_data[
                        (self.obs_seq_len - 1):((self.obs_seq_len - 1) + self.action_seq_len), :]
                else:
                    if key == 'rgb_static':
                        gen_img_static = stacked_data[self.obs_seq_len + img_gen_frame_diff - 1, :]
                    elif key == 'rgb_gripper':
                        gen_img_gripper = stacked_data[self.obs_seq_len + img_gen_frame_diff - 1, :]
                    episode[key] = stacked_data[:self.obs_seq_len, :]
            else:
                # Op2. reading actions from a single ep_rel_actions.npy file
                if key == "rel_actions" or key == 'actions':
                    episode[key] = ex_actions[(self.obs_seq_len - 1):((self.obs_seq_len - 1) + self.action_seq_len), :]
                else:
                    if gen_img_episode is not None:
                        if key == 'rgb_static':
                            gen_img_static = gen_img_episode[key]
                        elif key == 'rgb_gripper':
                            gen_img_gripper = gen_img_episode[key]

                    episode[key] = stacked_data[:, :]  # ori: stacked_data[:self.obs_seq_len, :]

            # print(f'[DEBUG] {key}: {episode[key].shape}, {stacked_data.shape}')
        # print(f'[DEBUG] gen:{gen_img_static.shape}, {gen_img_gripper.shape}')

        # if not self.debug_print:
        #     print(f'[DEBUG] start:{start_idx}, end:{end_idx}')
        #     print(episode['rgb_static'].mean(), episode['rgb_static'].min(), episode['rgb_static'].max(),)
        #     self.debug_print = True

        if self.with_lang:
            episode["language"] = self.lang_ann[self.lang_lookup[idx]][0]  # NOTE check  [0]
            episode["language_text"] = self.lang_text[self.lang_lookup[idx]]  # [0]  # NOTE check  [0]
            assert len(self.lang_text) < 2 * self.lang_lookup[-1], "lang_text length error!"

        '''
        2. Read from [goal-obs,goal]
        rgb_static, rgb_gripper, robot_obs, scene_obs
            goal_episode[rgb_static]: (1, 200, 200, 3), [:1,...]
            goal_episode[rgb_gripper]: (1, 84, 84, 3), [:1,...]
            goal_episode[robot_obs]: (1, 15), [:1,...]
            goal_episode[scene_obs]: (1, 24), [:1,...]
        '''
        # get the random future state as goal
        if window_size > 0:
            goal_idx = end_idx + window_size
            # print(start_idx, end_idx, goal_idx)
            eps_start_idx, eps_end_idx = self.find_sequence_boundaries(end_idx)

            # Check if future goal can be sampled
            if eps_end_idx < goal_idx:
                goal_idx = eps_end_idx

            goal_episodes = self.load_file(self._get_episode_name(goal_idx))  # should load [goal_idx-obs_len:goal_idx]
            goal_episodes = ep = dict(goal_episodes)
            # try to fix some missing keys
            if 'rel_actions' not in ep:
                goal_episodes['rel_actions'] = ep['rel_action']
            if 'scene_obs' not in ep:
                goal_episodes['scene_obs'] = np.zeros((1, 24))
            if 'robot_obs' not in ep:
                goal_episodes['robot_obs'] = np.zeros((1, 15))

            goal_episode = {}
            for key in keys:
                if 'gen' in key:
                    continue
                goal_stacked_data = np.stack([goal_episodes[key]])
                if key == "rel_actions" or key == 'actions':
                    pass
                else:
                    goal_episode[key] = goal_stacked_data[:self.obs_seq_len, :]
            # store for merging

            episode = self.merge_episodes(episode, goal_episode)

        '''
        3. Merge "obs" and "goal", append "gen"
        '''
        if img_gen_frame_diff >= 0:
            episode['gen_static'] = gen_img_static
            episode['gen_gripper'] = gen_img_gripper
            episode['future_frame_diff'] = np.array(img_gen_frame_diff)
        return episode

    def merge_episodes(self, episode1: Dict[str, np.ndarray], episode2: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        merged_episode = {}
        all_keys = set(episode1.keys()).union(set(episode2.keys()))
        for key in all_keys:
            if key in episode1 and key in episode2:
                # Merge logic here, for example:
                merged_episode[key] = np.concatenate([episode1[key], episode2[key]], axis=0)
            elif key in episode1:
                merged_episode[key] = episode1[key]
            else:
                merged_episode[key] = episode2[key]
        return merged_episode

    def _build_file_indices(self, abs_datasets_dir: Path) -> np.ndarray:
        """
        This method builds the mapping from index to file_name used for loading the episodes of the non language
        dataset.

        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.

        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
        """
        assert abs_datasets_dir.is_dir()

        episode_lookup = []

        ep_start_end_ids = np.load(abs_datasets_dir / "ep_start_end_ids.npy")
        print('[DEBUG] VIS ep_start_end_ids:', ep_start_end_ids.shape)
        print(f'[DEBUG] Found "ep_start_end_ids.npy" with {len(ep_start_end_ids)} episodes.')
        for start_idx, end_idx in ep_start_end_ids:
            assert end_idx > self.max_window_size
            for idx in range(start_idx, end_idx + 1 - self.min_window_size):
                episode_lookup.append(idx)
        print('[DEBUG] VIS episode_lookup len=', len(episode_lookup))
        return np.array(episode_lookup)

    #################### End ####################

    ### Cosmos-predict2 specific functions ###
    @staticmethod
    def denorm_action(in_action, **kwargs): return in_action
    @staticmethod
    def denorm_state_or_force(in_state, **kwargs): return in_state
