from typing import Dict
import copy

import numpy as np
import torch

from .pusht_utils import dict_apply
from .pusht_utils import ReplayBuffer
from .pusht_utils import (
SequenceSampler, get_val_mask, downsample_mask)
from .pusht_utils import LinearNormalizer
from .pusht_utils import get_image_range_normalizer


class BaseImageDataset(torch.utils.data.Dataset):
    def get_validation_dataset(self) -> 'BaseImageDataset':
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


class PushTImageDataset(BaseImageDataset):
    def __init__(self,
                 zarr_path,
                 max_obs=5,
                 max_act_out=12,  # a1 in {0,max_act_out}
                 pad_before=0,  # v1 in [1,max_obs]
                 pad_after=0,
                 seed=42,
                 val_ratio=0.0,
                 max_train_episodes=None,
                 out_resize=(256, 256),
                 ):
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['img', 'state', 'action'])
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed)

        # Sampling strategy
        assert max_obs >= 1 and max_obs % 4 == 1, "max_obs-1 must be non-negative and multiple of 4 to match 4x downsampled image size"
        assert max_act_out % 4 == 0, "max_act_out must be positive and multiple of 4 to match 4x downsampled image size"
        pad_before, pad_after = 0, 0
        self.max_obs = max_obs
        self.max_act_out = max_act_out
        self.max_seq_len = self.max_obs + max_act_out

        self.pad_before = max_obs - 1
        self.pad_after = pad_after
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.max_seq_len,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask

        self.out_resize = out_resize
        print(f"[PushTImageDataset] Loaded from: {zarr_path}, len={self.__len__()}. "
              f"max_obs={self.max_obs}, max_act_out={max_act_out}, "
              f"seq_len={self.max_seq_len}, pad_before={self.pad_before}, ")

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.max_seq_len,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'][..., :2]
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['image'] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['state'][:, :2].astype(np.float32)  # (agent_posx2, block_posex3)
        # image = np.moveaxis(sample['img'], -1, 1) / 255  # (T,H,W,C) -> (T,C,H,W)

        # print(sample.keys(), type(sample['img']))
        # print(sample['img'].mean(), sample['img'].max(),sample['img'].min())
        # from PIL import Image
        # s_img = Image.fromarray(sample['img'][0].astype(np.uint8))
        # s_img.save("/home/geyuan/code/cospred2nvidia/output/tmp_dataset_train_b0.png")

        # data = {
        #     'obs': {
        #         'image': image,  # T, 3, 96, 96, in [0,255]
        #         'agent_pos': agent_pos,  # T, 2, in [0,512]
        #     },
        #     'action': sample['action'].astype(np.float32)  # T, 2, in [0,512]
        # }

        image = torch.from_numpy(sample['img']).to(torch.uint8)  # (T,H,W,C) in [0,255]
        image = image.permute(3, 0, 1, 2)  # Rearrange from (T,H,W,C) to (C,T,H,W)
        # print("[DEBUG] image:", image.shape, image.dtype, image.min(), image.max())
        # from imaginaire.utils.io import save_image_or_video
        # save_image_or_video(image.float() / 255., "/home/geyuan/code/cospred2nvidia/output/tmp_dataset_train_b0.mp4", fps=5)

        action = sample['action'].astype(np.float32) / 512.0  # [0,512] -> [0,1]
        agent_pos = agent_pos.astype(np.float32) / 512.0  # [0,512] -> [0,1]

        action = action * 2.0 - 1.0  # (T,2) [0,1] -> [-1,1]
        agent_pos = agent_pos * 2.0 - 1.0  # (T,2) [0,1] -> [-1,1], above all share the same T

        # print("[DEBUG] image:", image.shape, image.dtype, image.min(), image.max(),
        #       "agent_pos:", agent_pos.shape, agent_pos.dtype, agent_pos.min(), agent_pos.max(),
        #       "action:", action.shape, action.dtype, action.min(), action.max())

        # Dataset returns all, we will sample condition and output in the training_step
        ret_video = image  # (3,v_cond+v_out,256,256)
        ret_action = action
        ret_agent_pos = agent_pos

        ''' Remap keys to match the cosmos-predict2 output format '''
        remapped_data = {
            "action": torch.from_numpy(ret_action),  # (a_out,2), [-1,1]
            "video": ret_video,  # (3,v1 or v1+v2,256,256), [0,255] torch.uint8
            "agent_pos": torch.from_numpy(ret_agent_pos),  # (v1,2), [-1,1]
            "annotation_file": "None",
            "__key__": "None",
            "t5_text_embeddings": torch.zeros(512, 1024, dtype=torch.bfloat16),
            "t5_text_mask": torch.ones(512, dtype=torch.int64),
            "fps": 10,
            "image_size": torch.tensor([
                self.out_resize[0], self.out_resize[1], 256, 256
            ]),
            "num_frames": ret_video.shape[1],  # v_cond (+v_out)
            "padding_mask": torch.zeros(1, 256, 256),  # (T,H,W) not used; cond mask is set in conditioner
            # "num_conditional_frames": n_v_cond,  # different across in a single batch
            # "num_conditional_actions": n_a_cond,
        }
        return remapped_data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        return data
        # torch_data = dict_apply(data, torch.from_numpy)
        # return torch_data


def test():
    import os
    zarr_path = os.path.expanduser('~/dev/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr')
    dataset = PushTImageDataset(zarr_path, horizon=16)
