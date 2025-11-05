from collections import deque
from typing import Dict, Any, List, Optional
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms

from cosmos_predict2.connects.utils import (
    replace_multiview_video_back_with_another,
    get_skipped_indices,
    get_frames_from_multiview_video,
)

from imaginaire.utils import log


class VideoAugmentation:
    def __init__(self, enable_augmentation: bool = True):
        self.enable_augmentation = enable_augmentation
        self.augmentation_prob = 0.7
        self.color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        self.gaussian_blur = transforms.GaussianBlur(kernel_size=(7, 7), sigma=(0.1, 2.0))

    def augment_video(self, video_B_C_T_H_W: torch.Tensor) -> torch.Tensor:
        """
        对整个视频进行增广。保证每一帧用相同的增广方式。

        Args:
            video_B_C_T_H_W: 输入视频 tensor，形状为 (B, C, T, H, W)，类型为 uint8

        Returns:
            augmented_video: 增广后的视频 tensor，形状与输入相同，类型为 uint8
        """
        assert video_B_C_T_H_W.dtype == torch.uint8, f"Input video must be of type uint8, got {video_B_C_T_H_W.dtype}"

        if not self.enable_augmentation:
            return video_B_C_T_H_W
        if np.random.rand() > self.augmentation_prob:
            return video_B_C_T_H_W

        # 转换为 float32 [0,1] 范围进行增广
        video_float = video_B_C_T_H_W.float() / 255.0
        video = video_float.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)

        B, T, C, H, W = video.shape
        augmented_video = []

        for b in range(B):
            # 为每个视频样本固定随机种子
            seed = np.random.randint(0, 2 ** 31)

            augmented_frames = []
            for t in range(T):
                frame = video[b, t]  # (C, H, W)

                # 设置随机种子保证每帧应用相同的随机增广
                torch.manual_seed(seed)
                np.random.seed(seed)

                frame = self.color_jitter(frame)
                frame = self.gaussian_blur(frame)

                augmented_frames.append(frame)

            augmented_video.append(torch.stack(augmented_frames, dim=0))

        augmented_video = torch.stack(augmented_video, dim=0)
        augmented_video = augmented_video.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)

        # 转换回 uint8
        augmented_video = torch.clamp(augmented_video * 255.0, 0, 255).byte()

        return augmented_video


class OnlineDataHandler:
    def __init__(self, max_batches: int = 100, device: str = "cuda",
                 future_skip_frames: int = 1,):
        """
        Online data handler for storing and concatenating batches used in Video2WorldExpertPipeline.

        Args:
            max_batches: Maximum number of batches to store.
            device: Device where the data will be stored.
        """
        self.device = device
        self.max_batches = max_batches
        self.future_skip_frames = future_skip_frames
        self.batch_queue = deque(maxlen=max_batches)
        self.video_augmentor = VideoAugmentation(enable_augmentation=True)
        log.info(f"Initialized OnlineDataHandler with "
                 f"max_batches={max_batches}, device={device}, future_skip_frames={future_skip_frames}")

    def add_batch(self, batch: Dict[str, Any]) -> None:
        """
        Add a data batch to the queue.

        Args:
            batch: A dictionary containing video, actions, and other data.
        """
        batch_on_device = self._move_batch_to_device(batch)
        self.batch_queue.append(batch_on_device)
        log.info(f"Added a batch to the queue. Current batch count: {len(self.batch_queue)}")

    def get_concatenated_batch(self) -> Dict[str, Any]:
        """
        Concatenate all batches in the queue into a single large batch.

        Returns:
            A concatenated batch where the B dimension is the total length of all batches.
        """
        if not self.batch_queue:
            raise ValueError("No batches in queue")

        if len(self.batch_queue) == 1:
            log.info("Only one batch in the queue. Returning it directly.")
            return self.batch_queue[0]

        batches = list(self.batch_queue)
        concatenated_batch = {}

        for key in batches[0].keys():
            if isinstance(batches[0][key], torch.Tensor):
                tensors = [batch[key] for batch in batches]
                concatenated_batch[key] = torch.cat(tensors, dim=0)
            elif isinstance(batches[0][key], np.ndarray):
                arrays = [torch.from_numpy(batch[key]) if isinstance(batch[key], np.ndarray)
                          else batch[key] for batch in batches]
                concatenated_batch[key] = torch.cat(arrays, dim=0).to(self.device)
            elif isinstance(batches[0][key], (int, float)):
                if key in ["num_conditional_frames", "num_conditional_actions", "guidance", "seed"]:
                    concatenated_batch[key] = batches[0][key]
                else:
                    values = [batch[key] for batch in batches]
                    concatenated_batch[key] = torch.tensor(values, device=self.device)
            elif isinstance(batches[0][key], str):
                concatenated_batch[key] = batches[0][key]
            elif isinstance(batches[0][key], list):
                concatenated_list = []
                for batch in batches:
                    concatenated_list.extend(batch[key])
                concatenated_batch[key] = concatenated_list
            else:
                concatenated_batch[key] = batches[0][key]

        log.info(f"Concatenated {len(self.batch_queue)} batches into one.")
        return concatenated_batch

    def get_batch_count(self) -> int:
        """Return the current number of batches in the queue."""
        count = len(self.batch_queue)
        log.info(f"Current batch count: {count}")
        return count

    def get_total_batch_size(self) -> int:
        """Return the total batch size across all batches in the queue."""
        if not self.batch_queue:
            return 0

        total_size = 0
        for batch in self.batch_queue:
            for value in batch.values():
                if isinstance(value, torch.Tensor):
                    total_size += value.shape[0]
                    break
        log.info(f"Total batch size across all batches: {total_size}")
        return total_size

    def clear_batches(self) -> None:
        """Clear all batches in the queue."""
        self.batch_queue.clear()
        log.info("Cleared all batches in the queue.")

    def update_latest_video(self, gt_video: torch.Tensor) -> None:
        """Update a specific key in the most recently added batch."""
        if not self.batch_queue:
            raise ValueError("No batches in queue")
        device, dtype = self.get_latest_batch()["video"].device, self.get_latest_batch()["video"].dtype
        gt_video = gt_video.to(device=device, dtype=dtype)  # (B,C,V*Ts,H,W), uint8

        sample_n_views = self.get_latest_batch()["sample_n_views"]
        action_horizon = self.get_latest_batch()["action"].shape[1]  # (B,v2,D)

        _, _, VTs, _, _ = gt_video.shape
        _, _, Vv12, _, _ = self.get_latest_batch()["video"].shape
        Ts = VTs // sample_n_views  # feedback frames in gt_video single view, without obs frames
        v12 = Vv12 // sample_n_views  # v1+v2 in a single view
        v2 = action_horizon // self.future_skip_frames  # v2 = skipped future prediction
        v1 = v12 - v2
        assert Ts == v2 * self.future_skip_frames
        gt_skipped_indices = np.arange(self.future_skip_frames - 1, Ts)[0::self.future_skip_frames]  # no obs frames
        assert gt_skipped_indices[-1] == Ts - 1, \
            f"Last skipped index {gt_skipped_indices[-1]} does not match Ts-1 {Ts - 1}"
        gt_skipped_video = get_frames_from_multiview_video(
            gt_video,
            sample_n_views=sample_n_views,
            skipped_indices=gt_skipped_indices,
        )  # (B,V*(v2),H,W,C)

        # NOTE: force data also needs to be handled

        self.batch_queue[-1]["video"] = replace_multiview_video_back_with_another(
            self.batch_queue[-1]["video"],  # (B,C,V*(v1+v2),H,W)
            gt_skipped_video,
            sample_n_views=sample_n_views,
            replace_length=v2,
        )
        log.info(f"Updated latest batch video with new ground truth video.")

    def get_latest_batch(self) -> Dict[str, Any]:
        """Get the most recently added batch."""
        if not self.batch_queue:
            raise ValueError("No batches in queue")

        # 深拷贝最新批次
        latest_batch = {}
        for key, value in self.batch_queue[-1].items():
            if isinstance(value, torch.Tensor):
                latest_batch[key] = value.clone()
            elif isinstance(value, np.ndarray):
                latest_batch[key] = value.copy()
            elif isinstance(value, list):
                latest_batch[key] = value.copy()
            elif isinstance(value, dict):
                latest_batch[key] = value.copy()
            else:
                # 对于基本类型（int, float, str），直接赋值即可
                latest_batch[key] = value

        latest_batch["video"] = self.video_augmentor.augment_video(
            latest_batch["video"])  # (B,C,V*Ts,H,W), uint8

        return latest_batch

    def get_oldest_batch(self) -> Dict[str, Any]:
        """Get the oldest batch in the queue."""
        if not self.batch_queue:
            raise ValueError("No batches in queue")
        return self.batch_queue[0]

    def remove_oldest_batch(self) -> Optional[Dict[str, Any]]:
        """Remove and return the oldest batch in the queue."""
        if not self.batch_queue:
            return None
        oldest_batch = self.batch_queue.popleft()
        log.info("Removed the oldest batch from the queue.")
        return oldest_batch

    def peek_batch_info(self) -> List[Dict[str, Any]]:
        """View basic information about all batches without returning actual data."""
        info_list = []
        for i, batch in enumerate(self.batch_queue):
            info = {"batch_index": i}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    info[key] = {
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                        "device": str(value.device)
                    }
                elif isinstance(value, np.ndarray):
                    info[key] = {
                        "shape": list(value.shape),
                        "dtype": str(value.dtype)
                    }
                else:
                    info[key] = {"type": type(value).__name__, "value": value}
            info_list.append(info)
        log.info(f"Peeked at batch info for {len(self.batch_queue)} batches.")
        return info_list

    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch data to the specified device."""
        moved_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved_batch[key] = value.to(self.device)
            elif isinstance(value, np.ndarray):
                moved_batch[key] = torch.from_numpy(value).to(self.device)
            else:
                moved_batch[key] = value
        return moved_batch

    def set_device(self, device: str) -> None:
        """Change the device and move all existing batches to the new device."""
        self.device = device
        for i in range(len(self.batch_queue)):
            self.batch_queue[i] = self._move_batch_to_device(self.batch_queue[i])
        log.info(f"Set new device to {device} and moved all batches to the new device.")
