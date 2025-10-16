from collections import deque
from typing import Dict, Any, List, Optional
import torch
import numpy as np

from cosmos_predict2.connects.utils import replace_multiview_video_back_with_another

from imaginaire.utils import log


class OnlineDataHandler:
    def __init__(self, max_batches: int = 100, device: str = "cuda"):
        """
        Online data handler for storing and concatenating batches used in Video2WorldExpertPipeline.

        Args:
            max_batches: Maximum number of batches to store.
            device: Device where the data will be stored.
        """
        self.device = device
        self.max_batches = max_batches
        self.batch_queue = deque(maxlen=max_batches)
        log.info(f"Initialized OnlineDataHandler with max_batches={max_batches}, device={device}")

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
        gt_video = gt_video.to(device=device, dtype=dtype)  # (B,C,V*Ts,H,W)

        sample_n_views = self.get_latest_batch()["sample_n_views"]
        action_horizon = self.get_latest_batch()["action"].shape[1]  # (B,v2,D)

        self.batch_queue[-1]["video"] = replace_multiview_video_back_with_another(
            self.batch_queue[-1]["video"],  # (B,C,V*(v1+v2),H,W)
            gt_video,
            sample_n_views=sample_n_views,
            replace_length=action_horizon,
        )
        log.info(f"Updated latest batch video with new ground truth video.")

    def get_latest_batch(self) -> Dict[str, Any]:
        """Get the most recently added batch."""
        if not self.batch_queue:
            raise ValueError("No batches in queue")
        return self.batch_queue[-1]

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
