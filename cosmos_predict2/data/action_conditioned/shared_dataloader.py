import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from typing import Iterator, Optional
import queue
import threading


class NodeSharedDataLoader:
    """
    在节点内的所有GPU之间共享一个DataLoader实例
    """

    def __init__(self, dataloader: DataLoader, gpus_per_node: int = 8):
        self.dataloader = dataloader
        self.gpus_per_node = gpus_per_node

        # 获取分布式信息
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            self.local_rank = self.rank % gpus_per_node
            self.node_id = self.rank // gpus_per_node
        else:
            self.world_size = 1
            self.rank = 0
            self.local_rank = 0
            self.node_id = 0

        # 只有每个节点的rank0创建真实的DataLoader迭代器
        self.is_data_loader_rank = (self.local_rank == 0)

        # 共享队列（通过共享内存或进程间通信）
        self._data_queue: Optional[queue.Queue] = None
        self._loader_thread: Optional[threading.Thread] = None

    def __iter__(self) -> Iterator:
        """
        返回一个迭代器，节点内所有GPU共享相同的数据
        """
        if self.is_data_loader_rank:
            # 只有节点内的rank0实际运行DataLoader
            return self._data_loader_iterator()
        else:
            # 其他GPU等待并接收来自rank0的数据
            return self._data_receiver_iterator()

    def _data_loader_iterator(self) -> Iterator:
        """
        DataLoader的主迭代器（仅在节点的rank0上运行）
        """
        for batch in self.dataloader:
            # 将数据广播到节点内的所有其他GPU
            self._broadcast_batch_to_node(batch)
            yield batch

    def _data_receiver_iterator(self) -> Iterator:
        """
        数据接收迭代器（在节点的其他ranks上运行）
        """
        while True:
            # 接收来自节点rank0的数据
            batch = self._receive_batch_from_node()
            if batch is None:
                break
            yield batch

    def _broadcast_batch_to_node(self, batch):
        """
        将batch广播到节点内的所有GPU
        """
        # 获取节点内的所有ranks
        node_ranks = [
            self.node_id * self.gpus_per_node + i
            for i in range(self.gpus_per_node)
        ]

        # 创建进程组（如果还未创建）
        if not hasattr(self, '_node_group'):
            self._node_group = dist.new_group(node_ranks)

        # 广播每个tensor
        self._broadcast_dict_recursive(batch, self._node_group)

    def _receive_batch_from_node(self):
        """
        从节点的rank0接收batch
        """
        # 创建空的接收缓冲区
        batch = self._create_receive_buffer()

        # 接收数据
        self._broadcast_dict_recursive(batch, self._node_group)

        return batch

    def _broadcast_dict_recursive(self, data, group):
        """
        递归广播字典中的所有tensors
        """
        if isinstance(data, dict):
            for key in data:
                self._broadcast_dict_recursive(data[key], group)
        elif isinstance(data, torch.Tensor):
            dist.broadcast(data, src=self.node_id * self.gpus_per_node, group=group)
        elif isinstance(data, (list, tuple)):
            for item in data:
                self._broadcast_dict_recursive(item, group)

    def _create_receive_buffer(self):
        """
        创建用于接收数据的空缓冲区
        需要预先知道数据结构
        """
        # 这里需要根据实际的batch结构来创建
        # 简化实现：第一次迭代时从rank0获取元数据
        pass
