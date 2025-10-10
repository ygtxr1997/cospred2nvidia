import logging
import os
from pathlib import Path
from typing import Dict, List

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
# import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
from torch.utils.data import IterableDataset
import torchvision

# from uha import make_pytorch_oxe_iterable_dataset, get_octo_dataset_tensorflow, get_single_dataset_tensorflow, multi_worker_iterable_dataset
from uha.uha_datamodule import UhaDataModule, UhaDataModuleNoValidationSet


logger = logging.getLogger(__name__)
DEFAULT_TRANSFORM = OmegaConf.create({"train": None, "val": None})
ONE_EP_DATASET_URL = "http://www.informatik.uni-freiburg.de/~meeso/50steps.tar.xz"


class NoEncoder(torch.nn.Module):
    def __init__(self, model_name="", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model_name = model_name
    def forward(self, batch_text):
        return batch_text


class NodeBroadcastDataLoader:
    """在节点的rank0上加载数据并广播到同节点的其他GPUs"""

    def __init__(self, dataloader, rank, gpus_per_node=8):
        self.dataloader = dataloader
        self.rank = rank
        self.gpus_per_node = gpus_per_node
        self.node_id = rank // gpus_per_node

        # 创建节点内的进程组
        node_ranks = [self.node_id * gpus_per_node + i for i in range(gpus_per_node)]
        self.node_group = dist.new_group(node_ranks)

    def __iter__(self):
        for batch in self.dataloader:
            # 广播batch到节点内的其他GPUs
            self._broadcast_batch(batch)
            yield batch

    def _broadcast_batch(self, batch):
        """递归广播batch中的所有tensors"""
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                if not value.is_cuda:
                    value = value.cuda()
                    batch[key] = value
                dist.broadcast(value, src=self.rank, group=self.node_group)
            elif isinstance(value, dict):
                self._broadcast_batch(value)

    def __len__(self):
        return len(self.dataloader)


class NodeReceiverDataLoader:
    """在节点的其他ranks上接收来自rank0的数据"""

    def __init__(self, batch_template, rank, gpus_per_node=8):
        self.batch_template = batch_template
        self.rank = rank
        self.gpus_per_node = gpus_per_node
        self.node_id = rank // gpus_per_node
        self.src_rank = self.node_id * gpus_per_node

        # 创建节点内的进程组
        node_ranks = [self.node_id * gpus_per_node + i for i in range(gpus_per_node)]
        self.node_group = dist.new_group(node_ranks)

        # 接收迭代次数
        self.num_iterations = None

    def __iter__(self):
        # 第一次迭代时接收总迭代次数
        if self.num_iterations is None:
            num_iter_tensor = torch.tensor([0], device='cuda')
            dist.broadcast(num_iter_tensor, src=self.src_rank, group=self.node_group)
            self.num_iterations = num_iter_tensor.item()

        for _ in range(self.num_iterations):
            # 创建空batch并接收数据
            batch = self._create_empty_batch()
            self._receive_batch(batch)
            yield batch

    def _create_empty_batch(self):
        """根据模板创建空batch"""
        batch = {}
        for key, value in self.batch_template.items():
            if isinstance(value, torch.Tensor):
                batch[key] = torch.empty_like(value, device='cuda')
            elif isinstance(value, dict):
                batch[key] = self._create_empty_dict(value)
            else:
                batch[key] = value
        return batch

    def _create_empty_dict(self, template_dict):
        """递归创建空字典"""
        result = {}
        for key, value in template_dict.items():
            if isinstance(value, torch.Tensor):
                result[key] = torch.empty_like(value, device='cuda')
            elif isinstance(value, dict):
                result[key] = self._create_empty_dict(value)
            else:
                result[key] = value
        return result

    def _receive_batch(self, batch):
        """递归接收batch"""
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                dist.broadcast(value, src=self.src_rank, group=self.node_group)
            elif isinstance(value, dict):
                self._receive_batch(value)

    def __len__(self):
        return self.num_iterations if self.num_iterations else 0


class DdpTfBroadcastDataset(IterableDataset):
    """
    Rank 0 读取 & collate & to(cuda) 后，先广播 meta（结构/形状/dtype），
    非主进程按 meta 正确建空张量，再用 NCCL broadcast 真正拷贝数据。
    然后仅对 batch 维（=B_super）切片，保证各 rank 样本互斥且不会破坏非 batch 张量。
    """

    def __init__(self, tf_dataloader, tf_iterator, first_batch,
                 rank, world_size, batch_size, collate_fn=None):
        self.tf_dataloader = tf_dataloader
        self.tf_iterator = tf_iterator
        self.first_batch = first_batch
        self.rank = rank
        self.world_size = world_size
        self.batch_size = batch_size
        self.B_super = batch_size * world_size
        self.collate_fn = collate_fn
        self.is_main_rank = (rank == 0)
        self.num_batches = None

        # 在第一次迭代后会设置
        self._receiver_template = None
        self._meta_spec = None

        if self.is_main_rank:
            self.stream = torch.cuda.Stream()

    # ---------- 元信息编解码 ----------

    @staticmethod
    def _tensor_meta(t: torch.Tensor):
        # 返回 (kind, shape, dtype_str)
        return ("tensor", tuple(t.shape), str(t.dtype))

    @staticmethod
    def _pack_meta(x):
        if isinstance(x, dict):
            # 需要固定顺序（按 key 排序），保证各 rank 递归遍历一致
            return ("dict", {k: DdpTfBroadcastDataset._pack_meta(x[k]) for k in sorted(x.keys())})
        elif isinstance(x, torch.Tensor):
            return DdpTfBroadcastDataset._tensor_meta(x)
        elif isinstance(x, (list, tuple)):
            return ("list", [DdpTfBroadcastDataset._pack_meta(v) for v in x])
        else:
            # 标量或字符串
            return ("obj", type(x).__name__)

    @staticmethod
    def _dtype_from_str(dtype_str: str):
        # str(t.dtype) 如 "torch.float32" / "torch.uint8"
        name = dtype_str.split(".")[-1]
        return getattr(torch, name)

    @staticmethod
    def _alloc_from_meta(meta):
        kind = meta[0]
        if kind == "tensor":
            _, shape, dtype_str = meta
            return torch.empty(shape, dtype=DdpTfBroadcastDataset._dtype_from_str(dtype_str), device="cuda")
        elif kind == "dict":
            _, child = meta
            return {k: DdpTfBroadcastDataset._alloc_from_meta(child[k]) for k in sorted(child.keys())}
        elif kind == "list":
            _, child = meta
            return [DdpTfBroadcastDataset._alloc_from_meta(m) for m in child]
        else:
            # 标量或字符串，接收端初值占位；稍后用 broadcast_object_list 同步其值
            return None

    @staticmethod
    def _clone_empty_like(obj):
        if isinstance(obj, dict):
            return {k: DdpTfBroadcastDataset._clone_empty_like(v) for k, v in obj.items()}
        elif isinstance(obj, torch.Tensor):
            return torch.empty_like(obj, device="cuda")
        elif isinstance(obj, (list, tuple)):
            return type(obj)([DdpTfBroadcastDataset._clone_empty_like(v) for v in obj])
        else:
            return None

    # ---------- 广播叶子 ----------

    def _broadcast_tensor(self, tensor, src=0):
        if tensor is None:
            return None
        if not tensor.is_cuda:
            tensor = tensor.cuda(non_blocking=True)
        dist.broadcast(tensor, src=src)
        return tensor

    def _broadcast_object(self, value, src=0):
        obj_list = [value if self.rank == src else None]
        dist.broadcast_object_list(obj_list, src=src)
        return obj_list[0]

    def _broadcast_batch_recursive(self, batch, meta, src=0):
        """用 meta 指导递归广播，保证所有 rank 的遍历次序和类型一致。"""
        kind = meta[0]

        if kind == "dict":
            _, child = meta
            out = {}
            for k in sorted(child.keys()):
                out[k] = self._broadcast_batch_recursive(batch[k], child[k], src)
            return out

        elif kind == "tensor":
            # 这里 batch 是 torch.Tensor
            return self._broadcast_tensor(batch, src)

        elif kind == "list":
            _, child_list = meta
            return [self._broadcast_batch_recursive(batch[idx], child_list[idx], src)
                    for idx in range(len(child_list))]

        else:
            # 标量/字符串
            return self._broadcast_object(batch, src)

    # ---------- 工具 ----------

    def _is_batched_tensor(self, t: torch.Tensor) -> bool:
        return t.dim() >= 1 and t.size(0) == self.B_super

    def _slice_batch(self, batch, start_idx, end_idx):
        """只切 batch 维=首维且等于 B_super 的张量；其他保持不变。"""
        if isinstance(batch, dict):
            return {k: self._slice_batch(v, start_idx, end_idx) for k, v in batch.items()}

        elif isinstance(batch, torch.Tensor):
            if self._is_batched_tensor(batch):
                return batch[start_idx:end_idx].contiguous()
            else:
                # 非 batch 张量（如 (512,), (4,) 等），不切片，保持一致
                return batch

        elif isinstance(batch, (list, tuple)):
            return type(batch)([self._slice_batch(v, start_idx, end_idx) for v in batch])

        else:
            # 标量/字符串
            return batch

    def _move_to_cuda(self, batch):
        if isinstance(batch, dict):
            return {k: self._move_to_cuda(v) for k, v in batch.items()}
        elif isinstance(batch, torch.Tensor):
            return batch.cuda(non_blocking=True) if not batch.is_cuda else batch
        elif isinstance(batch, (list, tuple)):
            return type(batch)([self._move_to_cuda(v) for v in batch])
        else:
            return batch

    # ---------- 迭代 ----------

    def __iter__(self):
        if self.num_batches is None:
            raise RuntimeError("num_batches not set!")

        num_iterations = self.num_batches
        rank = self.rank
        print(f"[INFO] Rank {rank} starting iteration with {num_iterations} batches")

        iterator = self.tf_iterator

        for i in range(num_iterations):
            if i % 100 == 0 or i < 3:
                print(f"[DEBUG] Rank {rank} - Iteration {i}/{num_iterations}")

            # --- 准备 super_batch ---
            if self.is_main_rank:
                # 取 batch（首个迭代复用提前取好的 first_batch）
                if i == 0 and self.first_batch is not None:
                    super_batch = self.first_batch
                else:
                    super_batch = next(iterator)

                # 应用 collate
                if self.collate_fn:
                    super_batch = self.collate_fn(super_batch)

                # 上 GPU
                with torch.cuda.stream(self.stream):
                    super_batch = self._move_to_cuda(super_batch)
                torch.cuda.current_stream().wait_stream(self.stream)

                # 在 i==0 时生成并广播 meta
                if i == 0:
                    self._meta_spec = self._pack_meta(super_batch)
                    meta_obj = [self._meta_spec]
                    dist.broadcast_object_list(meta_obj, src=0)
                else:
                    # 非首轮，仍然需要给其他 rank 一个同步点
                    meta_obj = [None]
                    dist.broadcast_object_list(meta_obj, src=0)

                # 主进程已有真实数据，无需分配模板
                pass

            else:
                # 非主进程：首轮先接 meta 再建模板
                meta_obj = [None]
                dist.broadcast_object_list(meta_obj, src=0)

                if i == 0:
                    self._meta_spec = meta_obj[0]
                    # 用 meta 构建一次“接收模板”
                    self._receiver_template = self._alloc_from_meta(self._meta_spec)

                # 每轮克隆一个空模板来接收实际数据
                super_batch = self._clone_empty_like(self._receiver_template)

            # --- 真正广播所有叶子 ---
            super_batch = self._broadcast_batch_recursive(super_batch, self._meta_spec, src=0)

            # --- 切片并 yield ---
            start_idx = rank * self.batch_size
            end_idx = start_idx + self.batch_size
            local_batch = self._slice_batch(super_batch, start_idx, end_idx)

            yield local_batch

        print(f"[INFO] Rank {rank} finished all {num_iterations} iterations")

    def __len__(self):
        if self.num_batches is None:
            raise RuntimeError("num_batches not set!")
        return self.num_batches



class OxeUhaDataModule(object):
    CAMERA_KEY_TO_VIEW = {
        "primary": "image_primary",
        "secondary": "image_secondary",
        "wrist": "image_wrist",
    }
    VIEW_CHOICES = ["image_primary", "image_secondary", "image_wrist"]
    CAMERA_TO_VIEW_ID = {
        "image_primary": 0,
        "image_secondary": 1,
        "image_wrist": 2,
    }
    def __init__(
        self,
        # OriUhaDataModule
        transforms: DictConfig,  # Replace with your default transforms
        language_encoders: DictConfig,
        datasets: DictConfig,  # 'DATA_NAME','DATA_PATH','load_camera_views','action_xxx_norm','interleaved_cfg'
        batch_size: int = 4,
        num_workers: int = 0,  # will have no effect
        pin_memory: bool = False,
        drop_last: bool = True,
        # T5 text embeddings
        t5_embedding_subdir: str = "lang_emb_t5xxl",  # under DATA_PATH
        # CosmosPredict2 specific
        use_ori_uha_data_collate: bool = False,  # use ori_uha_data_module's collate_fn
        p_camera_drop: float = 0.0,
        p_proprio_drop: float = 0.0,
        state_t: int = 1+1+5,  # latent video frames = cond + pred
        # Others
        **kwargs: Dict,
    ):
        super().__init__()
        self.transforms_cfg = transforms
        self.language_encoders_cfg = language_encoders
        self.datasets_cfg = datasets

        if isinstance(language_encoders, DictConfig):
            assert "_target_" in language_encoders, "Language encoder config must have a '_target_' field."
        else:
            language_encoders = OmegaConf.create({
                "_target_": "cosmos_predict2.data.action_conditioned.uha_dataset.NoEncoder",
                "model_name": "none",
            })  # NOTE: hard coded for now, instantiating inside the class is not a good idea

        self.ori_uha_cfg = OmegaConf.create({
            'datasets': datasets,
            'transforms': transforms,
            'language_encoders': language_encoders,
            'batch_size': batch_size,
            'num_workers': num_workers,
            'pin_memory': pin_memory,
            'drop_last': drop_last,
        })  # merge configs

        self.batch_size = batch_size
        self.modalities = ['lang']  # lang only

        self.t5_embedding_subdir = t5_embedding_subdir
        self.data_name = datasets['DATA_NAME']
        self.camera_keys = [self.CAMERA_KEY_TO_VIEW[k] for k in datasets['load_camera_views']]
        self.window_size = datasets['interleaved_dataset_cfg']['traj_transform_kwargs']['window_size']
        self.action_horizon = datasets['interleaved_dataset_cfg']['traj_transform_kwargs']['action_horizon']
        self.horizon = self.window_size + self.action_horizon

        self.use_ori_uha_data_collate = use_ori_uha_data_collate
        self.p_camera_drop = p_camera_drop
        self.p_proprio_drop = p_proprio_drop
        self.state_t = state_t

        self.prepare_data()
        self.setup()

    def prepare_data(self, *args, **kwargs):
        pass
        print('[DEBUG] OxeUhaDataModule prepare_data finished.')

    def setup(self, stage=None):
        """
        Called by trainer.fit()
        """
        cfg = self.ori_uha_cfg
        is_main_process = os.environ.get("LOCAL_RANK", "0") == "0"

        ''' 1. Get dataloaders '''
        self.ori_uha_data_module = UhaDataModuleNoValidationSet(
            datasets=self.ori_uha_cfg['datasets'],
            batch_size=self.ori_uha_cfg['batch_size'],
            num_workers=self.ori_uha_cfg['num_workers'],
            pin_memory=self.ori_uha_cfg['pin_memory'],
            drop_last=self.ori_uha_cfg['drop_last'],
            transforms=self.ori_uha_cfg['transforms'],
            language_encoders=self.ori_uha_cfg['language_encoders'],
        )
        self.train_loader = self.ori_uha_data_module.create_train_dataloader(main_process=is_main_process)
        # self.val_loader = self.ori_uha_data_module.create_val_dataloader()
        self.val_loader = None

        ''' 2. Get dataset info '''
        self.dataset_info = self.ori_uha_data_module.get_dataset_statistics()

        print(f'[DEBUG] OxeUhaDataModule setup finished (main={is_main_process}). '
              f'Train len={len(self.train_loader)}, '
              # f'Val len={len(self.val_loader)}. '
              f'Info: {self.dataset_info}')

    def custom_collate_fn(self, batch):  # used by DataLoader
        # 如果 batch 不是列表 (例如，在 DDP 模式下，它已经是一个整理好的字典),
        # [ {'action': tensor, 'video': tensor, ...} ]
        # 我们需要取出这个字典
        # from cosmos_predict2.utils.printer import print_batch
        # rank = dist.get_rank()
        # print_batch(f'[DEBUG]rank@{rank}:custom_collate_fn', batch)
        # return batch

        if isinstance(batch, list) and len(batch) == 1 and isinstance(batch[0], dict):
            batch = batch[0]
        # 如果 batch 仍然是列表 (例如，在非 DDP 模式下),
        # 则执行 default_collate。
        elif isinstance(batch, list):
            batch = torch.utils.data.default_collate(batch)

        # 检查 batch 是否为空或无效
        if not batch:
            # 在 DDP 模式下，非主进程可能会收到空 batch，直接返回 None 或空字典
            # 让训练循环来处理这种情况
            return None

        B, T, H, D = batch["action"].shape

        # 1. 转换、重排并拼接所有视频
        video_list = []
        for view in ['primary', 'secondary', 'wrist']:
            video_key = f'image_{view}'
            view_videos = []

            # 按时间顺序收集 observation 和 future_frames
            for key in ['observation', 'future_frames']:
                if key in batch and video_key in batch[key]:
                    # 转换类型和维度，然后添加到列表
                    view_videos.append(
                        batch[key][video_key].to(torch.uint8).permute(0, 2, 1, 3, 4)
                    )
                    del batch[key][video_key]  # 尽早删除以释放内存

            if view_videos:
                # 拼接当前视图的所有时间步并添加到主列表
                video_list.append(torch.cat(view_videos, dim=2))

        # 沿时间维度拼接所有视图
        concatenated_video = torch.cat(video_list, dim=2)
        del video_list  # 清理列表

        # 2. 准备视图索引
        ret_n_views = len(self.camera_keys)
        view_indices_selection = [self.CAMERA_TO_VIEW_ID[camera_key] for camera_key in self.camera_keys]
        view_indices_t = torch.tensor(view_indices_selection).repeat_interleave(self.horizon)
        latent_view_indices_t = torch.tensor(view_indices_selection).repeat_interleave(self.state_t)
        view_indices_t = view_indices_t.unsqueeze(0).expand(B, -1).clone()
        latent_view_indices_t = latent_view_indices_t.unsqueeze(0).expand(B, -1).clone()

        # 3. 提取并填充动作
        action = batch["action"][:, -1]
        action_prefix = torch.zeros((B, self.window_size, D), dtype=action.dtype, device=action.device)
        action = torch.cat([action_prefix, action], dim=1)

        # 4. 处理本体感受信息
        if "proprio" in batch["observation"]:
            agent_pos = batch["observation"]["proprio"]
            if self.p_proprio_drop > 0.0 and self.p_proprio_drop < np.random.rand():
                agent_pos = torch.zeros_like(agent_pos)
            agent_pos_suffix = torch.zeros((B, self.action_horizon, agent_pos.shape[-1]), dtype=agent_pos.dtype, device=agent_pos.device)
            agent_pos = torch.cat([agent_pos, agent_pos_suffix], dim=1)
        else:
            agent_pos = torch.zeros((B, self.horizon, 8), dtype=torch.float32)


        # 5. 提取文本嵌入
        ret_text_embeddings = batch["task"]["language_embedding"].to(torch.bfloat16)

        del batch # 删除原始批处理字典

        remapped_data = {
            "action": action,
            "video": concatenated_video,
            "agent_pos": agent_pos,
            "annotation_file": "None",
            "__key__": "None",
            "t5_text_embeddings": ret_text_embeddings,
            "t5_text_mask": torch.ones(512, dtype=torch.int64),
            "fps": torch.ones(B) * 10,
            "image_size": torch.tensor([176, 176, 176, 176]),
            "num_frames": self.horizon,
            "padding_mask": torch.zeros(B, 1, 176, 176, dtype=torch.bool),
            "sample_n_views": ret_n_views,
            "view_indices": view_indices_t,
            "latent_view_indices_B_T": latent_view_indices_t,
        }
        return remapped_data

    def get_gpus_per_node(self):
        try:
            if 'CUDA_VISIBLE_DEVICES' in os.environ:
                # 从CUDA_VISIBLE_DEVICES计算GPU数量
                cuda_devices = os.environ['CUDA_VISIBLE_DEVICES']
                if cuda_devices:
                    gpus_per_node = len([x.strip() for x in cuda_devices.split(',') if x.strip()])
                else:
                    gpus_per_node = torch.cuda.device_count()
            else:
                # 从torchrun参数或手动设置
                gpus_per_node = int(os.environ.get('WORLD_SIZE',
                                                   os.environ.get('LOCAL_WORLD_SIZE', '4')))
        except (ValueError, TypeError):
            # 后备方案：使用可见的GPU数量
            gpus_per_node = torch.cuda.device_count()
            print(f"[Warning]: cannot parse gpus_per_node, fallback to {gpus_per_node}")
        return gpus_per_node

    # def train_dataloader(self):
    #     original_loader = self.train_loader
    #
    #     if self.use_ori_uha_data_collate:
    #         return original_loader
    #         '''
    #         fractal: Dict,keys=dict_keys(['observation', 'task', 'action', 'action_pad_mask', 'future_frames'])
    #         observation: Dict,keys=dict_keys(['image_primary', 'image_secondary', 'image_wrist', 'proprio', 'timestep', 'pad_mask_dict', 'timestep_pad_mask', 'task_completed'])
    #         -image_primary,<class 'torch.Tensor'>,shape=torch.Size([4, 5, 3, 224, 224]),min=0.0000,max=255.0000
    #         -image_secondary,<class 'torch.Tensor'>,shape=torch.Size([4, 5, 3, 224, 224]),min=0.0000,max=0.0000
    #         -image_wrist,<class 'torch.Tensor'>,shape=torch.Size([4, 5, 3, 84, 84]),min=0.0000,max=0.0000
    #         -proprio,<class 'torch.Tensor'>,shape=torch.Size([4, 5, 8]),min=-1.0000,max=0.8496
    #         -timestep,<class 'torch.Tensor'>,shape=torch.Size([4, 5]),min=0.0000,max=9.0000
    #         -pad_mask_dict: Dict,keys=dict_keys(['image_primary', 'image_secondary', 'image_wrist', 'proprio', 'timestep'])
    #         --image_primary,<class 'torch.Tensor'>,shape=torch.Size([4, 5]),min=1.0000,max=1.0000
    #         --image_secondary,<class 'torch.Tensor'>,shape=torch.Size([4, 5]),min=0.0000,max=0.0000
    #         --image_wrist,<class 'torch.Tensor'>,shape=torch.Size([4, 5]),min=0.0000,max=0.0000
    #         --proprio,<class 'torch.Tensor'>,shape=torch.Size([4, 5]),min=1.0000,max=1.0000
    #         --timestep,<class 'torch.Tensor'>,shape=torch.Size([4, 5]),min=1.0000,max=1.0000
    #         -timestep_pad_mask,<class 'torch.Tensor'>,shape=torch.Size([4, 5]),min=0.0000,max=1.0000
    #         -task_completed,<class 'torch.Tensor'>,shape=torch.Size([4, 5, 20]),min=0.0000,max=0.0000
    #         task: Dict,keys=dict_keys(['language_instruction', 'language_key', 'language_embedding', 'pad_mask_dict'])
    #         -language_instruction: List,len=4,elem:<class 'str'>
    #         -language_key,<class 'torch.Tensor'>,shape=torch.Size([4]),min=0.0000,max=0.0000
    #         -language_embedding,<class 'torch.Tensor'>,shape=torch.Size([4, 512, 1024]),min=-0.6177,max=0.6455
    #         -pad_mask_dict: Dict,keys=dict_keys(['language_instruction', 'language_key', 'language_embedding'])
    #         --language_instruction,<class 'torch.Tensor'>,shape=torch.Size([4]),min=1.0000,max=1.0000
    #         --language_key,<class 'torch.Tensor'>,shape=torch.Size([4]),min=1.0000,max=1.0000
    #         --language_embedding,<class 'torch.Tensor'>,shape=torch.Size([4]),min=1.0000,max=1.0000
    #         action,<class 'torch.Tensor'>,shape=torch.Size([4, 5, 20, 7]),min=-1.0000,max=1.0000
    #         action_pad_mask,<class 'torch.Tensor'>,shape=torch.Size([4, 5, 20, 7]),min=1.0000,max=1.0000
    #         future_frames: Dict,keys=dict_keys(['image_primary', 'image_secondary', 'image_wrist', 'pad_mask_dict', 'timestep_pad_mask'])
    #         -image_primary,<class 'torch.Tensor'>,shape=torch.Size([4, 20, 3, 224, 224]),min=0.0000,max=255.0000
    #         -image_secondary,<class 'torch.Tensor'>,shape=torch.Size([4, 20, 3, 112, 112]),min=0.0000,max=0.0000
    #         -image_wrist,<class 'torch.Tensor'>,shape=torch.Size([4, 20, 3, 112, 112]),min=0.0000,max=0.0000
    #         -pad_mask_dict: Dict,keys=dict_keys(['image_primary', 'image_secondary', 'image_wrist', 'proprio', 'timestep'])
    #         --image_primary,<class 'torch.Tensor'>,shape=torch.Size([4, 20]),min=1.0000,max=1.0000
    #         --image_secondary,<class 'torch.Tensor'>,shape=torch.Size([4, 20]),min=0.0000,max=0.0000
    #         --image_wrist,<class 'torch.Tensor'>,shape=torch.Size([4, 20]),min=0.0000,max=0.0000
    #         --proprio,<class 'torch.Tensor'>,shape=torch.Size([4, 20]),min=1.0000,max=1.0000
    #         --timestep,<class 'torch.Tensor'>,shape=torch.Size([4, 20]),min=1.0000,max=1.0000
    #         -timestep_pad_mask,<class 'torch.Tensor'>,shape=torch.Size([4, 20]),min=1.0000,max=1.0000
    #         '''
    #     else:
    #         return DataLoader(
    #             dataset=original_loader.dataset,
    #             batch_size=original_loader.batch_size,
    #             num_workers=original_loader.num_workers,
    #             pin_memory=original_loader.pin_memory,
    #             drop_last=original_loader.drop_last,
    #             prefetch_factor=original_loader.prefetch_factor,
    #             shuffle=None,
    #             collate_fn=self.custom_collate_fn,
    #         )
    #         '''
    #         fractal: Dict,keys=dict_keys(['action', 'video', 'agent_pos', 'annotation_file', 't5_text_embedding', 't5_text_mask', 'fps', 'image_size', 'num_frames', 'padding_mask', 'sample_n_views', 'view_indices', 'latent_view_indices_B_T'])
    #         action,<class 'torch.Tensor'>,shape=torch.Size([32, 20, 7]),min=-1.0000,max=1.0000
    #         video,<class 'torch.Tensor'>,shape=torch.Size([32, 3, 50, 224, 224]),min=0.0000,max=255.0000
    #         agent_pos,<class 'torch.Tensor'>,shape=torch.Size([32, 5, 8]),min=-1.0000,max=1.0000
    #         annotation_file:<class 'str'>,len=4
    #         t5_text_embedding,<class 'torch.Tensor'>,shape=torch.Size([32, 512, 1024]),min=-0.6523,max=0.7485
    #         t5_text_mask,<class 'torch.Tensor'>,shape=torch.Size([512]),min=1.0000,max=1.0000
    #         fps:<class 'int'>,10
    #         image_size,<class 'torch.Tensor'>,shape=torch.Size([4]),min=224.0000,max=224.0000
    #         num_frames:<class 'int'>,25
    #         padding_mask,<class 'torch.Tensor'>,shape=torch.Size([32, 1, 224, 224]),min=0.0000,max=0.0000
    #         sample_n_views:<class 'int'>,2
    #         view_indices,<class 'torch.Tensor'>,shape=torch.Size([32, 50]),min=0.0000,max=1.0000
    #         latent_view_indices_B_T,<class 'torch.Tensor'>,shape=torch.Size([32, 14]),min=0.0000,max=1.0000
    #         '''

    def train_dataloader(self):
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            super_batch_size = self.batch_size * world_size

            tf_dataloader = None
            tf_iterator = None
            first_batch = None  # ✅ 新增：存储预加载的 batch
            num_batches = 0

            if rank == 0:
                print(f"[INFO] Rank 0 creating TF dataloader...")
                uha_train_cfg = self.ori_uha_cfg.copy()
                uha_train_cfg['batch_size'] = super_batch_size
                uha_train_cfg['num_workers'] = 0

                temp_uha_module = UhaDataModuleNoValidationSet(
                    datasets=uha_train_cfg['datasets'],
                    batch_size=uha_train_cfg['batch_size'],
                    num_workers=uha_train_cfg['num_workers'],
                    pin_memory=uha_train_cfg['pin_memory'],
                    drop_last=uha_train_cfg['drop_last'],
                    transforms=uha_train_cfg['transforms'],
                    language_encoders=uha_train_cfg['language_encoders'],
                )
                tf_dataloader = temp_uha_module.create_train_dataloader(main_process=True)
                num_batches = len(tf_dataloader)

                print(f"[INFO] Rank 0 initializing TF iterator...")
                tf_iterator = iter(tf_dataloader)

                # ✅ 预加载第一个 batch
                print(f"[DEBUG] Rank 0 pre-loading first batch...")
                try:
                    first_batch = next(tf_iterator)
                    print(
                        f"[INFO] Rank 0 finished initialization, first batch shape: {first_batch['observation']['image_primary'].shape}")
                except Exception as e:
                    print(f"[ERROR] Rank 0 failed to load first batch: {e}")
                    raise

                print(f"[INFO] Rank 0 ready with {num_batches} batches")
            else:
                print(f"[DEBUG] Rank {rank} waiting for rank 0 to finish initialization...")

            print(f"[DEBUG] Rank {rank} reaching barrier...")
            dist.barrier()
            print(f"[DEBUG] Rank {rank} passed barrier")

            # 同步数据集长度
            num_batches_tensor = torch.tensor([num_batches], dtype=torch.long, device='cuda')
            dist.broadcast(num_batches_tensor, src=0)
            if rank != 0:
                num_batches = num_batches_tensor.item()
            print(f"[INFO] Rank {rank} - Synced dataloader length: {num_batches}")

            collate_fn_for_broadcast = None if self.use_ori_uha_data_collate else self.custom_collate_fn

            # ✅ 传递 iterator 和预加载的 batch
            broadcast_dataset = DdpTfBroadcastDataset(
                tf_dataloader=tf_dataloader,
                tf_iterator=tf_iterator,
                first_batch=first_batch,  # ✅ 新增参数
                rank=rank,
                world_size=world_size,
                batch_size=self.batch_size,
                collate_fn=collate_fn_for_broadcast
            )

            broadcast_dataset.num_batches = num_batches
            print(f"[INFO] Rank {rank} - Dataset __len__: {len(broadcast_dataset)}")

            return DataLoader(
                dataset=broadcast_dataset,
                batch_size=None,
                num_workers=0,
                pin_memory=True,
                drop_last=True,
                collate_fn=lambda x: x[0] if isinstance(x, list) else x,
            )
        else:
            # 非 DDP 环境
            original_loader = self.train_loader
            if self.use_ori_uha_data_collate:
                return original_loader
            else:
                return DataLoader(
                    dataset=original_loader.dataset,
                    batch_size=original_loader.batch_size,
                    num_workers=original_loader.num_workers,
                    pin_memory=original_loader.pin_memory,
                    drop_last=original_loader.drop_last,
                    collate_fn=self.custom_collate_fn,
                )

    def _create_super_batch_template(self, B_super):
        """
        创建 super batch 模板（供非主进程接收数据用）
        形状必须与 rank 0 的 super batch 完全一致
        """
        if self.use_ori_uha_data_collate:
            # UHA 原始格式
            return {
                "action": torch.empty((B_super, self.window_size, self.action_horizon, 7),
                                      dtype=torch.float32, device='cuda'),
                "action_pad_mask": torch.empty((B_super, self.window_size, self.action_horizon, 7),
                                               dtype=torch.float32, device='cuda'),
                "future_frames": {
                    "image_primary": torch.empty((B_super, self.action_horizon, 3, 224, 224),
                                                 dtype=torch.float32, device='cuda'),
                    "image_secondary": torch.empty((B_super, self.action_horizon, 3, 112, 112),
                                                   dtype=torch.float32, device='cuda'),
                    "image_wrist": torch.empty((B_super, self.action_horizon, 3, 112, 112),
                                               dtype=torch.float32, device='cuda'),
                    "pad_mask_dict": {
                        "image_primary": torch.empty((B_super, self.action_horizon),
                                                     dtype=torch.float32, device='cuda'),
                        "image_secondary": torch.empty((B_super, self.action_horizon),
                                                       dtype=torch.float32, device='cuda'),
                        "image_wrist": torch.empty((B_super, self.action_horizon),
                                                   dtype=torch.float32, device='cuda'),
                        "proprio": torch.empty((B_super, self.action_horizon),
                                               dtype=torch.float32, device='cuda'),
                        "timestep": torch.empty((B_super, self.action_horizon),
                                                dtype=torch.float32, device='cuda'),
                    },
                    "timestep_pad_mask": torch.empty((B_super, self.action_horizon),
                                                     dtype=torch.float32, device='cuda'),
                },
                "observation": {
                    "image_primary": torch.empty((B_super, self.window_size, 3, 224, 224),
                                                 dtype=torch.float32, device='cuda'),
                    "image_secondary": torch.empty((B_super, self.window_size, 3, 224, 224),
                                                   dtype=torch.float32, device='cuda'),
                    "image_wrist": torch.empty((B_super, self.window_size, 3, 84, 84),
                                               dtype=torch.float32, device='cuda'),
                    "pad_mask_dict": {
                        "image_primary": torch.empty((B_super, self.window_size),
                                                     dtype=torch.float32, device='cuda'),
                        "image_secondary": torch.empty((B_super, self.window_size),
                                                       dtype=torch.float32, device='cuda'),
                        "image_wrist": torch.empty((B_super, self.window_size),
                                                   dtype=torch.float32, device='cuda'),
                        "proprio": torch.empty((B_super, self.window_size),
                                               dtype=torch.float32, device='cuda'),
                        "timestep": torch.empty((B_super, self.window_size),
                                                dtype=torch.float32, device='cuda'),
                    },
                    "proprio": torch.empty((B_super, self.window_size, 8),
                                           dtype=torch.float32, device='cuda'),
                    "task_completed": torch.empty((B_super, self.window_size, 20),
                                                  dtype=torch.float32, device='cuda'),
                    "timestep": torch.empty((B_super, self.window_size),
                                            dtype=torch.float32, device='cuda'),
                    "timestep_pad_mask": torch.empty((B_super, self.window_size),
                                                     dtype=torch.float32, device='cuda'),
                },
                "task": {
                    "language_embedding": torch.empty((B_super, 512, 1024),
                                                      dtype=torch.bfloat16, device='cuda'),
                    "language_key": torch.empty((B_super,), dtype=torch.float32, device='cuda'),
                    "pad_mask_dict": {
                        "language_embedding": torch.empty((B_super,),
                                                          dtype=torch.float32, device='cuda'),
                        "language_instruction": torch.empty((B_super,),
                                                            dtype=torch.float32, device='cuda'),
                        "language_key": torch.empty((B_super,),
                                                    dtype=torch.float32, device='cuda'),
                    },
                },
            }
        else:
            # 你的自定义格式 - 需要根据实际情况调整
            print("[WARNING] Using simplified template for custom format")
            return self._create_batch_template()

    def val_dataloader(self):
        # Directly reuse train_dataloader for validation to simplify
        return self.train_dataloader()
        # original_loader = self.val_loader
        # if self.use_ori_uha_data_collate:
        #     return original_loader
        # else:
        #     return DataLoader(
        #         dataset=original_loader.dataset,
        #         batch_size=original_loader.batch_size,
        #         num_workers=original_loader.num_workers,
        #         pin_memory=original_loader.pin_memory,
        #         drop_last=original_loader.drop_last,
        #         prefetch_factor=original_loader.prefetch_factor,
        #         shuffle=None,
        #         collate_fn=self.custom_collate_fn,
        #     )

    def _create_batch_template(self):
        """创建batch模板，用于接收数据"""
        # 注意：这里的 B 应该是 per-GPU 的 batch_size
        B = self.batch_size
        T = self.horizon
        D = 7  # action维度

        if self.use_ori_uha_data_collate:
            # UHA原始格式的模板
            return {
                "observation": {
                    "image_primary": torch.empty((B, self.window_size, 3, 224, 224), dtype=torch.float32),
                    "image_secondary": torch.empty((B, self.window_size, 3, 224, 224), dtype=torch.float32),
                    "image_wrist": torch.empty((B, self.window_size, 3, 84, 84), dtype=torch.float32),
                    "proprio": torch.empty((B, self.window_size, 8), dtype=torch.float32),
                },
                "task": {
                    "language_embedding": torch.empty((B, 512, 1024), dtype=torch.bfloat16),
                },
                "action": torch.empty((B, self.window_size, self.action_horizon, D), dtype=torch.float32),
                "future_frames": {
                    "image_primary": torch.empty((B, self.action_horizon, 3, 224, 224), dtype=torch.float32),
                    "image_secondary": torch.empty((B, self.action_horizon, 3, 112, 112), dtype=torch.float32),
                    "image_wrist": torch.empty((B, self.action_horizon, 3, 112, 112), dtype=torch.float32),
                },
            }
        else:
            # 重映射后的格式模板
            ret_n_views = len(self.camera_keys)
            return {
                "action": torch.empty((B, T, D), dtype=torch.float32),
                "video": torch.empty((B, 3, T * ret_n_views, 176, 176), dtype=torch.uint8),
                "agent_pos": torch.empty((B, T, 8), dtype=torch.float32),
                "annotation_file": "None",
                "__key__": "None",
                "t5_text_embeddings": torch.empty((B, 512, 1024), dtype=torch.bfloat16),
                "t5_text_mask": torch.ones((512,), dtype=torch.int64),
                "fps": torch.ones((B,)),
                "image_size": torch.tensor([176, 176, 176, 176]),
                "num_frames": T,
                "padding_mask": torch.zeros((B, 1, 176, 176), dtype=torch.bool),
                "sample_n_views": ret_n_views,
                "view_indices": torch.empty((B, T * ret_n_views), dtype=torch.long),
                "latent_view_indices_B_T": torch.empty((B, self.state_t * ret_n_views), dtype=torch.long),
            }