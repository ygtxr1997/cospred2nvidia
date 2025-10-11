import os

from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_predict2.data.action_conditioned.pusht_dataset import PushTImageDataset
from imaginaire.lazy_config import LazyCall as L


n_v_cond, n_v_out = 4 * 1 + 1, 4 * 5  # 4+1+20=25
n_a_out = n_v_out
n_latent_v_cond, n_latent_v_out = 1 * 1 + 1, 1 * 5  # 1+1+5=7
horizon = n_v_cond + n_v_out # 25
pad_before = n_v_cond - 1
pusht_train_dataset = L(PushTImageDataset)(
    # zarr_path="./datasets/pusht/pusht_cchi_v7_replay.zarr",
    zarr_path="/home/geyuan/code/cospred2nvidia/datasets/pusht/pusht_256.zarr",
    # zarr_path="./datasets/pusht/pusht_256_val.zarr",
    max_obs=n_v_cond,
    max_act_out=n_a_out,
)

pusht_val_dataset = L(PushTImageDataset)(
    zarr_path="/home/geyuan/code/cospred2nvidia/datasets/pusht/pusht_orange_random_v2.zarr",
    max_obs=n_v_cond,
    max_act_out=n_a_out,
)


def get_sampler(dataset):
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=3,  # ori:0
    )


pusht_train_dataloader = L(DataLoader)(
    dataset=pusht_train_dataset,
    sampler=L(get_sampler)(dataset=pusht_train_dataset),
    batch_size=4,  # ori: 1
    drop_last=True,
    num_workers=8,
    pin_memory=True,
)

pusht_val_dataloader = L(DataLoader)(
    dataset=pusht_val_dataset,
    sampler=L(get_sampler)(dataset=pusht_val_dataset),
    batch_size=1,
    drop_last=True,
)
