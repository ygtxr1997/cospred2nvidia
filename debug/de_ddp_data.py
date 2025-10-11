import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm
from cosmos_predict2.utils.printer import print_batch
from cosmos_predict2.data.action_conditioned.uha_dataset import OxeUhaDataModule
from imaginaire.lazy_config import LazyCall as L


def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup():
    dist.destroy_process_group()


def main():
    local_rank = setup_distributed()
    is_main_process = (local_rank == 0)

    # Dataset configs
    DEBUG_DATASET = "fractal"
    DEBUG_DATASET_MAPPING = {
        "fractal": "fractal",
        "bridge2": "bridge",
    }

    n_v_cond, n_v_out = 4 * 1 + 1, 4 * 5
    n_latent_v_cond, n_latent_v_out = 1 * 1 + 1, 1 * 5
    horizon = n_v_cond + n_v_out

    transforms_conf = dict(
        move_axis=True,
        bytes_to_string=True,
        adjust_type=None,
        add_robot_information=False
    )

    language_encoders_conf = dict(
        model_name=None
    )

    frame_transform_kwargs = dict(
        image_augment_kwargs=dict(
            primary=dict(
                random_resized_crop=dict(scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                random_brightness=(0.1,),
                random_contrast=(0.9, 1.1),
                random_saturation=(0.9, 1.1),
                random_hue=(0.05,),
                augment_order=("random_resized_crop", "random_brightness", "random_contrast", "random_saturation",
                               "random_hue"),
            ),
            secondary=dict(
                random_resized_crop=dict(scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                random_brightness=(0.1,),
                random_contrast=(0.9, 1.1),
                random_saturation=(0.9, 1.1),
                random_hue=(0.05,),
                augment_order=("random_resized_crop", "random_brightness", "random_contrast", "random_saturation",
                               "random_hue"),
            ),
            wrist=dict(
                random_brightness=(0.1,),
                random_contrast=(0.9, 1.1),
                random_saturation=(0.9, 1.1),
                random_hue=(0.05,),
                augment_order=("random_brightness", "random_contrast", "random_saturation", "random_hue"),
            ),
        ),
        resize_size=dict(
            primary=(176, 176),
            secondary=(160, 160),
            wrist=(84, 84),
        ),
        resize_size_future_obs=dict(
            primary=(176, 176),
            secondary=(160, 160),
            wrist=(84, 84),
        ),
        num_parallel_calls=6,
    )

    datasets_conf = dict(
        DATA_NAME=DEBUG_DATASET_MAPPING[DEBUG_DATASET],
        DATA_PATH="/home/geyuan/local_soft/huggingface/v1/",
        load_camera_views=["primary", "secondary", "wrist"],
        load_proprio=True,
        load_language_embeddings=True,
        action_proprio_normalization_type="bounds",
        interleaved_dataset_cfg=dict(
            shuffle_buffer_size=15000,
            balance_weights=True,
            traj_transform_kwargs=dict(
                goal_relabeling_strategy=None,
                goal_relabeling_kwargs=dict(
                    min_bound=20,
                    max_bound=50,
                    frame_diff=3
                ),
                window_size=n_v_cond,
                action_horizon=n_v_out,
                skip_unlabeled=True,
                load_future_frames=True,
            ),
            frame_transform_kwargs=frame_transform_kwargs,
            traj_transform_threads=16,
            traj_read_threads=8,
        )
    )

    uha_datamodule = OxeUhaDataModule(
        transforms=transforms_conf,
        language_encoders=language_encoders_conf,
        datasets=datasets_conf,
        batch_size=20,  # batch_size per GPU
        drop_last=True,
        use_ori_uha_data_collate=True,
        p_camera_drop=0.,
        p_proprio_drop=0.,
        state_t=n_latent_v_cond + n_latent_v_out,
    )

    train_loader = uha_datamodule.train_dataloader()
    train_loader.sampler = DistributedSampler(train_loader.dataset, num_replicas=dist.get_world_size(), rank=local_rank)

    vis_idx = 10000
    for idx, batch in enumerate(tqdm(train_loader)):
        if idx == 0 and is_main_process:
            print_batch(DEBUG_DATASET, batch)
        if idx < vis_idx:
            continue
        if is_main_process:
            print_batch(DEBUG_DATASET, batch)
        break

    cleanup()


if __name__ == "__main__":
    main()
