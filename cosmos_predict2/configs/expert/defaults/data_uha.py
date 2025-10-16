from cosmos_predict2.utils.printer import print_batch
from imaginaire.lazy_config import LazyCall as L
from cosmos_predict2.data.action_conditioned.uha_dataset import OxeUhaDataModule, NoEncoder



transforms_conf = dict(
    move_axis=True,
    bytes_to_string=True,
    adjust_type=None,
    add_robot_information=False
)

language_encoders_conf = dict(
    model_name=None
)

frame_transform_kwargs=dict(
    image_augment_kwargs=dict(
        primary=dict(
            # random_resized_crop=dict(
            #     scale=(0.8, 1.0),
            #     ratio=(0.9, 1.1)
            # ),
            random_brightness=(0.1,),
            random_contrast=(0.9, 1.1),
            random_saturation=(0.9, 1.1),
            random_hue=(0.05,),
            augment_order=(
                # "random_resized_crop",
                "random_brightness",
                "random_contrast",
                "random_saturation",
                "random_hue",
            ),
        ),
        secondary=dict(
            # random_resized_crop=dict(
            #     scale=(0.8, 1.0),
            #     ratio=(0.9, 1.1)
            # ),
            random_brightness=(0.1,),
            random_contrast=(0.9, 1.1),
            random_saturation=(0.9, 1.1),
            random_hue=(0.05,),
            augment_order=(
                # "random_resized_crop",
                "random_brightness",
                "random_contrast",
                "random_saturation",
                "random_hue",
            ),
        ),
        wrist=dict(
            random_brightness=(0.1,),
            random_contrast=(0.9, 1.1),
            random_saturation=(0.9, 1.1),
            random_hue=(0.05,),
            augment_order=(
                "random_brightness",
                "random_contrast",
                "random_saturation",
                "random_hue",
            ),
        ),
    ),
    resize_size=dict(
        primary=(128, 128),
        secondary=(128, 128),  # not used
        wrist=(128, 128),  # all black
    ),
    resize_size_future_obs=dict(
        primary=(128, 128),
        secondary=(128, 128),  # should be same as resize_size
        wrist=(128, 128),
    ),
    num_parallel_calls=24,
)


n_v_cond, n_v_out = 4 * 1 + 1, 4 * 5  # 4+1+20=25
n_a_out = n_v_out
n_latent_v_cond, n_latent_v_out = 1 * 1 + 1, 1 * 5  # 1+1+5=7
horizon = n_v_cond + n_v_out # 25
pad_before = n_v_cond - 1
datasets_conf = dict(
    DATA_NAME="fractal",  # ori: "fractal"
    DATA_PATH="/home/geyuan/local_soft/huggingface/v1/",
    load_camera_views=["primary"],  # ori: ["primary", "secondary", "wrist"],
    load_proprio=True,  # ori: False
    load_language_embeddings=True,  # ori: False
    action_proprio_normalization_type="bounds",
    interleaved_dataset_cfg=dict(
        shuffle_buffer_size=5000,  # ori: 5000
        balance_weights=True,
        traj_transform_kwargs=dict(
            goal_relabeling_strategy=None,
            goal_relabeling_kwargs=dict(
                min_bound=20,
                max_bound=50,
                frame_diff=3
            ),
            window_size=n_v_cond,
            action_horizon=n_a_out,
            skip_unlabeled=True,
            load_future_frames=True, # NOTE: ori: False
        ),
        frame_transform_kwargs=frame_transform_kwargs,
        traj_transform_threads=16,
        traj_read_threads=8,
    )
)

uha_datamodule = L(OxeUhaDataModule)(
    transforms=transforms_conf,
    language_encoders=L(NoEncoder)(**language_encoders_conf),  # will be merged by OmegaConf
    datasets=datasets_conf,
    batch_size=2,  # ori:20
    drop_last=True,
    # CosmosPredict2 specific
    use_ori_uha_data_collate=False,  # False: use modified collate fn in cosmos_predict2
    p_camera_drop=0.,
    p_proprio_drop=0.,
    state_t=n_latent_v_cond + n_latent_v_out,
)


def get_oxe_uha_train_dataloader(
        data_module: OxeUhaDataModule,
        is_train: bool = True,
):
    if is_train:
        return data_module.train_dataloader()
    else:
        return data_module.val_dataloader()


oxe_uha_train_dataloader = L(get_oxe_uha_train_dataloader)(
    data_module=uha_datamodule,
    is_train=True,
)
oxe_uha_val_dataloader = L(get_oxe_uha_train_dataloader)(
    data_module=uha_datamodule,
    is_train=False,
)

'''
bridge2: Dict,keys=dict_keys(['observation', 'task', 'action', 'action_pad_mask', 'future_frames'])
observation: Dict,keys=dict_keys(['image_primary', 'image_secondary', 'image_wrist', 'timestep', 'pad_mask_dict', 'timestep_pad_mask', 'task_completed'])
-image_primary,<class 'torch.Tensor'>,shape=torch.Size([4, 5, 3, 224, 224]),min=0.0000,max=255.0000
-image_secondary,<class 'torch.Tensor'>,shape=torch.Size([4, 5, 3, 224, 224]),min=0.0000,max=241.0000
-image_wrist,<class 'torch.Tensor'>,shape=torch.Size([4, 5, 3, 84, 84]),min=0.0000,max=0.0000
-timestep,<class 'torch.Tensor'>,shape=torch.Size([4, 5]),min=1.0000,max=12.0000
-pad_mask_dict: Dict,keys=dict_keys(['image_primary', 'image_secondary', 'image_wrist', 'timestep'])
--image_primary,<class 'torch.Tensor'>,shape=torch.Size([4, 5]),min=1.0000,max=1.0000
--image_secondary,<class 'torch.Tensor'>,shape=torch.Size([4, 5]),min=1.0000,max=1.0000
--image_wrist,<class 'torch.Tensor'>,shape=torch.Size([4, 5]),min=0.0000,max=0.0000
--timestep,<class 'torch.Tensor'>,shape=torch.Size([4, 5]),min=1.0000,max=1.0000
-timestep_pad_mask,<class 'torch.Tensor'>,shape=torch.Size([4, 5]),min=1.0000,max=1.0000
-task_completed,<class 'torch.Tensor'>,shape=torch.Size([4, 5, 20]),min=0.0000,max=1.0000
task: Dict,keys=dict_keys(['language_instruction', 'language_key', 'pad_mask_dict'])
-language_instruction: List,len=4,elem:<class 'str'>
-language_key,<class 'torch.Tensor'>,shape=torch.Size([4]),min=0.0000,max=0.0000
-pad_mask_dict: Dict,keys=dict_keys(['language_instruction', 'language_key'])
--language_instruction,<class 'torch.Tensor'>,shape=torch.Size([4]),min=1.0000,max=1.0000
--language_key,<class 'torch.Tensor'>,shape=torch.Size([4]),min=1.0000,max=1.0000
action,<class 'torch.Tensor'>,shape=torch.Size([4, 5, 20, 7]),min=-1.0000,max=1.0000
action_pad_mask,<class 'torch.Tensor'>,shape=torch.Size([4, 5, 20, 7]),min=0.0000,max=1.0000
future_frames: Dict,keys=dict_keys(['image_primary', 'image_secondary', 'image_wrist', 'pad_mask_dict', 'timestep_pad_mask'])
-image_primary,<class 'torch.Tensor'>,shape=torch.Size([4, 20, 3, 224, 224]),min=0.0000,max=255.0000
-image_secondary,<class 'torch.Tensor'>,shape=torch.Size([4, 20, 3, 112, 112]),min=0.0000,max=254.0000
-image_wrist,<class 'torch.Tensor'>,shape=torch.Size([4, 20, 3, 112, 112]),min=0.0000,max=0.0000
-pad_mask_dict: Dict,keys=dict_keys(['image_primary', 'image_secondary', 'image_wrist', 'timestep'])
--image_primary,<class 'torch.Tensor'>,shape=torch.Size([4, 20]),min=1.0000,max=1.0000
--image_secondary,<class 'torch.Tensor'>,shape=torch.Size([4, 20]),min=1.0000,max=1.0000
--image_wrist,<class 'torch.Tensor'>,shape=torch.Size([4, 20]),min=0.0000,max=0.0000
--timestep,<class 'torch.Tensor'>,shape=torch.Size([4, 20]),min=1.0000,max=1.0000
-timestep_pad_mask,<class 'torch.Tensor'>,shape=torch.Size([4, 20]),min=0.0000,max=1.0000
language: ['take corn out of bowl sink', 'Place the spatula just behind the egg plant', 'unfold the cloth from top right to bottom left', 'Trying to pic the blue table cloth.']
'''
