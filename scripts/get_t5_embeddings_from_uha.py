from typing import Callable, Union, Tuple, List, Dict
import os
import zarr
import argparse
from functools import partial
import re

from tqdm import tqdm
import numpy as np

from cosmos_predict2.auxiliary.cosmos_reason1 import CosmosReason1
from cosmos_predict2.auxiliary.text_encoder import CosmosT5TextEncoder
from cosmos_predict2.data.action_conditioned.uha_dataset import OxeUhaDataModule
from cosmos_predict2.configs.expert.defaults.data_uha import uha_datamodule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute T5 embeddings for text prompts")
    parser.add_argument("-d", "--dataset_name", type=str, default="example", help="Dataset mix name")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum length of the text embedding")
    parser.add_argument(
        "--cache_dir", type=str, default="checkpoints/google-t5/t5-11b", help="Directory to cache the T5 model"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    return parser.parse_args()


def text_to_t5_embedding(
        text: str,
        t5_model: CosmosT5TextEncoder,
        max_length: int = 512,
        prompt_refiner_model: CosmosReason1 = None,
        filter_scene_index: bool = False,
) -> tuple[np.ndarray, int]:
    """Mock function to convert text to T5 embeddings. Replace with actual model inference."""
    if filter_scene_index:
        # 删除从开头到包括首个“数字_”位置的所有字符
        text = re.sub(r'^.*?\d+\s', '', text)
        text = text.replace('_', ' ')
    print("[DEBUG] text:", text)

    if prompt_refiner_model is not None:
        print("[DEBUG] Refining prompt...")
        refined = prompt_refiner_model.refine_prompt(
            image_or_video_path="",
            prompt=text,
        )
        print("[DEBUG] Refined prompt:", refined)
        text = refined[0]

    if t5_model is not None:
        encoded_text, mask_bool = t5_model.encode_prompts(
            text, max_length=max_length, return_mask=True
        )  # list of np.ndarray in (len, 1024)
        attn_mask = mask_bool.long()
        lengths = attn_mask.sum(dim=1).cpu()  # tensor, (1,)

        encoded_text = encoded_text.cpu().numpy().astype(np.float16)  #（1,512,1024)

        full_embedding = encoded_text[0]  # (512, 1024)
        valid_length = int(lengths[0])  # scalar
    else:
        # Mock embedding for testing without a model
        full_embedding = np.random.randn(max_length, 1024).astype(np.float16)
        valid_length = 0

    return full_embedding, valid_length


def extract_lang_embeddings_with_t5(
        ori_data_path: str,
        ori_data_name: str,  # mixed dataset name (`fractal`), rather than the directory name (`fractal20220817_data`)
        text_to_t5_embedding_func: Callable,
        save_to_dir: str = None,
        force_rebuild: bool = False
):
    if save_to_dir is None:
        save_to_dir = os.path.join(ori_data_path, "lang_emb_t5xxl", ori_data_name)

    # 创建保存目录
    os.makedirs(save_to_dir, exist_ok=True)
    embeddings_file = os.path.join(save_to_dir, "t5_embeddings.npz")

    if os.path.exists(embeddings_file) and not force_rebuild:
        print(f"T5XXL language embeddings already exists: {embeddings_file}")
        return save_to_dir

    print(f"Extracting T5 embeddings from {ori_data_path} to {save_to_dir} ...")

    # 1. Load dataset and dataloader
    uha_datamodule['datasets']['DATA_PATH'] = ori_data_path
    uha_datamodule['datasets']['DATA_NAME'] = ori_data_name
    uha_datamodule['datasets']['load_camera_views'] = ['primary']  # to speed up
    uha_datamodule['datasets']['interleaved_dataset_cfg'] = {
        'shuffle_buffer_size': 10000,
        'traj_transform_kwargs': {
            'window_size': 1,
            'action_horizon': 1,
        }
    }  # use single-frame to speed up
    uha_datamodule['batch_size'] = 256
    uha_datamodule['drop_last'] = False

    uha_data = OxeUhaDataModule(**uha_datamodule)
    uha_data.prepare_data()
    uha_data.setup()
    train_dataloader = uha_data.train_dataloader()
    val_dataloader = uha_data.val_dataloader()

    print('[DEBUG] dataloader lens:', len(train_dataloader), len(val_dataloader))

    # 2. 计算 T5 embeddings（同时处理train和val）
    map_text_to_embeddings: Dict[str, np.ndarray] = {}
    unique_texts = {}  # text -> unique_id
    unique_texts_list = []  # List[str], 按ID顺序存储原始文本
    text_embeddings = []  # List[np.ndarray], 每个shape: (512, 1024)
    text_valid_lengths = []  # List[int], 每个文本的有效长度

    train_step_to_text_id = []  # List[int], 每个train step对应的文本ID
    val_step_to_text_id = []  # List[int], 每个val step对应的文本ID

    # 处理训练数据
    print("Processing training data...")
    for idx, batch in enumerate(tqdm(train_dataloader, desc="Train")):
        language_instructions = batch['task']['language_instruction']  # List[str]

        for text in language_instructions:
            if text not in unique_texts:
                # 新文本，分配ID并计算embedding
                text_id = len(unique_texts)
                unique_texts[text] = text_id
                unique_texts_list.append(text)

                # 计算T5 embedding（保持完整形状）
                full_embedding, valid_length = text_to_t5_embedding_func(text)

                map_text_to_embeddings[text] = full_embedding
                text_embeddings.append(full_embedding)  # (512, 1024)
                text_valid_lengths.append(valid_length)
            else:
                text_id = unique_texts[text]

            train_step_to_text_id.append(text_id)

        if idx >= len(train_dataloader) - 1:
            break  # 仅处理一个epoch

    # 处理验证数据
    # NOTE: in RLDSIterableDataset, val dataloader is same as train dataloader, so skip val here
    print("Processing validation data...")
    # for idx, batch in enumerate(tqdm(val_dataloader, desc="Val")):
    #     language_instructions = batch['task']['language_instruction']  # List[str]
    #
    #     for text in language_instructions:
    #         if text not in unique_texts:
    #             # 新文本，分配ID并计算embedding
    #             text_id = len(unique_texts)
    #             unique_texts[text] = text_id
    #             unique_texts_list.append(text)
    #
    #             # 计算T5 embedding（保持完整形状）
    #             full_embedding, valid_length = text_to_t5_embedding_func(text)
    #
    #             map_text_to_embeddings[text] = full_embedding
    #             text_embeddings.append(full_embedding)  # (512, 1024)
    #             text_valid_lengths.append(valid_length)
    #         else:
    #             text_id = unique_texts[text]
    #
    #         val_step_to_text_id.append(text_id)

    # 3. 准备保存数据
    n_unique = len(unique_texts)
    n_train_steps = len(train_step_to_text_id)
    n_val_steps = len(val_step_to_text_id)
    total_steps = n_train_steps + n_val_steps

    print(f"Found {n_unique} unique texts from {n_train_steps} train steps and {n_val_steps} val steps")

    # 计算存储空间
    embedding_storage = n_unique * 512 * 1024 * 2  # float16
    index_storage = total_steps * 4  # int32
    length_storage = n_unique * 4  # int32
    total_storage = embedding_storage + index_storage + length_storage

    print(f"Storage breakdown:")
    print(f"  - Embeddings: {embedding_storage / (1024 ** 3):.3f} GB")
    print(f"  - Indices: {index_storage / (1024 ** 2):.1f} MB")
    print(f"  - Lengths: {length_storage / 1024:.1f} KB")
    print(f"  - Total: {total_storage / (1024 ** 3):.3f} GB")

    # 4. 转换为numpy数组
    unique_embeddings = np.stack(text_embeddings, axis=0)  # (n_unique, 512, 1024)
    text_valid_lengths_array = np.array(text_valid_lengths, dtype=np.int32)  # (n_unique,)
    train_step_to_text_id_array = np.array(train_step_to_text_id, dtype=np.int32)  # (n_train_steps,)
    val_step_to_text_id_array = np.array(val_step_to_text_id, dtype=np.int32)  # (n_val_steps,)

    # 5. 保存所有数据到单个npz文件
    np.savez_compressed(
        embeddings_file,
        # 唯一embedding相关
        unique_embeddings=unique_embeddings,
        text_valid_lengths=text_valid_lengths_array,
        unique_texts=np.array(unique_texts_list, dtype=object),

        # step到文本ID的映射
        train_step_to_text_id=train_step_to_text_id_array,
        val_step_to_text_id=val_step_to_text_id_array,

        # 文本到embedding的直接映射（使用pickle序列化）
        text_to_embedding_map=np.array([map_text_to_embeddings], dtype=object)[0],

        # 元数据
        n_unique_texts=np.array([n_unique], dtype=np.int32),
        n_train_steps=np.array([n_train_steps], dtype=np.int32),
        n_val_steps=np.array([n_val_steps], dtype=np.int32)
    )

    print(f"Successfully saved T5 embeddings to: {embeddings_file}")

    # 打印存储信息
    file_size = os.path.getsize(embeddings_file) / (1024 ** 3)
    print(f"File size: {file_size:.3f} GB")

    return save_to_dir


def load_t5_embeddings(save_dir: str):
    """加载保存的T5 embeddings"""
    embeddings_file = os.path.join(save_dir, "t5_embeddings.npz")

    # 加载npz文件
    data = np.load(embeddings_file, allow_pickle=True)

    return {
        # 唯一embedding相关
        'unique_embeddings': data['unique_embeddings'],  # (n_unique, 512, 1024)
        'text_valid_lengths': data['text_valid_lengths'],  # (n_unique,)
        'unique_texts': data['unique_texts'],  # (n_unique,) object array

        # step到文本ID的映射
        'train_step_to_text_id': data['train_step_to_text_id'],  # (n_train_steps,)
        'val_step_to_text_id': data['val_step_to_text_id'],  # (n_val_steps,)

        # 文本到embedding的直接映射
        'text_to_embedding_map': data['text_to_embedding_map'].item(),  # Dict[str, np.ndarray]

        # 元数据
        'n_unique_texts': int(data['n_unique_texts'][0]),
        'n_train_steps': int(data['n_train_steps'][0]),
        'n_val_steps': int(data['n_val_steps'][0])
    }


if __name__ == "__main__":
    args = parse_args()

    if args.debug:
        encoder = None
    else:
        # Initialize T5
        print("[get_t5_embeddings_from_zarr] Initializing T5 model...")
        encoder = CosmosT5TextEncoder(
            cache_dir=args.cache_dir,
            local_files_only=True
        )
        text_to_t5_embedding_func = partial(
            text_to_t5_embedding,
            t5_model=encoder,
            max_length=args.max_length,
            filter_scene_index=False,
        )

    if not args.debug:
        extract_lang_embeddings_with_t5(
            ori_data_path="/home/geyuan/local_soft/huggingface/v1/",
            ori_data_name=args.dataset_name,  # `fractal`, `bridge`, `example`
            text_to_t5_embedding_func=text_to_t5_embedding_func,
            force_rebuild=True,
        )

    # Load and verify
    saved_data = load_t5_embeddings(
        save_dir=f"/home/geyuan/local_soft/huggingface/v1/lang_emb_t5xxl/{args.dataset_name}"
    )
    print("[DEBUG] loaded data keys:", saved_data.keys())
    print("[DEBUG] mapping:", len(saved_data['text_to_embedding_map'].items()),
          list(saved_data['text_to_embedding_map'].keys())[:10],
          list(saved_data['text_to_embedding_map'].values())[0].shape,
          list(saved_data['text_to_embedding_map'].values())[0].min(),
          list(saved_data['text_to_embedding_map'].values())[0].max(),
    )

    for embedding in saved_data['text_to_embedding_map'].values():
        assert embedding.shape == (args.max_length, 1024)
        assert embedding.dtype == np.float16
