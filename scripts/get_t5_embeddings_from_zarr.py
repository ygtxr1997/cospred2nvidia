from typing import Callable, Union, Tuple, List, Dict
import os
import zarr
import argparse
from functools import partial

from tqdm import tqdm
import numpy as np

from cosmos_predict2.auxiliary.text_encoder import CosmosT5TextEncoder
from cosmos_predict2.data.action_conditioned.libero_dataset import (
    ReplayBuffer,
    AutoTokenizer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute T5 embeddings for text prompts")
    parser.add_argument("--dataset_path", type=str, default="datasets/hdvila", help="Root path to the dataset")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum length of the text embedding")
    parser.add_argument(
        "--cache_dir", type=str, default="checkpoints/google-t5/t5-11b", help="Directory to cache the T5 model"
    )
    return parser.parse_args()


def text_to_t5_embedding(
        text: str,
        t5_model: CosmosT5TextEncoder,
        max_length: int = 512,
) -> tuple[np.ndarray, int]:
    """Mock function to convert text to T5 embeddings. Replace with actual model inference."""
    print("[DEBUG] text:", text)

    encoded_text, mask_bool = t5_model.encode_prompts(
        text, max_length=max_length, return_mask=True
    )  # list of np.ndarray in (len, 1024)
    attn_mask = mask_bool.long()
    lengths = attn_mask.sum(dim=1).cpu()  # tensor, (1,)

    encoded_text = encoded_text.cpu().numpy().astype(np.float16)  #（1,512,1024)

    full_embedding = encoded_text[0]  # (512, 1024)
    valid_length = int(lengths[0])  # scalar

    return full_embedding, valid_length


def convert_zarr_cache_with_t5(
        original_cache_path: str,
        t5_cache_path: str,
        text_to_t5_embedding_func: Callable,
        clip_tokenizer,
        force_rebuild: bool = False
):
    """将原始缓存转换为包含 T5 embeddings 的缓存"""

    if os.path.exists(t5_cache_path) and not force_rebuild:
        print(f"T5 cache already exists: {t5_cache_path}")
        return t5_cache_path

    print(f"Converting {original_cache_path} to {t5_cache_path} with T5 embeddings...")

    # 1. 读取原始数据
    with zarr.ZipStore(original_cache_path, mode="r") as src_store:
        src_replay_buffer = ReplayBuffer.copy_from_store(
            src_store=src_store,
            store=zarr.MemoryStore()
        )
    print("[DEBUG] keys: ", list(src_replay_buffer.keys()),
          "\nnumber of episodes:", src_replay_buffer.n_episodes,
          "\nnumber of steps:", src_replay_buffer.n_steps,
          "\naction shape:", src_replay_buffer.data['action'].shape,
          "\nlanguage shape:", src_replay_buffer.data['language'].shape,
          "\nagentview_rgb shape:", src_replay_buffer.data['agentview_rgb'].shape,
          )
    '''
    [DEBUG] keys:  ['action', 'agentview_rgb', 'language'] 
    number of episodes: 500 
    number of steps: 138090 
    action shape: (138090, 10) 
    language shape: (138090, 2, 30) 
    agentview_rgb shape: (138090, 128, 128, 3)
    '''

    # 2. 计算 T5 embeddings
    # 第一遍：收集唯一文本
    unique_texts = {}  # text -> unique_id
    unique_texts_list = []  # List[str], 按ID顺序存储原始文本
    text_embeddings = []  # List[np.ndarray], 每个shape: (512, 1024)
    text_valid_lengths = []  # List[int], 每个文本的有效长度
    step_to_text_id = []  # List[int], 每个step对应的文本ID

    n_steps = src_replay_buffer.n_steps
    for step_idx in tqdm(range(n_steps), desc="Computing T5 embeddings"):
        clip_data = src_replay_buffer['language'][step_idx]  # (2,30)
        clip_token = clip_data[0, :]  # (30,)
        clip_mask = clip_data[1, :]  # (30,)
        text = clip_tokenizer.decode(clip_token, skip_special_tokens=True).strip()

        if text not in unique_texts:
            # 新文本，分配ID并计算embedding
            text_id = len(unique_texts)
            unique_texts[text] = text_id
            unique_texts_list.append(text)

            # 计算T5 embedding（保持完整形状）
            full_embedding, valid_length = text_to_t5_embedding_func(text)
            text_embeddings.append(full_embedding)  # (512, 1024)
            text_valid_lengths.append(valid_length)
        else:
            text_id = unique_texts[text]

        step_to_text_id.append(text_id)

    # 3. 合并所有 T5 embeddings
    print(f"Found {len(unique_texts)} unique texts")

    # 计算存储空间
    n_unique = len(unique_texts)
    embedding_storage = n_unique * 512 * 1024 * 2  # float16
    index_storage = n_steps * 4  # int32
    length_storage = n_unique * 4  # int32
    total_storage = embedding_storage + index_storage + length_storage

    print(f"Storage breakdown:")
    print(f"  - Embeddings: {embedding_storage / (1024 ** 3):.3f} GB")
    print(f"  - Indices: {index_storage / (1024 ** 2):.1f} MB")
    print(f"  - Lengths: {length_storage / 1024:.1f} KB")
    print(f"  - Total: {total_storage / (1024 ** 3):.3f} GB")

    # 4. 创建新的 replay buffer 并添加 T5 embeddings
    max_text_length = 512
    text_str_array = np.array(unique_texts_list, dtype=f'U{max_text_length}')
    with zarr.ZipStore(t5_cache_path, mode="w") as dst_store:
        # 复制原始数据
        dst_replay_buffer = ReplayBuffer.copy_from_store(
            src_store=zarr.ZipStore(original_cache_path, mode="r"),
            store=dst_store
        )

        # 存储去重后的文本embeddings (n_unique, 512, 1024)
        unique_embeddings = np.stack(text_embeddings, axis=0)
        dst_replay_buffer.meta.array(
            name="t5_unique_embeddings",
            data=unique_embeddings,
            chunks=(min(100, n_unique), 512, 1024),
            compressor=ReplayBuffer.resolve_compressor("default"),
            dtype=unique_embeddings.dtype
        )

        # 存储原始文本 (n_unique, max_text_length)
        # 将文本转换为固定长度的字节数组
        text_bytes_array = np.zeros((len(unique_texts), 512), dtype='S1')
        for i, text in enumerate(unique_texts_list):
            text_bytes = text.encode('utf-8')
            text_bytes_array[i, :len(text_bytes)] = list(text_bytes)

        dst_replay_buffer.meta.array(
            name="t5_unique_texts",
            data=text_str_array,
            chunks=(len(unique_texts),),
            compressor=ReplayBuffer.resolve_compressor("default"),
            dtype=f'U{max_text_length}'
        )

        # 存储每个唯一文本的有效长度 (n_unique,)
        dst_replay_buffer.meta.array(
            name="t5_valid_lengths",
            data=np.array(text_valid_lengths, dtype=np.int32),
            chunks=(n_unique,),
            compressor=ReplayBuffer.resolve_compressor("default"),
            dtype=np.int32
        )

        # 存储每个step到唯一文本ID的映射 (n_steps,)
        dst_replay_buffer.data.array(
            name="t5_text_indices",
            data=np.array(step_to_text_id, dtype=np.int32),
            chunks=(min(10000, n_steps),),
            compressor=ReplayBuffer.resolve_compressor("default"),
            dtype=np.int32
        )

    print(f"Successfully created T5 cache: {t5_cache_path}")
    return t5_cache_path





if __name__ == "__main__":
    args = parse_args()

    # Initialize T5
    print("[get_t5_embeddings_from_zarr] Initializing T5 model...")
    encoder = CosmosT5TextEncoder(
        cache_dir=args.cache_dir,
        local_files_only=True
    )
    text_to_t5_embedding_func = partial(
        text_to_t5_embedding,
        t5_model=encoder,
        max_length=args.max_length
    )

    clip_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    # V1
    # convert_zarr_cache_with_t5(
    #     original_cache_path="/home/geyuan/datasets/LIBERO_uva25rss/libero_10_clip.zarr.zip",
    #     t5_cache_path="/home/geyuan/datasets/LIBERO_uva25rss/libero_10_clip_t5xxl.zarr.zip",
    #     text_to_t5_embedding_func=text_to_t5_embedding_func,
    #     clip_tokenizer=clip_tokenizer,
    #     force_rebuild=True,
    # )
    # V2
    convert_zarr_cache_with_t5(
        original_cache_path="/home/geyuan/datasets/LIBERO_uva25rss/libero_10_full_clip.zarr.zip",
        t5_cache_path="/home/geyuan/datasets/LIBERO_uva25rss/libero_10_full_clip_t5xxl.zarr.zip",
        text_to_t5_embedding_func=text_to_t5_embedding_func,
        clip_tokenizer=clip_tokenizer,
        force_rebuild=True,
    )
