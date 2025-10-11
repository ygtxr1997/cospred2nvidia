import argparse
import os
import logging
from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfFolder

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


"""
Usage:
export TARGET_HF_DATASET="IPEC-COMMUNITY/bridge_orig_lerobot"  # fractal20220817_data_lerobot, bridge_orig_lerobot
python scripts/download_hf_datasets.py \
    --repo_id "${TARGET_HF_DATASET}" \
    --local_dir "/home/geyuan/datasets/ipec_lerobot/${TARGET_HF_DATASET}" \
    --num_proc 8 \
    --token "${HF_TOKEN}"
"""
def download_hf_dataset(repo_id: str, local_dir: str, num_proc: int, token: str):
    """
    使用 snapshot_download 从 Hugging Face Hub 高效下载整个数据集。

    Args:
        repo_id (str): Hugging Face Hub 上的数据集仓库 ID。
        local_dir (str): 存储数据集的本地目标目录。
        num_proc (int): 用于并行下载的线程数。
        token (str): 用于认证的 Hugging Face 令牌。
    """
    logging.info(f"开始使用 snapshot_download 下载: '{repo_id}'")
    logging.info(f"将保存到目录: '{os.path.abspath(local_dir)}'")
    logging.info(f"使用 {num_proc} 个线程进行下载。")

    if not token:
        # 尝试从缓存中获取 token
        token = HfFolder.get_token()

    if token:
        logging.info("检测到认证令牌，将以认证用户身份下载。")
    else:
        logging.warning("未找到 Hugging Face 令牌。将以匿名用户身份下载，可能很快会遇到速率限制。")
        logging.warning("推荐使用 'huggingface-cli login' 登录或通过 --token 参数传入令牌。")

    try:
        # snapshot_download 专门用于高效下载仓库文件
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",  # 指明这是一个数据集
            local_dir=local_dir,
            local_dir_use_symlinks=False, # 建议设为 False，直接下载文件而非创建符号链接
            token=token,
            max_workers=num_proc, # 使用 max_workers 控制并行度
            resume_download=True, # 支持断点续传
        )

        logging.info("=" * 50)
        logging.info("数据集文件成功下载！")
        logging.info(f"文件已保存至: {os.path.abspath(local_dir)}")
        logging.info("=" * 50)

    except Exception as e:
        logging.error(f"下载过程中发生错误: {e}")
        if "401" in str(e):
            logging.error("发生 401 Client Error，请检查您的令牌是否正确且具有读取权限。")
        elif "429" in str(e):
            logging.error("仍然遇到 429 速率限制错误。请确认您的账户有足够的配额。")


def main():
    parser = argparse.ArgumentParser(description="使用 snapshot_download 从 Hugging Face Hub 高效下载数据集。")
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="要下载的数据集的 Hugging Face Hub 仓库 ID。"
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        required=True,
        help="用于存储数据集的本地目录路径。"
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=8,
        help="用于并行下载的线程数。"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="用于认证的 Hugging Face 访问令牌。如果未提供，将尝试使用 'huggingface-cli login' 保存的令牌。"
    )
    args = parser.parse_args()

    # 确保目标目录存在
    os.makedirs(args.local_dir, exist_ok=True)

    download_hf_dataset(
        repo_id=args.repo_id,
        local_dir=args.local_dir,
        num_proc=args.num_proc,
        token=args.token
    )

if __name__ == "__main__":
    main()
