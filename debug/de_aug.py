import torch
import cv2
import numpy as np
from pathlib import Path
from torchvision import transforms
import torchvision.io as io

from cosmos_predict2.online.data_handler import VideoAugmentation


def load_video_to_tensor(video_path: str, num_frames: int = None) -> torch.Tensor:
    """
    从 MP4 文件加载视频并转换为 tensor。

    Args:
        video_path: 视频文件路径
        num_frames: 要加载的帧数，None 表示加载所有帧

    Returns:
        video tensor，形状为 (C, T, H, W)，值域为 [0, 1]
    """
    video, _, info = io.read_video(video_path, output_format="TCHW")
    video = video.to(torch.uint8)

    if num_frames is not None and video.shape[1] > num_frames:
        video = video[:, :num_frames]

    return video


def save_video_tensor(video_tensor: torch.Tensor, output_path: str, fps: int = 30) -> None:
    """
    将 tensor 保存为 MP4 视频文件。

    Args:
        video_tensor: 视频 tensor，形状为 (T, C, H, W)，值域为 [0, 1]
        output_path: 输出视频文件路径
        fps: 帧率
    """
    # 转换为 uint8：(T, C, H, W) -> (T, H, W, C)
    video_uint8 = video_tensor
    video_uint8 = video_uint8.permute(0, 2, 3, 1)  # (T, H, W, C)

    io.write_video(output_path, video_uint8, fps=fps)
    print(f"视频已保存到: {output_path}")


def test_video_augmentation(input_video_path: str, output_dir: str = "./output"):
    """
    测试 VideoAugmentation 类。

    Args:
        input_video_path: 输入 MP4 视频路径
        output_dir: 输出目录
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"加载视频: {input_video_path}")
    video = load_video_to_tensor(input_video_path, num_frames=30)  # (T, C, H, W)
    print(f"原始视频形状: {video.shape}")

    # 转换为 (B, T, C, H, W) 格式用于增广
    video_batch = video.unsqueeze(0)  # (1, T, C, H, W)
    print(f"批次视频形状: {video_batch.shape}")

    augmentor = VideoAugmentation(enable_augmentation=True)
    print("应用增广...")
    augmented_video = augmentor.augment_video(video_batch.permute(0, 2, 1, 3, 4))  # (B, C, T, H, W)
    augmented_video = augmented_video.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
    print(f"增广后视频形状: {augmented_video.shape}")

    # 移除批次维度，转换回 (T, C, H, W)
    augmented_video_squeezed = augmented_video.squeeze(0)  # (T, C, H, W)

    save_video_tensor(video, str(Path(output_dir) / "original.mp4"), fps=30)
    save_video_tensor(augmented_video_squeezed, str(Path(output_dir) / "augmented.mp4"), fps=30)

    print(f"\n测试完成！")


if __name__ == "__main__":
    # 示例：替换为您的视频文件路径
    input_video = "./debug/fractal_success.mp4"

    # 检查文件是否存在
    if not Path(input_video).exists():
        print(f"错误：视频文件 '{input_video}' 不存在")
        print("请提供正确的视频文件路径")
    else:
        test_video_augmentation(input_video)
