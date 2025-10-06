import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from tqdm import tqdm
import time


class CrossConnectedModel(nn.Module):
    """
    一个具有跨层连接的非对称模型。
    左分支大，右分支小。
    """

    def __init__(self):
        super().__init__()
        # 左边大分支
        self.left_branch_1 = nn.Sequential(
            nn.Linear(5120, 1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.ReLU(),
        )
        self.left_branch_2 = nn.Sequential(
            nn.Linear(1024 + 128, 1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.ReLU(),
        )

        # 右边小分支
        self.right_branch_1 = nn.Sequential(nn.Linear(64, 128), nn.ReLU())
        # 第二层接收自己的前序输入和来自左分支的交互输入
        self.right_branch_2 = nn.Sequential(nn.Linear(128 + 1024, 256), nn.ReLU())

        # 头部
        self.head = nn.Linear(1024 + 256, 10)

    def forward(self, x_left, x_right, freeze_left=False):
        """
        前向传播逻辑。
        freeze_left=True 时，会跳过左分支的反向传播。
        """
        # --- 完整的前向传播始终执行 ---
        # 1. First layer
        out_left_1 = self.left_branch_1(x_left)
        out_right_1 = self.right_branch_1(x_right)

        # 2. Second layer with optional freezing
        if freeze_left:
            # 【关键】将左分支的输出从计算图中分离，以阻止梯度回传
            # 这样在 backward() 时，左分支的梯度计算将被跳过，从而加速
            interaction_tensor = out_left_1.detach()
        else:
            # 正常训练，保留计算图连接
            interaction_tensor = out_left_1

        # --- 交互和合并 ---
        combined_right_input = torch.cat((out_right_1, interaction_tensor), dim=1)
        out_left_2 = self.left_branch_2(combined_right_input)
        out_right_2 = self.right_branch_2(combined_right_input)

        # 使用可能被 detach 的 final_left_output
        final_combined = torch.cat((out_left_2, out_right_2), dim=1)

        return self.head(final_combined)


def run_experiment(freeze_prob):
    """
    运行一次完整的实验，并报告平均每步的耗时。
    """
    print(f"\n--- 运行实验: freeze_prob = {freeze_prob:.2f} ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    model = CrossConnectedModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    batch_size = 20480
    num_classes = 10
    num_steps = 500
    total_duration = 0.0

    iterator = tqdm(range(num_steps), desc=f"Prob={freeze_prob:.2f}")
    for i in iterator:
        # 使用 torch.cuda.Event 进行更精确的计时
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()

        # --- 模拟数据 ---
        x_left = torch.randn(batch_size, 5120, device=device)
        x_right = torch.randn(batch_size, 64, device=device)
        targets = torch.randint(0, num_classes, (batch_size,), device=device)

        # --- 核心训练逻辑 ---
        freeze_this_step = random.random() < freeze_prob

        optimizer.zero_grad()
        outputs = model(x_left, x_right, freeze_left=freeze_this_step)
        loss = criterion(outputs, targets)

        # 当 freeze_left=True 时，反向传播会因为 detach 而跳过左分支的计算
        loss.backward()

        optimizer.step()

        end_event.record()
        torch.cuda.synchronize()  # 等待 GPU 操作完成

        total_duration += start_event.elapsed_time(end_event)  # 记录毫秒

    avg_time_per_step = total_duration / num_steps
    print(f"实验完成。平均每步耗时: {avg_time_per_step:.2f} ms")
    return avg_time_per_step


if __name__ == "__main__":
    # 设置随机种子以保证可复现性
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # 运行不同概率的实验以对比速度
    run_experiment(freeze_prob=1.0)  # 100% 冻结，最快
    run_experiment(freeze_prob=0.0)  # 0% 冻结，最慢