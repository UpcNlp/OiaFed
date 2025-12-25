"""
MNIST LeNet模型 - 符合TPAMI 2025论文标准

从 methods/models/mnist_lenet.py 迁移到 src/

基于论文 "Fundamentals and Experimental Analysis of Federated Learning Algorithms:
A Comparative Study on Non-IID Data Silos" (TPAMI 2025) 的TABLE III实验设置
"""
import torch
import torch.nn as nn
from ...registry import model


@model(
    name='mnist_lenet',
    description='MNIST LeNet模型(TPAMI 2025论文标准架构)',
    version='1.0',
    task='classification',
    input_shape=(1, 28, 28),
    output_shape=(10,)
)
class MNISTLeNetModel(nn.Module):
    """
    MNIST LeNet模型 - 符合论文TABLE III使用的标准架构

    Architecture (from TPAMI 2025 paper, Section VI):
    -----------------------------------------------
    - 两个 5×5 卷积层(6和16通道)
    - 两个全连接层(120和84单元)
    - 最后一层为分类层

    Detailed Structure:
    ------------------
    Conv1: 1 → 6 channels, kernel_size=5
    ReLU + MaxPool(2×2)
    Conv2: 6 → 16 channels, kernel_size=5
    ReLU + MaxPool(2×2)
    Flatten: 16×4×4 = 256
    FC1: 256 → 120, ReLU
    FC2: 120 → 84, ReLU
    FC3: 84 → 10 (output layer)

    Parameters: ~61K
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # LeNet-like架构(严格按照论文标准)
        # 卷积层1: 1→6 通道, 5×5卷积核
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)     # 28×28 → 24×24
        self.pool1 = nn.MaxPool2d(2, 2)                 # 24×24 → 12×12

        # 卷积层2: 6→16 通道, 5×5卷积核
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)    # 12×12 → 8×8
        self.pool2 = nn.MaxPool2d(2, 2)                 # 8×8 → 4×4

        # 全连接层 (论文标准: 120 → 84 → 10)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)           # 256 → 120
        self.fc2 = nn.Linear(120, 84)                   # 120 → 84
        self.fc3 = nn.Linear(84, num_classes)           # 84 → 10

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量, shape=(batch_size, 1, 28, 28)

        Returns:
            logits: 分类logits, shape=(batch_size, num_classes)
        """
        # 第一组卷积+池化
        x = self.conv1(x)                               # (B, 1, 28, 28) → (B, 6, 24, 24)
        x = torch.relu(x)
        x = self.pool1(x)                               # (B, 6, 24, 24) → (B, 6, 12, 12)

        # 第二组卷积+池化
        x = self.conv2(x)                               # (B, 6, 12, 12) → (B, 16, 8, 8)
        x = torch.relu(x)
        x = self.pool2(x)                               # (B, 16, 8, 8) → (B, 16, 4, 4)

        # 展平
        x = x.view(-1, 16 * 4 * 4)                      # (B, 16, 4, 4) → (B, 256)

        # 全连接层1
        x = self.fc1(x)                                 # (B, 256) → (B, 120)
        x = torch.relu(x)

        # 全连接层2
        x = self.fc2(x)                                 # (B, 120) → (B, 84)
        x = torch.relu(x)

        # 输出层
        x = self.fc3(x)                                 # (B, 84) → (B, 10)

        return x
