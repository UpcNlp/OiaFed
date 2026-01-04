"""
MNIST 数据集模型

包含适用于 MNIST、Fashion-MNIST 等 28x28 灰度图像的模型。

注册名:
- mnist_cnn: 主力 CNN 模型
- mnist_lenet: LeNet 架构 (TPAMI 2025)
- mnist_paper_cnn: 论文标准 CNN (TPAMI 2025)
"""

import torch
import torch.nn as nn

from ...registry.decorators import model


# ==================== 主力 CNN 模型 ====================

@model(
    name='mnist_cnn',
    description='MNIST 主力 CNN 模型',
    task='classification',
    version='1.0',
    input_shape=(1, 28, 28),
    output_shape=(10,)
)
class MNISTCNN(nn.Module):
    """
    MNIST 主力 CNN 模型
    
    简单高效的 CNN，适合 MNIST 和 Fashion-MNIST。
    
    架构:
    - Conv1: 1 → 32, kernel=3
    - Conv2: 32 → 64, kernel=3
    - MaxPool + Dropout
    - FC1: 64*14*14 → 128
    - FC2: 128 → num_classes
    
    输入: (batch, 1, 28, 28)
    输出: (batch, num_classes)
    
    支持 VFL/SplitNN 分割: 通过 features/classifier 属性
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # 卷积部分（用于 VFL 分割）
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )
        
        # 全连接部分（用于 VFL 分割）
        self.classifier = nn.Sequential(
            nn.Linear(64 * 14 * 14, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x


# ==================== LeNet (TPAMI 2025) ====================

@model(
    name='mnist_lenet',
    description='MNIST LeNet 模型 (TPAMI 2025 论文标准)',
    task='classification',
    version='1.0',
    input_shape=(1, 28, 28),
    output_shape=(10,)
)
class MNISTLeNet(nn.Module):
    """
    MNIST LeNet 模型
    
    符合 TPAMI 2025 论文 Table III 标准架构。
    
    架构 (Section VI):
    - Conv1: 1 → 6, kernel=5  (28 → 24 → 12)
    - Conv2: 6 → 16, kernel=5 (12 → 8 → 4)
    - FC1: 256 → 120
    - FC2: 120 → 84
    - FC3: 84 → num_classes
    
    参数量: ~61K
    
    输入: (batch, 1, 28, 28)
    输出: (batch, num_classes)
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # 卷积层
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)     # 28 → 24
        self.pool1 = nn.MaxPool2d(2, 2)                 # 24 → 12
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)    # 12 → 8
        self.pool2 = nn.MaxPool2d(2, 2)                 # 8 → 4

        # 全连接层 (16 * 4 * 4 = 256)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(-1, 256)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ==================== 论文标准 CNN (TPAMI 2025) ====================

@model(
    name='mnist_paper_cnn',
    description='MNIST 论文标准 CNN (TPAMI 2025)',
    task='classification',
    version='1.0',
    input_shape=(1, 28, 28),
    output_shape=(10,)
)
class MNISTPaperCNN(nn.Module):
    """
    MNIST 论文标准 CNN
    
    符合 TPAMI 2025 论文标准架构（与 LeNet 相同结构）。
    
    架构:
    - Conv1: 1 → 6, kernel=5
    - ReLU + MaxPool(2×2)
    - Conv2: 6 → 16, kernel=5
    - ReLU + MaxPool(2×2)
    - FC1: 256 → 120, ReLU
    - FC2: 120 → 84, ReLU
    - FC3: 84 → num_classes
    
    输入: (batch, 1, 28, 28)
    输出: (batch, num_classes)
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # 卷积层
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2, 2)

        # 全连接层
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ==================== 导出 ====================

__all__ = [
    "MNISTCNN",
    "MNISTLeNet",
    "MNISTPaperCNN",
]
