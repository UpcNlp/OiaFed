"""
论文标准CNN模型 - TPAMI 2025

从 methods/models/paper_cnn.py 迁移到 src/

符合论文 "Fundamentals and Experimental Analysis of Federated Learning Algorithms:
A Comparative Study on Non-IID Data Silos" (TPAMI 2025) Table III实验设置
"""
import torch
import torch.nn as nn
from ...registry import model


@model(
    name='paper_cnn',
    description='论文标准CNN模型(TPAMI 2025) - 适配所有图像数据集',
    version='1.0',
    task='classification'
)
class PaperCNN(nn.Module):
    """
    论文标准CNN模型 - 适配所有图像数据集

    Architecture (TPAMI 2025 Paper, Section VI):
    -------------------------------------------
    - Conv1: in_channels → 6, kernel_size=5
    - ReLU + MaxPool(2×2)
    - Conv2: 6 → 16, kernel_size=5
    - ReLU + MaxPool(2×2)
    - Flatten
    - FC1: flatten_size → 120
    - ReLU
    - FC2: 120 → 84
    - ReLU
    - FC3: 84 → num_classes

    支持的数据集:
    - MNIST: 1×28×28 → flatten_size=256
    - FMNIST: 1×28×28 → flatten_size=256
    - CIFAR-10: 3×32×32 → flatten_size=400
    - SVHN: 3×32×32 → flatten_size=400
    - CINIC-10: 3×32×32 → flatten_size=400
    """

    def __init__(self,
                 num_classes: int = 10,
                 in_channels: int = 3,
                 input_height: int = 32,
                 input_width: int = 32):
        """
        初始化论文标准CNN模型

        Args:
            num_classes: 分类数量
            in_channels: 输入通道数(1为灰度图,3为彩色图)
            input_height: 输入图像高度
            input_width: 输入图像宽度
        """
        super().__init__()

        # 卷积层1: in_channels → 6, 5×5卷积核
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2, 2)

        # 卷积层2: 6 → 16, 5×5卷积核
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2, 2)

        # 计算卷积后的特征图尺寸
        conv_output_h = ((input_height - 4) // 2 - 4) // 2
        conv_output_w = ((input_width - 4) // 2 - 4) // 2
        flatten_size = 16 * conv_output_h * conv_output_w

        # 全连接层(论文标准: 120 → 84 → num_classes)
        self.fc1 = nn.Linear(flatten_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        """前向传播"""
        # 第一组卷积+池化
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool1(x)

        # 第二组卷积+池化
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool2(x)

        # 展平
        x = x.view(x.size(0), -1)

        # 全连接层
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)

        return x


# 为不同数据集创建便捷的工厂类

@model(
    name='cifar10_paper_cnn',
    description='CIFAR-10论文标准CNN(3×32×32输入,10分类)',
    version='1.0',
    task='classification',
    input_shape=(3, 32, 32),
    output_shape=(10,)
)
class CIFAR10_PaperCNN(PaperCNN):
    """CIFAR-10数据集的论文标准CNN"""
    def __init__(self, num_classes: int = 10):
        super().__init__(
            num_classes=num_classes,
            in_channels=3,
            input_height=32,
            input_width=32
        )


@model(
    name='mnist_paper_cnn',
    description='MNIST论文标准CNN(1×28×28输入,10分类)',
    version='1.0',
    task='classification',
    input_shape=(1, 28, 28),
    output_shape=(10,)
)
class MNIST_PaperCNN(PaperCNN):
    """MNIST数据集的论文标准CNN"""
    def __init__(self, num_classes: int = 10):
        super().__init__(
            num_classes=num_classes,
            in_channels=1,
            input_height=28,
            input_width=28
        )
