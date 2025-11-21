"""
论文标准CNN模型 - TPAMI 2025
fedcl/methods/models/paper_cnn.py

符合论文 "Fundamentals and Experimental Analysis of Federated Learning Algorithms:
A Comparative Study on Non-IID Data Silos" (TPAMI 2025) Table III实验设置

论文第11页描述（Section VI）：
"For the image datasets, we use a lightweight CNN architecture similar to that
used in foundational FL studies, featuring two 5×5 convolution layers (6 and 16 channels)
followed by two fully connected layers (120 and 84 units)."
"""
import torch
import torch.nn as nn
from typing import Dict, Tuple

from fedcl.api.decorators import model
from fedcl.methods.models.base import FederatedModel


@model(
    name='PaperCNN',
    description='论文标准CNN模型（TPAMI 2025）- 适配所有图像数据集',
    task='classification',
    version='1.0'
)
class PaperCNN(FederatedModel):
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

    支持的数据集：
    - MNIST: 1×28×28 → flatten_size=256
    - FMNIST: 1×28×28 → flatten_size=256
    - CIFAR-10: 3×32×32 → flatten_size=400
    - SVHN: 3×32×32 → flatten_size=400
    - CINIC-10: 3×32×32 → flatten_size=400
    - FED-ISIC2019: 3×32×32 → flatten_size=400
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
            in_channels: 输入通道数（1为灰度图，3为彩色图）
            input_height: 输入图像高度
            input_width: 输入图像宽度
        """
        super().__init__()

        # 设置元数据
        self.set_metadata(
            task_type='classification',
            input_shape=(in_channels, input_height, input_width),
            output_shape=(num_classes,)
        )

        # 卷积层1: in_channels → 6, 5×5卷积核
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2, 2)

        # 卷积层2: 6 → 16, 5×5卷积核
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2, 2)

        # 计算卷积后的特征图尺寸
        # Conv1: H → H-4, Pool1: →(H-4)/2
        # Conv2: (H-4)/2 → (H-4)/2-4, Pool2: →((H-4)/2-4)/2
        conv_output_h = ((input_height - 4) // 2 - 4) // 2
        conv_output_w = ((input_width - 4) // 2 - 4) // 2
        flatten_size = 16 * conv_output_h * conv_output_w

        # 全连接层（论文标准: 120 → 84 → num_classes）
        self.fc1 = nn.Linear(flatten_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

        # 损失函数
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量, shape=(batch_size, in_channels, H, W)

        Returns:
            logits: 分类logits, shape=(batch_size, num_classes)
        """
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

        # 全连接层1
        x = self.fc1(x)
        x = torch.relu(x)

        # 全连接层2
        x = self.fc2(x)
        x = torch.relu(x)

        # 输出层
        x = self.fc3(x)

        return x

    def get_weights_as_dict(self) -> Dict[str, torch.Tensor]:
        """获取模型权重为字典格式"""
        return {k: v.cpu().clone() for k, v in self.state_dict().items()}

    def set_weights_from_dict(self, weights: Dict[str, torch.Tensor], strict: bool = True):
        """从字典设置模型权重"""
        self.load_state_dict(weights, strict=strict)

    def get_param_count(self) -> int:
        """获取参数数量"""
        return sum(p.numel() for p in self.parameters())


# 为不同数据集创建便捷的工厂函数

@model(
    name='CIFAR10_PaperCNN',
    description='CIFAR-10论文标准CNN（3×32×32输入，10分类）',
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
    name='MNIST_PaperCNN',
    description='MNIST论文标准CNN（1×28×28输入，10分类）',
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


@model(
    name='FedISIC2019_PaperCNN',
    description='FED-ISIC2019论文标准CNN（3×32×32输入，8分类）',
    task='classification',
    input_shape=(3, 32, 32),
    output_shape=(8,)
)
class FedISIC2019_PaperCNN(PaperCNN):
    """FED-ISIC2019数据集的论文标准CNN"""
    def __init__(self, num_classes: int = 8):
        super().__init__(
            num_classes=num_classes,
            in_channels=3,
            input_height=32,
            input_width=32
        )
