"""
MNIST LeNet模型 - 符合TPAMI 2025论文标准
fedcl/methods/models/mnist_lenet.py

基于论文 "Fundamentals and Experimental Analysis of Federated Learning Algorithms:
A Comparative Study on Non-IID Data Silos" (TPAMI 2025) 的TABLE III实验设置
"""
import torch
import torch.nn as nn
from typing import Dict

from fedcl.api.decorators import model
from fedcl.methods.models.base import FederatedModel


@model(
    name='MNIST_LeNet',
    description='MNIST LeNet模型（TPAMI 2025论文标准架构）',
    task='classification',
    input_shape=(1, 28, 28),
    output_shape=(10,)
)
class MNISTLeNetModel(FederatedModel):
    """
    MNIST LeNet模型 - 符合论文TABLE III使用的标准架构

    Architecture (from TPAMI 2025 paper, Section VI):
    -----------------------------------------------
    论文第10页描述：使用轻量级CNN架构，类似于LeNet
    - 两个 5×5 卷积层（6和16通道）
    - 两个全连接层（120和84单元）
    - 最后一层为分类层

    Detailed Structure:
    ------------------
    Input: 1×28×28 (MNIST grayscale images)

    Conv1: 1 → 6 channels, kernel_size=5, stride=1
           Output: 6×24×24
    ReLU + MaxPool(2×2)
           Output: 6×12×12

    Conv2: 6 → 16 channels, kernel_size=5, stride=1
           Output: 16×8×8
    ReLU + MaxPool(2×2)
           Output: 16×4×4

    Flatten: 16×4×4 = 256

    FC1: 256 → 120
    ReLU

    FC2: 120 → 84
    ReLU

    FC3: 84 → 10 (output layer)

    Parameters:
    -----------
    - Total params: ~61K (比原来的模型小，训练更快)
    - 无Dropout层（论文标准）
    - 使用ReLU激活函数

    Training Settings (from paper):
    ------------------------------
    - Batch size: 64
    - Learning rate: 0.01
    - Momentum: 0.9
    - Local epochs: 10
    - Communication rounds: 50

    Expected Performance (IID setting):
    ----------------------------------
    - MNIST IID: ~99.1% accuracy
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # 设置元数据
        self.set_metadata(
            task_type='classification',
            input_shape=(1, 28, 28),
            output_shape=(num_classes,)
        )

        # LeNet-like架构（严格按照论文标准）
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

        # 损失函数
        self.criterion = nn.CrossEntropyLoss()

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

    def get_weights_as_dict(self) -> Dict[str, torch.Tensor]:
        """获取模型权重为字典格式"""
        return {k: v.cpu().clone() for k, v in self.state_dict().items()}

    def set_weights_from_dict(self, weights: Dict[str, torch.Tensor], strict: bool = True):
        """从字典设置模型权重"""
        self.load_state_dict(weights, strict=strict)

    def get_param_count(self) -> int:
        """
        获取参数数量

        Returns:
            int: 总参数数量 (~61K)
        """
        return sum(p.numel() for p in self.parameters())

    def get_layer_names(self):
        """获取所有层的名称（用于调试）"""
        return [name for name, _ in self.named_parameters()]
