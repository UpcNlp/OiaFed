"""
MNIST CNN 分类模型
fedcl/methods/models/mnist_cnn.py
"""
from typing import Dict
import torch
import torch.nn as nn

from fedcl.api.decorators import model
from fedcl.methods.models.base import FederatedModel


@model(
    name='MNIST_CNN',
    description='MNIST CNN分类模型',
    task='classification',
    input_shape=(1, 28, 28),
    output_shape=(10,)
)
class MNISTCNNModel(FederatedModel):
    """MNIST CNN模型

    一个简单的卷积神经网络，用于MNIST手写数字分类。

    网络结构：
    - Conv2d(1, 32) + ReLU
    - Conv2d(32, 64) + ReLU
    - MaxPool2d(2, 2)
    - Dropout(0.25)
    - Linear(64*14*14, 128) + ReLU
    - Dropout(0.5)
    - Linear(128, 10)
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # 设置元数据
        self.set_metadata(
            task_type='classification',
            input_shape=(1, 28, 28),
            output_shape=(num_classes,)
        )

        # 定义网络结构
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, num_classes)

        # 损失函数
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        """前向传播"""
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = x.view(-1, 64 * 14 * 14)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
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
