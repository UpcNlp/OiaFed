"""
联邦学习模型模块
fedcl/methods/models/__init__.py

提供：
1. 可选基类：FederatedModel
2. 辅助函数：参数统计、模型摘要等
3. 常见模型实现
"""

from .base import (
    FederatedModel,
    count_parameters,
    get_model_size_mb,
    print_model_summary,
)

# 自动导入内置模型以触发装饰器注册
from .mnist_cnn import MNISTCNNModel  # noqa: F401
from .mnist_lenet import MNISTLeNetModel  # noqa: F401
from .cifar10_cnn import CNNModel, SimpleCNNModel  # noqa: F401
from .paper_cnn import (  # noqa: F401
    PaperCNN,
    CIFAR10_PaperCNN,
    MNIST_PaperCNN,
    FedISIC2019_PaperCNN
)
from .paper_mlp import (  # noqa: F401
    PaperMLP,
    Adult_PaperMLP,
    FCUBE_PaperMLP
)
from .resnet import ResNet18_CIFAR, ResNet34_CIFAR  # noqa: F401

__all__ = [
    # 基类
    'FederatedModel',

    # 辅助函数
    'count_parameters',
    'get_model_size_mb',
    'print_model_summary',

    # 内置模型
    'MNISTCNNModel',
    'MNISTLeNetModel',
    'CNNModel',
    'SimpleCNNModel',

    # 论文标准模型（TPAMI 2025）
    'PaperCNN',
    'CIFAR10_PaperCNN',
    'MNIST_PaperCNN',
    'FedISIC2019_PaperCNN',
    'PaperMLP',
    'Adult_PaperMLP',
    'FCUBE_PaperMLP',

    # ResNet models
    'ResNet18_CIFAR',
    'ResNet34_CIFAR',
]
