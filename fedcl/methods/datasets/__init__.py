"""
联邦数据集模块
fedcl/methods/datasets/__init__.py

提供：
1. 抽象基类：FederatedDataset
2. 数据划分策略：IID, Label Skew, Dirichlet, Shard
3. 常见数据集实现
"""

from .base import FederatedDataset
from .partition import (
    DataPartitioner,
    IIDPartitioner,
    LabelSkewPartitioner,
    DirichletPartitioner,
    ShardPartitioner,
    create_partitioner
)

# 自动导入内置数据集以触发装饰器注册
from .mnist import MNISTFederatedDataset  # noqa: F401
from .fmnist import FMNISTFederatedDataset  # noqa: F401
from .cifar10 import CIFAR10FederatedDataset  # noqa: F401
from .svhn import SVHNFederatedDataset  # noqa: F401
from .cinic10 import CINIC10FederatedDataset  # noqa: F401
from .adult import AdultFederatedDataset  # noqa: F401
from .fedisic2019 import FedISIC2019Dataset  # noqa: F401

__all__ = [
    # 基类
    'FederatedDataset',

    # 划分器
    'DataPartitioner',
    'IIDPartitioner',
    'LabelSkewPartitioner',
    'DirichletPartitioner',
    'ShardPartitioner',
    'create_partitioner',

    # 内置数据集
    'MNISTFederatedDataset',
    'FMNISTFederatedDataset',
    'CIFAR10FederatedDataset',
    'SVHNFederatedDataset',
    'CINIC10FederatedDataset',
    'AdultFederatedDataset',
    'FedISIC2019Dataset',
]
