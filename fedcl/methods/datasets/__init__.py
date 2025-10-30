"""
联邦数据集模块
fedcl/methods/datasets/__init__.py

提供：
1. 抽象基类：FederatedDataset
2. 数据划分策略：IID, Label Skew, Dirichlet, Shard
3. 常见数据集实现（将在后续添加）
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
]
