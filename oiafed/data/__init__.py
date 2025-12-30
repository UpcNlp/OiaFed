"""
联邦学习数据管理
"""

from .provider import DataProvider, InMemoryDataProvider, SubsetDataProvider
from .partitioner import (
    Partitioner,
    IIDPartitioner,
    LabelPartitioner,
    DirichletPartitioner,
    QuantityPartitioner,
)
from .manager import DataManager
from .partitioned_dataset import (
    create_partitioned_dataset,
    create_dataset_with_partition,
)

__all__ = [
    # 数据提供者
    "DataProvider",
    "InMemoryDataProvider",
    "SubsetDataProvider",

    # 划分器
    "Partitioner",
    "IIDPartitioner",
    "LabelPartitioner",
    "DirichletPartitioner",
    "QuantityPartitioner",

    # 数据管理器
    "DataManager",

    # 划分工厂函数
    "create_partitioned_dataset",
    "create_dataset_with_partition",
]
