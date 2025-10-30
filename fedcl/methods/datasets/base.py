"""
联邦数据集抽象基类
fedcl/methods/datasets/base.py

提供模式无关的联邦数据集接口，支持：
1. 集中式划分（Memory/Process模式）
2. 分布式划分（Network模式）
3. 统一的数据统计接口
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import torch
from torch.utils.data import Dataset


class FederatedDataset(ABC):
    """
    联邦数据集抽象基类

    设计原则：
    - 模式无关：同时支持集中式和分布式划分
    - 确定性：相同参数保证相同划分结果
    - 灵活性：支持多种划分策略（IID/Non-IID）

    使用场景：
    1. Memory/Process模式：Server端调用 partition() 集中划分
    2. Network模式：Client端调用 get_client_partition() 独立加载
    """

    def __init__(self, root: str, train: bool = True, download: bool = False, **kwargs):
        """
        初始化联邦数据集

        Args:
            root: 数据集根目录
            train: 是否为训练集
            download: 是否自动下载
            **kwargs: 其他参数
        """
        self.root = root
        self.train = train
        self.download = download
        self.kwargs = kwargs

        # 子类需要设置这些属性
        self.dataset: Optional[Dataset] = None
        self.num_classes: Optional[int] = None
        self.input_shape: Optional[tuple] = None

    # ==================== 集中式划分接口（Memory/Process）====================

    def partition(self,
                  num_clients: int,
                  strategy: str = 'iid',
                  **kwargs) -> Dict[int, Dataset]:
        """
        集中式数据划分（Server端调用）

        适用于 Memory/Process 模式，Server端一次性划分所有客户端数据。

        Args:
            num_clients: 客户端数量
            strategy: 划分策略，可选：
                - 'iid': 独立同分布（均匀随机）
                - 'non_iid_label': 按标签数量控制Non-IID
                - 'non_iid_dirichlet': 狄利克雷分布
                - 'shard': 分片划分（类似FedAvg论文）
            **kwargs: 策略特定参数，例如：
                - alpha (float): Dirichlet分布参数
                - num_shards (int): 分片数量
                - labels_per_client (int): 每个客户端的标签数

        Returns:
            Dict[client_id, Dataset]: 每个客户端的数据集

        Examples:
            >>> fed_dataset = MNISTFederated(root='./data', train=True)
            >>> client_datasets = fed_dataset.partition(num_clients=10, strategy='iid')
            >>> print(f"Client 0 has {len(client_datasets[0])} samples")
        """
        if self.dataset is None:
            raise ValueError("数据集未初始化，请在子类中设置 self.dataset")

        from .partition import create_partitioner

        # 创建划分器
        partitioner = create_partitioner(strategy, **kwargs)

        # 执行划分，获取索引字典
        client_indices = partitioner.partition(self.dataset, num_clients, **kwargs)

        # 将索引转换为Dataset对象
        client_datasets = {}
        for client_id, indices in client_indices.items():
            client_datasets[client_id] = torch.utils.data.Subset(self.dataset, indices)

        return client_datasets

    # ==================== 分布式划分接口（Network）====================

    def get_client_partition(self,
                            client_id: int,
                            num_clients: int,
                            strategy: str = 'iid',
                            **kwargs) -> Dataset:
        """
        分布式数据划分（Client端调用）

        适用于 Network 模式，每个客户端独立加载自己的数据分片。
        使用确定性算法保证相同参数下结果一致。

        Args:
            client_id: 客户端ID（0-based）
            num_clients: 总客户端数量
            strategy: 划分策略（同 partition）
            **kwargs: 策略特定参数

        Returns:
            Dataset: 该客户端的数据集

        Examples:
            >>> # Client 0
            >>> fed_dataset = MNISTFederated(root='./data', train=True)
            >>> my_dataset = fed_dataset.get_client_partition(
            ...     client_id=0, num_clients=10, strategy='iid'
            ... )
        """
        if self.dataset is None:
            raise ValueError("数据集未初始化，请在子类中设置 self.dataset")

        from .partition import create_partitioner

        # 创建划分器
        partitioner = create_partitioner(strategy, **kwargs)

        # 获取该客户端的数据分片
        return partitioner.get_client_partition(
            self.dataset, client_id, num_clients, **kwargs
        )

    # ==================== 统计信息接口 ====================

    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取数据集统计信息

        子类必须实现此方法，返回数据集的元数据。

        Returns:
            Dict包含：
            - dataset_name: 数据集名称
            - num_samples: 总样本数
            - num_classes: 类别数量
            - input_shape: 输入形状
            - train: 是否为训练集
            - 其他统计信息
        """
        pass

    def get_class_distribution(self, indices: Optional[List[int]] = None) -> Dict[int, int]:
        """
        获取类别分布

        Args:
            indices: 样本索引列表，None表示全部样本

        Returns:
            Dict[class_id, count]: 每个类别的样本数量
        """
        if self.dataset is None:
            raise ValueError("数据集未初始化")

        if indices is None:
            indices = list(range(len(self.dataset)))

        class_counts = {}
        for idx in indices:
            _, label = self.dataset[idx]
            if isinstance(label, torch.Tensor):
                label = label.item()
            class_counts[label] = class_counts.get(label, 0) + 1

        return class_counts

    # ==================== 辅助方法 ====================

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.dataset) if self.dataset else 0

    def __repr__(self) -> str:
        """字符串表示"""
        return (f"{self.__class__.__name__}("
                f"root='{self.root}', train={self.train}, "
                f"samples={len(self)})")
