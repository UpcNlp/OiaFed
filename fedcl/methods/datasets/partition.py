"""
数据划分策略
fedcl/methods/datasets/partition.py

实现多种数据划分策略，支持IID和Non-IID场景。
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, Subset


class DataPartitioner(ABC):
    """
    数据划分器抽象基类

    定义统一的数据划分接口，所有划分策略都需要继承此类。
    """

    def __init__(self, seed: int = 42):
        """
        初始化划分器

        Args:
            seed: 随机种子（保证可重复性）
        """
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    @abstractmethod
    def partition(self,
                  dataset: Dataset,
                  num_clients: int,
                  **kwargs) -> Dict[int, List[int]]:
        """
        划分数据集

        Args:
            dataset: 要划分的数据集
            num_clients: 客户端数量
            **kwargs: 策略特定参数

        Returns:
            Dict[client_id, indices]: 每个客户端的样本索引列表
        """
        pass

    def get_client_partition(self,
                            dataset: Dataset,
                            client_id: int,
                            num_clients: int,
                            **kwargs) -> Subset:
        """
        获取特定客户端的数据分片（确定性）

        Args:
            dataset: 数据集
            client_id: 客户端ID
            num_clients: 总客户端数
            **kwargs: 策略特定参数

        Returns:
            Subset: 该客户端的数据子集
        """
        # 先划分所有客户端
        all_indices = self.partition(dataset, num_clients, **kwargs)

        # 返回指定客户端的数据
        if client_id not in all_indices:
            raise ValueError(f"Invalid client_id {client_id}, expected 0-{num_clients-1}")

        return Subset(dataset, all_indices[client_id])

    def _get_labels(self, dataset: Dataset, indices: Optional[List[int]] = None) -> np.ndarray:
        """
        提取数据集的标签

        Args:
            dataset: 数据集
            indices: 样本索引，None表示全部

        Returns:
            标签数组
        """
        if indices is None:
            indices = list(range(len(dataset)))

        labels = []
        for idx in indices:
            _, label = dataset[idx]
            if isinstance(label, torch.Tensor):
                label = label.item()
            labels.append(label)

        return np.array(labels)


# ==================== IID 划分 ====================

class IIDPartitioner(DataPartitioner):
    """
    IID（独立同分布）划分器

    将数据随机均匀划分给所有客户端，保证每个客户端的数据分布相似。
    """

    def partition(self,
                  dataset: Dataset,
                  num_clients: int,
                  **kwargs) -> Dict[int, List[int]]:
        """
        执行IID划分

        Args:
            dataset: 数据集
            num_clients: 客户端数量

        Returns:
            Dict[client_id, indices]: 每个客户端的索引
        """
        # 随机打乱所有索引
        indices = self.rng.permutation(len(dataset)).tolist()

        # 计算每个客户端的数据量
        split_size = len(dataset) // num_clients

        # 均匀划分
        client_indices = {}
        for i in range(num_clients):
            start = i * split_size
            end = start + split_size if i < num_clients - 1 else len(dataset)
            client_indices[i] = indices[start:end]

        return client_indices


# ==================== Label-based Non-IID 划分 ====================

class LabelSkewPartitioner(DataPartitioner):
    """
    按标签数量的Non-IID划分器

    控制每个客户端持有的类别数量，实现标签偏斜的Non-IID分布。
    """

    def partition(self,
                  dataset: Dataset,
                  num_clients: int,
                  labels_per_client: int = 2,
                  **kwargs) -> Dict[int, List[int]]:
        """
        执行Label Skew划分

        Args:
            dataset: 数据集
            num_clients: 客户端数量
            labels_per_client: 每个客户端持有的类别数量

        Returns:
            Dict[client_id, indices]: 每个客户端的索引
        """
        # 获取所有标签
        labels = self._get_labels(dataset)
        num_classes = len(np.unique(labels))

        if labels_per_client > num_classes:
            raise ValueError(f"labels_per_client ({labels_per_client}) 不能大于总类别数 ({num_classes})")

        # 按类别分组索引
        label_to_indices = {label: [] for label in range(num_classes)}
        for idx, label in enumerate(labels):
            label_to_indices[label].append(idx)

        # 为每个客户端分配类别
        client_indices = {i: [] for i in range(num_clients)}

        for client_id in range(num_clients):
            # 随机选择 labels_per_client 个类别
            selected_labels = self.rng.choice(
                num_classes, labels_per_client, replace=False
            ).tolist()

            # 从选中的类别中随机抽取样本
            for label in selected_labels:
                available_indices = label_to_indices[label]
                # 每个类别分配大约相同数量的样本
                num_samples = len(available_indices) // (num_clients // labels_per_client + 1)
                if num_samples > 0:
                    selected = self.rng.choice(
                        available_indices, min(num_samples, len(available_indices)), replace=False
                    ).tolist()
                    client_indices[client_id].extend(selected)

                    # 从可用索引中移除已分配的
                    label_to_indices[label] = [i for i in available_indices if i not in selected]

        return client_indices


# ==================== Dirichlet Non-IID 划分 ====================

class DirichletPartitioner(DataPartitioner):
    """
    狄利克雷分布Non-IID划分器

    使用Dirichlet分布控制数据的Non-IID程度。
    alpha 越小，Non-IID程度越高。
    """

    def partition(self,
                  dataset: Dataset,
                  num_clients: int,
                  alpha: float = 0.5,
                  **kwargs) -> Dict[int, List[int]]:
        """
        执行Dirichlet划分

        Args:
            dataset: 数据集
            num_clients: 客户端数量
            alpha: Dirichlet分布参数，越小Non-IID程度越高
                  - alpha < 0.1: 极度Non-IID
                  - alpha = 0.5: 中度Non-IID
                  - alpha = 1.0: 轻度Non-IID
                  - alpha > 10: 接近IID

        Returns:
            Dict[client_id, indices]: 每个客户端的索引
        """
        # 获取所有标签
        labels = self._get_labels(dataset)
        num_classes = len(np.unique(labels))

        # 按类别分组索引
        label_to_indices = {label: [] for label in range(num_classes)}
        for idx, label in enumerate(labels):
            label_to_indices[label].append(idx)

        # 初始化客户端数据
        client_indices = {i: [] for i in range(num_clients)}

        # 为每个类别使用Dirichlet分布分配给客户端
        for label in range(num_classes):
            indices = np.array(label_to_indices[label])
            self.rng.shuffle(indices)

            # 使用Dirichlet分布生成分配比例
            proportions = self.rng.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.array([p * (len(indices) < num_clients) for p in proportions])
            proportions = proportions / proportions.sum()

            # 按比例分配样本
            proportions = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
            split_indices = np.split(indices, proportions)

            for client_id in range(num_clients):
                client_indices[client_id].extend(split_indices[client_id].tolist())

        # 打乱每个客户端的数据
        for client_id in client_indices:
            self.rng.shuffle(client_indices[client_id])

        return client_indices


# ==================== Shard 划分（FedAvg论文风格）====================

class ShardPartitioner(DataPartitioner):
    """
    分片划分器（FedAvg论文风格）

    将数据按标签排序后分成若干分片，每个客户端获得几个连续的分片。
    """

    def partition(self,
                  dataset: Dataset,
                  num_clients: int,
                  shards_per_client: int = 2,
                  **kwargs) -> Dict[int, List[int]]:
        """
        执行Shard划分

        Args:
            dataset: 数据集
            num_clients: 客户端数量
            shards_per_client: 每个客户端的分片数量

        Returns:
            Dict[client_id, indices]: 每个客户端的索引
        """
        # 获取所有标签
        labels = self._get_labels(dataset)

        # 按标签排序索引
        sorted_indices = np.argsort(labels)

        # 计算分片数量
        num_shards = num_clients * shards_per_client
        shard_size = len(dataset) // num_shards

        # 创建分片
        shards = []
        for i in range(num_shards):
            start = i * shard_size
            end = start + shard_size if i < num_shards - 1 else len(dataset)
            shards.append(sorted_indices[start:end].tolist())

        # 随机分配分片给客户端
        shard_ids = list(range(num_shards))
        self.rng.shuffle(shard_ids)

        client_indices = {i: [] for i in range(num_clients)}
        for client_id in range(num_clients):
            # 每个客户端获得 shards_per_client 个分片
            start = client_id * shards_per_client
            end = start + shards_per_client
            for shard_id in shard_ids[start:end]:
                client_indices[client_id].extend(shards[shard_id])

        return client_indices


# ==================== 工厂函数 ====================

def create_partitioner(strategy: str, seed: int = 42, **kwargs) -> DataPartitioner:
    """
    创建数据划分器

    Args:
        strategy: 划分策略名称
            - 'iid': IID划分
            - 'non_iid_label' / 'label_skew': 按标签数量Non-IID
            - 'non_iid_dirichlet' / 'dirichlet': Dirichlet分布Non-IID
            - 'shard': 分片划分
        seed: 随机种子
        **kwargs: 传递给划分器的参数

    Returns:
        DataPartitioner: 划分器实例

    Examples:
        >>> partitioner = create_partitioner('iid', seed=42)
        >>> partitioner = create_partitioner('dirichlet', seed=42, alpha=0.5)
    """
    strategy = strategy.lower()

    partitioner_map = {
        'iid': IIDPartitioner,
        'non_iid_label': LabelSkewPartitioner,
        'label_skew': LabelSkewPartitioner,
        'non_iid_dirichlet': DirichletPartitioner,
        'dirichlet': DirichletPartitioner,
        'shard': ShardPartitioner,
    }

    if strategy not in partitioner_map:
        raise ValueError(
            f"Unknown partition strategy: {strategy}. "
            f"Available: {list(partitioner_map.keys())}"
        )

    return partitioner_map[strategy](seed=seed)
