"""
数据划分策略
fedcl/methods/datasets/partition.py

实现多种数据划分策略，支持IID和Non-IID场景。

参考文献：
- Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification (arXiv 2019)
- Federated Learning on Non-IID Data Silos: An Experimental Study (arXiv 2021)
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
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


# ==================== 病理性标签倾斜（Pathological Label Skew）====================

class PathologicalLabelSkewPartitioner(DataPartitioner):
    """
    病理性标签倾斜划分器 (Pathological Label Distribution Skew)

    这是最严重的Non-IID场景，每个客户端仅拥有极少数类别的数据。
    使用Dirichlet分布控制每个客户端内的标签分布。

    参考文献：
    - Measuring the Effects of Non-Identical Data Distribution (arXiv 2019)

    数学表示：
        pk ~ Dir(α), #C = k
        其中 pk 是客户端k的标签分布，#C是客户端拥有的类别数

    影响：
        - CIFAR-10上，FedAvg准确率从IID的72.59%降至9.64% (当#C=1时)
        - 几乎等同于随机猜测
        - 在所有算法和数据集上普遍导致最严重的性能下降

    参数：
        alpha: Dirichlet分布参数，控制标签分布的集中程度
            - α = 0.1: 极度集中（每个客户端1-2个类别占主导）
            - α = 0.5: 中度集中（默认，论文中的设置）
            - α = 1.0: 轻度集中
        classes_per_client: 每个客户端拥有的最大类别数 (默认None表示不限制)
        min_samples_per_class: 每个类别的最小样本数 (默认10)
    """

    def partition(self,
                  dataset: Dataset,
                  num_clients: int,
                  alpha: float = 0.5,
                  classes_per_client: Optional[int] = None,
                  min_samples_per_class: int = 10,
                  **kwargs) -> Dict[int, List[int]]:
        """
        执行病理性标签倾斜划分

        Args:
            dataset: 数据集
            num_clients: 客户端数量
            alpha: Dirichlet分布参数（默认0.5，论文设置）
            classes_per_client: 每个客户端最多拥有的类别数（None表示不限制）
            min_samples_per_class: 每个类别的最小样本数

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
        client_label_counts = {i: {} for i in range(num_clients)}

        # 为每个类别分配到客户端
        for label in range(num_classes):
            indices = np.array(label_to_indices[label])
            self.rng.shuffle(indices)

            # 使用Dirichlet分布生成分配比例
            proportions = self.rng.dirichlet(np.repeat(alpha, num_clients))

            # 如果指定了classes_per_client，只分配给部分客户端
            if classes_per_client is not None:
                # 为该类别随机选择 num_clients // (num_classes // classes_per_client) 个客户端
                num_clients_for_class = max(1, num_clients * classes_per_client // num_classes)
                selected_clients = self.rng.choice(num_clients, num_clients_for_class, replace=False)

                # 重新分配比例：非选中客户端比例设为0
                new_proportions = np.zeros(num_clients)
                new_proportions[selected_clients] = proportions[selected_clients]
                new_proportions = new_proportions / new_proportions.sum() if new_proportions.sum() > 0 else new_proportions
                proportions = new_proportions

            # 按比例分配样本
            proportions = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
            split_indices = np.split(indices, proportions)

            for client_id in range(num_clients):
                client_data = split_indices[client_id].tolist()
                if len(client_data) >= min_samples_per_class:
                    client_indices[client_id].extend(client_data)
                    client_label_counts[client_id][label] = len(client_data)

        # 打乱每个客户端的数据
        for client_id in client_indices:
            self.rng.shuffle(client_indices[client_id])

        # 打印统计信息
        self._print_partition_stats(client_label_counts, num_classes, alpha, classes_per_client)

        return client_indices

    def _print_partition_stats(self, client_label_counts: Dict[int, Dict[int, int]],
                              num_classes: int, alpha: float, classes_per_client: Optional[int]):
        """打印划分统计信息"""
        print(f"\n=== Pathological Label Skew Partition (α={alpha}) ===")

        classes_per_client_actual = []
        for client_id, label_counts in client_label_counts.items():
            num_labels = len([c for c, count in label_counts.items() if count > 0])
            classes_per_client_actual.append(num_labels)

        print(f"每个客户端的类别数统计:")
        print(f"  平均: {np.mean(classes_per_client_actual):.2f}")
        print(f"  最小: {np.min(classes_per_client_actual)}")
        print(f"  最大: {np.max(classes_per_client_actual)}")
        if classes_per_client:
            print(f"  期望: {classes_per_client}")


# ==================== 特征分布倾斜（Feature Distribution Skew）====================

class NoisyDataset(Dataset):
    """
    添加高斯噪声的数据集包装器

    用于实现特征分布倾斜(Feature Distribution Skew)
    """

    def __init__(self, dataset: Dataset, noise_std: float = 0.1, seed: int = 42):
        """
        Args:
            dataset: 原始数据集
            noise_std: 高斯噪声标准差
            seed: 随机种子
        """
        self.dataset = dataset
        self.noise_std = noise_std
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, label = self.dataset[idx]

        # 添加高斯噪声
        if isinstance(data, torch.Tensor):
            noise = torch.from_numpy(
                self.rng.normal(0, self.noise_std, data.shape).astype(np.float32)
            )
            noisy_data = data + noise
            # 裁剪到有效范围 [0, 1]（假设数据已归一化）
            noisy_data = torch.clamp(noisy_data, 0.0, 1.0)
        else:
            # 处理numpy数组
            noise = self.rng.normal(0, self.noise_std, data.shape).astype(data.dtype)
            noisy_data = np.clip(data + noise, 0.0, 1.0)

        return noisy_data, label


class FeatureSkewPartitioner(DataPartitioner):
    """
    特征分布倾斜划分器 (Feature Distribution Skew)

    对每个客户端的特征添加不同的高斯噪声，模拟特征分布差异。

    参考文献：
    - Measuring the Effects of Non-Identical Data Distribution (arXiv 2019)

    数学表示：
        x̃ ~ Gau(μ, σ)
        其中 x̃ 是添加噪声后的特征，μ=0, σ=0.1

    影响：
        - CIFAR-10上，FedAvg准确率降至64.02%
        - 比病理性标签倾斜温和，但仍有明显影响
        - 模拟不同设备/环境下的数据采集差异

    参数：
        noise_std: 高斯噪声标准差（默认0.1，论文设置）
        different_noise_per_client: 是否为每个客户端使用不同的噪声
    """

    def partition(self,
                  dataset: Dataset,
                  num_clients: int,
                  noise_std: float = 0.1,
                  different_noise_per_client: bool = True,
                  base_partition: str = 'iid',
                  **kwargs) -> Dict[int, Tuple[Subset, float]]:
        """
        执行特征倾斜划分

        Args:
            dataset: 数据集
            num_clients: 客户端数量
            noise_std: 高斯噪声标准差
            different_noise_per_client: 是否为每个客户端使用不同的噪声级别
            base_partition: 基础划分策略（'iid' 或 'dirichlet'）

        Returns:
            Dict[client_id, (subset, noise_level)]: 每个客户端的数据子集和噪声级别
        """
        # 首先使用基础策略划分数据
        if base_partition == 'iid':
            base_partitioner = IIDPartitioner(seed=self.seed)
        elif base_partition == 'dirichlet':
            alpha = kwargs.get('alpha', 0.5)
            base_partitioner = DirichletPartitioner(seed=self.seed)
            indices_dict = base_partitioner.partition(dataset, num_clients, alpha=alpha)
        else:
            base_partitioner = IIDPartitioner(seed=self.seed)

        if base_partition != 'dirichlet':
            indices_dict = base_partitioner.partition(dataset, num_clients)

        # 为每个客户端分配噪声级别
        client_data = {}
        for client_id in range(num_clients):
            if different_noise_per_client:
                # 为每个客户端使用不同的噪声级别
                client_noise_std = noise_std * self.rng.uniform(0.5, 1.5)
            else:
                client_noise_std = noise_std

            # 创建该客户端的子集
            subset = Subset(dataset, indices_dict[client_id])
            client_data[client_id] = (subset, client_noise_std)

        print(f"\n=== Feature Distribution Skew Partition ===")
        print(f"基础噪声标准差: {noise_std}")
        if different_noise_per_client:
            noise_levels = [noise for _, noise in client_data.values()]
            print(f"客户端噪声级别范围: [{min(noise_levels):.3f}, {max(noise_levels):.3f}]")

        return client_data

    def get_client_partition(self,
                            dataset: Dataset,
                            client_id: int,
                            num_clients: int,
                            **kwargs) -> NoisyDataset:
        """
        获取特定客户端的数据分片（带噪声）

        Returns:
            NoisyDataset: 添加了噪声的数据集
        """
        client_data = self.partition(dataset, num_clients, **kwargs)

        if client_id not in client_data:
            raise ValueError(f"Invalid client_id {client_id}")

        subset, noise_std = client_data[client_id]
        return NoisyDataset(subset, noise_std=noise_std, seed=self.seed + client_id)


# ==================== 数量倾斜（Quantity Skew）====================

class QuantitySkewPartitioner(DataPartitioner):
    """
    数量倾斜划分器 (Quantity Skew)

    使用Dirichlet分布控制每个客户端的数据量，但保持标签分布相对均匀。
    这是最温和的Non-IID形式。

    参考文献：
    - Measuring the Effects of Non-Identical Data Distribution (arXiv 2019)

    数学表示：
        q ~ Dir(α)
        其中 q 是客户端的样本数量分布

    影响：
        - FM-NIST上，FedAvg准确率88.80%
        - 与IID的89.27%非常接近，仅略微下降
        - 大多数算法性能仅比IID略低
        - 模拟现实中不同客户端数据量差异

    参数：
        alpha: Dirichlet分布参数（默认0.5）
            - α < 0.1: 极度不均衡（某些客户端数据很少）
            - α = 0.5: 中度不均衡（默认，论文设置）
            - α = 1.0: 轻度不均衡
            - α > 10: 接近均匀分布
        min_samples: 每个客户端的最小样本数
    """

    def partition(self,
                  dataset: Dataset,
                  num_clients: int,
                  alpha: float = 0.5,
                  min_samples: int = 10,
                  **kwargs) -> Dict[int, List[int]]:
        """
        执行数量倾斜划分

        Args:
            dataset: 数据集
            num_clients: 客户端数量
            alpha: Dirichlet分布参数
            min_samples: 每个客户端的最小样本数

        Returns:
            Dict[client_id, indices]: 每个客户端的索引
        """
        # 获取所有索引
        indices = np.arange(len(dataset))
        self.rng.shuffle(indices)

        # 使用Dirichlet分布生成数据量比例
        proportions = self.rng.dirichlet(np.repeat(alpha, num_clients))

        # 确保每个客户端至少有min_samples个样本
        min_proportion = min_samples / len(dataset)
        if np.any(proportions < min_proportion):
            # 调整过小的比例
            proportions = np.maximum(proportions, min_proportion)
            proportions = proportions / proportions.sum()

        # 计算每个客户端的样本数量
        sample_counts = (proportions * len(dataset)).astype(int)

        # 确保总和等于数据集大小
        sample_counts[-1] = len(dataset) - sample_counts[:-1].sum()

        # 分配数据
        client_indices = {}
        start_idx = 0
        for client_id in range(num_clients):
            end_idx = start_idx + sample_counts[client_id]
            client_indices[client_id] = indices[start_idx:end_idx].tolist()
            start_idx = end_idx

        # 打印统计信息
        self._print_partition_stats(sample_counts, alpha)

        return client_indices

    def _print_partition_stats(self, sample_counts: np.ndarray, alpha: float):
        """打印划分统计信息"""
        print(f"\n=== Quantity Skew Partition (α={alpha}) ===")
        print(f"每个客户端的样本数统计:")
        print(f"  平均: {np.mean(sample_counts):.0f}")
        print(f"  最小: {np.min(sample_counts)}")
        print(f"  最大: {np.max(sample_counts)}")
        print(f"  标准差: {np.std(sample_counts):.0f}")
        print(f"  变异系数: {np.std(sample_counts) / np.mean(sample_counts):.2f}")


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
            - 'pathological_label_skew' / 'pathological': 病理性标签倾斜（最严重）
            - 'feature_skew': 特征分布倾斜（添加高斯噪声）
            - 'quantity_skew': 数量倾斜（最温和）
        seed: 随机种子
        **kwargs: 传递给划分器的参数

    Returns:
        DataPartitioner: 划分器实例

    Examples:
        >>> # IID划分
        >>> partitioner = create_partitioner('iid', seed=42)

        >>> # 病理性标签倾斜 (#C=2，每个客户端2个类别)
        >>> partitioner = create_partitioner('pathological', seed=42, alpha=0.5, classes_per_client=2)

        >>> # 特征分布倾斜
        >>> partitioner = create_partitioner('feature_skew', seed=42, noise_std=0.1)

        >>> # 数量倾斜
        >>> partitioner = create_partitioner('quantity_skew', seed=42, alpha=0.5)

        >>> # Dirichlet标签分布倾斜
        >>> partitioner = create_partitioner('dirichlet', seed=42, alpha=0.5)
    """
    strategy = strategy.lower()

    partitioner_map = {
        # 基础划分
        'iid': IIDPartitioner,
        'non_iid_label': LabelSkewPartitioner,
        'label_skew': LabelSkewPartitioner,
        'non_iid_dirichlet': DirichletPartitioner,
        'dirichlet': DirichletPartitioner,
        'shard': ShardPartitioner,

        # 三种主要Non-IID类型（论文分类）
        'pathological_label_skew': PathologicalLabelSkewPartitioner,
        'pathological': PathologicalLabelSkewPartitioner,
        'feature_skew': FeatureSkewPartitioner,
        'feature_distribution_skew': FeatureSkewPartitioner,
        'quantity_skew': QuantitySkewPartitioner,
        'quantity_distribution_skew': QuantitySkewPartitioner,
    }

    if strategy not in partitioner_map:
        raise ValueError(
            f"Unknown partition strategy: {strategy}. "
            f"Available: {list(partitioner_map.keys())}"
        )

    return partitioner_map[strategy](seed=seed)
