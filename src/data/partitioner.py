"""
Partitioner - 数据划分器

支持多种划分策略（IID, Label-based, Dirichlet 等）
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import random
from ..infra import get_module_logger

logger = get_module_logger(__name__)


class Partitioner(ABC):
    """
    数据划分器抽象基类

    职责：
    - 将数据集划分为多个子集

    属性：
    - needs_labels: 是否需要标签信息（子类可以覆盖）
    """

    # 子类可以覆盖此属性
    needs_labels: bool = False

    @abstractmethod
    def partition(
        self,
        dataset_size: int,
        num_clients: int,
        labels: Optional[List[Any]] = None,
    ) -> Dict[int, List[int]]:
        """
        划分数据

        Args:
            dataset_size: 数据集大小
            num_clients: 客户端数量
            labels: 标签列表（某些划分策略需要）

        Returns:
            划分结果 {client_idx: [sample_indices]}
        """
        pass


class IIDPartitioner(Partitioner):
    """
    IID 划分器
    
    随机均匀划分数据
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        初始化
        
        Args:
            seed: 随机种子
        """
        self._seed = seed
    
    def partition(
        self,
        dataset_size: int,
        num_clients: int,
        labels: Optional[List[Any]] = None,
    ) -> Dict[int, List[int]]:
        """IID 划分"""
        if self._seed is not None:
            random.seed(self._seed)
        
        # 打乱索引
        indices = list(range(dataset_size))
        random.shuffle(indices)
        
        # 均匀划分
        result = {}
        chunk_size = dataset_size // num_clients
        remainder = dataset_size % num_clients
        
        start = 0
        for i in range(num_clients):
            # 前 remainder 个客户端多分一个样本
            size = chunk_size + (1 if i < remainder else 0)
            result[i] = indices[start:start + size]
            start += size
        
        return result


class LabelPartitioner(Partitioner):
    """
    标签划分器

    每个客户端只有部分类别的数据。

    实现逻辑：
    1. 循环分配类别给客户端（每个客户端获得 labels_per_client 个类别）
    2. 统计每个类别被多少个客户端需要
    3. 将每个类别的样本均匀分割给需要该类别的所有客户端
    4. 确保没有数据重复
    """

    needs_labels = True  # 需要标签信息

    def __init__(
        self,
        labels_per_client: int = 2,
        seed: Optional[int] = None,
    ):
        """
        初始化

        Args:
            labels_per_client: 每个客户端的标签数量
            seed: 随机种子
        """
        self._labels_per_client = labels_per_client
        self._seed = seed

    def partition(
        self,
        dataset_size: int,
        num_clients: int,
        labels: Optional[List[Any]] = None,
    ) -> Dict[int, List[int]]:
        """按标签划分"""
        if labels is None:
            raise ValueError("LabelPartitioner requires labels")

        if self._seed is not None:
            random.seed(self._seed)

        # 按标签分组
        label_to_indices: Dict[Any, List[int]] = {}
        for idx, label in enumerate(labels):
            if label not in label_to_indices:
                label_to_indices[label] = []
            label_to_indices[label].append(idx)

        unique_labels = list(label_to_indices.keys())
        num_labels = len(unique_labels)

        # 先打乱每个类别的样本（保证随机性）
        for label in unique_labels:
            random.shuffle(label_to_indices[label])

        # 步骤1: 为每个客户端分配标签（确定谁需要哪些类别）
        client_to_labels: Dict[int, List[Any]] = {i: [] for i in range(num_clients)}

        for i in range(num_clients):
            start_label = (i * self._labels_per_client) % num_labels
            for j in range(self._labels_per_client):
                label_idx = (start_label + j) % num_labels
                client_to_labels[i].append(unique_labels[label_idx])

        # 步骤2: 统计每个类别被多少客户端需要
        label_to_clients: Dict[Any, List[int]] = {label: [] for label in unique_labels}
        for client_id, client_labels in client_to_labels.items():
            for label in client_labels:
                label_to_clients[label].append(client_id)

        # 步骤3: 将每个类别的样本分割给需要它的客户端
        result = {i: [] for i in range(num_clients)}

        for label in unique_labels:
            indices = label_to_indices[label]
            clients_need_this_label = label_to_clients[label]
            num_clients_sharing = len(clients_need_this_label)

            if num_clients_sharing == 0:
                continue

            # 将样本尽可能均匀分割
            samples_per_client = len(indices) // num_clients_sharing
            remainder = len(indices) % num_clients_sharing

            start_idx = 0
            for i, client_id in enumerate(clients_need_this_label):
                # 前 remainder 个客户端多分一个样本
                num_samples = samples_per_client + (1 if i < remainder else 0)
                end_idx = start_idx + num_samples
                result[client_id].extend(indices[start_idx:end_idx])
                start_idx = end_idx

        # 打乱每个客户端的数据
        for i in range(num_clients):
            random.shuffle(result[i])

        return result


class DirichletPartitioner(Partitioner):
    """
    Dirichlet 划分器

    使用 Dirichlet 分布划分数据，alpha 越小越不均衡
    """

    needs_labels = True  # 需要标签信息

    def __init__(
        self,
        alpha: float = 0.5,
        seed: Optional[int] = None,
    ):
        """
        初始化
        
        Args:
            alpha: Dirichlet 分布浓度参数
            seed: 随机种子
        """
        self._alpha = alpha
        self._seed = seed
    
    def partition(
        self,
        dataset_size: int,
        num_clients: int,
        labels: Optional[List[Any]] = None,
    ) -> Dict[int, List[int]]:
        """Dirichlet 划分"""
        try:
            import numpy as np
        except ImportError:
            logger.warning("numpy not available, falling back to IID partition")
            return IIDPartitioner(self._seed).partition(dataset_size, num_clients, labels)
        
        if labels is None:
            raise ValueError("DirichletPartitioner requires labels")
        
        if self._seed is not None:
            np.random.seed(self._seed)
        
        # 按标签分组
        label_to_indices: Dict[Any, List[int]] = {}
        for idx, label in enumerate(labels):
            if label not in label_to_indices:
                label_to_indices[label] = []
            label_to_indices[label].append(idx)
        
        unique_labels = list(label_to_indices.keys())
        
        result = {i: [] for i in range(num_clients)}
        
        # 对每个标签，按 Dirichlet 分布分配给各客户端
        for label in unique_labels:
            indices = np.array(label_to_indices[label])
            np.random.shuffle(indices)
            
            # 生成 Dirichlet 分布
            proportions = np.random.dirichlet([self._alpha] * num_clients)
            
            # 按比例分配
            proportions = (proportions * len(indices)).astype(int)
            
            # 处理舍入误差
            diff = len(indices) - proportions.sum()
            for i in range(abs(diff)):
                idx = i % num_clients
                proportions[idx] += 1 if diff > 0 else -1
            
            # 分配样本
            start = 0
            for i, count in enumerate(proportions):
                result[i].extend(indices[start:start + count].tolist())
                start += count
        
        # 打乱每个客户端的数据
        for i in range(num_clients):
            random.shuffle(result[i])
        
        return result


class QuantityPartitioner(Partitioner):
    """
    数量划分器
    
    按指定数量或比例划分
    """
    
    def __init__(
        self,
        quantities: Optional[List[int]] = None,
        ratios: Optional[List[float]] = None,
        seed: Optional[int] = None,
    ):
        """
        初始化
        
        Args:
            quantities: 每个客户端的样本数量
            ratios: 每个客户端的比例
            seed: 随机种子
        """
        self._quantities = quantities
        self._ratios = ratios
        self._seed = seed
    
    def partition(
        self,
        dataset_size: int,
        num_clients: int,
        labels: Optional[List[Any]] = None,
    ) -> Dict[int, List[int]]:
        """按数量划分"""
        if self._seed is not None:
            random.seed(self._seed)
        
        # 打乱索引
        indices = list(range(dataset_size))
        random.shuffle(indices)
        
        # 确定每个客户端的数量
        if self._quantities:
            quantities = self._quantities[:num_clients]
            if len(quantities) < num_clients:
                quantities.extend([0] * (num_clients - len(quantities)))
        elif self._ratios:
            ratios = self._ratios[:num_clients]
            if len(ratios) < num_clients:
                ratios.extend([0.0] * (num_clients - len(ratios)))
            total_ratio = sum(ratios)
            quantities = [int(r / total_ratio * dataset_size) for r in ratios]
        else:
            # 默认均匀分配
            base = dataset_size // num_clients
            quantities = [base] * num_clients

        # 分配
        result = {}
        start = 0
        for i in range(num_clients):
            end = min(start + quantities[i], dataset_size)
            result[i] = indices[start:end]
            start = end

        return result


class DirichletQuantityPartitioner(Partitioner):
    """
    Dirichlet数量划分器

    使用Dirichlet分布决定每个客户端的样本数量
    论文中的Quantity Skew: q ~ Dir(α)，然后按q的比例分配样本
    """

    def __init__(
        self,
        alpha: float = 0.5,
        seed: Optional[int] = None,
    ):
        """
        初始化

        Args:
            alpha: Dirichlet分布浓度参数，越小越不均衡
            seed: 随机种子
        """
        self._alpha = alpha
        self._seed = seed

    def partition(
        self,
        dataset_size: int,
        num_clients: int,
        labels: Optional[List[Any]] = None,
    ) -> Dict[int, List[int]]:
        """使用Dirichlet分布划分样本数量"""
        try:
            import numpy as np
        except ImportError:
            logger.warning("numpy not available, falling back to IID partition")
            return IIDPartitioner(self._seed).partition(dataset_size, num_clients, labels)

        if self._seed is not None:
            np.random.seed(self._seed)
            random.seed(self._seed)

        # 使用Dirichlet分布采样比例: q ~ Dir_N(α)
        proportions = np.random.dirichlet([self._alpha] * num_clients)

        # 计算每个客户端的样本数量
        quantities = (proportions * dataset_size).astype(int)

        # 处理舍入误差，确保总和等于dataset_size
        diff = dataset_size - quantities.sum()
        for i in range(abs(diff)):
            idx = i % num_clients
            quantities[idx] += 1 if diff > 0 else -1

        # 打乱索引
        indices = list(range(dataset_size))
        random.shuffle(indices)

        # 按比例分配样本
        result = {}
        start = 0
        for i, count in enumerate(quantities):
            result[i] = indices[start:start + count]
            start += count

        logger.info(f"DirichletQuantity partition (α={self._alpha}): {[len(result[i]) for i in range(num_clients)]}")

        return result
