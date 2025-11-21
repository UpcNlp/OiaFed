"""
持续学习数据划分策略
fedcl/methods/datasets/continual_partition.py

实现持续学习场景的数据划分策略：
- Task-Incremental Learning (TIL)
- Class-Incremental Learning (CIL)
- Domain-Incremental Learning (DIL)

支持的数据流模式：
- Sequential: 任务顺序执行
- Interleaved: 任务交替出现
- Overlapping: 任务时间重叠
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from .partition import DataPartitioner


class ContinualLearningPartitioner(DataPartitioner):
    """
    持续学习数据划分器基类

    支持将数据集划分为多个任务序列
    """

    def __init__(self,
                 num_tasks: int,
                 scenario: str = 'class_incremental',
                 seed: int = 42):
        """
        初始化持续学习划分器

        Args:
            num_tasks: 任务数量
            scenario: 场景类型 ('task_incremental', 'class_incremental', 'domain_incremental')
            seed: 随机种子
        """
        super().__init__(seed)
        self.num_tasks = num_tasks
        self.scenario = scenario

        if scenario not in ['task_incremental', 'class_incremental', 'domain_incremental']:
            raise ValueError(f"Unknown scenario: {scenario}")

    def partition(self,
                  dataset: Dataset,
                  num_clients: int,
                  **kwargs) -> Dict[int, List[int]]:
        """
        划分数据集（持续学习场景）

        Returns:
            Dict[client_id, indices]: 每个客户端的样本索引
        """
        # 先按任务划分
        task_indices = self.partition_tasks(dataset, **kwargs)

        # 再按客户端划分
        client_task_indices = self.partition_clients(
            dataset, num_clients, task_indices, **kwargs
        )

        return client_task_indices

    def partition_tasks(self,
                       dataset: Dataset,
                       **kwargs) -> Dict[int, List[int]]:
        """
        将数据集划分为多个任务

        Args:
            dataset: 数据集
            **kwargs: 任务划分参数

        Returns:
            Dict[task_id, indices]: 每个任务的样本索引
        """
        raise NotImplementedError("Subclass must implement partition_tasks")

    def partition_clients(self,
                         dataset: Dataset,
                         num_clients: int,
                         task_indices: Dict[int, List[int]],
                         **kwargs) -> Dict[int, Dict[int, List[int]]]:
        """
        将任务数据划分给多个客户端

        Args:
            dataset: 数据集
            num_clients: 客户端数量
            task_indices: 任务划分结果
            **kwargs: 客户端划分参数

        Returns:
            Dict[client_id, Dict[task_id, indices]]: 每个客户端的每个任务的索引
        """
        client_task_indices = {
            client_id: {} for client_id in range(num_clients)
        }

        # 对每个任务，划分给所有客户端
        for task_id, indices in task_indices.items():
            # 使用IID或Non-IID策略划分
            partition_type = kwargs.get('partition_type', 'iid')

            if partition_type == 'iid':
                client_indices = self._partition_iid(indices, num_clients)
            elif partition_type == 'label_skew':
                # 需要标签信息
                labels = self._get_labels(dataset, indices)
                alpha = kwargs.get('alpha', 0.5)
                client_indices = self._partition_dirichlet(indices, labels, num_clients, alpha)
            else:
                raise ValueError(f"Unknown partition_type: {partition_type}")

            # 分配给各个客户端
            for client_id, client_task_indices_list in client_indices.items():
                client_task_indices[client_id][task_id] = client_task_indices_list

        return client_task_indices

    def _partition_iid(self,
                      indices: List[int],
                      num_clients: int) -> Dict[int, List[int]]:
        """
        IID划分

        Args:
            indices: 要划分的索引
            num_clients: 客户端数量

        Returns:
            Dict[client_id, indices]
        """
        # 随机打乱
        shuffled_indices = self.rng.permutation(indices).tolist()

        # 均匀划分
        split_size = len(shuffled_indices) // num_clients
        client_indices = {}

        for client_id in range(num_clients):
            start = client_id * split_size
            end = start + split_size if client_id < num_clients - 1 else len(shuffled_indices)
            client_indices[client_id] = shuffled_indices[start:end]

        return client_indices

    def _partition_dirichlet(self,
                           indices: List[int],
                           labels: np.ndarray,
                           num_clients: int,
                           alpha: float) -> Dict[int, List[int]]:
        """
        使用Dirichlet分布划分（Non-IID）

        Args:
            indices: 要划分的索引
            labels: 对应的标签
            num_clients: 客户端数量
            alpha: Dirichlet参数（越小越Non-IID）

        Returns:
            Dict[client_id, indices]
        """
        num_classes = len(np.unique(labels))
        client_indices = {i: [] for i in range(num_clients)}

        # 对每个类别
        for class_id in range(num_classes):
            # 找到该类别的所有样本
            class_mask = (labels == class_id)
            class_indices = [indices[i] for i in range(len(indices)) if class_mask[i]]

            # Dirichlet分布生成客户端比例
            proportions = self.rng.dirichlet([alpha] * num_clients)
            proportions = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]

            # 按比例划分
            split_indices = np.split(
                self.rng.permutation(class_indices),
                proportions
            )

            # 分配给各客户端
            for client_id, client_class_indices in enumerate(split_indices):
                client_indices[client_id].extend(client_class_indices.tolist())

        # 打乱每个客户端的样本
        for client_id in client_indices:
            self.rng.shuffle(client_indices[client_id])

        return client_indices


class ClassIncrementalPartitioner(ContinualLearningPartitioner):
    """
    类增量学习（Class-Incremental Learning）数据划分器

    将数据集按类别划分为多个任务，每个任务包含不同的类别
    """

    def __init__(self,
                 num_tasks: int,
                 classes_per_task: Optional[int] = None,
                 seed: int = 42):
        """
        初始化类增量学习划分器

        Args:
            num_tasks: 任务数量
            classes_per_task: 每个任务的类别数（None表示自动均分）
            seed: 随机种子
        """
        super().__init__(num_tasks, 'class_incremental', seed)
        self.classes_per_task = classes_per_task

    def partition_tasks(self,
                       dataset: Dataset,
                       **kwargs) -> Dict[int, List[int]]:
        """
        按类别划分任务

        Args:
            dataset: 数据集
            **kwargs: 其他参数

        Returns:
            Dict[task_id, indices]
        """
        # 获取所有标签
        all_labels = self._get_labels(dataset)
        num_classes = len(np.unique(all_labels))

        # 计算每个任务的类别数
        if self.classes_per_task is None:
            self.classes_per_task = num_classes // self.num_tasks

        # 验证参数
        if self.classes_per_task * self.num_tasks > num_classes:
            raise ValueError(
                f"classes_per_task ({self.classes_per_task}) * num_tasks ({self.num_tasks}) "
                f"> num_classes ({num_classes})"
            )

        # 随机打乱类别顺序（可选）
        shuffle_classes = kwargs.get('shuffle_classes', False)
        class_order = list(range(num_classes))
        if shuffle_classes:
            self.rng.shuffle(class_order)

        # 按任务分配类别
        task_indices = {}
        for task_id in range(self.num_tasks):
            # 确定该任务的类别
            start_class = task_id * self.classes_per_task
            end_class = min(start_class + self.classes_per_task, num_classes)
            task_classes = class_order[start_class:end_class]

            # 收集这些类别的所有样本
            task_indices[task_id] = [
                idx for idx in range(len(dataset))
                if all_labels[idx] in task_classes
            ]

        return task_indices


class TaskIncrementalPartitioner(ContinualLearningPartitioner):
    """
    任务增量学习（Task-Incremental Learning）数据划分器

    每个任务可以包含所有类别，但在不同的数据分布或环境下
    """

    def __init__(self,
                 num_tasks: int,
                 seed: int = 42):
        super().__init__(num_tasks, 'task_incremental', seed)

    def partition_tasks(self,
                       dataset: Dataset,
                       **kwargs) -> Dict[int, List[int]]:
        """
        按时间/批次划分任务

        Args:
            dataset: 数据集
            **kwargs: 其他参数

        Returns:
            Dict[task_id, indices]
        """
        total_samples = len(dataset)

        # 简单均分
        task_size = total_samples // self.num_tasks
        task_indices = {}

        all_indices = list(range(total_samples))
        self.rng.shuffle(all_indices)

        for task_id in range(self.num_tasks):
            start = task_id * task_size
            end = start + task_size if task_id < self.num_tasks - 1 else total_samples
            task_indices[task_id] = all_indices[start:end]

        return task_indices


class DomainIncrementalPartitioner(ContinualLearningPartitioner):
    """
    域增量学习（Domain-Incremental Learning）数据划分器

    假设数据集包含多个域（如不同的数据增强、风格等）
    每个任务对应一个域
    """

    def __init__(self,
                 num_tasks: int,
                 seed: int = 42):
        super().__init__(num_tasks, 'domain_incremental', seed)

    def partition_tasks(self,
                       dataset: Dataset,
                       domain_ids: Optional[List[int]] = None,
                       **kwargs) -> Dict[int, List[int]]:
        """
        按域划分任务

        Args:
            dataset: 数据集
            domain_ids: 每个样本的域ID（如果数据集不提供，需要外部指定）
            **kwargs: 其他参数

        Returns:
            Dict[task_id, indices]
        """
        if domain_ids is None:
            # 如果数据集有domain属性
            if hasattr(dataset, 'domain_ids'):
                domain_ids = dataset.domain_ids
            else:
                # 否则，简单按顺序划分（假设数据按域排序）
                total_samples = len(dataset)
                task_size = total_samples // self.num_tasks
                task_indices = {}

                for task_id in range(self.num_tasks):
                    start = task_id * task_size
                    end = start + task_size if task_id < self.num_tasks - 1 else total_samples
                    task_indices[task_id] = list(range(start, end))

                return task_indices

        # 根据domain_ids划分
        domain_ids = np.array(domain_ids)
        unique_domains = np.unique(domain_ids)

        if len(unique_domains) != self.num_tasks:
            raise ValueError(
                f"Number of unique domains ({len(unique_domains)}) "
                f"!= num_tasks ({self.num_tasks})"
            )

        task_indices = {}
        for task_id, domain_id in enumerate(unique_domains):
            task_indices[task_id] = np.where(domain_ids == domain_id)[0].tolist()

        return task_indices


class ReplayBufferPartitioner:
    """
    重放缓冲区（Replay Buffer）管理器

    用于基于重放的持续学习方法（如经验回放）
    """

    def __init__(self,
                 buffer_size: int,
                 sampling_strategy: str = 'random',
                 seed: int = 42):
        """
        初始化重放缓冲区

        Args:
            buffer_size: 缓冲区大小
            sampling_strategy: 采样策略 ('random', 'reservoir', 'balanced')
            seed: 随机种子
        """
        self.buffer_size = buffer_size
        self.sampling_strategy = sampling_strategy
        self.rng = np.random.RandomState(seed)

        self.buffer_indices = []  # 缓冲区中的样本索引
        self.buffer_labels = []   # 对应的标签（用于balanced采样）

    def update_buffer(self,
                     new_indices: List[int],
                     new_labels: Optional[List[int]] = None):
        """
        更新缓冲区

        Args:
            new_indices: 新任务的样本索引
            new_labels: 新任务的样本标签
        """
        if self.sampling_strategy == 'random':
            # 随机采样
            if len(new_indices) > self.buffer_size:
                sampled = self.rng.choice(
                    new_indices,
                    size=self.buffer_size,
                    replace=False
                )
                self.buffer_indices = sampled.tolist()
                if new_labels is not None:
                    self.buffer_labels = [
                        new_labels[new_indices.index(idx)]
                        for idx in sampled
                    ]
            else:
                self.buffer_indices = new_indices
                self.buffer_labels = new_labels if new_labels is not None else []

        elif self.sampling_strategy == 'reservoir':
            # Reservoir采样（适合数据流）
            for idx in new_indices:
                if len(self.buffer_indices) < self.buffer_size:
                    self.buffer_indices.append(idx)
                else:
                    # 随机替换
                    j = self.rng.randint(0, len(self.buffer_indices))
                    if j < self.buffer_size:
                        self.buffer_indices[j] = idx

        elif self.sampling_strategy == 'balanced':
            # 类别平衡采样
            if new_labels is None:
                raise ValueError("Balanced sampling requires labels")

            # 合并旧缓冲区和新数据
            all_indices = self.buffer_indices + new_indices
            all_labels = self.buffer_labels + new_labels

            # 按类别分组
            class_indices = {}
            for idx, label in zip(all_indices, all_labels):
                if label not in class_indices:
                    class_indices[label] = []
                class_indices[label].append(idx)

            # 每个类别均匀采样
            num_classes = len(class_indices)
            samples_per_class = self.buffer_size // num_classes

            self.buffer_indices = []
            self.buffer_labels = []

            for class_id, indices in class_indices.items():
                if len(indices) > samples_per_class:
                    sampled = self.rng.choice(
                        indices,
                        size=samples_per_class,
                        replace=False
                    )
                else:
                    sampled = indices

                self.buffer_indices.extend(sampled)
                self.buffer_labels.extend([class_id] * len(sampled))

    def get_buffer(self) -> Tuple[List[int], List[int]]:
        """
        获取当前缓冲区

        Returns:
            (indices, labels)
        """
        return self.buffer_indices, self.buffer_labels

    def sample_from_buffer(self, num_samples: int) -> List[int]:
        """
        从缓冲区采样

        Args:
            num_samples: 采样数量

        Returns:
            采样的索引
        """
        if num_samples > len(self.buffer_indices):
            return self.buffer_indices

        sampled = self.rng.choice(
            self.buffer_indices,
            size=num_samples,
            replace=False
        )
        return sampled.tolist()


def create_continual_partitioner(scenario: str,
                                 num_tasks: int,
                                 **kwargs) -> ContinualLearningPartitioner:
    """
    工厂函数：创建持续学习划分器

    Args:
        scenario: 场景类型
        num_tasks: 任务数量
        **kwargs: 其他参数

    Returns:
        对应的划分器实例
    """
    if scenario == 'class_incremental':
        return ClassIncrementalPartitioner(
            num_tasks=num_tasks,
            classes_per_task=kwargs.get('classes_per_task'),
            seed=kwargs.get('seed', 42)
        )
    elif scenario == 'task_incremental':
        return TaskIncrementalPartitioner(
            num_tasks=num_tasks,
            seed=kwargs.get('seed', 42)
        )
    elif scenario == 'domain_incremental':
        return DomainIncrementalPartitioner(
            num_tasks=num_tasks,
            seed=kwargs.get('seed', 42)
        )
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
