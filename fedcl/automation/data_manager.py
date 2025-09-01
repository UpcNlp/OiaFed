# fedcl/automation/data_manager.py
"""
自动数据管理器

处理真联邦（多机）和伪联邦（本地）环境下的数据分发和管理。
支持IID和Non-IID数据分布，以及数据隐私保护。
"""

import hashlib
import json
import pickle
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from loguru import logger

from .communication import TransparentCommunication, Message


class DataDistributionType(Enum):
    """数据分布类型"""
    IID = "iid"                        # 独立同分布
    NON_IID_LABEL = "non_iid_label"    # 标签不均衡
    NON_IID_FEATURE = "non_iid_feature" # 特征分布不同
    NON_IID_QUANTITY = "non_iid_quantity" # 数据量不均衡
    TEMPORAL = "temporal"              # 时序分布


@dataclass
class DataPartition:
    """数据分区"""
    client_id: str
    indices: List[int]
    labels: List[int]
    size: int
    distribution_info: Dict[str, Any]


@dataclass
class DataConfig:
    """数据配置"""
    distribution_type: DataDistributionType = DataDistributionType.IID
    num_clients: int = 3
    min_samples_per_client: int = 100
    alpha: float = 0.5  # Dirichlet分布参数
    num_classes: int = 10
    seed: int = 42
    validation_split: float = 0.1
    batch_size: int = 32
    shuffle: bool = True


class BaseDataPartitioner(ABC):
    """数据分区器基类"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.logger = logger.bind(component="DataPartitioner")
        np.random.seed(config.seed)
        random.seed(config.seed)
    
    @abstractmethod
    def partition_data(self, dataset: Dataset) -> List[DataPartition]:
        """分区数据"""
        pass
    
    def _get_labels(self, dataset: Dataset) -> np.ndarray:
        """获取数据集标签"""
        if hasattr(dataset, 'targets'):
            return np.array(dataset.targets)
        elif hasattr(dataset, 'labels'):
            return np.array(dataset.labels)
        else:
            # 遍历数据集获取标签
            labels = []
            for i in range(len(dataset)):
                _, label = dataset[i]
                labels.append(label)
            return np.array(labels)


class IIDPartitioner(BaseDataPartitioner):
    """IID数据分区器"""
    
    def partition_data(self, dataset: Dataset) -> List[DataPartition]:
        """IID分区"""
        self.logger.info(f"🎲 开始IID数据分区 - {self.config.num_clients}个客户端")
        
        total_size = len(dataset)
        labels = self._get_labels(dataset)
        
        # 随机打乱索引
        indices = np.random.permutation(total_size)
        
        # 平均分配
        partition_size = total_size // self.config.num_clients
        partitions = []
        
        for i in range(self.config.num_clients):
            start_idx = i * partition_size
            if i == self.config.num_clients - 1:
                end_idx = total_size  # 最后一个客户端获取剩余所有数据
            else:
                end_idx = (i + 1) * partition_size
            
            client_indices = indices[start_idx:end_idx].tolist()
            client_labels = labels[client_indices].tolist()
            
            # 计算标签分布
            unique_labels, counts = np.unique(client_labels, return_counts=True)
            label_distribution = dict(zip(unique_labels.tolist(), counts.tolist()))
            
            partition = DataPartition(
                client_id=f"client_{i}",
                indices=client_indices,
                labels=client_labels,
                size=len(client_indices),
                distribution_info={
                    "type": "iid",
                    "label_distribution": label_distribution,
                    "num_classes": len(unique_labels)
                }
            )
            partitions.append(partition)
        
        self.logger.info(f"✅ IID分区完成 - 平均每客户端: {partition_size}样本")
        return partitions


class NonIIDLabelPartitioner(BaseDataPartitioner):
    """Non-IID标签分区器（基于Dirichlet分布）"""
    
    def partition_data(self, dataset: Dataset) -> List[DataPartition]:
        """Non-IID标签分区"""
        self.logger.info(f"🎯 开始Non-IID标签分区 - α={self.config.alpha}")
        
        labels = self._get_labels(dataset)
        num_classes = len(np.unique(labels))
        
        # 按类别组织数据
        class_indices = {i: np.where(labels == i)[0] for i in range(num_classes)}
        
        # 使用Dirichlet分布为每个客户端分配类别比例
        partitions = []
        
        for client_id in range(self.config.num_clients):
            # 为当前客户端生成类别分布
            proportions = np.random.dirichlet([self.config.alpha] * num_classes)
            
            client_indices = []
            client_labels = []
            
            # 根据比例从每个类别采样
            for class_id, proportion in enumerate(proportions):
                if class_id in class_indices:
                    available_indices = class_indices[class_id]
                    num_samples = int(proportion * self.config.min_samples_per_client)
                    
                    if num_samples > 0 and len(available_indices) > 0:
                        # 随机采样
                        sampled_indices = np.random.choice(
                            available_indices, 
                            size=min(num_samples, len(available_indices)), 
                            replace=False
                        )
                        client_indices.extend(sampled_indices.tolist())
                        client_labels.extend([class_id] * len(sampled_indices))
                        
                        # 从可用索引中移除已使用的
                        class_indices[class_id] = np.setdiff1d(available_indices, sampled_indices)
            
            # 计算标签分布
            unique_labels, counts = np.unique(client_labels, return_counts=True)
            label_distribution = dict(zip(unique_labels.tolist(), counts.tolist()))
            
            partition = DataPartition(
                client_id=f"client_{client_id}",
                indices=client_indices,
                labels=client_labels,
                size=len(client_indices),
                distribution_info={
                    "type": "non_iid_label",
                    "alpha": self.config.alpha,
                    "label_distribution": label_distribution,
                    "num_classes": len(unique_labels),
                    "proportions": proportions.tolist()
                }
            )
            partitions.append(partition)
        
        self.logger.info(f"✅ Non-IID标签分区完成")
        return partitions


class NonIIDQuantityPartitioner(BaseDataPartitioner):
    """Non-IID数量分区器（不同客户端数据量不同）"""
    
    def partition_data(self, dataset: Dataset) -> List[DataPartition]:
        """Non-IID数量分区"""
        self.logger.info(f"📊 开始Non-IID数量分区")
        
        total_size = len(dataset)
        labels = self._get_labels(dataset)
        
        # 生成不均匀的数据量分布
        sizes = np.random.lognormal(mean=np.log(total_size / self.config.num_clients), sigma=0.5, size=self.config.num_clients)
        sizes = sizes / sizes.sum() * total_size  # 归一化到总数据量
        sizes = np.maximum(sizes, self.config.min_samples_per_client)  # 确保最小样本数
        sizes = sizes.astype(int)
        
        # 调整最后一个客户端的大小以匹配总数据量
        sizes[-1] = total_size - sizes[:-1].sum()
        
        indices = np.random.permutation(total_size)
        partitions = []
        current_idx = 0
        
        for i in range(self.config.num_clients):
            size = sizes[i]
            client_indices = indices[current_idx:current_idx + size].tolist()
            client_labels = labels[client_indices].tolist()
            
            # 计算标签分布
            unique_labels, counts = np.unique(client_labels, return_counts=True)
            label_distribution = dict(zip(unique_labels.tolist(), counts.tolist()))
            
            partition = DataPartition(
                client_id=f"client_{i}",
                indices=client_indices,
                labels=client_labels,
                size=len(client_indices),
                distribution_info={
                    "type": "non_iid_quantity",
                    "expected_size": int(total_size / self.config.num_clients),
                    "actual_size": size,
                    "label_distribution": label_distribution,
                    "size_ratio": size / (total_size / self.config.num_clients)
                }
            )
            partitions.append(partition)
            current_idx += size
        
        self.logger.info(f"✅ Non-IID数量分区完成 - 大小范围: {min(sizes)}-{max(sizes)}")
        return partitions


class FederatedDataLoader:
    """联邦数据加载器"""
    
    def __init__(self, dataset: Dataset, partition: DataPartition, config: DataConfig):
        self.dataset = dataset
        self.partition = partition
        self.config = config
        self.logger = logger.bind(component="FederatedDataLoader", client=partition.client_id)
        
        # 创建子数据集
        self.client_dataset = Subset(dataset, partition.indices)
        
        # 分割训练和验证数据
        self.train_loader, self.val_loader = self._create_data_loaders()
    
    def _create_data_loaders(self) -> Tuple[DataLoader, Optional[DataLoader]]:
        """创建数据加载器"""
        dataset_size = len(self.client_dataset)
        
        if self.config.validation_split > 0:
            # 分割训练和验证数据
            val_size = int(dataset_size * self.config.validation_split)
            train_size = dataset_size - val_size
            
            train_dataset, val_dataset = torch.utils.data.random_split(
                self.client_dataset, [train_size, val_size]
            )
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=self.config.shuffle
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False
            )
            
            return train_loader, val_loader
        else:
            train_loader = DataLoader(
                self.client_dataset,
                batch_size=self.config.batch_size,
                shuffle=self.config.shuffle
            )
            return train_loader, None
    
    def get_stats(self) -> Dict[str, Any]:
        """获取数据统计信息"""
        return {
            "client_id": self.partition.client_id,
            "total_samples": self.partition.size,
            "train_batches": len(self.train_loader),
            "val_batches": len(self.val_loader) if self.val_loader else 0,
            "distribution_info": self.partition.distribution_info
        }


class AutoDataManager:
    """
    自动数据管理器
    
    提供联邦学习环境下的自动数据分发和管理功能
    """
    
    def __init__(
        self,
        communication: Optional[TransparentCommunication] = None,
        config: Optional[DataConfig] = None
    ):
        self.communication = communication
        self.config = config or DataConfig()
        self.logger = logger.bind(component="AutoDataManager")
        
        # 数据分区器映射
        self.partitioners = {
            DataDistributionType.IID: IIDPartitioner,
            DataDistributionType.NON_IID_LABEL: NonIIDLabelPartitioner,
            DataDistributionType.NON_IID_QUANTITY: NonIIDQuantityPartitioner,
        }
        
        self.partitions: List[DataPartition] = []
        self.data_loaders: Dict[str, FederatedDataLoader] = {}
    
    def distribute_data(
        self, 
        dataset: Dataset, 
        num_clients: Optional[int] = None,
        distribution_type: Optional[DataDistributionType] = None
    ) -> List[DataPartition]:
        """
        自动分发数据到各客户端
        
        Args:
            dataset: 原始数据集
            num_clients: 客户端数量
            distribution_type: 分布类型
            
        Returns:
            数据分区列表
        """
        # 更新配置
        if num_clients:
            self.config.num_clients = num_clients
        if distribution_type:
            self.config.distribution_type = distribution_type
        
        self.logger.info(f"🎯 开始数据分发 - 类型: {self.config.distribution_type.value}")
        
        # 选择分区器
        partitioner_class = self.partitioners.get(self.config.distribution_type)
        if not partitioner_class:
            raise ValueError(f"不支持的分布类型: {self.config.distribution_type}")
        
        partitioner = partitioner_class(self.config)
        self.partitions = partitioner.partition_data(dataset)
        
        # 打印分区统计信息
        self._log_partition_stats()
        
        return self.partitions
    
    def create_federated_dataloaders(
        self, 
        dataset: Dataset,
        client_ids: Optional[List[str]] = None
    ) -> Dict[str, FederatedDataLoader]:
        """
        创建联邦数据加载器
        
        Args:
            dataset: 原始数据集
            client_ids: 指定的客户端ID列表
            
        Returns:
            客户端数据加载器字典
        """
        if not self.partitions:
            raise ValueError("请先调用 distribute_data 进行数据分发")
        
        self.logger.info(f"📚 创建联邦数据加载器")
        
        target_partitions = self.partitions
        if client_ids:
            target_partitions = [p for p in self.partitions if p.client_id in client_ids]
        
        for partition in target_partitions:
            data_loader = FederatedDataLoader(dataset, partition, self.config)
            self.data_loaders[partition.client_id] = data_loader
        
        self.logger.info(f"✅ 创建了 {len(self.data_loaders)} 个数据加载器")
        return self.data_loaders
    
    def handle_data_heterogeneity(self, datasets: List[Dataset]) -> Dict[str, Any]:
        """
        处理数据异构性
        
        Args:
            datasets: 不同客户端的数据集列表
            
        Returns:
            异构性分析结果
        """
        self.logger.info("🔍 分析数据异构性")
        
        heterogeneity_stats = {
            "size_variance": 0.0,
            "label_distribution_variance": 0.0,
            "feature_similarity": 0.0,
            "recommendations": []
        }
        
        # 分析数据大小异构性
        sizes = [len(partition.indices) for partition in self.partitions]
        size_mean = np.mean(sizes)
        size_variance = np.var(sizes) / (size_mean ** 2)  # 变异系数
        heterogeneity_stats["size_variance"] = float(size_variance)
        
        # 分析标签分布异构性
        label_distributions = []
        for partition in self.partitions:
            dist = partition.distribution_info.get("label_distribution", {})
            total_samples = sum(dist.values()) if dist else 1
            normalized_dist = {k: v/total_samples for k, v in dist.items()}
            label_distributions.append(normalized_dist)
        
        # 计算标签分布的KL散度
        kl_divergences = []
        for i in range(len(label_distributions)):
            for j in range(i+1, len(label_distributions)):
                kl_div = self._calculate_kl_divergence(label_distributions[i], label_distributions[j])
                kl_divergences.append(kl_div)
        
        if kl_divergences:
            heterogeneity_stats["label_distribution_variance"] = float(np.mean(kl_divergences))
        
        # 生成推荐
        recommendations = []
        if size_variance > 0.5:
            recommendations.append("数据大小差异较大，建议使用加权聚合")
        if heterogeneity_stats["label_distribution_variance"] > 1.0:
            recommendations.append("标签分布差异较大，建议使用个性化联邦学习")
        
        heterogeneity_stats["recommendations"] = recommendations
        
        self.logger.info(f"📊 异构性分析完成 - 大小变异: {size_variance:.3f}")
        return heterogeneity_stats
    
    def _calculate_kl_divergence(self, p: Dict[str, float], q: Dict[str, float]) -> float:
        """计算KL散度"""
        all_keys = set(p.keys()) | set(q.keys())
        kl_div = 0.0
        
        for key in all_keys:
            p_val = p.get(key, 1e-10)
            q_val = q.get(key, 1e-10)
            kl_div += p_val * np.log(p_val / q_val)
        
        return kl_div
    
    def _log_partition_stats(self):
        """打印分区统计信息"""
        self.logger.info("📊 数据分区统计:")
        
        for partition in self.partitions:
            dist_info = partition.distribution_info
            self.logger.info(
                f"  {partition.client_id}: {partition.size}样本, "
                f"{dist_info.get('num_classes', 0)}类别"
            )
    
    def get_data_summary(self) -> Dict[str, Any]:
        """获取数据分发摘要"""
        if not self.partitions:
            return {"status": "no_partitions"}
        
        total_samples = sum(p.size for p in self.partitions)
        avg_samples = total_samples / len(self.partitions)
        
        return {
            "num_clients": len(self.partitions),
            "total_samples": total_samples,
            "avg_samples_per_client": avg_samples,
            "distribution_type": self.config.distribution_type.value,
            "partitions": [
                {
                    "client_id": p.client_id,
                    "size": p.size,
                    "distribution_info": p.distribution_info
                }
                for p in self.partitions
            ]
        }
    
    def save_partitions(self, filepath: Union[str, Path]):
        """保存数据分区信息"""
        filepath = Path(filepath)
        
        partition_data = {
            "config": {
                "distribution_type": self.config.distribution_type.value,
                "num_clients": self.config.num_clients,
                "seed": self.config.seed
            },
            "partitions": [
                {
                    "client_id": p.client_id,
                    "indices": p.indices,
                    "labels": p.labels,
                    "size": p.size,
                    "distribution_info": p.distribution_info
                }
                for p in self.partitions
            ]
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(partition_data, f)
        
        self.logger.info(f"💾 数据分区已保存到: {filepath}")
    
    def load_partitions(self, filepath: Union[str, Path]) -> List[DataPartition]:
        """加载数据分区信息"""
        filepath = Path(filepath)
        
        with open(filepath, 'rb') as f:
            partition_data = pickle.load(f)
        
        self.partitions = [
            DataPartition(
                client_id=p_data["client_id"],
                indices=p_data["indices"],
                labels=p_data["labels"],
                size=p_data["size"],
                distribution_info=p_data["distribution_info"]
            )
            for p_data in partition_data["partitions"]
        ]
        
        self.logger.info(f"📂 数据分区已从 {filepath} 加载")
        return self.partitions