# fedcl/data/split_strategy.py
"""
数据分割策略模块

提供不同的联邦学习数据分割策略，包括IID和Non-IID分割方法。
支持多种数据分割算法和配置选项。
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import random
import pickle
from collections import defaultdict, Counter

from loguru import logger

from ..config.config_manager import DictConfig
from ..config.exceptions import ConfigValidationError
from .dataset import Dataset
from ..exceptions import FedCLError


class SplitStrategyError(FedCLError):
    """数据分割策略错误"""
    pass


class DataSplitValidationError(SplitStrategyError):
    """数据分割验证错误"""
    pass


@dataclass
class SplitStatistics:
    """分割统计信息"""
    total_samples: int
    num_clients: int
    samples_per_client: Dict[str, int]
    class_distribution: Dict[str, Dict[str, int]]
    iid_score: float  # 0-1, 1表示完全IID
    balance_score: float  # 0-1, 1表示完全平衡
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'total_samples': self.total_samples,
            'num_客户端': self.num_clients,
            'samples_per_client': self.samples_per_client,
            'class_distribution': self.class_distribution,
            'iid_score': self.iid_score,
            'balance_score': self.balance_score
        }


class SplitStrategy(ABC):
    """
    数据分割策略抽象基类
    
    定义联邦学习数据分割的标准接口，所有具体分割策略都应继承此类。
    """
    
    def __init__(self, config: DictConfig):
        """
        初始化分割策略
        
        Args:
            config: 分割配置
        """
        self.config = config
        self.random_seed = config.get('random_seed', 42)
        self.min_samples_per_client = config.get('min_samples_per_client', 1)
        
        # 设置随机种子以保证可重现性
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        logger.debug(f"Initialized {self.__class__.__name__} with seed {self.random_seed}")
    
    @abstractmethod
    def split_data(self, dataset: Dataset, num_clients: int) -> Dict[str, Dataset]:
        """
        分割数据集
        
        Args:
            dataset: 要分割的数据集
            num_clients: 客户端数量
            
        Returns:
            客户端ID到数据集的映射
            
        Raises:
            SplitStrategyError: 分割失败
        """
        pass
    
    @abstractmethod
    def validate_split(self, split_data: Dict[str, Dataset]) -> bool:
        """
        验证分割结果
        
        Args:
            split_data: 分割后的数据
            
        Returns:
            验证是否通过
        """
        pass
    
    def get_split_statistics(self, split_data: Dict[str, Dataset]) -> SplitStatistics:
        """
        获取分割统计信息
        
        Args:
            split_data: 分割后的数据
            
        Returns:
            分割统计信息
        """
        try:
            total_samples = sum(len(dataset) for dataset in split_data.values())
            num_clients = len(split_data)
            
            # 计算每个客户端的样本数
            samples_per_client = {
                client_id: len(dataset) 
                for client_id, dataset in split_data.items()
            }
            
            # 计算类别分布
            class_distribution = {}
            for client_id, dataset in split_data.items():
                labels = [dataset[i][1] for i in range(len(dataset))]
                class_distribution[client_id] = dict(Counter(labels))
            
            # 计算IID得分（类别分布的相似性）
            iid_score = self._calculate_iid_score(class_distribution)
            
            # 计算平衡得分（样本数的平衡性）
            balance_score = self._calculate_balance_score(samples_per_client)
            
            stats = SplitStatistics(
                total_samples=total_samples,
                num_clients=num_clients,
                samples_per_client=samples_per_client,
                class_distribution=class_distribution,
                iid_score=iid_score,
                balance_score=balance_score
            )
            
            logger.debug(f"Split statistics: IID={iid_score:.3f}, Balance={balance_score:.3f}")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to calculate split statistics: {e}")
            raise SplitStrategyError(f"Failed to calculate statistics: {str(e)}") from e
    
    def visualize_split(self, split_data: Dict[str, Dataset], 
                       save_path: Optional[Path] = None) -> None:
        """
        可视化分割结果
        
        Args:
            split_data: 分割后的数据
            save_path: 保存路径（可选）
        """
        try:
            import matplotlib.pyplot as plt
            
            stats = self.get_split_statistics(split_data)
            
            # 创建子图
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. 样本数分布
            clients = list(stats.samples_per_client.keys())
            samples = list(stats.samples_per_client.values())
            ax1.bar(clients, samples)
            ax1.set_title('Samples per Client')
            ax1.set_xlabel('Client ID')
            ax1.set_ylabel('Number of Samples')
            ax1.tick_params(axis='x', rotation=45)
            
            # 2. 类别分布热图
            all_classes = set()
            for client_dist in stats.class_distribution.values():
                all_classes.update(client_dist.keys())
            all_classes = sorted(list(all_classes))
            
            class_matrix = []
            for client_id in clients:
                client_dist = stats.class_distribution[client_id]
                row = [client_dist.get(cls, 0) for cls in all_classes]
                class_matrix.append(row)
            
            im = ax2.imshow(class_matrix, cmap='Blues', aspect='auto')
            ax2.set_title('Class Distribution per Client')
            ax2.set_xlabel('Class')
            ax2.set_ylabel('Client ID')
            ax2.set_xticks(range(len(all_classes)))
            ax2.set_xticklabels(all_classes)
            ax2.set_yticks(range(len(clients)))
            ax2.set_yticklabels(clients)
            plt.colorbar(im, ax=ax2)
            
            # 3. IID vs Balance 散点图
            ax3.scatter([stats.iid_score], [stats.balance_score], s=100, color='red')
            ax3.set_xlim(0, 1)
            ax3.set_ylim(0, 1)
            ax3.set_xlabel('IID Score')
            ax3.set_ylabel('Balance Score')
            ax3.set_title('Data Split Quality')
            ax3.grid(True, alpha=0.3)
            
            # 4. 统计摘要文本
            ax4.axis('off')
            summary_text = f"""
Split Summary:
Total Samples: {stats.total_samples}
Number of Clients: {stats.num_clients}
IID Score: {stats.iid_score:.3f}
Balance Score: {stats.balance_score:.3f}
Min Samples: {min(samples)}
Max Samples: {max(samples)}
Avg Samples: {np.mean(samples):.1f}
Std Samples: {np.std(samples):.1f}
            """
            ax4.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.debug(f"Split visualization saved to {save_path}")
            else:
                plt.show()
                
        except ImportError:
            logger.warning("matplotlib not available, skipping visualization")
        except Exception as e:
            logger.error(f"Failed to visualize split: {e}")
            raise SplitStrategyError(f"Visualization failed: {str(e)}") from e
    
    def _calculate_iid_score(self, class_distribution: Dict[str, Dict[str, int]]) -> float:
        """计算IID得分"""
        try:
            if not class_distribution:
                return 0.0
            
            # 获取所有类别
            all_classes = set()
            for client_dist in class_distribution.values():
                all_classes.update(client_dist.keys())
            
            if len(all_classes) <= 1:
                return 1.0
            
            # 计算全局类别分布
            global_dist = defaultdict(int)
            for client_dist in class_distribution.values():
                for cls, count in client_dist.items():
                    global_dist[cls] += count
            
            total_samples = sum(global_dist.values())
            global_ratios = {cls: count / total_samples for cls, count in global_dist.items()}
            
            # 计算每个客户端与全局分布的相似性
            similarities = []
            for client_id, client_dist in class_distribution.items():
                client_total = sum(client_dist.values())
                if client_total == 0:
                    continue
                
                client_ratios = {cls: client_dist.get(cls, 0) / client_total for cls in all_classes}
                
                # 使用余弦相似度
                similarity = self._cosine_similarity(
                    [global_ratios.get(cls, 0) for cls in all_classes],
                    [client_ratios.get(cls, 0) for cls in all_classes]
                )
                similarities.append(similarity)
            
            return np.mean(similarities) if similarities else 0.0
            
        except Exception as e:
            logger.warning(f"Failed to calculate IID score: {e}")
            return 0.0
    
    def _calculate_balance_score(self, samples_per_client: Dict[str, int]) -> float:
        """计算平衡得分"""
        try:
            if not samples_per_client:
                return 0.0
            
            samples = list(samples_per_client.values())
            if len(samples) <= 1:
                return 1.0
            
            # 使用变异系数的倒数作为平衡得分
            mean_samples = np.mean(samples)
            std_samples = np.std(samples)
            
            if mean_samples == 0:
                return 0.0
            
            cv = std_samples / mean_samples  # 变异系数
            balance_score = 1.0 / (1.0 + cv)  # 转换为0-1范围
            
            return balance_score
            
        except Exception as e:
            logger.warning(f"Failed to calculate balance score: {e}")
            return 0.0
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        try:
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return np.dot(vec1, vec2) / (norm1 * norm2)
            
        except Exception:
            return 0.0
    
    def save_split(self, split_data: Dict[str, Dataset], save_path: Path) -> None:
        """
        保存分割结果
        
        Args:
            split_data: 分割后的数据
            save_path: 保存路径
        """
        try:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存分割索引而不是完整数据以节省空间
            split_indices = {}
            for client_id, dataset in split_data.items():
                # 假设dataset有indices属性存储原始索引
                if hasattr(dataset, 'indices'):
                    split_indices[client_id] = dataset.indices
                else:
                    # 如果没有索引，创建顺序索引
                    split_indices[client_id] = list(range(len(dataset)))
            
            with open(save_path, 'wb') as f:
                pickle.dump({
                    'split_indices': split_indices,
                    'config': self.config,
                    'statistics': self.get_split_statistics(split_data).to_dict()
                }, f)
            
            logger.debug(f"Split data saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save split data: {e}")
            raise SplitStrategyError(f"Failed to save split: {str(e)}") from e
    
    @classmethod
    def load_split(cls, load_path: Path) -> Tuple[Dict[str, List[int]], DictConfig, Dict[str, Any]]:
        """
        加载分割结果
        
        Args:
            load_path: 加载路径
            
        Returns:
            (分割索引, 配置, 统计信息)
        """
        try:
            with open(load_path, 'rb') as f:
                data = pickle.load(f)
            
            return data['split_indices'], data['config'], data['statistics']
            
        except Exception as e:
            logger.error(f"Failed to load split data: {e}")
            raise SplitStrategyError(f"Failed to load split: {str(e)}") from e


class IIDSplitStrategy(SplitStrategy):
    """
    IID（独立同分布）数据分割策略
    
    将数据随机分配给客户端，保持数据分布的IID性质。
    支持纯随机分割和分层分割两种模式。
    """
    
    def __init__(self, config: DictConfig, random_seed: int = 42):
        """
        初始化IID分割策略
        
        Args:
            config: 分割配置
            random_seed: 随机种子
        """
        config = DictConfig(config) if isinstance(config, dict) else config
        config.random_seed = random_seed
        super().__init__(config)
        
        self.stratified = config.get('stratified', True)
        self.shuffle = config.get('shuffle', True)
        
        logger.debug(f"IID strategy initialized: stratified={self.stratified}")
    
    def split_data(self, dataset: Dataset, num_clients: int) -> Dict[str, Dataset]:
        """
        执行IID数据分割
        
        Args:
            dataset: 要分割的数据集
            num_clients: 客户端数量
            
        Returns:
            客户端ID到数据集的映射
        """
        try:
            if num_clients <= 0:
                raise ValueError("Number of clients must be positive")
            
            if len(dataset) < num_clients:
                raise ValueError("Dataset size must be >= number of 客户端")
            
            logger.debug(f"Starting IID split: {len(dataset)} samples -> {num_clients} 客户端")
            
            if self.stratified:
                return self.stratified_split(dataset, num_clients)
            else:
                return self.random_split(dataset, num_clients)
                
        except Exception as e:
            logger.error(f"IID split failed: {e}")
            raise SplitStrategyError(f"IID split failed: {str(e)}") from e
    
    def random_split(self, dataset: Dataset, num_clients: int) -> Dict[str, Dataset]:
        """
        随机分割数据集
        
        Args:
            dataset: 数据集
            num_clients: 客户端数量
            
        Returns:
            分割后的数据集字典
        """
        try:
            # 生成随机索引
            indices = list(range(len(dataset)))
            if self.shuffle:
                random.shuffle(indices)
            
            # 计算每个客户端的样本数
            samples_per_client = self._calculate_samples_per_client(len(dataset), num_clients)
            
            # 分割索引
            split_indices = {}
            start_idx = 0
            
            for i in range(num_clients):
                end_idx = start_idx + samples_per_client[i]
                client_indices = indices[start_idx:end_idx]
                
                # 确保每个客户端至少有最小样本数
                if len(client_indices) < self.min_samples_per_client:
                    logger.warning(f"Client {i} has only {len(client_indices)} samples")
                
                split_indices[f"client_{i}"] = client_indices
                start_idx = end_idx
            
            # 创建子数据集
            split_datasets = {}
            for client_id, client_indices in split_indices.items():
                subset = dataset.create_subset(client_indices)
                split_datasets[client_id] = subset
            
            logger.debug(f"Random split completed: {len(split_datasets)} clients created")
            return split_datasets
            
        except Exception as e:
            raise SplitStrategyError(f"Random split failed: {str(e)}") from e
    
    def stratified_split(self, dataset: Dataset, num_clients: int) -> Dict[str, Dataset]:
        """
        分层分割保持类别比例
        
        Args:
            dataset: 数据集
            num_clients: 客户端数量
            
        Returns:
            分割后的数据集字典
        """
        try:
            # 按类别分组样本
            class_indices = defaultdict(list)
            for idx in range(len(dataset)):
                label = dataset[idx][1]
                class_indices[label].append(idx)
            
            # 为每个类别分配样本到客户端
            client_indices = defaultdict(list)
            
            for class_label, indices in class_indices.items():
                if self.shuffle:
                    random.shuffle(indices)
                
                # 计算每个客户端应该获得这个类别的样本数
                samples_per_client = self._calculate_samples_per_client(len(indices), num_clients)
                
                start_idx = 0
                for i in range(num_clients):
                    end_idx = start_idx + samples_per_client[i]
                    client_indices[f"client_{i}"].extend(indices[start_idx:end_idx])
                    start_idx = end_idx
            
            # 打乱每个客户端的索引
            if self.shuffle:
                for client_id in client_indices:
                    random.shuffle(client_indices[client_id])
            
            # 验证最小样本数要求
            for client_id, indices in client_indices.items():
                if len(indices) < self.min_samples_per_client:
                    logger.warning(f"{client_id} has only {len(indices)} samples")
            
            # 创建子数据集
            split_datasets = {}
            for client_id, indices in client_indices.items():
                if indices:  # 只创建非空数据集
                    subset = dataset.create_subset(indices)
                    split_datasets[client_id] = subset
            
            logger.debug(f"Stratified split completed: {len(split_datasets)} clients created")
            return split_datasets
            
        except Exception as e:
            raise SplitStrategyError(f"Stratified split failed: {str(e)}") from e
    
    def validate_split(self, split_data: Dict[str, Dataset]) -> bool:
        """
        验证IID分割结果
        
        Args:
            split_data: 分割后的数据
            
        Returns:
            验证是否通过
        """
        try:
            if not split_data:
                return False
            
            # 检查基本约束
            for client_id, dataset in split_data.items():
                if len(dataset) < self.min_samples_per_client:
                    logger.warning(f"Client {client_id} violates min_samples constraint")
                    return False
            
            # 计算IID得分
            stats = self.get_split_statistics(split_data)
            iid_threshold = self.config.get('iid_threshold', 0.8)
            
            if stats.iid_score < iid_threshold:
                logger.warning(f"IID score {stats.iid_score:.3f} below threshold {iid_threshold}")
                return False
            
            logger.debug(f"IID validation passed: score={stats.iid_score:.3f}")
            return True
            
        except Exception as e:
            logger.error(f"IID validation failed: {e}")
            return False
    
    def _calculate_samples_per_client(self, total_samples: int, num_clients: int) -> List[int]:
        """计算每个客户端的样本数"""
        base_samples = total_samples // num_clients
        extra_samples = total_samples % num_clients
        
        samples_per_client = [base_samples] * num_clients
        
        # 将额外的样本分配给前几个客户端
        for i in range(extra_samples):
            samples_per_client[i] += 1
        
        return samples_per_client


class NonIIDSplitStrategy(SplitStrategy):
    """
    Non-IID（非独立同分布）数据分割策略
    
    创建具有统计异质性的数据分割，模拟真实联邦学习场景中的数据分布。
    支持多种Non-IID分割方法，包括Dirichlet分布、病理性分割等。
    """
    
    def __init__(self, config: DictConfig, alpha: float = 0.5, min_samples_per_client: int = 10):
        """
        初始化Non-IID分割策略
        
        Args:
            config: 分割配置
            alpha: Dirichlet分布参数，越小越不平衡（仅当配置中没有alpha时使用）
            min_samples_per_client: 每个客户端最小样本数（仅当配置中没有时使用）
        """
        config = DictConfig(config) if isinstance(config, dict) else config
        
        # 从配置中读取alpha值，如果没有则使用传入的默认值
        if 'alpha' not in config:
            config.alpha = alpha
        
        # 从配置中读取min_samples_per_client，如果没有则使用传入的默认值
        if 'min_samples_per_client' not in config:
            config.min_samples_per_client = min_samples_per_client
            
        super().__init__(config)
        
        self.alpha = config.get('alpha', alpha)
        self.method = config.get('method', 'dirichlet')  # dirichlet, pathological, label_skew
        self.shards_per_client = config.get('shards_per_client', 2)
        self.num_classes_per_client = config.get('num_classes_per_client', 2)
        
        logger.debug(f"Non-IID strategy initialized: method={self.method}, alpha={self.alpha}")
    
    def split_data(self, dataset: Dataset, num_clients: int) -> Dict[str, Dataset]:
        """
        执行Non-IID数据分割
        
        Args:
            dataset: 要分割的数据集
            num_clients: 客户端数量
            
        Returns:
            客户端ID到数据集的映射
        """
        try:
            if num_clients <= 0:
                raise ValueError("Number of clients must be positive")
            
            if len(dataset) < num_clients * self.min_samples_per_client:
                raise ValueError("Dataset too small for required constraints")
            
            logger.debug(f"Starting Non-IID split: method={self.method}, {len(dataset)} samples -> {num_clients} 客户端")
            
            if self.method == 'dirichlet':
                return self.dirichlet_split(dataset, self.alpha, num_clients)
            elif self.method == 'pathological':
                return self.pathological_split(dataset, self.shards_per_client, num_clients)
            elif self.method == 'label_skew':
                return self.label_skew_split(dataset, num_clients, self.num_classes_per_client)
            elif self.method == 'feature_skew':
                return self.feature_skew_split(dataset, num_clients)
            else:
                raise ValueError(f"Unknown Non-IID method: {self.method}")
                
        except Exception as e:
            logger.error(f"Non-IID split failed: {e}")
            raise SplitStrategyError(f"Non-IID split failed: {str(e)}") from e
    
    def dirichlet_split(self, dataset: Dataset, alpha: float, num_clients: int) -> Dict[str, Dataset]:
        """
        使用Dirichlet分布分割数据
        
        Args:
            dataset: 数据集
            alpha: Dirichlet参数
            num_clients: 客户端数量
            
        Returns:
            分割后的数据集字典
        """
        try:
            # 按类别分组
            class_indices = defaultdict(list)
            for idx in range(len(dataset)):
                label = dataset[idx][1]
                class_indices[label].append(idx)
            
            num_classes = len(class_indices)
            client_indices = [[] for _ in range(num_clients)]
            
            # 为每个类别使用Dirichlet分布分配给客户端
            for class_label, indices in class_indices.items():
                # 生成Dirichlet分布的权重
                weights = np.random.dirichlet([alpha] * num_clients)
                
                # 根据权重分配样本
                np.random.shuffle(indices)
                cum_weights = np.cumsum(weights)
                
                start_idx = 0
                for client_id in range(num_clients):
                    # 计算这个客户端应该获得的样本数
                    end_ratio = cum_weights[client_id]
                    end_idx = int(end_ratio * len(indices))
                    
                    # 确保不超出范围
                    end_idx = min(end_idx, len(indices))
                    
                    if start_idx < end_idx:
                        client_indices[client_id].extend(indices[start_idx:end_idx])
                    
                    start_idx = end_idx
            
            # 确保每个客户端都有最小样本数
            self._enforce_min_samples(client_indices, dataset)
            
            # 创建数据集
            split_datasets = {}
            for i, indices in enumerate(client_indices):
                if indices:
                    random.shuffle(indices)  # 打乱顺序
                    subset = dataset.create_subset(indices)
                    split_datasets[f"client_{i}"] = subset
            
            logger.debug(f"Dirichlet split completed: {len(split_datasets)} clients created")
            return split_datasets
            
        except Exception as e:
            raise SplitStrategyError(f"Dirichlet split failed: {str(e)}") from e
    
    def pathological_split(self, dataset: Dataset, shards_per_client: int, num_clients: int) -> Dict[str, Dataset]:
        """
        病理性Non-IID分割
        
        每个客户端只获得有限数量的类别（shards）
        
        Args:
            dataset: 数据集
            shards_per_client: 每个客户端的分片数
            num_clients: 客户端数量
            
        Returns:
            分割后的数据集字典
        """
        try:
            # 按类别分组
            class_indices = defaultdict(list)
            for idx in range(len(dataset)):
                label = dataset[idx][1]
                class_indices[label].append(idx)
            
            num_classes = len(class_indices)
            total_shards = shards_per_client * num_clients
            
            if total_shards > num_classes * 2:
                logger.warning(f"Too many shards ({total_shards}) for {num_classes} classes")
            
            # 为每个类别创建分片
            all_shards = []
            for class_label, indices in class_indices.items():
                random.shuffle(indices)
                
                # 将类别分成两个分片
                mid_point = len(indices) // 2
                all_shards.append((class_label, indices[:mid_point]))
                all_shards.append((class_label, indices[mid_point:]))
            
            # 随机分配分片给客户端
            random.shuffle(all_shards)
            
            client_indices = [[] for _ in range(num_clients)]
            shard_idx = 0
            
            for client_id in range(num_clients):
                for _ in range(shards_per_client):
                    if shard_idx < len(all_shards):
                        _, indices = all_shards[shard_idx]
                        client_indices[client_id].extend(indices)
                        shard_idx += 1
            
            # 分配剩余的分片
            while shard_idx < len(all_shards):
                client_id = shard_idx % num_clients
                _, indices = all_shards[shard_idx]
                client_indices[client_id].extend(indices)
                shard_idx += 1
            
            # 确保最小样本数
            self._enforce_min_samples(client_indices, dataset)
            
            # 创建数据集
            split_datasets = {}
            for i, indices in enumerate(client_indices):
                if indices:
                    random.shuffle(indices)
                    subset = dataset.create_subset(indices)
                    split_datasets[f"client_{i}"] = subset
            
            logger.debug(f"Pathological split completed: {len(split_datasets)} clients created")
            return split_datasets
            
        except Exception as e:
            raise SplitStrategyError(f"Pathological split failed: {str(e)}") from e
    
    def label_skew_split(self, dataset: Dataset, num_clients: int, num_classes_per_client: int) -> Dict[str, Dataset]:
        """
        标签偏斜分割
        
        每个客户端只获得指定数量的类别
        
        Args:
            dataset: 数据集
            num_clients: 客户端数量
            num_classes_per_client: 每个客户端的类别数
            
        Returns:
            分割后的数据集字典
        """
        try:
            # 按类别分组
            class_indices = defaultdict(list)
            for idx in range(len(dataset)):
                label = dataset[idx][1]
                class_indices[label].append(idx)
            
            all_classes = list(class_indices.keys())
            num_classes = len(all_classes)
            
            if num_classes_per_client > num_classes:
                logger.warning(f"num_classes_per_client ({num_classes_per_client}) > total classes ({num_classes})")
                num_classes_per_client = num_classes
            
            client_indices = [[] for _ in range(num_clients)]
            
            # 为每个客户端分配类别
            for client_id in range(num_clients):
                # 随机选择类别
                selected_classes = random.sample(all_classes, num_classes_per_client)
                
                for class_label in selected_classes:
                    indices = class_indices[class_label].copy()
                    random.shuffle(indices)
                    
                    # 将这个类别的部分样本分配给当前客户端
                    samples_for_client = len(indices) // num_clients
                    if samples_for_client > 0:
                        start_idx = client_id * samples_for_client
                        end_idx = min(start_idx + samples_for_client, len(indices))
                        client_indices[client_id].extend(indices[start_idx:end_idx])
            
            # 确保最小样本数
            self._enforce_min_samples(client_indices, dataset)
            
            # 创建数据集
            split_datasets = {}
            for i, indices in enumerate(client_indices):
                if indices:
                    random.shuffle(indices)
                    subset = dataset.create_subset(indices)
                    split_datasets[f"client_{i}"] = subset
            
            logger.debug(f"Label skew split completed: {len(split_datasets)} clients created")
            return split_datasets
            
        except Exception as e:
            raise SplitStrategyError(f"Label skew split failed: {str(e)}") from e
    
    def feature_skew_split(self, dataset: Dataset, num_clients: int) -> Dict[str, Dataset]:
        """
        特征偏斜分割（简化版本）
        
        Args:
            dataset: 数据集
            num_clients: 客户端数量
            
        Returns:
            分割后的数据集字典
        """
        try:
            # 特征偏斜需要更复杂的实现，这里提供基础版本
            # 可以根据具体需求扩展
            logger.warning("Feature skew split is simplified - using Dirichlet as fallback")
            return self.dirichlet_split(dataset, self.alpha, num_clients)
            
        except Exception as e:
            raise SplitStrategyError(f"Feature skew split failed: {str(e)}") from e
    
    def validate_split(self, split_data: Dict[str, Dataset]) -> bool:
        """
        验证Non-IID分割结果
        
        Args:
            split_data: 分割后的数据
            
        Returns:
            验证是否通过
        """
        try:
            if not split_data:
                return False
            
            # 检查基本约束
            for client_id, dataset in split_data.items():
                if len(dataset) < self.min_samples_per_client:
                    logger.warning(f"Client {client_id} violates min_samples constraint")
                    return False
            
            # 检查Non-IID特性
            stats = self.get_split_statistics(split_data)
            iid_threshold = self.config.get('max_iid_score', 0.7)
            
            if stats.iid_score > iid_threshold:
                logger.warning(f"Data too IID for Non-IID split: score={stats.iid_score:.3f}")
                return False
            
            logger.debug(f"Non-IID validation passed: score={stats.iid_score:.3f}")
            return True
            
        except Exception as e:
            logger.error(f"Non-IID validation failed: {e}")
            return False
    
    def _enforce_min_samples(self, client_indices: List[List[int]], dataset: Dataset) -> None:
        """确保每个客户端都有最小样本数"""
        try:
            # 收集所有未分配的样本
            all_assigned = set()
            for indices in client_indices:
                all_assigned.update(indices)
            
            unassigned = []
            for idx in range(len(dataset)):
                if idx not in all_assigned:
                    unassigned.append(idx)
            
            # 为样本数不足的客户端分配额外样本
            for client_id, indices in enumerate(client_indices):
                needed = self.min_samples_per_client - len(indices)
                if needed > 0 and unassigned:
                    # 从未分配样本中取出需要的数量
                    additional = unassigned[:needed]
                    indices.extend(additional)
                    unassigned = unassigned[needed:]
                    
                    logger.debug(f"Added {len(additional)} samples to client_{client_id}")
            
        except Exception as e:
            logger.warning(f"Failed to enforce min samples: {e}")
