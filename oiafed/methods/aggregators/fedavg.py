"""
内置聚合器实现
"""

from typing import Any, List, Optional, TYPE_CHECKING

from ...core.aggregator import Aggregator
from ...core.types import ClientUpdate
from ...registry import aggregator
from ...infra import get_module_logger

if TYPE_CHECKING:
    from ...core.model import Model

logger = get_module_logger(__name__)


@aggregator(
    name='fedavg',
    description='FedAvg聚合器 - 加权平均聚合，权重为样本数量',
    version='1.0',
    author='Federation Framework',
    weighted=True
)
class FedAvgAggregator(Aggregator):
    """
    FedAvg 聚合器
    
    加权平均聚合，权重为样本数量
    """
    
    def __init__(self, weighted: bool = True, **kwargs):
        """
        初始化
        
        Args:
            weighted: 是否按样本数量加权
        """
        self._weighted = weighted
    
    def aggregate(
        self,
        updates: List[ClientUpdate],
        global_model: Optional["Model"] = None,
    ) -> Any:
        """聚合客户端更新"""
        if not updates:
            raise ValueError("No updates to aggregate")
        
        # 预处理
        updates = self.pre_aggregate(updates)
        
        # 计算权重
        if self._weighted:
            total_samples = sum(u.num_samples for u in updates)
            if total_samples == 0:
                # 如果所有客户端的样本数都是0，使用均匀权重
                logger.warning("Total samples is 0, using uniform weights instead")
                weights = [1.0 / len(updates)] * len(updates)
            else:
                weights = [u.num_samples / total_samples for u in updates]
        else:
            weights = [1.0 / len(updates)] * len(updates)
        
        # 加权平均
        aggregated = None
        
        for update, weight in zip(updates, weights):
            client_weights = update.weights
            
            if aggregated is None:
                # 初始化
                aggregated = self._scale_weights(client_weights, weight)
            else:
                # 累加
                scaled = self._scale_weights(client_weights, weight)
                aggregated = self._add_weights(aggregated, scaled)
        
        # 后处理
        aggregated = self.post_aggregate(aggregated, updates)
        
        return aggregated
    
    def _scale_weights(self, weights: Any, scale: float) -> Any:
        """缩放权重"""
        if isinstance(weights, list):
            return [self._scale_weights(w, scale) for w in weights]
        elif isinstance(weights, dict):
            return {k: self._scale_weights(v, scale) for k, v in weights.items()}
        else:
            # 假设是 numpy array 或 tensor
            return weights * scale
    
    def _add_weights(self, a: Any, b: Any) -> Any:
        """相加权重"""
        if isinstance(a, list):
            return [self._add_weights(x, y) for x, y in zip(a, b)]
        elif isinstance(a, dict):
            return {k: self._add_weights(a[k], b[k]) for k in a}
        else:
            return a + b


@aggregator(
    name='median',
    description='中位数聚合器 - 抗拜占庭攻击的鲁棒聚合',
    version='1.0',
    author='Federation Framework',
    weighted=False
)
class MedianAggregator(Aggregator):
    """
    中位数聚合器
    
    对每个参数取中位数，更鲁棒
    """
    
    def aggregate(
        self,
        updates: List[ClientUpdate],
        global_model: Optional["Model"] = None,
    ) -> Any:
        """聚合客户端更新"""
        if not updates:
            raise ValueError("No updates to aggregate")
        
        try:
            import numpy as np
        except ImportError:
            logger.warning("numpy not available, falling back to FedAvg")
            return FedAvgAggregator().aggregate(updates, global_model)
        
        # 收集所有权重
        all_weights = [u.weights for u in updates]
        
        # 计算中位数
        if isinstance(all_weights[0], list):
            # List of arrays
            aggregated = []
            for i in range(len(all_weights[0])):
                stacked = np.stack([w[i] for w in all_weights])
                aggregated.append(np.median(stacked, axis=0))
            return aggregated
        elif isinstance(all_weights[0], dict):
            # Dict of arrays
            aggregated = {}
            for key in all_weights[0]:
                stacked = np.stack([w[key] for w in all_weights])
                aggregated[key] = np.median(stacked, axis=0)
            return aggregated
        else:
            stacked = np.stack(all_weights)
            return np.median(stacked, axis=0)


@aggregator(
    name='trimmed_mean',
    description='修剪均值聚合器 - 去掉最大最小值后取平均',
    version='1.0',
    author='Federation Framework',
    weighted=False
)
class TrimmedMeanAggregator(Aggregator):
    """
    修剪均值聚合器
    
    去掉最大最小值后取平均
    """
    
    def __init__(self, trim_ratio: float = 0.1, **kwargs):
        """
        初始化
        
        Args:
            trim_ratio: 修剪比例（两端各去掉该比例）
        """
        self._trim_ratio = trim_ratio
    
    def aggregate(
        self,
        updates: List[ClientUpdate],
        global_model: Optional["Model"] = None,
    ) -> Any:
        """聚合客户端更新"""
        if not updates:
            raise ValueError("No updates to aggregate")
        
        try:
            import numpy as np
            from scipy import stats
        except ImportError:
            logger.warning("scipy not available, falling back to FedAvg")
            return FedAvgAggregator().aggregate(updates, global_model)
        
        # 收集所有权重
        all_weights = [u.weights for u in updates]
        
        # 计算修剪比例
        n = len(all_weights)
        trim_count = int(n * self._trim_ratio)
        
        if isinstance(all_weights[0], list):
            aggregated = []
            for i in range(len(all_weights[0])):
                stacked = np.stack([w[i] for w in all_weights])
                trimmed = stats.trim_mean(stacked, self._trim_ratio, axis=0)
                aggregated.append(trimmed)
            return aggregated
        elif isinstance(all_weights[0], dict):
            aggregated = {}
            for key in all_weights[0]:
                stacked = np.stack([w[key] for w in all_weights])
                aggregated[key] = stats.trim_mean(stacked, self._trim_ratio, axis=0)
            return aggregated
        else:
            stacked = np.stack(all_weights)
            return stats.trim_mean(stacked, self._trim_ratio, axis=0)
