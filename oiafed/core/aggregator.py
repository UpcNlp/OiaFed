"""
Aggregator（聚合器）抽象基类
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, TYPE_CHECKING

from .types import ClientUpdate

if TYPE_CHECKING:
    from .model import Model


class Aggregator(ABC):
    """
    聚合器抽象基类
    
    职责：
    - 聚合多个客户端的模型更新
    
    用户需要实现 aggregate() 方法
    """
    
    @abstractmethod
    def aggregate(
        self,
        updates: List[ClientUpdate],
        global_model: Optional["Model"] = None,
    ) -> Any:
        """
        聚合客户端更新
        
        Args:
            updates: 客户端更新列表，每个包含:
                - client_id: 客户端 ID
                - weights: 模型权重
                - num_samples: 样本数
                - metrics: 指标
            global_model: 当前全局模型（某些算法需要，如 FedProx）
            
        Returns:
            聚合后的模型权重
            
        Example:
            # FedAvg 实现
            def aggregate(self, updates, global_model=None):
                total_samples = sum(u.num_samples for u in updates)
                weighted_weights = None
                
                for update in updates:
                    weight = update.num_samples / total_samples
                    if weighted_weights is None:
                        weighted_weights = [w * weight for w in update.weights]
                    else:
                        for i, w in enumerate(update.weights):
                            weighted_weights[i] += w * weight
                
                return weighted_weights
        """
        pass
    
    def pre_aggregate(self, updates: List[ClientUpdate]) -> List[ClientUpdate]:
        """
        聚合前处理（可选覆盖）
        
        可用于过滤、排序、预处理更新
        
        Args:
            updates: 原始更新列表
            
        Returns:
            处理后的更新列表
        """
        return updates
    
    def post_aggregate(self, weights: Any, updates: List[ClientUpdate]) -> Any:
        """
        聚合后处理（可选覆盖）
        
        可用于后处理聚合结果
        
        Args:
            weights: 聚合后的权重
            updates: 参与聚合的更新列表
            
        Returns:
            处理后的权重
        """
        return weights
