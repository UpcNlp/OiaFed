"""
FedDyn 聚合器

实现 FedDyn (Federated Learning with Dynamic Regularization) 聚合算法。
通过动态正则化项改善联邦学习的收敛性。

论文：Federated Learning with Only Positive Labels
作者：Felix X. Yu et al.
发表：ICML 2020
"""

import torch
from typing import Dict, List, Any
from loguru import logger

from ...api.decorators import aggregator


@aggregator("feddyn", description="FedDyn动态正则化联邦聚合器")
class FedDynAggregator:
    """FedDyn 聚合器实现"""
    
    def __init__(self, config: Dict[str, Any] = None, **kwargs):
        self.config = config or {}
        self.alpha = self.config.get("alpha", 0.01)  # 动态正则化系数
        
        self.device = self.config.get("device", "auto")
        if self.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 状态变量
        self.h_state = {}  # 动态正则化状态
        self.round_count = 0
        
        self.global_model = kwargs.get("global_model")
        logger.info(f"✅ FedDyn聚合器初始化完成 - α: {self.alpha}")
    
    def aggregate(self, client_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """执行FedDyn聚合"""
        if not client_updates:
            raise ValueError("没有客户端更新可聚合")
        
        self.round_count += 1
        
        # 计算加权平均
        total_samples = sum(update.get("num_samples", 1) for update in client_updates)
        weights = [update.get("num_samples", 1) / total_samples for update in client_updates]
        
        # 聚合权重
        aggregated_weights = {}
        first_weights = client_updates[0]["model_weights"]
        
        for param_name in first_weights:
            aggregated_weights[param_name] = torch.zeros_like(
                first_weights[param_name], device=self.device
            )
            
            for i, update in enumerate(client_updates):
                param_value = update["model_weights"][param_name].to(self.device)
                aggregated_weights[param_name] += weights[i] * param_value
        
        # 初始化h状态
        if not self.h_state:
            for param_name in aggregated_weights:
                self.h_state[param_name] = torch.zeros_like(
                    aggregated_weights[param_name], device=self.device
                )
        
        # 更新h状态和全局模型
        for param_name in aggregated_weights:
            self.h_state[param_name] += self.alpha * aggregated_weights[param_name]
            aggregated_weights[param_name] -= (1.0 / self.alpha) * self.h_state[param_name]
        
        return {
            "aggregated_weights": aggregated_weights,
            "algorithm": "FedDyn",
            "alpha": self.alpha,
            "round": self.round_count
        }
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "algorithm": "FedDyn",
            "alpha": self.alpha,
            "rounds": self.round_count
        }