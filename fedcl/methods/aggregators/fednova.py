"""
FedNova 聚合器

实现 FedNova (Federated Optimization in Heterogeneous Networks) 聚合算法。
通过标准化客户端更新来处理客户端异构性问题。

论文：Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization
作者：Jianyu Wang et al.
发表：NeurIPS 2020

算法特点：
1. 标准化不同客户端的本地更新步数
2. 更好地处理系统异构性
3. 改善收敛速度和稳定性
"""

import torch
from typing import Dict, List, Any
from loguru import logger

from ...api.decorators import aggregator


@aggregator("fednova", description="FedNova标准化联邦聚合器")
class FedNovaAggregator:
    """FedNova 聚合器实现"""
    
    def __init__(self, config: Dict[str, Any] = None, **kwargs):
        self.config = config or {}
        self.device = self.config.get("device", "auto")
        if self.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.round_count = 0
        self.global_model = kwargs.get("global_model")
        
        logger.info("✅ FedNova聚合器初始化完成")
    
    def aggregate(self, client_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """执行FedNova聚合"""
        if not client_updates:
            raise ValueError("没有客户端更新可聚合")
        
        self.round_count += 1
        
        # 计算有效批次数进行标准化
        effective_batches = [update.get("local_epochs", 1) * update.get("num_samples", 1) 
                           for update in client_updates]
        total_effective = sum(effective_batches)
        
        # 标准化权重
        weights = [eff / total_effective for eff in effective_batches]
        
        # 聚合
        aggregated_weights = {}
        first_weights = client_updates[0]["model_weights"]
        
        for param_name in first_weights:
            aggregated_weights[param_name] = torch.zeros_like(
                first_weights[param_name], device=self.device
            )
            
            for i, update in enumerate(client_updates):
                param_value = update["model_weights"][param_name].to(self.device)
                aggregated_weights[param_name] += weights[i] * param_value
        
        return {
            "aggregated_weights": aggregated_weights,
            "algorithm": "FedNova",
            "round": self.round_count,
            "normalization_weights": weights
        }
    
    def get_stats(self) -> Dict[str, Any]:
        return {"algorithm": "FedNova", "rounds": self.round_count}