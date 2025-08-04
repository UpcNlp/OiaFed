# fedcl/implementations/aggregators/fedprox_aggregator.py
"""
FedProx 聚合器实现

基于论文: https://arxiv.org/abs/1812.06127
FedProx在FedAvg基础上增加了正则化项，提高了非IID数据分布下的性能。
"""

from typing import Dict, List, Any, Tuple
import torch
from omegaconf import DictConfig
from loguru import logger

from ...core.base_aggregator import BaseAggregator
from ...core.execution_context import ExecutionContext
from ...registry import registry
from .fedavg_aggregator import FedAvgAggregator


@registry.aggregator("fedprox", metadata={
    "description": "FedProx: Federated Optimization in Heterogeneous Networks",
    "paper": "https://arxiv.org/abs/1812.06127",
    "extends": "fedavg"
})
class FedProxAggregator(FedAvgAggregator):
    """
    FedProx聚合器实现
    
    在FedAvg基础上增加了正则化处理，适用于异构网络环境。
    """
    
    def __init__(self, context: ExecutionContext, config: DictConfig):
        super().__init__(context, config)
        self.mu = config.get("mu", 0.01)  # 正则化参数
        logger.debug(f"FedProxAggregator initialized with mu={self.mu}")
        
    def aggregate(
        self,
        client_updates: List[Dict[str, Any]],
        global_model: torch.nn.Module,
        round_id: int
    ) -> Tuple[torch.nn.Module, Dict[str, Any]]:
        """执行FedProx聚合"""
        # FedProx的聚合策略与FedAvg相同，区别在于客户端训练时的正则化
        aggregated_model, info = super().aggregate(client_updates, global_model, round_id)
        info["aggregation_method"] = "fedprox"
        info["mu"] = self.mu
        return aggregated_model, info
