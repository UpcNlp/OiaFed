# fedcl/implementations/aggregators/fednova_aggregator.py
"""
FedNova 聚合器实现

基于论文: https://arxiv.org/abs/2007.07481
FedNova通过归一化和方差减少技术来处理客户端异构性问题。
"""

from typing import Dict, List, Any, Tuple
import torch
from omegaconf import DictConfig
from loguru import logger

from ...core.base_aggregator import BaseAggregator
from ...core.execution_context import ExecutionContext
from ...registry import registry


@registry.aggregator("fednova", metadata={
    "description": "FedNova: Tackling the Objective Inconsistency Problem",
    "paper": "https://arxiv.org/abs/2007.07481",
    "handles_heterogeneity": True
})
class FedNovaAggregator(BaseAggregator):
    """
    FedNova聚合器实现
    
    通过归一化技术处理客户端异构性和目标不一致问题。
    """
    
    def __init__(self, context: ExecutionContext, config: DictConfig):
        super().__init__(context, config)
        self.momentum = config.get("momentum", 0.9)
        logger.debug(f"FedNovaAggregator initialized with momentum={self.momentum}")
        
    def aggregate(
        self,
        client_updates: List[Dict[str, Any]],
        global_model: torch.nn.Module,
        round_id: int
    ) -> Tuple[torch.nn.Module, Dict[str, Any]]:
        """执行FedNova聚合"""
        logger.info(f"FedNova aggregation for round {round_id}")
        
        # 简化的FedNova实现
        model_states = [update["model_state"] for update in client_updates]
        local_steps = [update.get("local_steps", 1) for update in client_updates]
        
        # 计算归一化权重
        total_steps = sum(local_steps)
        weights = [steps / total_steps for steps in local_steps]
        
        # 执行加权聚合
        aggregated_state = {}
        for param_name in model_states[0].keys():
            weighted_sum = torch.zeros_like(model_states[0][param_name])
            for state, weight in zip(model_states, weights):
                weighted_sum += weight * state[param_name]
            aggregated_state[param_name] = weighted_sum
            
        global_model.load_state_dict(aggregated_state)
        
        info = {
            "round_id": round_id,
            "num_客户端": len(client_updates),
            "total_local_steps": total_steps,
            "aggregation_method": "fednova",
            "momentum": self.momentum
        }
        
        return global_model, info
