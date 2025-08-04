# fedcl/implementations/aggregators/scaffold_aggregator.py
"""
SCAFFOLD 聚合器实现

基于论文: https://arxiv.org/abs/1910.06378
SCAFFOLD通过控制变量来减少客户端漂移，提高联邦学习的收敛性。
"""

from typing import Dict, List, Any, Tuple
import torch
from omegaconf import DictConfig
from loguru import logger

from ...core.base_aggregator import BaseAggregator
from ...core.execution_context import ExecutionContext
from ...registry import registry


@registry.aggregator("scaffold", metadata={
    "description": "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning",
    "paper": "https://arxiv.org/abs/1910.06378",
    "requires_control_variates": True
})
class SCAFFOLDAggregator(BaseAggregator):
    """
    SCAFFOLD聚合器实现
    
    使用控制变量来纠正客户端漂移，提高非IID数据下的性能。
    """
    
    def __init__(self, context: ExecutionContext, config: DictConfig):
        super().__init__(context, config)
        self.server_control = {}  # 服务器控制变量
        self.client_controls = {}  # 客户端控制变量
        logger.debug("SCAFFOLDAggregator initialized")
        
    def aggregate(
        self,
        client_updates: List[Dict[str, Any]],
        global_model: torch.nn.Module,
        round_id: int
    ) -> Tuple[torch.nn.Module, Dict[str, Any]]:
        """执行SCAFFOLD聚合"""
        logger.info(f"SCAFFOLD aggregation for round {round_id}")
        
        # 简化的SCAFFOLD实现
        # 实际应该包含控制变量的更新逻辑
        
        model_states = [update["model_state"] for update in client_updates]
        num_clients = len(client_updates)
        
        # 简单平均聚合（应该包含控制变量校正）
        aggregated_state = {}
        for param_name in model_states[0].keys():
            param_sum = torch.zeros_like(model_states[0][param_name])
            for state in model_states:
                param_sum += state[param_name]
            aggregated_state[param_name] = param_sum / num_clients
            
        global_model.load_state_dict(aggregated_state)
        
        info = {
            "round_id": round_id,
            "num_客户端": num_clients,
            "aggregation_method": "scaffold"
        }
        
        return global_model, info
