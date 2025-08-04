# fedcl/implementations/aggregators/fedavg_aggregator.py
"""
Federated Averaging (FedAvg) 聚合器实现

基于论文: https://arxiv.org/abs/1602.05629
FedAvg是最经典的联邦学习聚合算法，通过加权平均客户端模型更新来更新全局模型。

主要特性:
- 基于数据量的加权平均
- 支持部分客户端参与
- 简单高效的聚合策略
- 适用于IID和Non-IID数据分布
"""

from typing import Dict, List, Any, Optional, Tuple
import torch
import torch.nn as nn
from omegaconf import DictConfig
from loguru import logger

from ...core.base_aggregator import BaseAggregator
from ...core.execution_context import ExecutionContext
from ...registry import registry


@registry.aggregator("fedavg", metadata={
    "description": "Federated Averaging Algorithm",
    "paper": "https://arxiv.org/abs/1602.05629",
    "complexity": "O(n*p)",  # n=clients, p=parameters
    "memory_efficient": True
})
class FedAvgAggregator(BaseAggregator):
    """
    FedAvg (Federated Averaging) 聚合器实现
    
    FedAvg通过按数据量加权平均的方式聚合客户端模型更新：
    w_global = Σ(n_k / n_total) * w_k
    
    主要特点：
    1. 基于客户端数据量的加权平均
    2. 支持异步和同步聚合
    3. 内存效率高，计算简单
    4. 广泛适用的基准算法
    """
    
    def __init__(self, context: ExecutionContext, config: DictConfig):
        """
        初始化FedAvg聚合器
        
        Args:
            context: 执行上下文
            config: 配置参数，可包含：
                - weighted_average: 是否使用加权平均 (默认True)
                - min_clients: 最小客户端数量
                - aggregation_weights: 自定义聚合权重策略
        """
        super().__init__(context, config)
        
        # FedAvg特定配置
        self.weighted_average = config.get("weighted_average", True)
        self.min_clients = config.get("min_客户端", 1)
        self.aggregation_weights = config.get("aggregation_weights", "uniform")
        
        # 统计信息
        self.aggregation_count = 0
        self.total_clients_aggregated = 0
        
        logger.debug(f"FedAvgAggregator initialized with weighted_average={self.weighted_average}")
    
    def aggregate(self, client_updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        实现BaseAggregator的aggregate接口
        
        Args:
            client_updates: 客户端更新列表，每个更新是参数名到张量的映射
            
        Returns:
            Dict[str, torch.Tensor]: 聚合后的参数更新
        """
        logger.debug(f"FedAvgAggregator.aggregate called with {len(client_updates)} updates")
        
        if len(client_updates) < self.min_clients:
            raise ValueError(f"Insufficient clients: {len(client_updates)} < {self.min_clients}")
        
        if not client_updates:
            logger.warning("No client updates provided to aggregate")
            return {}
        
        # 获取第一个更新的参数名
        param_names = list(client_updates[0].keys())
        logger.debug(f"Aggregating parameters: {param_names[:5]}...")  # 只显示前5个参数名
        
        # 计算聚合权重（均匀权重，因为无法从参数更新中得知样本数量）
        num_clients = len(client_updates)
        weights = [1.0 / num_clients] * num_clients
        
        # 聚合参数
        aggregated_params = {}
        for param_name in param_names:
            # 加权平均所有客户端的参数
            param_sum = None
            for i, update in enumerate(client_updates):
                if param_name in update:
                    param_tensor = update[param_name]
                    if param_sum is None:
                        param_sum = weights[i] * param_tensor
                    else:
                        param_sum += weights[i] * param_tensor
            
            if param_sum is not None:
                aggregated_params[param_name] = param_sum
        
        logger.debug(f"FedAvgAggregator.aggregate completed, aggregated {len(aggregated_params)} parameters")
        return aggregated_params
        
    def aggregate_full(
        self,
        client_updates: List[Dict[str, Any]],
        global_model: torch.nn.Module,
        round_id: int
    ) -> Tuple[torch.nn.Module, Dict[str, Any]]:
        """
        执行FedAvg聚合
        
        Args:
            client_updates: 客户端更新列表，每个更新包含：
                - model_state: 模型状态字典
                - num_samples: 训练样本数量
                - client_id: 客户端ID
                - metrics: 训练指标
            global_model: 全局模型
            round_id: 当前轮次ID
            
        Returns:
            (aggregated_model, aggregation_info): 聚合后的模型和聚合信息
            
        Raises:
            AggregationError: 聚合过程中出现错误
        """
        if len(client_updates) < self.min_clients:
            raise ValueError(f"Insufficient clients: {len(client_updates)} < {self.min_clients}")
            
        logger.debug(f"Starting FedAvg aggregation for round {round_id} "
                   f"with {len(client_updates)} 客户端")
        
        # 提取模型状态和权重
        model_states = []
        client_weights = []
        total_samples = 0
        
        for update in client_updates:
            model_states.append(update["model_state"])
            num_samples = update.get("num_samples", 1)
            client_weights.append(num_samples)
            total_samples += num_samples
            
        # 计算聚合权重
        if self.weighted_average and total_samples > 0:
            # 基于数据量的加权平均
            weights = [w / total_samples for w in client_weights]
        else:
            # 均匀权重
            weights = [1.0 / len(client_updates)] * len(client_updates)
            
        # 执行模型聚合
        aggregated_state = self._aggregate_model_states(model_states, weights)
        
        # 更新全局模型
        global_model.load_state_dict(aggregated_state)
        
        # 更新统计信息
        self.aggregation_count += 1
        self.total_clients_aggregated += len(client_updates)
        
        # 构建聚合信息
        aggregation_info = {
            "round_id": round_id,
            "num_客户端": len(client_updates),
            "total_samples": total_samples,
            "client_weights": client_weights,
            "aggregation_weights": weights,
            "aggregation_method": "fedavg",
            "weighted_average": self.weighted_average
        }
        
        logger.debug(f"FedAvg aggregation completed for round {round_id}")
        
        return global_model, aggregation_info
        
    def _aggregate_model_states(
        self,
        model_states: List[Dict[str, torch.Tensor]],
        weights: List[float]
    ) -> Dict[str, torch.Tensor]:
        """
        聚合模型状态字典
        
        Args:
            model_states: 客户端模型状态列表
            weights: 聚合权重列表
            
        Returns:
            聚合后的模型状态字典
        """
        if not model_states:
            raise ValueError("Empty model states list")
            
        # 初始化聚合状态
        aggregated_state = {}
        
        # 获取参数名称
        param_names = model_states[0].keys()
        
        # 对每个参数进行加权平均
        for param_name in param_names:
            # 检查所有客户端都有该参数
            if not all(param_name in state for state in model_states):
                logger.warning(f"Parameter {param_name} not found in all client states")
                continue
                
            # 获取参数张量列表
            param_tensors = [state[param_name] for state in model_states]
            
            # 检查参数形状一致性
            if not all(tensor.shape == param_tensors[0].shape for tensor in param_tensors):
                logger.warning(f"Parameter {param_name} has inconsistent shapes across 客户端")
                continue
                
            # 执行加权平均
            aggregated_param = torch.zeros_like(param_tensors[0])
            for tensor, weight in zip(param_tensors, weights):
                aggregated_param += weight * tensor.to(aggregated_param.device)
                
            aggregated_state[param_name] = aggregated_param
            
        return aggregated_state
        
    def compute_client_weights(
        self,
        client_updates: List[Dict[str, Any]],
        strategy: str = "uniform"
    ) -> List[float]:
        """
        计算客户端聚合权重
        
        Args:
            client_updates: 客户端更新列表
            strategy: 权重计算策略
                - "uniform": 均匀权重
                - "samples": 基于样本数量
                - "loss": 基于损失值
                - "accuracy": 基于准确率
                
        Returns:
            归一化的权重列表
        """
        if strategy == "uniform":
            weights = [1.0] * len(client_updates)
        elif strategy == "samples":
            weights = [update.get("num_samples", 1) for update in client_updates]
        elif strategy == "loss":
            # 基于损失的权重（损失越低权重越高）
            losses = [update.get("metrics", {}).get("loss", 1.0) for update in client_updates]
            weights = [1.0 / (loss + 1e-8) for loss in losses]
        elif strategy == "accuracy":
            # 基于准确率的权重
            accuracies = [update.get("metrics", {}).get("accuracy", 0.5) for update in client_updates]
            weights = accuracies
        else:
            raise ValueError(f"Unknown weight strategy: {strategy}")
            
        # 归一化权重
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(client_updates)] * len(client_updates)
            
        return weights
        
    def get_aggregation_statistics(self) -> Dict[str, Any]:
        """
        获取聚合统计信息
        
        Returns:
            聚合统计信息字典
        """
        return {
            "algorithm": "FedAvg",
            "aggregation_count": self.aggregation_count,
            "total_clients_aggregated": self.total_clients_aggregated,
            "average_clients_per_round": (
                self.total_clients_aggregated / self.aggregation_count
                if self.aggregation_count > 0 else 0
            ),
            "weighted_average": self.weighted_average,
            "min_客户端": self.min_clients
        }
        
    def validate_client_updates(self, client_updates: List[Dict[str, Any]]) -> bool:
        """
        验证客户端更新的有效性
        
        Args:
            client_updates: 客户端更新列表
            
        Returns:
            是否所有更新都有效
        """
        if not client_updates:
            logger.error("Empty client updates list")
            return False
            
        required_keys = ["model_state", "client_id"]
        
        for i, update in enumerate(client_updates):
            # 检查必需字段
            for key in required_keys:
                if key not in update:
                    logger.error(f"Client update {i} missing required key: {key}")
                    return False
                    
            # 检查模型状态
            model_state = update["model_state"]
            if not isinstance(model_state, dict):
                logger.error(f"Client update {i} has invalid model_state type")
                return False
                
            # 检查模型状态不为空
            if not model_state:
                logger.error(f"Client update {i} has empty model_state")
                return False
                
        # 检查模型状态一致性
        if not self._check_model_state_consistency(client_updates):
            return False
            
        return True
        
    def _check_model_state_consistency(self, client_updates: List[Dict[str, Any]]) -> bool:
        """
        检查客户端模型状态的一致性
        
        Args:
            client_updates: 客户端更新列表
            
        Returns:
            模型状态是否一致
        """
        if len(client_updates) < 2:
            return True
            
        reference_state = client_updates[0]["model_state"]
        reference_keys = set(reference_state.keys())
        
        for i, update in enumerate(client_updates[1:], 1):
            current_state = update["model_state"]
            current_keys = set(current_state.keys())
            
            # 检查参数名称一致性
            if current_keys != reference_keys:
                logger.error(f"Client {i} has inconsistent parameter names")
                return False
                
            # 检查参数形状一致性
            for key in reference_keys:
                ref_shape = reference_state[key].shape
                cur_shape = current_state[key].shape
                if ref_shape != cur_shape:
                    logger.error(f"Client {i} parameter {key} has inconsistent shape: "
                               f"{cur_shape} vs {ref_shape}")
                    return False
                    
        return True
    
    def weight_updates(self, updates: List[Dict[str, torch.Tensor]]) -> List[float]:
        """
        计算客户端权重（FedAvg基于数据量的加权）
        
        Args:
            updates: 客户端更新列表，需包含num_samples字段
            
        Returns:
            List[float]: 标准化的权重列表
        """
        if not self.weighted_average:
            # 均匀权重
            num_clients = len(updates)
            return [1.0 / num_clients] * num_clients
        
        # 基于数据量的权重
        num_samples = []
        for update in updates:
            if "num_samples" in update:
                num_samples.append(update["num_samples"])
            else:
                # 如果没有num_samples，使用默认值1
                num_samples.append(1)
                
        total_samples = sum(num_samples)
        if total_samples == 0:
            # 防止除零错误
            num_clients = len(updates)
            return [1.0 / num_clients] * num_clients
            
        weights = [samples / total_samples for samples in num_samples]
        return weights
