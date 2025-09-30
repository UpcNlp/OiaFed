"""
FedYogi 聚合器

实现 FedYogi 聚合算法，基于Yogi优化器的联邦学习版本。
提供比FedAdam更稳定的自适应聚合。

论文：Adaptive Federated Optimization  
作者：Sashank J. Reddi et al.
发表：ICLR 2021
"""

import torch
from typing import Dict, List, Any
from loguru import logger

from ...api.decorators import aggregator


@aggregator("fedyogi", description="FedYogi自适应联邦聚合器")
class FedYogiAggregator:
    """FedYogi 聚合器实现"""
    
    def __init__(self, config: Dict[str, Any] = None, **kwargs):
        self.config = config or {}
        self.server_lr = self.config.get("server_lr", 1e-2)
        self.beta1 = self.config.get("beta1", 0.9)
        self.beta2 = self.config.get("beta2", 0.99)
        self.eps = self.config.get("eps", 1e-3)
        
        self.device = self.config.get("device", "auto")
        if self.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Yogi状态
        self.m_state = {}
        self.v_state = {}
        self.round_count = 0
        
        self.global_model = kwargs.get("global_model")
        logger.info(f"✅ FedYogi聚合器初始化完成 - LR: {self.server_lr}")
    
    def aggregate(self, client_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """执行FedYogi聚合"""
        if not client_updates:
            raise ValueError("没有客户端更新可聚合")
        
        self.round_count += 1
        
        # 计算聚合梯度
        aggregated_gradient = self._compute_pseudo_gradient(client_updates)
        
        # 初始化Yogi状态
        if not self.m_state:
            self._initialize_yogi_states(aggregated_gradient)
        
        # Yogi更新
        updated_weights = self._yogi_update(aggregated_gradient)
        
        return {
            "aggregated_weights": updated_weights,
            "algorithm": "FedYogi",
            "round": self.round_count
        }
    
    def _compute_pseudo_gradient(self, client_updates: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """计算伪梯度"""
        total_samples = sum(update.get("num_samples", 1) for update in client_updates)
        pseudo_gradient = {}
        
        first_weights = client_updates[0]["model_weights"]
        for param_name in first_weights:
            pseudo_gradient[param_name] = torch.zeros_like(
                first_weights[param_name], device=self.device
            )
            
            for update in client_updates:
                weight = update.get("num_samples", 1) / total_samples
                param_value = update["model_weights"][param_name].to(self.device)
                pseudo_gradient[param_name] += weight * param_value
        
        return pseudo_gradient
    
    def _initialize_yogi_states(self, gradient: Dict[str, torch.Tensor]):
        """初始化Yogi状态"""
        for param_name, grad in gradient.items():
            self.m_state[param_name] = torch.zeros_like(grad, device=self.device)
            self.v_state[param_name] = torch.zeros_like(grad, device=self.device)
    
    def _yogi_update(self, gradient: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """执行Yogi更新"""
        updated_weights = {}
        
        for param_name, grad in gradient.items():
            # 更新一阶矩
            self.m_state[param_name] = self.beta1 * self.m_state[param_name] + (1 - self.beta1) * grad
            
            # Yogi的二阶矩更新（与Adam不同）
            grad_squared = grad ** 2
            v_diff = grad_squared - self.v_state[param_name]
            self.v_state[param_name] = self.v_state[param_name] + (1 - self.beta2) * torch.sign(v_diff) * v_diff
            
            # 偏置修正
            m_corrected = self.m_state[param_name] / (1 - self.beta1 ** self.round_count)
            v_corrected = self.v_state[param_name] / (1 - self.beta2 ** self.round_count)
            
            # Yogi更新
            updated_weights[param_name] = grad - self.server_lr * m_corrected / (torch.sqrt(torch.abs(v_corrected)) + self.eps)
        
        return updated_weights
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "algorithm": "FedYogi",
            "server_lr": self.server_lr, 
            "rounds": self.round_count
        }