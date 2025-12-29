"""
FedYogi 聚合器

从 methods/aggregators/fedyogi.py 迁移到 src/

实现 FedYogi 聚合算法,基于Yogi优化器的联邦学习版本。
提供比FedAdam更稳定的自适应聚合。

论文: Adaptive Federated Optimization
作者: Sashank J. Reddi et al.
发表: ICLR 2021
"""

import torch
from typing import List, Dict, Any
from loguru import logger

from ...core.aggregator import Aggregator
from ...core.types import ClientUpdate
from ...registry import aggregator


@aggregator(
    name='fedyogi',
    description='FedYogi自适应联邦聚合器',
    version='1.0'
)
class FedYogiAggregator(Aggregator):
    """FedYogi 聚合器实现"""

    def __init__(self, server_lr: float = 1e-2, beta1: float = 0.9,
                 beta2: float = 0.99, eps: float = 1e-3, **kwargs):
        """初始化FedYogi聚合器"""
        self.server_lr = server_lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        device = kwargs.get("device", "auto")
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Yogi状态
        self.m_state = {}
        self.v_state = {}
        self.round_count = 0

        # 全局模型权重(需要用于计算梯度)
        self.global_weights = None

        logger.info(f"✅ FedYogi聚合器初始化完成 - LR: {self.server_lr}")

    def aggregate(self, updates: List[ClientUpdate], global_model=None) -> Dict[str, torch.Tensor]:
        """
        执行FedYogi聚合

        Args:
            updates: 客户端更新列表 (List[ClientUpdate])
            global_model: 全局模型 (可选)

        Returns:
            聚合后的模型权重字典
        """
        if not updates:
            raise ValueError("没有客户端更新可聚合")

        self.round_count += 1

        # 初始化全局权重(第一轮)
        if self.global_weights is None:
            first_weights = updates[0].weights
            self.global_weights = {}
            for k, v in first_weights.items():
                v_tensor = v.clone().to(self.device)
                # 转换为float以便后续计算
                if v_tensor.dtype in [torch.long, torch.int, torch.int32, torch.int64]:
                    v_tensor = v_tensor.float()
                self.global_weights[k] = v_tensor

        # 计算伪梯度: Δ = weighted_avg(client_weights) - global_weights
        aggregated_delta = self._compute_pseudo_gradient(updates)

        # 初始化Yogi状态
        if not self.m_state:
            self._initialize_yogi_states(aggregated_delta)

        # Yogi更新全局权重
        self.global_weights = self._yogi_update(aggregated_delta)

        # 转换回原始类型
        result = {}
        first_weights = updates[0].weights
        for param_name, param_value in self.global_weights.items():
            if first_weights[param_name].dtype in [torch.long, torch.int, torch.int32, torch.int64]:
                result[param_name] = param_value.long()
            else:
                result[param_name] = param_value

        return result

    def _compute_pseudo_gradient(self, updates: List[ClientUpdate]) -> Dict[str, torch.Tensor]:
        """
        计算伪梯度: Δ = Σ(w_i * client_weights_i) - global_weights
        这代表客户端权重加权平均与当前全局权重的差值
        """
        total_samples = sum(update.num_samples for update in updates)
        weighted_avg = {}

        first_weights = updates[0].weights
        for param_name in first_weights:
            # 使用float32避免Long类型转换错误
            weighted_avg[param_name] = torch.zeros_like(
                first_weights[param_name], dtype=torch.float32, device=self.device
            )

            # 计算加权平均
            for update in updates:
                weight = update.num_samples / total_samples
                param_value = update.weights[param_name].to(self.device)
                # 转换整数类型为float
                if param_value.dtype in [torch.long, torch.int, torch.int32, torch.int64]:
                    param_value = param_value.float()
                weighted_avg[param_name] += weight * param_value

            # 如果原始参数是整数类型，转换回去
            if first_weights[param_name].dtype in [torch.long, torch.int, torch.int32, torch.int64]:
                weighted_avg[param_name] = weighted_avg[param_name].long()

        # 计算伪梯度: Δ = weighted_avg - global_weights
        pseudo_gradient = {}
        for param_name in weighted_avg:
            pseudo_gradient[param_name] = weighted_avg[param_name] - self.global_weights[param_name]

        return pseudo_gradient

    def _initialize_yogi_states(self, gradient: Dict[str, torch.Tensor]):
        """初始化Yogi状态"""
        for param_name, grad in gradient.items():
            self.m_state[param_name] = torch.zeros_like(grad, device=self.device)
            self.v_state[param_name] = torch.zeros_like(grad, device=self.device)

    def _yogi_update(self, delta: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        执行Yogi更新
        global_weights = global_weights + server_lr * m / (sqrt(|v|) + eps)
        """
        updated_weights = {}

        for param_name, grad in delta.items():
            # 更新一阶矩
            self.m_state[param_name] = self.beta1 * self.m_state[param_name] + (1 - self.beta1) * grad

            # Yogi的二阶矩更新(与Adam不同)
            grad_squared = grad ** 2
            v_diff = grad_squared - self.v_state[param_name]
            self.v_state[param_name] = self.v_state[param_name] + (1 - self.beta2) * torch.sign(v_diff) * v_diff

            # 偏置修正
            m_corrected = self.m_state[param_name] / (1 - self.beta1 ** self.round_count)
            v_corrected = self.v_state[param_name] / (1 - self.beta2 ** self.round_count)

            # Yogi更新: w = w + lr * m / (sqrt(|v|) + eps)
            updated_weights[param_name] = (
                self.global_weights[param_name] +
                self.server_lr * m_corrected / (torch.sqrt(torch.abs(v_corrected)) + self.eps)
            )

        return updated_weights

    def get_stats(self) -> Dict[str, Any]:
        """获取聚合器统计信息"""
        return {
            "algorithm": "FedYogi",
            "server_lr": self.server_lr,
            "rounds": self.round_count
        }
