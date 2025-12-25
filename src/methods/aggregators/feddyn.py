"""
FedDyn 聚合器

从 methods/aggregators/feddyn.py 迁移到 src/

实现 FedDyn (Federated Learning with Dynamic Regularization) 聚合算法。
通过动态正则化项改善联邦学习的收敛性。

论文: Federated Learning with Only Positive Labels
作者: Felix X. Yu et al.
发表: ICML 2020
"""

import torch
from typing import List, Dict, Any
from loguru import logger

from ...core.aggregator import Aggregator
from ...core.types import ClientUpdate
from ...registry import aggregator


@aggregator(
    name='feddyn',
    description='FedDyn动态正则化联邦聚合器',
    version='1.0'
)
class FedDynAggregator(Aggregator):
    """FedDyn 聚合器实现"""

    def __init__(self, alpha: float = 0.01, **kwargs):
        """初始化FedDyn聚合器"""
        self.alpha = alpha  # 动态正则化系数

        device = kwargs.get("device", "auto")
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # 状态变量
        self.h_state = {}  # 动态正则化状态
        self.round_count = 0

        logger.info(f"✅ FedDyn聚合器初始化完成 - α: {self.alpha}")

    def aggregate(self, updates: List[ClientUpdate], global_model=None) -> Dict[str, torch.Tensor]:
        """
        执行FedDyn聚合

        Args:
            updates: 客户端更新列表 (List[ClientUpdate])
            global_model: 全局模型 (可选)

        Returns:
            聚合后的模型权重字典
        """
        if not updates:
            raise ValueError("没有客户端更新可聚合")

        self.round_count += 1

        # 计算加权平均
        total_samples = sum(update.num_samples for update in updates)
        weights = [update.num_samples / total_samples for update in updates]

        # 聚合权重
        aggregated_weights = {}
        first_weights = updates[0].weights

        for param_name in first_weights:
            # 使用float32避免Long类型转换错误
            aggregated_weights[param_name] = torch.zeros_like(
                first_weights[param_name], dtype=torch.float32, device=self.device
            )

            for i, update in enumerate(updates):
                param_value = update.weights[param_name].to(self.device)
                # 转换整数类型为float
                if param_value.dtype in [torch.long, torch.int, torch.int32, torch.int64]:
                    param_value = param_value.float()
                aggregated_weights[param_name] += weights[i] * param_value

        # 初始化h状态
        if not self.h_state:
            for param_name in aggregated_weights:
                # 使用float32避免Long类型转换错误
                self.h_state[param_name] = torch.zeros_like(
                    aggregated_weights[param_name], dtype=torch.float32, device=self.device
                )

        # 更新h状态和全局模型
        for param_name in aggregated_weights:
            self.h_state[param_name] += self.alpha * aggregated_weights[param_name]
            aggregated_weights[param_name] -= (1.0 / self.alpha) * self.h_state[param_name]

            # 转换回原始类型
            if first_weights[param_name].dtype in [torch.long, torch.int, torch.int32, torch.int64]:
                aggregated_weights[param_name] = aggregated_weights[param_name].long()

        logger.debug(f"✅ FedDyn聚合完成 - 轮次 {self.round_count}")

        return aggregated_weights

    def get_stats(self) -> Dict[str, Any]:
        """获取聚合器统计信息"""
        return {
            "algorithm": "FedDyn",
            "alpha": self.alpha,
            "rounds": self.round_count
        }
