"""
FedNova 聚合器

从 methods/aggregators/fednova.py 迁移到 src/

实现 FedNova (Federated Optimization in Heterogeneous Networks) 聚合算法。
通过标准化客户端更新来处理客户端异构性问题。

论文: Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization
作者: Jianyu Wang et al.
发表: NeurIPS 2020

算法特点:
1. 标准化不同客户端的本地更新步数
2. 更好地处理系统异构性
3. 改善收敛速度和稳定性
"""

import torch
from typing import List, Dict, Any
from loguru import logger

from ...core.aggregator import Aggregator
from ...core.types import ClientUpdate
from ...registry import aggregator


@aggregator(
    name='fednova',
    description='FedNova标准化联邦聚合器',
    version='1.0'
)
class FedNovaAggregator(Aggregator):
    """FedNova 聚合器实现"""

    def __init__(self, **kwargs):
        """初始化FedNova聚合器"""
        device = kwargs.get("device", "auto")
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.round_count = 0

        logger.info("✅ FedNova聚合器初始化完成")

    def aggregate(self, updates: List[ClientUpdate], global_model=None) -> Dict[str, torch.Tensor]:
        """
        执行FedNova聚合

        Args:
            updates: 客户端更新列表 (List[ClientUpdate])
            global_model: 全局模型 (可选)

        Returns:
            聚合后的模型权重字典
        """
        if not updates:
            raise ValueError("没有客户端更新可聚合")

        self.round_count += 1

        # 计算有效批次数进行标准化
        effective_batches = []
        for update in updates:
            local_epochs = 1
            if hasattr(update, 'metadata') and update.metadata:
                local_epochs = update.metadata.get("local_epochs", 1)
            effective_batches.append(local_epochs * update.num_samples)

        total_effective = sum(effective_batches)

        # 标准化权重
        weights = [eff / total_effective for eff in effective_batches]

        # 聚合
        aggregated_weights = {}
        first_weights = updates[0].weights

        for param_name in first_weights:
            # 创建float类型的零张量，避免Long类型转换错误
            aggregated_weights[param_name] = torch.zeros_like(
                first_weights[param_name], dtype=torch.float32, device=self.device
            )

            for i, update in enumerate(updates):
                param_value = update.weights[param_name].to(self.device)
                # 如果原始参数是Long类型，将其转换为float以便聚合
                if param_value.dtype in [torch.long, torch.int, torch.int32, torch.int64]:
                    param_value = param_value.float()
                aggregated_weights[param_name] += weights[i] * param_value

            # 如果原始参数是Long类型，将聚合结果转换回Long
            if first_weights[param_name].dtype in [torch.long, torch.int, torch.int32, torch.int64]:
                aggregated_weights[param_name] = aggregated_weights[param_name].long()

        logger.debug(f"✅ FedNova聚合完成 - 轮次 {self.round_count}")

        return aggregated_weights

    def get_stats(self) -> Dict[str, Any]:
        """获取聚合器统计信息"""
        return {
            "algorithm": "FedNova",
            "rounds": self.round_count
        }
