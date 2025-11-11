"""
FedBN (Federated Learning with Batch Normalization) 聚合器实现
fedcl/methods/aggregators/fedbn.py

论文：FedBN: Federated Learning on Non-IID Features via Local Batch Normalization
作者：Xiaoxiao Li et al.
发表：ICLR 2021

FedBN的核心思想：
- 在聚合时跳过BatchNorm层的参数（running_mean, running_var, weight, bias）
- 只聚合其他层的参数（卷积层、全连接层等）
- 这样可以保留每个客户端的本地数据分布特征
"""

import torch
from typing import Dict, List, Any
from loguru import logger

from ...api.decorators import aggregator


@aggregator("fedbn", description="FedBN: 跳过BatchNorm层的联邦聚合器")
class FedBNAggregator:
    """FedBN 聚合器实现

    在聚合时跳过BatchNorm相关的参数，包括：
    - 包含'bn'的层（如：bn1, bn2, batch_norm等）
    - 包含'running_mean'和'running_var'的参数
    - 包含'num_batches_tracked'的参数
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs):
        self.config = config or {}

        # 配置要跳过的层名称关键词
        self.skip_keywords = self.config.get(
            "skip_keywords",
            ['bn', 'batch_norm', 'batchnorm', 'running_mean', 'running_var', 'num_batches_tracked']
        )

        self.device = self.config.get("device", "auto")
        if self.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.round_count = 0
        self.global_model = kwargs.get("global_model")

        logger.info(f"✅ FedBN聚合器初始化完成 - 跳过包含以下关键词的层: {self.skip_keywords}")

    def _should_skip_param(self, param_name: str) -> bool:
        """判断参数是否应该跳过（不聚合）"""
        param_lower = param_name.lower()
        for keyword in self.skip_keywords:
            if keyword.lower() in param_lower:
                return True
        return False

    def aggregate(self, client_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """执行FedBN聚合"""
        if not client_updates:
            raise ValueError("没有客户端更新可聚合")

        self.round_count += 1

        # 计算总样本数
        total_samples = sum(update.get("num_samples", 1) for update in client_updates)

        # 初始化聚合权重字典
        aggregated_weights = {}

        # 统计跳过和聚合的参数
        skipped_params = []
        aggregated_params = []

        # 获取第一个客户端的参数名称
        first_weights = client_updates[0]["model_weights"]

        for param_name in first_weights:
            # 判断是否跳过该参数
            if self._should_skip_param(param_name):
                # 跳过BatchNorm相关参数，使用第一个客户端的值（或者不包含在结果中）
                # 这里我们选择不包含在聚合结果中，让客户端保留自己的BN参数
                skipped_params.append(param_name)
                continue

            # 聚合非BatchNorm参数
            aggregated_params.append(param_name)
            aggregated_weights[param_name] = torch.zeros_like(
                first_weights[param_name], device=self.device
            )

            # 加权平均
            for update in client_updates:
                weight = update.get("num_samples", 1) / total_samples
                param_value = update["model_weights"][param_name]

                # 转换为tensor（如果是numpy数组）
                if not isinstance(param_value, torch.Tensor):
                    param_value = torch.from_numpy(param_value)

                param_value = param_value.to(self.device)
                aggregated_weights[param_name] += weight * param_value

        logger.info(
            f"  [FedBN Round {self.round_count}] "
            f"聚合了 {len(aggregated_params)} 个参数, "
            f"跳过了 {len(skipped_params)} 个BatchNorm参数"
        )

        if len(skipped_params) > 0:
            logger.debug(f"  跳过的参数示例: {skipped_params[:5]}")

        return {
            "aggregated_weights": aggregated_weights,
            "algorithm": "FedBN",
            "round": self.round_count,
            "num_clients": len(client_updates),
            "aggregated_params": len(aggregated_params),
            "skipped_params": len(skipped_params)
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            "algorithm": "FedBN",
            "rounds": self.round_count,
            "skip_keywords": self.skip_keywords
        }
