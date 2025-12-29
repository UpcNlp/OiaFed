"""
FedProx 聚合器

从 methods/aggregators/fedprox.py 迁移到 src/

实现 FedProx (Federated Optimization in Heterogeneous Networks) 聚合算法。
在FedAvg基础上添加正则化项,更好地处理异构数据和系统异构性。

论文: Federated Optimization in Heterogeneous Networks
作者: Tian Li et al.
发表: MLSys 2020

算法特点:
1. 在客户端本地目标函数中添加近端项
2. 约束客户端模型不要偏离全局模型太远
3. 更好的收敛性和稳定性
4. 适合数据异构的联邦学习场景
"""

import torch
from typing import List, Dict, Any, Optional
from loguru import logger

from ...core.aggregator import Aggregator
from ...core.types import ClientUpdate
from ...registry import aggregator


@aggregator(
    name='fedprox',
    description='FedProx带正则化的联邦聚合器',
    version='1.0'
)
class FedProxAggregator(Aggregator):
    """
    FedProx 聚合器实现

    在FedAvg基础上,考虑近端项的影响:
    客户端损失: L_k(w) + (μ/2)||w - w_t||²

    其中:
    - L_k(w): 客户端k的原始损失函数
    - μ: 近端项系数,控制与全局模型的偏离程度
    - w_t: 当前轮次的全局模型参数

    参数:
    - mu: 近端项系数,默认0.01
    - weighted: 是否按样本数量加权,默认True
    - normalize_weights: 是否标准化权重,默认True
    """

    def __init__(self, mu: float = 0.01, weighted: bool = True,
                 normalize_weights: bool = True, **kwargs):
        """初始化FedProx聚合器"""
        # FedProx特定参数
        self.mu = mu  # 近端项系数
        self._weighted = weighted
        self.normalize_weights = normalize_weights

        # 设备配置
        device = kwargs.get("device", "auto")
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # 聚合历史
        self.round_count = 0
        self.convergence_history = []

        logger.info(f"✅ FedProx聚合器初始化完成 - μ: {self.mu}, 加权: {self._weighted}")

    def aggregate(self, updates: List[ClientUpdate], global_model=None) -> Dict[str, torch.Tensor]:
        """
        执行FedProx聚合

        Args:
            updates: 客户端更新列表 (List[ClientUpdate])
            global_model: 全局模型 (可选)

        Returns:
            聚合后的模型权重字典
        """
        if not updates:
            raise ValueError("没有客户端更新可聚合")

        self.round_count += 1
        logger.debug(f"🔄 FedProx聚合轮次 {self.round_count} - {len(updates)} 个客户端")

        # 1. 计算聚合权重 (考虑近端项影响)
        weights = self._compute_fedprox_weights(updates)

        # 2. 执行加权聚合
        aggregated_weights = self._fedprox_aggregation(updates, weights)

        # 3. 计算收敛指标
        convergence_metrics = self._compute_convergence_metrics(updates, aggregated_weights)
        self.convergence_history.append(convergence_metrics)

        logger.debug(f"✅ FedProx聚合完成 - 收敛度: {convergence_metrics.get('avg_divergence', 0):.6f}")

        return aggregated_weights

    def _compute_fedprox_weights(self, updates: List[ClientUpdate]) -> List[float]:
        """计算FedProx聚合权重"""
        if not self._weighted:
            num_clients = len(updates)
            return [1.0 / num_clients] * num_clients

        # 基础权重: 按样本数量
        sample_counts = [update.num_samples for update in updates]
        total_samples = sum(sample_counts)

        if total_samples == 0:
            num_clients = len(updates)
            return [1.0 / num_clients] * num_clients

        base_weights = [count / total_samples for count in sample_counts]

        # FedProx调整: 考虑近端项的影响
        # 近端项值越小(越接近全局模型),权重越大
        if self.mu > 0 and hasattr(updates[0], 'metadata') and updates[0].metadata:
            adjusted_weights = []
            for i, update in enumerate(updates):
                proximal_term = update.metadata.get("proximal_term", 0.0) if update.metadata else 0.0
                # 使用指数衰减调整权重
                adjustment = torch.exp(torch.tensor(-self.mu * proximal_term))
                adjusted_weights.append(base_weights[i] * adjustment.item())

            # 重新标准化
            if self.normalize_weights:
                total_weight = sum(adjusted_weights)
                if total_weight > 0:
                    adjusted_weights = [w / total_weight for w in adjusted_weights]

            return adjusted_weights

        return base_weights

    def _fedprox_aggregation(self, updates: List[ClientUpdate],
                            weights: List[float]) -> Dict[str, torch.Tensor]:
        """执行FedProx聚合"""
        aggregated_weights = {}

        # 获取模型结构
        first_weights = updates[0].weights
        param_names = list(first_weights.keys())

        # 初始化聚合结果
        for param_name in param_names:
            param_tensor = first_weights[param_name]
            if isinstance(param_tensor, torch.Tensor):
                # 使用float32避免Long类型转换错误
                aggregated_weights[param_name] = torch.zeros_like(param_tensor, dtype=torch.float32, device=self.device)
            else:
                aggregated_weights[param_name] = 0.0

        # 加权聚合
        for i, update in enumerate(updates):
            client_weights = update.weights
            weight = weights[i]

            for param_name in param_names:
                if param_name in client_weights:
                    param_value = client_weights[param_name]

                    if isinstance(param_value, torch.Tensor):
                        param_value = param_value.to(self.device)
                        # 转换整数类型为float
                        if param_value.dtype in [torch.long, torch.int, torch.int32, torch.int64]:
                            param_value = param_value.float()
                        aggregated_weights[param_name] += weight * param_value
                    else:
                        aggregated_weights[param_name] += weight * param_value

        # 转换回原始类型
        for param_name in param_names:
            if isinstance(first_weights[param_name], torch.Tensor):
                if first_weights[param_name].dtype in [torch.long, torch.int, torch.int32, torch.int64]:
                    aggregated_weights[param_name] = aggregated_weights[param_name].long()

        return aggregated_weights

    def _compute_convergence_metrics(self, updates: List[ClientUpdate],
                                   aggregated_weights: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """计算收敛指标"""
        metrics = {}

        # 计算客户端模型与聚合模型的差异
        divergences = []

        for update in updates:
            client_weights = update.weights
            divergence = 0.0
            param_count = 0

            for param_name in aggregated_weights.keys():
                if param_name in client_weights:
                    global_param = aggregated_weights[param_name]
                    client_param = client_weights[param_name]

                    if isinstance(global_param, torch.Tensor) and isinstance(client_param, torch.Tensor):
                        client_param = client_param.to(self.device)
                        # 转换为float以便计算norm
                        if global_param.dtype in [torch.long, torch.int, torch.int32, torch.int64]:
                            global_param = global_param.float()
                        if client_param.dtype in [torch.long, torch.int, torch.int32, torch.int64]:
                            client_param = client_param.float()
                        diff = torch.norm(global_param - client_param).item()
                        divergence += diff
                        param_count += 1

            if param_count > 0:
                divergences.append(divergence / param_count)

        if divergences:
            metrics["avg_divergence"] = sum(divergences) / len(divergences)
            metrics["max_divergence"] = max(divergences)
            metrics["min_divergence"] = min(divergences)
            metrics["std_divergence"] = torch.std(torch.tensor(divergences)).item()

        # 近端项统计
        proximal_terms = []
        for update in updates:
            if hasattr(update, 'metadata') and update.metadata:
                proximal_terms.append(update.metadata.get("proximal_term", 0.0))

        if proximal_terms:
            metrics["avg_proximal_term"] = sum(proximal_terms) / len(proximal_terms)
            metrics["max_proximal_term"] = max(proximal_terms)

        return metrics

    def get_convergence_trend(self) -> Dict[str, List[float]]:
        """获取收敛趋势"""
        if not self.convergence_history:
            return {}

        trends = {}
        metric_names = self.convergence_history[0].keys()

        for metric_name in metric_names:
            trends[metric_name] = [round_metrics.get(metric_name, 0.0)
                                 for round_metrics in self.convergence_history]

        return trends

    def get_stats(self) -> Dict[str, Any]:
        """获取聚合器统计信息"""
        stats = {
            "algorithm": "FedProx",
            "mu": self.mu,
            "weighted": self._weighted,
            "total_rounds": self.round_count,
            "device": str(self.device)
        }

        # 添加收敛统计
        if self.convergence_history:
            latest_metrics = self.convergence_history[-1]
            stats["latest_convergence"] = latest_metrics

            # 计算整体趋势
            divergences = [m.get("avg_divergence", 0) for m in self.convergence_history]
            if len(divergences) > 1:
                stats["convergence_trend"] = "improving" if divergences[-1] < divergences[0] else "degrading"

        return stats

    def reset_stats(self):
        """重置统计信息"""
        self.round_count = 0
        self.convergence_history.clear()
        logger.info("🔄 FedProx聚合器统计信息已重置")

    def __repr__(self) -> str:
        return f"FedProxAggregator(mu={self.mu}, weighted={self._weighted}, rounds={self.round_count})"
