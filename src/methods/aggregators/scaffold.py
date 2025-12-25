"""
SCAFFOLD 聚合器

从 methods/aggregators/scaffold.py 迁移到 src/

实现 SCAFFOLD (Stochastic Controlled Averaging for Federated Learning) 聚合算法。
使用控制变量减少客户端漂移,提高联邦学习的收敛速度。

论文: SCAFFOLD: Stochastic Controlled Averaging for Federated Learning
作者: Sai Praneeth Karimireddy et al.
发表: ICML 2020

算法特点:
1. 使用控制变量纠正客户端和服务器的更新偏差
2. 更好的收敛保证,特别是在数据异构情况下
3. 需要额外存储和传输控制变量
4. 适合数据分布差异较大的联邦学习场景
"""

import torch
from typing import List, Dict, Any, Optional
from loguru import logger

from ...core.aggregator import Aggregator
from ...core.types import ClientUpdate
from ...registry import aggregator


@aggregator(
    name='scaffold',
    description='SCAFFOLD控制变量联邦聚合器',
    version='1.0'
)
class SCAFFOLDAggregator(Aggregator):
    """
    SCAFFOLD 聚合器实现

    算法核心:
    1. 维护全局控制变量 c
    2. 每个客户端维护本地控制变量 c_i
    3. 客户端更新考虑控制变量的梯度修正
    4. 服务器聚合时同时更新模型和控制变量

    参数:
    - learning_rate: 全局学习率,默认1.0
    - control_lr: 控制变量学习率,默认None(自动计算)
    - weighted: 是否按样本数量加权,默认True
    - momentum: 动量系数,默认0.0
    """

    def __init__(self, learning_rate: float = 1.0, control_lr: Optional[float] = None,
                 weighted: bool = True, momentum: float = 0.0, **kwargs):
        """初始化SCAFFOLD聚合器"""
        # SCAFFOLD特定参数
        self.learning_rate = learning_rate
        self.control_lr = control_lr  # 自动计算
        self._weighted = weighted
        self.momentum = momentum

        # 设备配置
        device = kwargs.get("device", "auto")
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # 控制变量状态
        self.global_control_variate: Optional[Dict[str, torch.Tensor]] = None
        self.client_control_variates: Dict[str, Dict[str, torch.Tensor]] = {}

        # 统计信息
        self.round_count = 0
        self.control_variate_norm_history = []

        logger.info(f"✅ SCAFFOLD聚合器初始化完成 - LR: {self.learning_rate}, 动量: {self.momentum}")

    def aggregate(self, updates: List[ClientUpdate], global_model=None) -> Dict[str, torch.Tensor]:
        """
        执行SCAFFOLD聚合

        Args:
            updates: 客户端更新列表 (List[ClientUpdate])
            global_model: 全局模型 (可选)

        Returns:
            聚合后的模型权重字典
        """
        if not updates:
            raise ValueError("没有客户端更新可聚合")

        self.round_count += 1
        logger.debug(f"🔄 SCAFFOLD聚合轮次 {self.round_count} - {len(updates)} 个客户端")

        # 1. 初始化全局控制变量(如果是第一次)
        if self.global_control_variate is None:
            self._initialize_global_control_variate(updates[0])

        # 2. 计算聚合权重
        weights = self._compute_aggregation_weights(updates)

        # 3. 聚合模型参数
        aggregated_weights = self._aggregate_model_weights(updates, weights)

        # 4. 更新控制变量
        control_stats = self._update_control_variates(updates, weights)

        logger.debug(f"✅ SCAFFOLD聚合完成 - 控制变量范数: {control_stats.get('global_cv_norm', 0):.6f}")

        return aggregated_weights

    def _initialize_global_control_variate(self, sample_update: ClientUpdate):
        """初始化全局控制变量"""
        model_weights = sample_update.weights
        self.global_control_variate = {}

        for param_name, param_value in model_weights.items():
            if isinstance(param_value, torch.Tensor):
                self.global_control_variate[param_name] = torch.zeros_like(
                    param_value, device=self.device
                )
            else:
                self.global_control_variate[param_name] = 0.0

        logger.debug("🔧 全局控制变量已初始化")

    def _compute_aggregation_weights(self, updates: List[ClientUpdate]) -> List[float]:
        """计算聚合权重"""
        if not self._weighted:
            num_clients = len(updates)
            return [1.0 / num_clients] * num_clients

        sample_counts = [update.num_samples for update in updates]
        total_samples = sum(sample_counts)

        if total_samples == 0:
            num_clients = len(updates)
            return [1.0 / num_clients] * num_clients

        return [count / total_samples for count in sample_counts]

    def _aggregate_model_weights(self, updates: List[ClientUpdate],
                                weights: List[float]) -> Dict[str, torch.Tensor]:
        """聚合模型权重"""
        aggregated_weights = {}

        # 获取参数结构
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

    def _update_control_variates(self, updates: List[ClientUpdate],
                               weights: List[float]) -> Dict[str, float]:
        """更新控制变量"""
        control_stats = {}

        # 计算控制变量学习率
        if self.control_lr is None:
            # 自动计算: 基于平均本地epoch数
            avg_local_epochs = 1.0
            if hasattr(updates[0], 'metadata') and updates[0].metadata:
                local_epochs_list = [u.metadata.get("local_epochs", 1) for u in updates if u.metadata]
                if local_epochs_list:
                    avg_local_epochs = sum(local_epochs_list) / len(local_epochs_list)
            effective_lr = self.learning_rate / avg_local_epochs
        else:
            effective_lr = self.control_lr

        # 更新全局控制变量
        control_variate_deltas = {}

        # 聚合控制变量增量
        for param_name in self.global_control_variate.keys():
            control_variate_deltas[param_name] = torch.zeros_like(
                self.global_control_variate[param_name], device=self.device
            )

        for i, update in enumerate(updates):
            client_id = update.node_id if hasattr(update, 'node_id') else f"client_{i}"
            weight = weights[i]

            # 获取客户端控制变量增量
            if hasattr(update, 'metadata') and update.metadata and "control_variate_delta" in update.metadata:
                cv_delta = update.metadata["control_variate_delta"]

                for param_name in control_variate_deltas.keys():
                    if param_name in cv_delta:
                        delta_value = cv_delta[param_name]
                        if isinstance(delta_value, torch.Tensor):
                            delta_value = delta_value.to(self.device)
                            control_variate_deltas[param_name] += weight * delta_value

            # 更新客户端控制变量缓存
            if hasattr(update, 'metadata') and update.metadata and "control_variate" in update.metadata:
                self.client_control_variates[client_id] = update.metadata["control_variate"]

        # 应用控制变量更新
        global_cv_norm = 0.0
        for param_name, delta in control_variate_deltas.items():
            if isinstance(delta, torch.Tensor):
                # 使用动量更新
                if self.momentum > 0:
                    self.global_control_variate[param_name] = (
                        self.momentum * self.global_control_variate[param_name] +
                        (1 - self.momentum) * effective_lr * delta
                    )
                else:
                    self.global_control_variate[param_name] += effective_lr * delta

                # 计算范数 - 转换为float以便计算norm
                cv_tensor = self.global_control_variate[param_name]
                if cv_tensor.dtype in [torch.long, torch.int, torch.int32, torch.int64]:
                    cv_tensor = cv_tensor.float()
                global_cv_norm += torch.norm(cv_tensor).item() ** 2

        global_cv_norm = global_cv_norm ** 0.5
        self.control_variate_norm_history.append(global_cv_norm)

        control_stats = {
            "global_cv_norm": global_cv_norm,
            "effective_control_lr": effective_lr,
            "num_client_cv_updates": len([u for u in updates if hasattr(u, 'metadata') and u.metadata and "control_variate_delta" in u.metadata])
        }

        return control_stats

    def get_client_control_variate(self, client_id: str) -> Optional[Dict[str, torch.Tensor]]:
        """获取指定客户端的控制变量"""
        return self.client_control_variates.get(client_id)

    def get_control_variate_trend(self) -> List[float]:
        """获取控制变量范数的历史趋势"""
        return self.control_variate_norm_history.copy()

    def get_stats(self) -> Dict[str, Any]:
        """获取聚合器统计信息"""
        stats = {
            "algorithm": "SCAFFOLD",
            "learning_rate": self.learning_rate,
            "control_lr": self.control_lr,
            "momentum": self.momentum,
            "total_rounds": self.round_count,
            "num_registered_clients": len(self.client_control_variates),
            "device": str(self.device)
        }

        # 添加控制变量统计
        if self.control_variate_norm_history:
            stats["latest_cv_norm"] = self.control_variate_norm_history[-1]
            stats["avg_cv_norm"] = sum(self.control_variate_norm_history) / len(self.control_variate_norm_history)

            if len(self.control_variate_norm_history) > 1:
                trend = "increasing" if (self.control_variate_norm_history[-1] >
                                       self.control_variate_norm_history[0]) else "decreasing"
                stats["cv_norm_trend"] = trend

        return stats

    def reset_stats(self):
        """重置统计信息"""
        self.round_count = 0
        self.control_variate_norm_history.clear()
        self.client_control_variates.clear()
        self.global_control_variate = None
        logger.info("🔄 SCAFFOLD聚合器统计信息已重置")

    def __repr__(self) -> str:
        return (f"SCAFFOLDAggregator(lr={self.learning_rate}, momentum={self.momentum}, "
                f"rounds={self.round_count})")
