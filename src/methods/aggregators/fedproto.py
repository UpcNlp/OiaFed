"""
FedProto 聚合器

从 methods/aggregators/fedproto.py 迁移到 src/

实现 FedProto (Federated Prototypical Learning) 聚合算法。
除了聚合模型权重,还需要聚合各个客户端的类别原型(prototypes)。

论文: FedProto: Federated Prototype Learning across Heterogeneous Clients
作者: Yue Tan et al.
发表: AAAI 2022

算法特点:
1. 使用FedAvg聚合模型权重
2. 聚合客户端的类别原型(按样本数加权)
3. 原型用于客户端的知识蒸馏
"""

import torch
from typing import List, Dict, Any
from loguru import logger

from ...core.aggregator import Aggregator
from ...core.types import ClientUpdate
from ...registry import aggregator


@aggregator(
    name='fedproto',
    description='FedProto原型聚合器',
    version='1.0'
)
class FedProtoAggregator(Aggregator):
    """
    FedProto 聚合器实现

    执行两个层面的聚合:
    1. 模型权重聚合: 使用FedAvg加权平均
    2. 原型聚合: 聚合各客户端的类别原型

    原型聚合公式:
    proto_global[c] = Σ(n_k * proto_k[c]) / Σ(n_k)

    其中:
    - proto_k[c]: 客户端k对类别c的原型
    - n_k: 客户端k中类别c的样本数量
    - proto_global[c]: 类别c的全局原型

    参数:
    - weighted: 是否按样本数量加权,默认True
    - device: 计算设备,默认自动检测
    """

    def __init__(self, weighted: bool = True, **kwargs):
        """初始化FedProto聚合器"""
        # 聚合配置
        self._weighted = weighted

        # 自动检测设备
        device = kwargs.get("device", "auto")
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # 统计信息
        self.round_count = 0
        self.total_aggregations = 0

        logger.info(f"✅ FedProto聚合器初始化完成 - 加权: {self._weighted}, 设备: {self.device}")

    def aggregate(self, updates: List[ClientUpdate], global_model=None) -> Dict[str, Any]:
        """
        执行FedProto聚合

        Args:
            updates: 客户端更新列表 (List[ClientUpdate])
            global_model: 全局模型 (可选)

        Returns:
            聚合结果字典,包含:
            - aggregated_weights: 聚合后的模型权重
            - global_prototypes: 聚合后的全局原型
        """
        if not updates:
            raise ValueError("没有客户端更新可聚合")

        self.round_count += 1
        self.total_aggregations += 1

        logger.debug(f"FedProto聚合轮次 {self.round_count} - {len(updates)} 个客户端")

        # 1. 计算聚合权重
        weights = self._compute_aggregation_weights(updates)

        # 2. 聚合模型权重(使用FedAvg)
        aggregated_weights = self._aggregate_model_weights(updates, weights)

        # 3. 聚合原型
        global_prototypes = self._aggregate_prototypes(updates, weights)

        logger.debug(
            f"✅ FedProto聚合完成 - 总样本: {sum(u.num_samples for u in updates)}, "
            f"全局原型数: {len(global_prototypes)}"
        )

        # 返回字典需要包含global_prototypes字段
        return {
            "weights": aggregated_weights,  # 兼容性: 返回weights字段
            "global_prototypes": global_prototypes  # FedProto特有
        }

    def _compute_aggregation_weights(self, updates: List[ClientUpdate]) -> List[float]:
        """计算聚合权重"""
        if not self._weighted:
            # 均等权重
            num_clients = len(updates)
            return [1.0 / num_clients] * num_clients

        # 按样本数量加权
        sample_counts = [update.num_samples for update in updates]
        total_samples = sum(sample_counts)

        if total_samples == 0:
            raise ValueError("所有客户端的样本数都为0,无法进行加权聚合")

        weights = [count / total_samples for count in sample_counts]
        return weights

    def _aggregate_model_weights(self, updates: List[ClientUpdate],
                                 weights: List[float]) -> Dict[str, torch.Tensor]:
        """聚合模型权重(使用FedAvg)"""
        aggregated_weights = {}

        # 获取参数结构
        first_update = updates[0]
        model_weights = first_update.weights
        param_names = list(model_weights.keys())

        # 初始化聚合权重
        for param_name in param_names:
            param_shape = model_weights[param_name].shape
            aggregated_weights[param_name] = torch.zeros(param_shape, device=self.device)

        # 加权聚合
        for i, update in enumerate(updates):
            client_weights = update.weights
            client_weight = weights[i]

            for param_name in param_names:
                if param_name not in client_weights:
                    logger.warning(f"客户端 {i} 缺少参数 {param_name}")
                    continue

                # 将参数移到正确设备并加权
                param_value = client_weights[param_name]
                if isinstance(param_value, torch.Tensor):
                    param_value = param_value.to(self.device)
                    aggregated_weights[param_name] += client_weight * param_value
                else:
                    aggregated_weights[param_name] += client_weight * param_value

        return aggregated_weights

    def _aggregate_prototypes(self, updates: List[ClientUpdate],
                             weights: List[float]) -> Dict[int, torch.Tensor]:
        """
        聚合客户端原型

        原型聚合策略:
        - 对于每个类别,收集所有拥有该类别的客户端的原型
        - 按照客户端权重进行加权平均
        - 如果某个客户端没有某个类别的样本,则跳过该客户端对该类别的贡献
        """
        global_prototypes = {}

        # 收集所有出现的类别
        all_classes = set()
        for update in updates:
            if hasattr(update, 'metadata') and update.metadata and "prototypes" in update.metadata:
                prototypes = update.metadata["prototypes"]
                if prototypes:
                    all_classes.update(prototypes.keys())

        if not all_classes:
            logger.warning("没有客户端提供原型,返回空原型字典")
            return {}

        logger.debug(f"  聚合 {len(all_classes)} 个类别的原型")

        # 对每个类别进行聚合
        for class_id in all_classes:
            class_prototypes = []
            class_weights = []

            # 收集该类别的所有客户端原型
            for i, update in enumerate(updates):
                if not (hasattr(update, 'metadata') and update.metadata and "prototypes" in update.metadata):
                    continue

                prototypes = update.metadata["prototypes"]

                # 如果该客户端有这个类别的原型
                if class_id in prototypes:
                    proto = prototypes[class_id]

                    # 跳过零向量(表示该客户端没有该类别的样本)
                    if isinstance(proto, torch.Tensor):
                        if proto.sum().item() != 0:
                            class_prototypes.append(proto)
                            class_weights.append(weights[i])
                    else:
                        # 处理numpy数组
                        proto_tensor = torch.tensor(proto)
                        if proto_tensor.sum().item() != 0:
                            class_prototypes.append(proto_tensor)
                            class_weights.append(weights[i])

            # 计算该类别的全局原型(加权平均)
            if class_prototypes:
                # 归一化权重
                total_weight = sum(class_weights)
                if total_weight > 0:
                    normalized_weights = [w / total_weight for w in class_weights]

                    # 加权平均
                    global_proto = torch.zeros_like(class_prototypes[0])
                    for proto, weight in zip(class_prototypes, normalized_weights):
                        global_proto += weight * proto.to(global_proto.device)

                    global_prototypes[class_id] = global_proto
                    logger.debug(
                        f"    类别 {class_id}: {len(class_prototypes)} 个客户端贡献原型"
                    )
            else:
                logger.warning(f"    类别 {class_id}: 没有有效的客户端原型")

        return global_prototypes

    def get_stats(self) -> Dict[str, Any]:
        """获取聚合器统计信息"""
        return {
            "algorithm": "FedProto",
            "total_rounds": self.round_count,
            "total_aggregations": self.total_aggregations,
            "weighted": self._weighted,
            "device": str(self.device)
        }

    def reset_stats(self):
        """重置统计信息"""
        self.round_count = 0
        self.total_aggregations = 0
        logger.info("🔄 FedProto聚合器统计信息已重置")

    def __repr__(self) -> str:
        return f"FedProtoAggregator(weighted={self._weighted}, device={self.device}, rounds={self.round_count})"
