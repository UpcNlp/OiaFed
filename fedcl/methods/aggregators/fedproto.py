"""
FedProto 聚合器

实现 FedProto (Federated Prototypical Learning) 聚合算法。
除了聚合模型权重，还需要聚合各个客户端的类别原型（prototypes）。

论文：FedProto: Federated Prototype Learning across Heterogeneous Clients
作者：Yue Tan et al.
发表：AAAI 2022

算法特点：
1. 使用FedAvg聚合模型权重
2. 聚合客户端的类别原型（按样本数加权）
3. 原型用于客户端的知识蒸馏
"""

import torch
from typing import Dict, List, Any
from loguru import logger

from ...api.decorators import aggregator


@aggregator("fedproto", description="FedProto原型聚合器")
class FedProtoAggregator:
    """
    FedProto 聚合器实现

    执行两个层面的聚合：
    1. 模型权重聚合：使用FedAvg加权平均
    2. 原型聚合：聚合各客户端的类别原型

    原型聚合公式：
    proto_global[c] = Σ(n_k * proto_k[c]) / Σ(n_k)

    其中：
    - proto_k[c]: 客户端k对类别c的原型
    - n_k: 客户端k中类别c的样本数量
    - proto_global[c]: 类别c的全局原型

    参数：
    - weighted: 是否按样本数量加权，默认True
    - device: 计算设备，默认自动检测
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs):
        """初始化FedProto聚合器"""
        self.config = config or {}

        # 聚合配置
        self.weighted = self.config.get("weighted", True)
        self.device = self.config.get("device", "auto")

        # 自动检测设备
        if self.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 统计信息
        self.round_count = 0
        self.total_aggregations = 0

        logger.info(f"✅ FedProto聚合器初始化完成 - 加权: {self.weighted}, 设备: {self.device}")

    def aggregate(self, client_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        执行FedProto聚合

        Args:
            client_updates: 客户端更新列表，每个元素包含：
                - model_weights: 模型权重字典
                - num_samples: 样本数量
                - client_id: 客户端ID
                - prototypes: 本地原型字典 {class_id: prototype_tensor}

        Returns:
            聚合结果字典，包含：
                - aggregated_weights: 聚合后的模型权重
                - global_prototypes: 聚合后的全局原型
                - total_samples: 总样本数
                - num_participants: 参与客户端数量
                - aggregation_weights: 各客户端权重
        """
        if not client_updates:
            raise ValueError("没有客户端更新可聚合")

        # 验证必需字段
        for i, update in enumerate(client_updates):
            if "model_weights" not in update:
                raise ValueError(f"客户端更新 {i} 缺少必需字段 'model_weights'")
            if "num_samples" not in update:
                raise ValueError(f"客户端更新 {i} 缺少必需字段 'num_samples'")

        self.round_count += 1
        self.total_aggregations += 1

        logger.debug(f"FedProto聚合轮次 {self.round_count} - {len(client_updates)} 个客户端")

        # 1. 计算聚合权重
        weights = self._compute_aggregation_weights(client_updates)

        # 2. 聚合模型权重（使用FedAvg）
        aggregated_weights = self._aggregate_model_weights(client_updates, weights)

        # 3. 聚合原型
        global_prototypes = self._aggregate_prototypes(client_updates, weights)

        # 4. 计算基础统计
        total_samples = sum(update["num_samples"] for update in client_updates)

        # 5. 构建结果
        result = {
            "aggregated_weights": aggregated_weights,
            "global_prototypes": global_prototypes,  # FedProto特有
            "total_samples": total_samples,
            "num_participants": len(client_updates),
            "aggregation_weights": {
                update.get("client_id", f"client_{i}"): weights[i]
                for i, update in enumerate(client_updates)
            },
            "algorithm": "FedProto",
            "round": self.round_count
        }

        logger.debug(
            f"✅ FedProto聚合完成 - 总样本: {total_samples}, "
            f"全局原型数: {len(global_prototypes)}"
        )
        return result

    def _compute_aggregation_weights(self, client_updates: List[Dict[str, Any]]) -> List[float]:
        """计算聚合权重"""
        if not self.weighted:
            # 均等权重
            num_clients = len(client_updates)
            return [1.0 / num_clients] * num_clients

        # 按样本数量加权
        sample_counts = [update["num_samples"] for update in client_updates]
        total_samples = sum(sample_counts)

        if total_samples == 0:
            raise ValueError("所有客户端的样本数都为0，无法进行加权聚合")

        weights = [count / total_samples for count in sample_counts]
        return weights

    def _aggregate_model_weights(self, client_updates: List[Dict[str, Any]],
                                 weights: List[float]) -> Dict[str, torch.Tensor]:
        """聚合模型权重（使用FedAvg）"""
        aggregated_weights = {}

        # 获取参数结构
        first_update = client_updates[0]
        model_weights = first_update["model_weights"]
        param_names = list(model_weights.keys())

        # 初始化聚合权重
        for param_name in param_names:
            param_shape = model_weights[param_name].shape
            aggregated_weights[param_name] = torch.zeros(param_shape, device=self.device)

        # 加权聚合
        for i, update in enumerate(client_updates):
            client_weights = update["model_weights"]
            client_weight = weights[i]

            for param_name in param_names:
                if param_name not in client_weights:
                    logger.warning(f"客户端 {update.get('client_id', i)} 缺少参数 {param_name}")
                    continue

                # 将参数移到正确设备并加权
                param_value = client_weights[param_name]
                if isinstance(param_value, torch.Tensor):
                    param_value = param_value.to(self.device)
                    aggregated_weights[param_name] += client_weight * param_value
                else:
                    aggregated_weights[param_name] += client_weight * param_value

        return aggregated_weights

    def _aggregate_prototypes(self, client_updates: List[Dict[str, Any]],
                             weights: List[float]) -> Dict[int, torch.Tensor]:
        """
        聚合客户端原型

        原型聚合策略：
        - 对于每个类别，收集所有拥有该类别的客户端的原型
        - 按照客户端权重进行加权平均
        - 如果某个客户端没有某个类别的样本，则跳过该客户端对该类别的贡献
        """
        global_prototypes = {}

        # 收集所有出现的类别
        all_classes = set()
        for update in client_updates:
            if "prototypes" in update and update["prototypes"]:
                all_classes.update(update["prototypes"].keys())

        if not all_classes:
            logger.warning("没有客户端提供原型，返回空原型字典")
            return {}

        logger.debug(f"  聚合 {len(all_classes)} 个类别的原型")

        # 对每个类别进行聚合
        for class_id in all_classes:
            class_prototypes = []
            class_weights = []

            # 收集该类别的所有客户端原型
            for i, update in enumerate(client_updates):
                if "prototypes" not in update or not update["prototypes"]:
                    continue

                prototypes = update["prototypes"]

                # 如果该客户端有这个类别的原型
                if class_id in prototypes:
                    proto = prototypes[class_id]

                    # 跳过零向量（表示该客户端没有该类别的样本）
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

            # 计算该类别的全局原型（加权平均）
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
            "weighted": self.weighted,
            "device": str(self.device)
        }

    def reset_stats(self):
        """重置统计信息"""
        self.round_count = 0
        self.total_aggregations = 0
        logger.info("🔄 FedProto聚合器统计信息已重置")

    def __repr__(self) -> str:
        return f"FedProtoAggregator(weighted={self.weighted}, device={self.device}, rounds={self.round_count})"
