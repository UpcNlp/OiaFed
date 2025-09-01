"""
FedAvg 聚合器

实现经典的 FedAvg (Federated Averaging) 聚合算法。
按客户端样本数量进行加权平均聚合，是最基础和广泛使用的联邦学习聚合方法。

论文：Communication-Efficient Learning of Deep Networks from Decentralized Data
作者：H. Brendan McMahan et al.
发表：AISTATS 2017

算法特点：
1. 简单高效的加权平均聚合
2. 按客户端数据量分配权重
3. 适合大多数联邦学习场景
"""

import torch
from typing import Dict, List, Any, Union
from loguru import logger

from ...api.decorators import aggregator


@aggregator("fedavg", description="经典FedAvg加权平均聚合器")
class FedAvgAggregator:
    """
    FedAvg 聚合器实现
    
    执行按样本数量加权的参数平均聚合：
    w_global = Σ(n_k * w_k) / Σ(n_k)
    
    其中：
    - w_k: 客户端k的模型参数
    - n_k: 客户端k的样本数量
    - w_global: 聚合后的全局模型参数
    
    参数：
    - weighted: 是否按样本数量加权，默认True
    - clip_norm: 梯度裁剪阈值，默认None（不裁剪）
    - device: 计算设备，默认自动检测
    """
    
    def __init__(self, config: Dict[str, Any] = None, **kwargs):
        """初始化FedAvg聚合器"""
        self.config = config or {}
        
        # 聚合配置
        self.weighted = self.config.get("weighted", True)
        self.clip_norm = self.config.get("clip_norm", None)
        self.device = self.config.get("device", "auto")
        
        # 自动检测设备
        if self.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 统计信息
        self.round_count = 0
        self.total_aggregations = 0
        
        # 兼容现有框架的参数
        self.global_model = kwargs.get("global_model")
        
        logger.info(f"✅ FedAvg聚合器初始化完成 - 加权: {self.weighted}, 设备: {self.device}")
    
    def aggregate(self, client_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        执行FedAvg聚合
        
        Args:
            client_updates: 客户端更新列表，每个元素必须包含：
                - model_weights: 模型权重字典
                - num_samples: 样本数量  
                - client_id: 客户端ID
                - 其他动态指标（支持任意指标名称）
        
        Returns:
            聚合结果字典，包含：
                - aggregated_weights: 聚合后的模型权重
                - total_samples: 总样本数
                - num_participants: 参与客户端数量
                - aggregation_weights: 各客户端权重
                - 动态聚合的指标
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
        
        logger.debug(f"🔄 FedAvg聚合轮次 {self.round_count} - {len(client_updates)} 个客户端")
        
        # 1. 计算聚合权重
        weights = self._compute_aggregation_weights(client_updates)
        
        # 2. 执行加权聚合
        aggregated_weights = self._fedavg_aggregation(client_updates, weights)
        
        # 3. 聚合所有数值指标（动态支持任意指标）
        aggregated_metrics = self._aggregate_metrics(client_updates, weights)
        
        # 4. 计算基础统计
        total_samples = sum(update["num_samples"] for update in client_updates)
        
        # 5. 构建结果
        result = {
            "aggregated_weights": aggregated_weights,
            "total_samples": total_samples,
            "num_participants": len(client_updates),
            "aggregation_weights": {
                update.get("client_id", f"client_{i}"): weights[i] 
                for i, update in enumerate(client_updates)
            },
            "algorithm": "FedAvg",
            "round": self.round_count
        }
        
        # 添加聚合的指标
        result.update(aggregated_metrics)
        
        logger.debug(f"✅ FedAvg聚合完成 - 总样本: {total_samples}")
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
    
    def _aggregate_metrics(self, client_updates: List[Dict[str, Any]], weights: List[float]) -> Dict[str, Any]:
        """聚合所有数值指标（支持任意指标名称）"""
        aggregated_metrics = {}
        
        # 获取所有可聚合的数值指标
        metric_names = set()
        for update in client_updates:
            for key, value in update.items():
                # 跳过非数值字段
                if key in ["model_weights", "client_id"]:
                    continue
                # 只聚合数值类型
                if isinstance(value, (int, float, torch.Tensor)):
                    if isinstance(value, torch.Tensor) and value.numel() == 1:
                        metric_names.add(key)
                    elif not isinstance(value, torch.Tensor):
                        metric_names.add(key)
        
        # 对每个指标进行加权聚合
        for metric_name in metric_names:
            if metric_name == "num_samples":  # 跳过样本数，已经单独处理
                continue
                
            metric_values = []
            metric_weights = []
            
            for i, update in enumerate(client_updates):
                if metric_name in update:
                    value = update[metric_name]
                    if isinstance(value, torch.Tensor):
                        value = value.item() if value.numel() == 1 else value
                    
                    if isinstance(value, (int, float)):
                        metric_values.append(value)
                        metric_weights.append(weights[i])
            
            # 计算加权平均
            if metric_values:
                weighted_sum = sum(val * weight for val, weight in zip(metric_values, metric_weights))
                weight_sum = sum(metric_weights)
                
                if weight_sum > 0:
                    aggregated_metrics[f"avg_{metric_name}"] = weighted_sum / weight_sum
                    aggregated_metrics[f"total_{metric_name}"] = sum(metric_values)
                    aggregated_metrics[f"min_{metric_name}"] = min(metric_values)
                    aggregated_metrics[f"max_{metric_name}"] = max(metric_values)
        
        return aggregated_metrics
    
    def _fedavg_aggregation(self, client_updates: List[Dict[str, Any]], 
                           weights: List[float]) -> Dict[str, torch.Tensor]:
        """执行FedAvg加权平均聚合"""
        aggregated_weights = {}
        
        # 获取参数结构
        first_update = client_updates[0]
        if "model_weights" not in first_update:
            raise ValueError("客户端更新缺少 model_weights 字段")
        
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
                    
                    # 可选：梯度裁剪
                    if self.clip_norm is not None:
                        param_value = torch.clamp(param_value, -self.clip_norm, self.clip_norm)
                    
                    aggregated_weights[param_name] += client_weight * param_value
                else:
                    # 处理非张量参数（如标量）
                    aggregated_weights[param_name] += client_weight * param_value
        
        return aggregated_weights
    
    def get_stats(self) -> Dict[str, Any]:
        """获取聚合器统计信息"""
        return {
            "algorithm": "FedAvg",
            "total_rounds": self.round_count,
            "total_aggregations": self.total_aggregations,
            "weighted": self.weighted,
            "clip_norm": self.clip_norm,
            "device": str(self.device)
        }
    
    def reset_stats(self):
        """重置统计信息"""
        self.round_count = 0
        self.total_aggregations = 0
        logger.info("🔄 FedAvg聚合器统计信息已重置")
    
    def __repr__(self) -> str:
        return f"FedAvgAggregator(weighted={self.weighted}, device={self.device}, rounds={self.round_count})"