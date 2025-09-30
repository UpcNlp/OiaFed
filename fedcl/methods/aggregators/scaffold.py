"""
SCAFFOLD 聚合器

实现 SCAFFOLD (Stochastic Controlled Averaging for Federated Learning) 聚合算法。
使用控制变量减少客户端漂移，提高联邦学习的收敛速度。

论文：SCAFFOLD: Stochastic Controlled Averaging for Federated Learning
作者：Sai Praneeth Karimireddy et al.
发表：ICML 2020

算法特点：
1. 使用控制变量纠正客户端和服务器的更新偏差
2. 更好的收敛保证，特别是在数据异构情况下
3. 需要额外存储和传输控制变量
4. 适合数据分布差异较大的联邦学习场景
"""

import torch
from typing import Dict, List, Any, Optional
from loguru import logger

from ...api.decorators import aggregator


@aggregator("scaffold", description="SCAFFOLD控制变量联邦聚合器")
class SCAFFOLDAggregator:
    """
    SCAFFOLD 聚合器实现
    
    算法核心：
    1. 维护全局控制变量 c
    2. 每个客户端维护本地控制变量 c_i
    3. 客户端更新考虑控制变量的梯度修正
    4. 服务器聚合时同时更新模型和控制变量
    
    参数：
    - learning_rate: 全局学习率，默认1.0
    - control_lr: 控制变量学习率，默认None（自动计算）
    - weighted: 是否按样本数量加权，默认True
    - momentum: 动量系数，默认0.0
    """
    
    def __init__(self, config: Dict[str, Any] = None, **kwargs):
        """初始化SCAFFOLD聚合器"""
        self.config = config or {}
        
        # SCAFFOLD特定参数
        self.learning_rate = self.config.get("learning_rate", 1.0)
        self.control_lr = self.config.get("control_lr", None)  # 自动计算
        self.weighted = self.config.get("weighted", True)
        self.momentum = self.config.get("momentum", 0.0)
        
        # 设备配置
        self.device = self.config.get("device", "auto")
        if self.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 控制变量状态
        self.global_control_variate: Optional[Dict[str, torch.Tensor]] = None
        self.client_control_variates: Dict[str, Dict[str, torch.Tensor]] = {}
        
        # 统计信息
        self.round_count = 0
        self.control_variate_norm_history = []
        
        # 兼容参数
        self.global_model = kwargs.get("global_model")
        
        logger.info(f"✅ SCAFFOLD聚合器初始化完成 - LR: {self.learning_rate}, 动量: {self.momentum}")
    
    def aggregate(self, client_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        执行SCAFFOLD聚合
        
        Args:
            client_updates: 客户端更新列表，需包含：
                - model_weights: 模型权重
                - control_variate: 客户端控制变量（可选）
                - control_variate_delta: 控制变量更新（可选）
                - num_samples: 样本数量
                - local_epochs: 本地训练轮数（用于计算控制变量学习率）
        
        Returns:
            聚合结果，包含更新的模型和控制变量
        """
        if not client_updates:
            raise ValueError("没有客户端更新可聚合")
        
        self.round_count += 1
        logger.debug(f"🔄 SCAFFOLD聚合轮次 {self.round_count} - {len(client_updates)} 个客户端")
        
        # 1. 初始化全局控制变量（如果是第一次）
        if self.global_control_variate is None:
            self._initialize_global_control_variate(client_updates[0])
        
        # 2. 计算聚合权重
        weights = self._compute_aggregation_weights(client_updates)
        
        # 3. 聚合模型参数
        aggregated_weights = self._aggregate_model_weights(client_updates, weights)
        
        # 4. 更新控制变量
        control_stats = self._update_control_variates(client_updates, weights)
        
        # 5. 构建结果
        total_samples = sum(update.get("num_samples", 0) for update in client_updates)
        
        result = {
            "aggregated_weights": aggregated_weights,
            "global_control_variate": self.global_control_variate.copy(),
            "total_samples": total_samples,
            "num_participants": len(client_updates),
            "aggregation_weights": {
                update.get("client_id", f"client_{i}"): weights[i] 
                for i, update in enumerate(client_updates)
            },
            "algorithm": "SCAFFOLD",
            "round": self.round_count,
            "control_stats": control_stats
        }
        
        logger.debug(f"✅ SCAFFOLD聚合完成 - 控制变量范数: {control_stats.get('global_cv_norm', 0):.6f}")
        return result
    
    def _initialize_global_control_variate(self, sample_update: Dict[str, Any]):
        """初始化全局控制变量"""
        model_weights = sample_update["model_weights"]
        self.global_control_variate = {}
        
        for param_name, param_value in model_weights.items():
            if isinstance(param_value, torch.Tensor):
                self.global_control_variate[param_name] = torch.zeros_like(
                    param_value, device=self.device
                )
            else:
                self.global_control_variate[param_name] = 0.0
        
        logger.debug("🔧 全局控制变量已初始化")
    
    def _compute_aggregation_weights(self, client_updates: List[Dict[str, Any]]) -> List[float]:
        """计算聚合权重"""
        if not self.weighted:
            num_clients = len(client_updates)
            return [1.0 / num_clients] * num_clients
        
        sample_counts = [update.get("num_samples", 1) for update in client_updates]
        total_samples = sum(sample_counts)
        
        if total_samples == 0:
            num_clients = len(client_updates)
            return [1.0 / num_clients] * num_clients
        
        return [count / total_samples for count in sample_counts]
    
    def _aggregate_model_weights(self, client_updates: List[Dict[str, Any]], 
                                weights: List[float]) -> Dict[str, torch.Tensor]:
        """聚合模型权重"""
        aggregated_weights = {}
        
        # 获取参数结构
        first_weights = client_updates[0]["model_weights"]
        param_names = list(first_weights.keys())
        
        # 初始化聚合结果
        for param_name in param_names:
            param_tensor = first_weights[param_name]
            if isinstance(param_tensor, torch.Tensor):
                aggregated_weights[param_name] = torch.zeros_like(param_tensor, device=self.device)
            else:
                aggregated_weights[param_name] = 0.0
        
        # 加权聚合
        for i, update in enumerate(client_updates):
            client_weights = update["model_weights"]
            weight = weights[i]
            
            for param_name in param_names:
                if param_name in client_weights:
                    param_value = client_weights[param_name]
                    
                    if isinstance(param_value, torch.Tensor):
                        param_value = param_value.to(self.device)
                        aggregated_weights[param_name] += weight * param_value
                    else:
                        aggregated_weights[param_name] += weight * param_value
        
        return aggregated_weights
    
    def _update_control_variates(self, client_updates: List[Dict[str, Any]], 
                               weights: List[float]) -> Dict[str, float]:
        """更新控制变量"""
        control_stats = {}
        
        # 计算控制变量学习率
        if self.control_lr is None:
            # 自动计算：基于平均本地epoch数
            avg_local_epochs = sum(update.get("local_epochs", 1) for update in client_updates) / len(client_updates)
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
        
        for i, update in enumerate(client_updates):
            client_id = update.get("client_id", f"client_{i}")
            weight = weights[i]
            
            # 获取客户端控制变量增量
            if "control_variate_delta" in update:
                cv_delta = update["control_variate_delta"]
                
                for param_name in control_variate_deltas.keys():
                    if param_name in cv_delta:
                        delta_value = cv_delta[param_name]
                        if isinstance(delta_value, torch.Tensor):
                            delta_value = delta_value.to(self.device)
                            control_variate_deltas[param_name] += weight * delta_value
            
            # 更新客户端控制变量缓存
            if "control_variate" in update:
                self.client_control_variates[client_id] = update["control_variate"]
        
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
                
                # 计算范数
                global_cv_norm += torch.norm(self.global_control_variate[param_name]).item() ** 2
        
        global_cv_norm = global_cv_norm ** 0.5
        self.control_variate_norm_history.append(global_cv_norm)
        
        control_stats = {
            "global_cv_norm": global_cv_norm,
            "effective_control_lr": effective_lr,
            "num_client_cv_updates": len([u for u in client_updates if "control_variate_delta" in u])
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