"""
个性化联邦训练器

实现个性化联邦学习的全局协调逻辑，管理多个客户端的个性化权重。
这是联邦训练的协调层，而不是客户端学习层。
"""

import torch
from typing import Dict, Any, List, Optional
from loguru import logger

from ...fl.server import FLTrainerBase
from ...api.decorators import trainer


@trainer("personalized_federation", description="个性化联邦训练协调器")
class PersonalizedFederationTrainer(FLTrainerBase):
    """
    个性化联邦训练器
    
    职责：
    1. 协调多个客户端的个性化训练
    2. 管理全局状态和个性化权重
    3. 实现个性化聚合策略
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        
        # 个性化配置
        self.personalization_ratio = self.config.get("personalization_ratio", 0.2)
        self.personal_layer_patterns = self.config.get("personal_layer_patterns", ["classifier", "fc", "output"])
        
        # 全局状态管理（这些应该在trainer层）
        self.client_personal_weights: Dict[str, Dict[str, Any]] = {}
        self.client_global_weights: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"✅ 个性化联邦训练器初始化完成 - 个性化比例: {self.personalization_ratio}")
    
    def setup_training(self, **kwargs) -> None:
        """设置个性化联邦训练环境"""
        federation_context = kwargs.get("federation_context")
        implementation = kwargs.get("implementation")
        
        # 初始化全局模型
        self.global_model_weights = self._create_initial_model_weights()
        self.update_global_state("global_model_weights", self.global_model_weights)
        
        logger.info("🎯 个性化联邦训练环境设置完成")
    
    def execute_client_round(self, client_id: str, round_num: int, 
                           global_model_weights: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """执行客户端个性化训练轮次"""
        logger.debug(f"👤 客户端 {client_id} 开始个性化训练轮次 {round_num}")
        
        # 1. 初始化客户端个性化权重（如果需要）
        if client_id not in self.client_personal_weights:
            self._initialize_client_personal_weights(client_id, global_model_weights)
        
        # 2. 获取客户端的混合权重（全局 + 个性化）
        mixed_weights = self._get_mixed_weights(client_id, global_model_weights)
        
        # 3. 使用客户端学习器进行本地训练（这里调用真实的客户端学习器）
        client_learner = self._get_client_learner(client_id)
        
        training_result = client_learner.train_task({
            "client_id": client_id,
            "round_num": round_num,
            "model_weights": mixed_weights,
            "train_data": self._get_client_data(client_id),
            "num_samples": self._get_client_sample_count(client_id)
        })
        
        # 4. 分离个性化权重更新
        self._update_client_personal_weights(client_id, training_result.get("updated_weights", {}))
        
        # 5. 返回用于聚合的结果
        return {
            "model_weights": training_result.get("global_weights", mixed_weights),
            "personal_weights": training_result.get("personal_weights", {}),
            "num_samples": training_result.get("num_samples", 0),
            "client_id": client_id,
            **{k: v for k, v in training_result.items() if isinstance(v, (int, float))}
        }
    
    def execute_server_aggregation(self, client_results: List[Dict[str, Any]], 
                                  round_num: int, **kwargs) -> Dict[str, Any]:
        """执行个性化联邦聚合"""
        logger.debug(f"🔄 服务器执行个性化聚合 - 轮次 {round_num}")
        
        # 1. 分离全局权重和个性化权重
        global_updates = []
        personal_updates = {}
        
        for result in client_results:
            client_id = result["client_id"]
            
            # 全局权重用于联邦聚合
            global_updates.append({
                "model_weights": result["model_weights"],
                "num_samples": result["num_samples"],
                "client_id": client_id
            })
            
            # 个性化权重只在本地保存
            if "personal_weights" in result:
                personal_updates[client_id] = result["personal_weights"]
        
        # 2. 聚合全局权重（使用标准聚合器）
        aggregator = self._get_aggregator()
        aggregated_result = aggregator.aggregate(global_updates)
        
        # 3. 更新个性化权重状态
        for client_id, personal_weights in personal_updates.items():
            if personal_weights:
                self.update_client_personal_weights(client_id, personal_weights)
        
        # 4. 计算聚合指标
        aggregated_metrics = self._compute_aggregated_metrics(client_results)
        
        return {
            "aggregated_weights": aggregated_result["aggregated_weights"],
            "num_participants": len(client_results),
            "total_samples": aggregated_result["total_samples"],
            "personalization_stats": self._compute_personalization_stats(),
            **aggregated_metrics
        }
    
    # ================== 个性化权重管理方法（这些应该在trainer层）==================
    
    def get_client_personal_weights(self, client_id: str) -> Optional[Dict[str, Any]]:
        """获取客户端的个性化权重"""
        return self.client_personal_weights.get(client_id)
    
    def update_client_personal_weights(self, client_id: str, weights: Dict[str, Any]):
        """更新客户端的个性化权重"""
        if client_id not in self.client_personal_weights:
            self.client_personal_weights[client_id] = {}
        
        self.client_personal_weights[client_id].update(weights)
        logger.debug(f"✅ 更新客户端 {client_id} 个性化权重")
    
    def _initialize_client_personal_weights(self, client_id: str, global_weights: Dict[str, Any]):
        """初始化客户端个性化权重"""
        self.client_personal_weights[client_id] = {}
        self.client_global_weights[client_id] = global_weights.copy()
        
        # 选择需要个性化的层
        for layer_name, weight in global_weights.items():
            should_personalize = any(pattern in layer_name.lower() 
                                   for pattern in self.personal_layer_patterns)
            
            if should_personalize and hasattr(weight, 'clone'):
                self.client_personal_weights[client_id][layer_name] = weight.clone().detach()
                logger.debug(f"👤 客户端 {client_id} 个性化层: {layer_name}")
        
        logger.info(f"✅ 客户端 {client_id} 个性化权重初始化完成 - "
                   f"{len(self.client_personal_weights[client_id])} 层")
    
    def _get_mixed_weights(self, client_id: str, global_weights: Dict[str, Any]) -> Dict[str, Any]:
        """获取混合权重（全局 + 个性化）"""
        mixed_weights = global_weights.copy()
        
        # 用个性化权重覆盖对应层
        personal_weights = self.get_client_personal_weights(client_id)
        if personal_weights:
            mixed_weights.update(personal_weights)
        
        return mixed_weights
    
    def _compute_personalization_stats(self) -> Dict[str, Any]:
        """计算个性化统计信息"""
        total_clients = len(self.client_personal_weights)
        avg_personal_layers = 0
        
        if total_clients > 0:
            total_personal_layers = sum(len(weights) for weights in self.client_personal_weights.values())
            avg_personal_layers = total_personal_layers / total_clients
        
        return {
            "total_personalized_clients": total_clients,
            "avg_personal_layers_per_client": avg_personal_layers,
            "personalization_ratio": self.personalization_ratio
        }
    
    # ================== 辅助方法 ==================
    
    def _initialize_global_model_weights(self, **kwargs) -> Dict[str, Any]:
        """初始化全局模型权重"""
        return self._create_initial_model_weights()
    
    def _create_initial_model_weights(self) -> Dict[str, Any]:
        """创建初始模型权重"""
        # 为个性化训练器提供一个简单的默认实现
        input_dim = self.config.get("input_dim", 784)
        hidden_dim = self.config.get("hidden_dim", 64)
        output_dim = self.config.get("output_dim", 10)
        
        weights = {
            "linear1.weight": torch.randn(hidden_dim, input_dim) * 0.01,
            "linear1.bias": torch.zeros(hidden_dim),
            "linear2.weight": torch.randn(hidden_dim // 2, hidden_dim) * 0.01,
            "linear2.bias": torch.zeros(hidden_dim // 2),
            "linear3.weight": torch.randn(output_dim, hidden_dim // 2) * 0.01,
            "linear3.bias": torch.zeros(output_dim)
        }
        
        return weights
    
    def _get_client_learner(self, client_id: str):
        """获取客户端学习器（必须由子类实现）"""
        raise NotImplementedError(
            "个性化联邦训练器必须实现真实的客户端学习器获取。"
            "请重写此方法以提供真实的学习器实例。"
        )
    
    def _get_client_data(self, client_id: str):
        """获取客户端数据（必须由子类实现）"""
        raise NotImplementedError(
            "个性化联邦训练器必须实现真实的客户端数据获取。"
            "请重写此方法以提供真实的训练数据。"
        )
    
    def _get_client_sample_count(self, client_id: str) -> int:
        """获取客户端样本数量（必须由子类实现）"""
        raise NotImplementedError(
            "个性化联邦训练器必须实现真实的样本数量统计。"
            "请重写此方法以返回真实的样本数量。"
        )
    
    def _get_aggregator(self):
        """获取聚合器（必须由子类实现）"""
        raise NotImplementedError(
            "个性化联邦训练器必须实现真实的聚合器获取。"
            "请重写此方法以提供真实的聚合器实例。"
        )
    
    def _compute_aggregated_metrics(self, client_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算聚合指标（支持任意指标名称）"""
        if not client_results:
            return {}
        
        # 收集所有数值指标
        all_metrics = {}
        total_samples = sum(result.get("num_samples", 0) for result in client_results)
        
        # 获取所有可能的指标名称
        metric_names = set()
        for result in client_results:
            for key, value in result.items():
                if isinstance(value, (int, float)) and key not in ["num_samples", "client_id"]:
                    metric_names.add(key)
        
        # 计算加权平均
        for metric_name in metric_names:
            weighted_sum = 0
            total_weight = 0
            
            for result in client_results:
                if metric_name in result:
                    weight = result.get("num_samples", 1)
                    weighted_sum += result[metric_name] * weight
                    total_weight += weight
            
            if total_weight > 0:
                all_metrics[f"avg_{metric_name}"] = weighted_sum / total_weight
        
        return all_metrics