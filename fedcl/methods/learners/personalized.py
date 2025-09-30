"""
个性化客户端学习器

实现纯粹的客户端个性化学习策略，专注于单个客户端的本地学习过程。
这是客户端学习层，不包含跨客户端的全局状态管理功能。
"""

import torch
from typing import Dict, Any, Optional
from loguru import logger

from ...api.decorators import learner


@learner("personalized_client", description="客户端个性化学习器")
class PersonalizedClientLearner:
    """
    客户端个性化学习器
    
    职责：
    1. 在单个客户端上执行个性化学习
    2. 处理全局权重和个性化权重的混合训练
    3. 返回训练结果给trainer层协调
    """
    
    def __init__(self, config: Dict[str, Any] = None, context: Optional[Any] = None):
        self.config = config or {}
        self.context = context
        
        # 个性化学习参数
        self.local_epochs = self.config.get("local_epochs", 3)
        self.learning_rate = self.config.get("learning_rate", 0.01)
        self.personalization_ratio = self.config.get("personalization_ratio", 0.2)
        self.personal_layer_patterns = self.config.get("personal_layer_patterns", ["classifier", "fc", "output"])
        
        # 当前客户端的本地状态（不跨客户端）
        self.current_personal_weights: Dict[str, Any] = {}
        
        logger.info(f"✅ 个性化客户端学习器初始化完成 - 本地轮次: {self.local_epochs}")
    
    def train_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行客户端个性化训练"""
        # 验证必需数据
        if "model_weights" not in task_data:
            raise ValueError("缺少必需的模型权重 'model_weights'")
        
        if "train_data" not in task_data:
            raise ValueError("缺少必需的训练数据 'train_data'")
        
        client_id = task_data.get("client_id", "unknown")
        model_weights = task_data["model_weights"]
        train_data = task_data["train_data"]
        num_samples = task_data.get("num_samples", len(train_data) if hasattr(train_data, '__len__') else 0)
        
        # 执行本地训练（必须使用真实的训练逻辑）
        if not hasattr(self, '_trainer') or self._trainer is None:
            raise NotImplementedError(
                "个性化客户端学习器必须提供真实的训练器。"
                "请设置self._trainer或重写此方法。"
                "不允许使用模拟训练结果，这是生产环境。"
            )
        
        # 使用真实训练器进行本地训练
        training_result = self._trainer.train(
            data=train_data,
            initial_weights=model_weights,
            epochs=self.local_epochs,
            learning_rate=self.learning_rate
        )
        
        # 分离全局权重和个性化权重更新
        updated_weights = training_result.get("updated_weights", model_weights)
        personal_weights, global_weights = self._separate_weight_updates(updated_weights)
        
        # 更新本地个性化权重
        self.current_personal_weights.update(personal_weights)
        
        return {
            "global_weights": global_weights,
            "personal_weights": personal_weights,
            "num_samples": num_samples,
            "client_id": client_id,
            # 包含所有训练指标（支持任意指标名称）
            **{k: v for k, v in training_result.items() 
               if isinstance(v, (int, float)) and k not in ["updated_weights"]}
        }
    
    def evaluate_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """评估客户端个性化效果"""
        # 验证必需数据
        if "model_weights" not in task_data:
            raise ValueError("缺少必需的模型权重 'model_weights'")
        
        if "test_data" not in task_data:
            raise ValueError("缺少必需的测试数据 'test_data'")
        
        if not hasattr(self, '_evaluator') or self._evaluator is None:
            raise NotImplementedError(
                "个性化客户端学习器必须提供真实的评估器。"
                "请设置self._evaluator或重写此方法。"
                "不允许使用模拟评估结果，这是生产环境。"
            )
        
        model_weights = task_data["model_weights"]
        test_data = task_data["test_data"]
        
        # 使用真实评估器进行评估
        evaluation_result = self._evaluator.evaluate(
            data=test_data,
            weights=model_weights
        )
        
        return evaluation_result
    
    def _separate_weight_updates(self, updated_weights: Dict[str, Any]) -> tuple:
        """分离权重更新为个性化和全局部分"""
        personal_updates = {}
        global_updates = {}
        
        for layer_name, weight in updated_weights.items():
            should_personalize = any(pattern in layer_name.lower() 
                                   for pattern in self.personal_layer_patterns)
            
            if should_personalize:
                personal_updates[layer_name] = weight
            else:
                global_updates[layer_name] = weight
        
        return personal_updates, global_updates
    
    def initialize_personal_weights(self, global_weights: Dict[str, Any]):
        """从全局权重初始化当前客户端的个性化权重"""
        if not global_weights:
            raise ValueError(
                "必须提供真实的全局权重来初始化个性化权重。"
                "不允许使用空权重，这是生产环境。"
            )
        
        self.current_personal_weights = {}
        
        # 只复制需要个性化的层
        for layer_name, weight in global_weights.items():
            should_personalize = any(pattern in layer_name.lower() 
                                   for pattern in self.personal_layer_patterns)
            
            if should_personalize and isinstance(weight, torch.Tensor):
                # 初始化为全局权重的副本
                self.current_personal_weights[layer_name] = weight.clone().detach()
                logger.debug(f"个性化层: {layer_name}")
        
        logger.debug(f"个性化权重初始化完成 - {len(self.current_personal_weights)} 层")
    
    def get_current_personal_weights(self) -> Dict[str, Any]:
        """获取当前客户端的个性化权重"""
        return self.current_personal_weights.copy()
    
    def set_personal_weights(self, weights: Dict[str, Any]):
        """设置当前客户端的个性化权重"""
        self.current_personal_weights = weights.copy()
    
    def merge_weights(self, global_weights: Dict[str, Any]) -> Dict[str, Any]:
        """合并全局权重和个性化权重"""
        merged_weights = global_weights.copy()
        merged_weights.update(self.current_personal_weights)
        return merged_weights