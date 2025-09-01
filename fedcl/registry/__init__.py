# fedcl/registry/__init__.py
"""
简化的组件注册系统

提供装饰器注册和组件查找功能，支持学习器、聚合器等组件的注册。
"""

from typing import Dict, Any, Optional, Callable
from loguru import logger


class ComponentRegistry:
    """组件注册表"""
    
    def __init__(self):
        self.learners: Dict[str, Any] = {}
        self.aggregators: Dict[str, Any] = {}
        self.evaluators: Dict[str, Any] = {}
        self.trainers: Dict[str, Any] = {}  # 新增联邦训练器注册
        self.logger = logger.bind(component="ComponentRegistry")
    
    def register_learner(self, name: str, learner_cls: Any):
        """注册学习器"""
        self.learners[name] = learner_cls
        self.logger.debug(f"注册学习器: {name}")
    
    def register_aggregator(self, name: str, aggregator_cls: Any):
        """注册聚合器"""
        self.aggregators[name] = aggregator_cls
        self.logger.debug(f"注册聚合器: {name}")
    
    def register_evaluator(self, name: str, evaluator_cls: Any):
        """注册评估器"""
        self.evaluators[name] = evaluator_cls
        self.logger.debug(f"注册评估器: {name}")
    
    def register_trainer(self, name: str, trainer_cls: Any):
        """注册联邦训练器"""
        self.trainers[name] = trainer_cls
        self.logger.debug(f"注册联邦训练器: {name}")
    
    def get_learner(self, name: str) -> Optional[Any]:
        """获取学习器"""
        learner_cls = self.learners.get(name)
        if learner_cls is None:
            raise ValueError(f"学习器 '{name}' 未注册。可用的学习器: {list(self.learners.keys())}")
        return learner_cls
    
    def get_aggregator(self, name: str) -> Optional[Any]:
        """获取聚合器"""
        aggregator_cls = self.aggregators.get(name)
        if aggregator_cls is None:
            raise ValueError(f"聚合器 '{name}' 未注册。可用的聚合器: {list(self.aggregators.keys())}")
        return aggregator_cls
    
    def get_evaluator(self, name: str) -> Optional[Any]:
        """获取评估器"""
        evaluator_cls = self.evaluators.get(name)
        if evaluator_cls is None:
            raise ValueError(f"评估器 '{name}' 未注册。可用的评估器: {list(self.evaluators.keys())}")
        return evaluator_cls
    
    def get_trainer(self, name: str) -> Optional[Any]:
        """获取联邦训练器"""
        trainer_cls = self.trainers.get(name)
        if trainer_cls is None:
            raise ValueError(f"联邦训练器 '{name}' 未注册。可用的联邦训练器: {list(self.trainers.keys())}")
        return trainer_cls



# 全局注册表实例
registry = ComponentRegistry()

# 导出
__all__ = [
    "ComponentRegistry",
    "registry"
]