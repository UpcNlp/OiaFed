# fedcl/registry/__init__.py
"""
简化的组件注册系统

提供装饰器注册和组件查找功能，支持学习器、聚合器等组件的注册。
"""

from typing import Dict, Any, Optional, Callable, List
from loguru import logger


class ComponentRegistry:
    """组件注册表"""

    def __init__(self):
        self.learners: Dict[str, Any] = {}
        self.aggregators: Dict[str, Any] = {}
        self.evaluators: Dict[str, Any] = {}
        self.trainers: Dict[str, Any] = {}  # 联邦训练器注册
        self.datasets: Dict[str, Any] = {}  # 数据集注册
        self.models: Dict[str, Any] = {}    # 模型注册
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

    def register_dataset(self, name: str, dataset_cls: Any):
        """注册数据集"""
        self.datasets[name] = dataset_cls
        self.logger.debug(f"注册数据集: {name}")

    def register_model(self, name: str, model_cls: Any):
        """注册模型"""
        self.models[name] = model_cls
        self.logger.debug(f"注册模型: {name}")
    
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

    def get_dataset(self, name: str) -> Optional[Any]:
        """获取数据集"""
        dataset_cls = self.datasets.get(name)
        if dataset_cls is None:
            raise ValueError(f"数据集 '{name}' 未注册。可用的数据集: {list(self.datasets.keys())}")
        return dataset_cls

    def get_model(self, name: str) -> Optional[Any]:
        """获取模型"""
        model_cls = self.models.get(name)
        if model_cls is None:
            raise ValueError(f"模型 '{name}' 未注册。可用的模型: {list(self.models.keys())}")
        return model_cls

    def list_all_components(self) -> Dict[str, List[str]]:
        """列出所有已注册的组件"""
        return {
            'learners': list(self.learners.keys()),
            'trainers': list(self.trainers.keys()),
            'aggregators': list(self.aggregators.keys()),
            'evaluators': list(self.evaluators.keys()),
            'datasets': list(self.datasets.keys()),
            'models': list(self.models.keys())
        }

    def get_component_count(self) -> Dict[str, int]:
        """获取各类型组件的数量"""
        return {
            'learners': len(self.learners),
            'trainers': len(self.trainers),
            'aggregators': len(self.aggregators),
            'evaluators': len(self.evaluators),
            'datasets': len(self.datasets),
            'models': len(self.models)
        }
    
    def has_component(self, name: str, component_type: str) -> bool:
        """检查组件是否已注册"""
        registry_map = {
            'learner': self.learners,
            'trainer': self.trainers,
            'aggregator': self.aggregators,
            'evaluator': self.evaluators,
            'dataset': self.datasets,
            'model': self.models
        }

        if component_type not in registry_map:
            return False

        return name in registry_map[component_type]

    def get(self, name: str, component_type: str) -> Optional[Any]:
        """获取组件"""
        registry_map = {
            'learner': self.learners,
            'trainer': self.trainers,
            'aggregator': self.aggregators,
            'evaluator': self.evaluators,
            'dataset': self.datasets,
            'model': self.models
        }

        if component_type not in registry_map:
            return None

        return registry_map[component_type].get(name)

    def unregister_component(self, name: str, component_type: str) -> bool:
        """注销组件"""
        registry_map = {
            'learner': self.learners,
            'trainer': self.trainers,
            'aggregator': self.aggregators,
            'evaluator': self.evaluators,
            'dataset': self.datasets,
            'model': self.models
        }

        if component_type not in registry_map:
            return False

        if name in registry_map[component_type]:
            del registry_map[component_type][name]
            self.logger.debug(f"注销{component_type}: {name}")
            return True

        return False


# 全局注册表实例
registry = ComponentRegistry()

# 导出
__all__ = [
    "ComponentRegistry",
    "registry"
]