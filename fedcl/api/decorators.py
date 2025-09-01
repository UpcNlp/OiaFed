# fedcl/api/decorators.py
"""
全新的透明装饰器系统

专为透明联邦学习设计的简洁装饰器API，让用户能够：
1. 极简地定义联邦学习组件
2. 专注于算法逻辑而非分布式细节
3. 自动处理注册和元数据管理
"""

from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type

from loguru import logger
from ..registry import registry


def learner(name: str, **metadata) -> Callable:
    """
    学习器装饰器 - 极简设计

    Args:
        name: 学习器名称
        **metadata: 可选的元数据

    Example:
        @fedcl.learner("my_continual_learner")
        class MyContinualLearner:
            def __init__(self, config, context):
                self.config = config
                self.context = context

            def train_task(self, task_data):
                # 专注算法逻辑，框架自动处理分布式细节
                for batch in task_data:
                    # 训练逻辑
                    pass
                return {"accuracy": 0.95, "loss": 0.1}

            def evaluate_task(self, task_data):
                # 评估逻辑
                return {"accuracy": 0.93}
    """

    def decorator(cls: Type) -> Type:
        # 自动添加基础方法（如果没有的话）
        if not hasattr(cls, "train_task") and not hasattr(cls, "train_on_client"):
            logger.warning(f"学习器 {name} 缺少 train_task 或 train_on_client 方法")

        if not hasattr(cls, "evaluate_task") and not hasattr(cls, "evaluate"):
            logger.warning(f"学习器 {name} 缺少 evaluate_task 或 evaluate 方法")

        # 注册到注册表
        registry.register_learner(name, cls)

        # 添加元数据
        cls._fedcl_name = name
        cls._fedcl_type = "learner"
        cls._fedcl_metadata = metadata

        return cls

    return decorator


def aggregator(name: str, **metadata) -> Callable:
    """
    聚合器装饰器 - 极简设计

    Args:
        name: 聚合器名称
        **metadata: 可选的元数据

    Example:
        @fedcl.aggregator("my_weighted_avg")
        class MyWeightedAvgAggregator:
            def aggregate(self, client_updates):
                # 专注聚合算法，框架自动处理通信
                total_samples = sum(update['num_samples'] for update in client_updates)

                aggregated_weights = {}
                for param_name in client_updates[0]['weights']:
                    weighted_sum = sum(
                        update['weights'][param_name] * (update['num_samples'] / total_samples)
                        for update in client_updates
                    )
                    aggregated_weights[param_name] = weighted_sum

                return aggregated_weights
    """

    def decorator(cls: Type) -> Type:
        if not hasattr(cls, "aggregate"):
            logger.warning(f"聚合器 {name} 缺少 aggregate 方法")

        registry.register_aggregator(name, cls)

        cls._fedcl_name = name
        cls._fedcl_type = "aggregator"
        cls._fedcl_metadata = metadata

        return cls

    return decorator


def evaluator(name: str, **metadata) -> Callable:
    """
    评估器装饰器 - 极简设计

    Args:
        name: 评估器名称
        **metadata: 可选的元数据

    Example:
        @fedcl.evaluator("my_accuracy")
        class MyAccuracyEvaluator:
            def evaluate(self, model, test_data):
                # 专注评估逻辑，框架自动处理数据分发
                correct = 0
                total = 0

                for batch in test_data:
                    predictions = model(batch['input'])
                    correct += (predictions.argmax(1) == batch['target']).sum()
                    total += len(batch['target'])

                return {"accuracy": correct / total}
    """

    def decorator(cls: Type) -> Type:
        if not hasattr(cls, "evaluate"):
            logger.warning(f"评估器 {name} 缺少 evaluate 方法")

        registry.register_evaluator(name, cls)

        cls._fedcl_name = name
        cls._fedcl_type = "evaluator"
        cls._fedcl_metadata = metadata

        return cls

    return decorator


def trainer(name: str, **metadata) -> Callable:
    """
    联邦训练器装饰器

    Args:
        name: 训练器名称
        **metadata: 可选的元数据

    Example:
        @fedcl.trainer("diffusion_trainer")
        class DiffusionFederationTrainer(AbstractFederationTrainer):
            def setup_training(self, **kwargs):
                # 设置扩散模型训练环境
                pass
            
            def execute_client_round(self, client_id, round_num, global_model_weights, **kwargs):
                # 客户端训练：分类器 + 扩散模型
                return training_result
            
            def execute_server_aggregation(self, client_results, round_num, **kwargs):
                # 服务器聚合：FedAvg + 扩散模型更新
                return aggregation_result
    """

    def decorator(cls: Type) -> Type:
        # 检查是否继承了正确的基类
        from ..fl.abstract_trainer import AbstractFederationTrainer
        if not issubclass(cls, AbstractFederationTrainer):
            logger.warning(f"联邦训练器 {name} 应该继承 AbstractFederationTrainer")

        # 检查必需的方法
        required_methods = ["setup_training", "execute_client_round", "execute_server_aggregation"]
        for method in required_methods:
            if not hasattr(cls, method):
                logger.warning(f"联邦训练器 {name} 缺少 {method} 方法")

        # 注册到注册表
        registry.register_trainer(name, cls)

        # 添加元数据
        cls._fedcl_name = name
        cls._fedcl_type = "trainer"
        cls._fedcl_metadata = metadata

        return cls

    return decorator


def list_components() -> Dict[str, List[str]]:
    """列出所有已注册的组件"""
    return {
        "learners": list(registry.learners.keys()),
        "aggregators": list(registry.aggregators.keys()),
        "evaluators": list(registry.evaluators.keys()),
        "trainers": list(registry.trainers.keys()),
    }


def get_component_info(component_type: str, name: str) -> Optional[Dict[str, Any]]:
    """获取组件信息"""
    if component_type == "learner":
        cls = registry.get_learner(name)
    elif component_type == "aggregator":
        cls = registry.get_aggregator(name)
    elif component_type == "evaluator":
        cls = registry.get_evaluator(name)
    elif component_type == "trainer":
        cls = registry.get_trainer(name)
    else:
        return None

    if cls is None:
        return None

    return {
        "name": getattr(cls, "_fedcl_name", name),
        "type": getattr(cls, "_fedcl_type", component_type),
        "class": cls.__name__ if hasattr(cls, '__name__') else str(cls),
        "module": getattr(cls, '__module__', 'unknown'),
        "metadata": getattr(cls, "_fedcl_metadata", {}),
    }


def clear_registry():
    """清空注册表（主要用于测试）"""
    registry.learners.clear()
    registry.aggregators.clear()
    registry.evaluators.clear()
    registry.trainers.clear()
    logger.info("🔄 注册表已清空")


# 导出所有装饰器和工具函数
__all__ = [
    "learner",
    "aggregator", 
    "evaluator",
    "trainer",
    "list_components",
    "get_component_info",
    "clear_registry"
]