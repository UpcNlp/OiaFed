# fedcl/api/__init__.py
"""
FedCL API模块

提供统一、简洁的联邦学习API接口，实现真伪联邦完全透明的用户体验。

主要组件：
- FederatedTrainer: 统一的联邦训练器
- 装饰器系统: @learner, @aggregator, @evaluator
- 快速启动接口: train(), train_from_config(), quick_experiment()

使用示例：
    import fedcl

    # 一行代码启动联邦学习
    result = fedcl.train(
        learner="ewc_mnist",
        dataset="mnist",
        num_clients=3,
        num_rounds=10
    )
"""

from .decorators import (aggregator, clear_registry, evaluator,
                         get_component_info, learner, list_components)
from .experiments import quick_experiment, train, train_from_config
from .trainer import EvaluationResult, FederatedTrainer, TrainingResult

__all__ = [
    # 核心类
    "FederatedTrainer",
    "TrainingResult",
    "EvaluationResult",
    # 装饰器
    "learner",
    "aggregator",
    "evaluator",
    # 快速启动接口
    "train",
    "train_from_config",
    "quick_experiment",
    # 工具函数
    "list_components",
    "get_component_info",
    "clear_registry",
]
