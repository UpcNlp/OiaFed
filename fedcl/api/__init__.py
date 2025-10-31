"""
MOE-FedCL API装饰器模块
fedcl/api/__init__.py

提供组件注册装饰器，支持自动发现和管理用户自定义的组件。
"""

from .decorators import learner, trainer, aggregator, evaluator, dataset, model
from .discovery import auto_discover_components, register_from_module
from .builder import ComponentBuilder, get_builder, build_from_config

__all__ = [
    "learner",
    "trainer",
    "aggregator",
    "evaluator",
    "dataset",
    "model",
    "auto_discover_components",
    "register_from_module",
    "ComponentBuilder",
    "get_builder",
    "build_from_config"
]
