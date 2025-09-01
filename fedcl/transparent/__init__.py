# fedcl/transparent/__init__.py
"""
透明抽象层

实现运行环境自动检测和执行策略选择，为真正的透明性奠定基础。
"""

from .mode_detector import ModeDetector, ExecutionMode, SystemResources, NetworkEnvironment
from .strategy_selector import StrategySelector, ExecutionStrategy, StrategyType
from .base_federation_engine import TrainingResult, EvaluationResult
from .base_federation_engine import BaseFederationEngine as TransparentExecutionEngine

__all__ = [
    "ModeDetector",
    "ExecutionMode", 
    "SystemResources",
    "NetworkEnvironment",
    "StrategySelector",
    "ExecutionStrategy",
    "StrategyType",
    "TransparentExecutionEngine",
    "TrainingResult",
    "EvaluationResult",
]