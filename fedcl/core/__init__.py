# fedcl/core/__init__.py
"""
FedCL核心模块

提供联邦学习框架的核心组件，包括基础抽象类、执行上下文、
Hook系统和异常处理等核心功能。
"""

from .base_aggregator import BaseAggregator
from .base_evaluator import BaseEvaluator
from .base_learner import BaseLearner
from .base_dispatcher import BaseDispatcher  # 新增下发钩子基类
from .execution_context import ExecutionContext
from .hook import Hook, HookPhase, HookPriority
from .hook_executor import HookExecutor
from .metrics_hook import MetricsHook
from .checkpoint_hook import CheckpointHook
from ..exceptions import (
    FedCLError,
    ConfigurationError,
    HookExecutionError,
    ContextError,
    LearnerError,
    ModelStateError,
    AggregationError,
    EvaluationError,
    DataError,
    NetworkError,
    ResourceError
)

__all__ = [
    # 基础抽象类
    'BaseAggregator',
    'BaseEvaluator', 
    'BaseLearner',
    'BaseDispatcher',  # 新增下发钩子基类
    
    # 执行上下文
    'ExecutionContext',
    
    # Hook系统
    'Hook',
    'HookPhase',
    'HookPriority',
    'HookExecutor',
    'MetricsHook',
    'CheckpointHook',
    
    # 异常类
    'FedCLError',
    'ConfigurationError',
    'HookExecutionError',
    'ContextError',
    'LearnerError',
    'ModelStateError',
    'AggregationError',
    'EvaluationError',
    'DataError',
    'NetworkError',
    'ResourceError'
]