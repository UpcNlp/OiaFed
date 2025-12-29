"""
联邦学习核心组件
"""

from .types import (
    TrainResult,
    EvalResult,
    ClientUpdate,
    ClientInfo,
    FitConfig,
    EvalConfig,
    FitStatus,
)
from .learner import Learner
from .trainer import Trainer
from .aggregator import Aggregator
from .system import FederatedSystem

# scenario_trainers 和 scenario_learners 已删除
# 使用显式 callback 调用替代 __init_subclass__ 自动装饰器机制

__all__ = [
    # 类型
    "TrainResult",
    "EvalResult",
    "ClientUpdate",
    "ClientInfo",
    "FitConfig",
    "EvalConfig",
    "FitStatus",

    # 抽象基类
    "Learner",
    "Trainer",
    "Aggregator",

    # 系统容器
    "FederatedSystem",
]
