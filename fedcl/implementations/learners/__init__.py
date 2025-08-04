# fedcl/implementations/learners/__init__.py
"""
学习器实现模块

这个模块包含各种持续学习算法的具体实现。
每个学习器实现都必须继承自BaseLearner并注册到registry中。
"""

from .l2p_learner import L2PLearner
from .ewc_learner import EWCLearner
from .replay_learner import ReplayLearner
from .lwf_learner import LwFLearner
from .packnet_learner import PackNetLearner
from .default import DefaultLearner

__all__ = [
    "L2PLearner",
    "EWCLearner", 
    "ReplayLearner",
    "LwFLearner",
    "PackNetLearner",
    "DefaultLearner"
]
