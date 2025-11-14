"""
经典联邦学习学习器

本模块实现了特殊的客户端学习策略，用于处理特定的联邦学习场景。
"""

from .contrastive import ContrastiveLearner
from .personalized import PersonalizedClientLearner
from .meta import MetaLearner
# from .default import DefaultLearner  # Temporarily commented out due to broken import
# from .test import SimpleLearnerStub  # Temporarily commented out due to broken import

# 自动导入内置学习器以触发装饰器注册
from .generic import GenericLearner  # noqa: F401
from .mnist_learner import MNISTLearner  # noqa: F401
from .moon import MOONLearner  # noqa: F401
from .fedper import FedPerLearner  # noqa: F401
from .fedrep import FedRepLearner  # noqa: F401
from .fedrod import FedRoDLearner  # noqa: F401
from .fedbabu import FedBABULearner  # noqa: F401
from .fedproto import FedProtoLearner  # noqa: F401
from .feddistill import FedDistillLearner  # noqa: F401
from .fedcp import FedCPLearner  # noqa: F401
from .gpfl import GPFLLearner  # noqa: F401
from .feddbe import FedDBELearner  # noqa: F401

__all__ = [
    "ContrastiveLearner",
    "PersonalizedClientLearner",
    "MetaLearner",
    # "DefaultLearner",
    # "SimpleLearnerStub"

    # 内置学习器
    "GenericLearner",
    "MNISTLearner",
    "MOONLearner",
    "FedPerLearner",
    "FedRepLearner",
    "FedRoDLearner",
    "FedBABULearner",
    "FedProtoLearner",
    "FedDistillLearner",
    "FedCPLearner",
    "GPFLLearner",
    "FedDBELearner",
]