"""
经典联邦学习学习器

本模块实现了特殊的客户端学习策略，用于处理特定的联邦学习场景。
"""

from .contrastive import ContrastiveLearner
from .personalized import PersonalizedClientLearner
from .meta import MetaLearner
from .default import DefaultLearner
from .test import SimpleLearnerStub  # 导入测试学习器，触发@learner装饰器

__all__ = [
    "ContrastiveLearner",
    "PersonalizedClientLearner", 
    "MetaLearner",
    "DefaultLearner",
    "SimpleLearnerStub"
]