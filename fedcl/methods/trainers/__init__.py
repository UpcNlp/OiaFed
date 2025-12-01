"""
预设联邦训练器

本模块实现了各种经典和特殊的联邦训练协调器。
所有训练器都继承自AbstractFederationTrainer，负责协调整个联邦学习流程。
"""

# 自动导入内置训练器以触发装饰器注册
from .generic import GenericTrainer  # noqa: F401
from .continual import ContinualTrainer  # noqa: F401
from .target_trainer import TARGETTrainer  # noqa: F401

__all__ = [
    # 内置训练器
    "GenericTrainer",
    "ContinualTrainer",
    "TARGETTrainer",
]