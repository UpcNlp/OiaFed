"""
经典联邦学习评估器

本模块实现了特殊的评估方法，用于联邦学习场景的性能评估。
"""

from .prototype import PrototypeEvaluator
from .fairness import FairnessEvaluator

__all__ = [
    "PrototypeEvaluator",
    "FairnessEvaluator"
]