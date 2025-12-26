"""
内置学习器
"""

from .default import DefaultLearner
from . import fl
from . import cl
from . import vfl  # ← 确保这行存在

__all__ = [
    "DefaultLearner",
]
