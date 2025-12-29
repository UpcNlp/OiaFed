"""
内置训练器
"""

from .default import DefaultTrainer, AsyncTrainer
from .continual import ContinualTrainer
from .target import TARGETTrainer

__all__ = [
    "DefaultTrainer",
    "AsyncTrainer",
    "ContinualTrainer",
    "TARGETTrainer",
]
