"""
内置训练器
"""

from .default import DefaultTrainer, AsyncTrainer
from .continual import ContinualTrainer
from .target import TARGETTrainer
from .fedsra import FedSRATrainer

__all__ = [
    "DefaultTrainer",
    "AsyncTrainer",
    "ContinualTrainer",
    "TARGETTrainer",
    "FedSRATrainer",
]
