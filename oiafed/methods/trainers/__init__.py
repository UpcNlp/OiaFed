"""
内置训练器
"""

from .default import DefaultTrainer, AsyncTrainer
from .continual import ContinualTrainer
from .target import TARGETTrainer
from .fot_continual import FOTContinualTrainer
from .faderaser_trainer import FedEraserTrainer
from .splitnn_trainer import SplitNNTrainer

__all__ = [
    "DefaultTrainer",
    "AsyncTrainer",
    "ContinualTrainer",
    "TARGETTrainer",
    "FedEraserTrainer",
    "FOTContinualTrainer",
    "SplitNNTrainer"
]
