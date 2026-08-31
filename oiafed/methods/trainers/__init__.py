"""
内置训练器
"""

from .default import DefaultTrainer, AsyncTrainer
from .continual import ContinualTrainer
from .target import TARGETTrainer
from .fot_continual import FOTContinualTrainer
from .faderaser_trainer import FedEraserTrainer
from .splitnn_trainer import SplitNNTrainer
from .fedsra import FedSRATrainer
from .oneshot import (
    OFedAvgTrainer,
    EnsembleTrainer,
    FedCGSTrainer,
)

__all__ = [
    "DefaultTrainer",
    "AsyncTrainer",
    "ContinualTrainer",
    "TARGETTrainer",
    "FedEraserTrainer",
    "FOTContinualTrainer",
    "SplitNNTrainer",
    "FedSRATrainer",
    "OFedAvgTrainer",
    "EnsembleTrainer",
    "FedCGSTrainer",
]
