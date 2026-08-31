"""
内置训练器
"""

from .default import DefaultTrainer, AsyncTrainer
from .continual import ContinualTrainer
from .target import TARGETTrainer
from .fedsra import FedSRATrainer
from .fedemoe import FedEMoETrainer
from .oneshot import (
    OFedAvgTrainer,
    EnsembleTrainer,
    FAFITrainer,
    FedCGSTrainer,
    FuseFLTrainer,
    CoBoostingTrainer,
)

__all__ = [
    "DefaultTrainer",
    "AsyncTrainer",
    "ContinualTrainer",
    "TARGETTrainer",
    "FedSRATrainer",
    "FedEMoETrainer",
    "OFedAvgTrainer",
    "EnsembleTrainer",
    "FAFITrainer",
    "FedCGSTrainer",
    "FuseFLTrainer",
    "CoBoostingTrainer",
]
