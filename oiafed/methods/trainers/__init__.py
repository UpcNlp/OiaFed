"""
内置训练器
"""

from .default import DefaultTrainer, AsyncTrainer
from .continual import ContinualTrainer
from .target import TARGETTrainer
from .fedsra import FedSRATrainer
from .fedemoe import FedEMoETrainer
from .fedemoe_baselines import FedEMoEBaselineTrainer
from .oneshot import (
    OFedAvgTrainer,
    EnsembleTrainer,
    FAFITrainer,
    FedCGSTrainer,
    FuseFLTrainer,
    CoBoostingTrainer,
)
from .fot_continual import FOTContinualTrainer
from .faderaser_trainer import FedEraserTrainer
from .splitnn_trainer import SplitNNTrainer

__all__ = [
    "DefaultTrainer",
    "AsyncTrainer",
    "ContinualTrainer",
    "TARGETTrainer",
    "FedSRATrainer",
    "FedEMoETrainer",
    "FedEMoEBaselineTrainer",
    "OFedAvgTrainer",
    "EnsembleTrainer",
    "FAFITrainer",
    "FedCGSTrainer",
    "FuseFLTrainer",
    "CoBoostingTrainer",
    "FedEraserTrainer",
    "FOTContinualTrainer",
    "SplitNNTrainer",
]
