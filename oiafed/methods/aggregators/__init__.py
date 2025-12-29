"""
内置聚合器
"""

from .fedavg import (
    FedAvgAggregator,
    MedianAggregator,
    TrimmedMeanAggregator,
)
from .fedprox import FedProxAggregator
from .scaffold import SCAFFOLDAggregator
from .fednova import FedNovaAggregator
from .fedadam import FedAdamAggregator
from .fedyogi import FedYogiAggregator
from .feddyn import FedDynAggregator
from .fedbn import FedBNAggregator
from .fedproto import FedProtoAggregator
from .faderaser import FedEraserAggregator, FedEraserPlusAggregator

__all__ = [
    "FedAvgAggregator",
    "FedProxAggregator",
    "MedianAggregator",
    "TrimmedMeanAggregator",
    "SCAFFOLDAggregator",
    "FedNovaAggregator",
    "FedAdamAggregator",
    "FedYogiAggregator",
    "FedDynAggregator",
    "FedBNAggregator",
    "FedProtoAggregator",
    "FedEraserAggregator",
    "FedEraserPlusAggregator",
]
