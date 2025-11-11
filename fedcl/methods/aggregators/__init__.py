"""
经典联邦学习聚合器

本模块实现了各种经典的联邦学习聚合算法，用户可以通过配置文件选择使用。
所有聚合器都继承自基础聚合器类，保持统一的接口。
"""

from .fedavg import FedAvgAggregator
from .fedprox import FedProxAggregator
from .scaffold import SCAFFOLDAggregator
from .fednova import FedNovaAggregator
from .fedadam import FedAdamAggregator
from .fedyogi import FedYogiAggregator
from .feddyn import FedDynAggregator
from .fedbn import FedBNAggregator

__all__ = [
    "FedAvgAggregator",
    "FedProxAggregator",
    "SCAFFOLDAggregator",
    "FedNovaAggregator",
    "FedAdamAggregator",
    "FedYogiAggregator",
    "FedDynAggregator",
    "FedBNAggregator"
]