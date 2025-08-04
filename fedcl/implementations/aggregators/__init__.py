# fedcl/implementations/aggregators/__init__.py
"""
联邦聚合算法实现模块

提供各种联邦学习聚合算法的具体实现，包括经典算法和先进算法。
所有实现都继承自BaseAggregator基类。

可用算法:
- FedAvgAggregator: Federated Averaging
- FedProxAggregator: Federated Proximal
- SCAFFOLDAggregator: SCAFFOLD Algorithm
- FedNovaAggregator: FedNova Algorithm

使用示例:
    from fedcl.implementations.aggregators import FedAvgAggregator
    
    # 或通过注册系统
    from fedcl.registry import registry
    fedavg_cls = registry.get_aggregator("fedavg")
"""

from .fedavg_aggregator import FedAvgAggregator
from .fedprox_aggregator import FedProxAggregator
from .scaffold_aggregator import SCAFFOLDAggregator
from .fednova_aggregator import FedNovaAggregator

__all__ = [
    "FedAvgAggregator",
    "FedProxAggregator", 
    "SCAFFOLDAggregator",
    "FedNovaAggregator"
]
