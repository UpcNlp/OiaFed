# fedcl/federation/__init__.py
"""
联邦学习核心组件模块

该模块提供联邦学习的核心组件实现，包括：
- LocalTrainer: 本地训练器
- ModelManager: 模型管理器
- ClientManager: 客户端管理器
- ClientStatus: 客户端状态枚举

支持真联邦和伪联邦两种模式。
"""

from .local_trainer import LocalTrainer
from .model_manager import ModelManager
from .client_manager import ClientManager, ClientStatus

__all__ = [
    "LocalTrainer",
    "ModelManager", 
    "ClientManager",
    "ClientStatus"
]
