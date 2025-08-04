# fedcl/federation/coordinators/__init__.py
"""
联邦协调器模块

包含客户端和服务端协调器，负责联邦学习的工作流程协调。
"""

from .federated_client import MultiLearnerClient
from .federated_server import FederatedServer

__all__ = [
    'MultiLearnerClient',
    'FederatedServer'
]
