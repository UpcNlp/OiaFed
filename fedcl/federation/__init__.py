# fedcl/federation/__init__.py
"""
联邦学习模块

重构后的联邦学习组件，包含：
- coordinators: 客户端和服务端协调器
- managers: 客户端管理器和模型管理器
- trainers: 本地训练器实现
- exceptions: 联邦学习相关异常
"""

# 协调器
from .coordinators.federated_client import MultiLearnerClient
from .coordinators.federated_server import FederatedServer

# 管理器
from .managers.client_manager import ClientManager, ClientStatus
from .managers.model_manager import ModelManager

# 训练器
from .trainers.local_trainer import LocalTrainer

__all__ = [
    # 协调器
    'MultiLearnerClient',
    'FederatedServer',
    'ModelUpdate',
    'RoundResults',
    
    # 管理器
    'ClientManager',
    'ClientInfo',
    'ClientStatus',
    'ModelManager',
    
    # 训练器
    'LocalTrainer',
    
    # 异常
    'FederationError',
    'ClientError',
    'ServerError',
    'CommunicationError',
    'AggregationError',
    'ModelSyncError'
]



