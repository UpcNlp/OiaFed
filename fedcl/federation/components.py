"""
组件数据类 - 封装通信层和业务层的组件集合
fedcl/federation/components.py
"""

from dataclasses import dataclass
from typing import Optional, Any

from ..communication.base import CommunicationManagerBase
from ..communication.business_layer import BusinessCommunicationLayer
from ..connection.manager import ConnectionManager
from ..learner.base_learner import BaseLearner
from ..trainer.trainer import BaseTrainer
from ..transport.base import TransportBase
from ..types import ModelData


@dataclass
class CommunicationComponents:
    """
    通信层组件集合

    包含初始化后的5层通信栈组件：
    - Layer 5: Transport（传输层）
    - Layer 4: CommunicationManager（通信管理层）
    - Layer 3: ConnectionManager（连接管理层）
    - Layer 2: BusinessCommunicationLayer（业务通信层，仅服务端）
    """
    transport: TransportBase
    communication_manager: CommunicationManagerBase
    connection_manager: ConnectionManager
    business_layer: Optional[BusinessCommunicationLayer] = None  # 仅服务端需要


@dataclass
class ServerBusinessComponents:
    """
    服务端业务组件集合

    包含服务端业务逻辑所需的所有组件：
    - trainer: 训练器（必需）
    - global_model: 全局模型（必需）
    - aggregator: 聚合器（可选）
    """
    trainer: BaseTrainer
    global_model: ModelData
    aggregator: Optional[Any] = None


@dataclass
class ClientBusinessComponents:
    """
    客户端业务组件集合

    包含客户端业务逻辑所需的所有组件：
    - learner: 学习器（必需）
    - dataset: 数据集（可选）
    """
    learner: BaseLearner
    dataset: Optional[Any] = None
