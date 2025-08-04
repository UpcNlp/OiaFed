# fedcl/communication/__init__.py
"""
FedCL通信模块

提供联邦学习系统的通信功能，包括：
- 消息协议和序列化
- 网络接口和安全模块
- 通信处理器和管理器
- 多线程异步通信支持
"""
from .message_protocol import MessageProtocol, Message
from .data_serializer import DataSerializer
from .network_interface import NetworkInterface, Connection, ConnectionPool, ConnectionStatus, ProtocolType
from .security_module import SecurityModule
from .communication_handler import CommunicationHandler
from .communication_manager import CommunicationManager
from .exceptions import (
    CommunicationError, 
    NetworkError, 
    SerializationError, 
    ProtocolError,
    ConnectionTimeoutError,
    SecurityError
)

__all__ = [
    'MessageProtocol',
    'Message',
    'DataSerializer',
    'NetworkInterface',
    'Connection',
    'ConnectionPool',
    'ConnectionStatus',
    'ProtocolType',
    'SecurityModule',
    'CommunicationHandler',
    'CommunicationManager',
    'CommunicationError',
    'NetworkError',
    'SerializationError',
    'ProtocolError',
    'ConnectionTimeoutError',
    'SecurityError',
]
