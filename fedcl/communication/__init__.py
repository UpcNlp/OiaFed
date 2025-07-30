# fedcl/communication/__init__.py
"""
通信模块初始化文件

提供FedCL框架的通信相关组件，包括消息协议、网络接口、数据序列化、
安全模块、通信处理器和通信管理器等。
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
