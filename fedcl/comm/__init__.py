"""
通信层 - 支持多种通信方式的传输器

提供统一的通信接口，支持：
1. 内存传输 - 用于本地开发调试
2. 进程传输 - 用于本地多进程测试
3. 网络传输 - 用于真实分布式部署
4. 透明通信 - 用于进程内模拟通信
"""

from .memory_transport import MemoryTransport
from .process_transport import ProcessTransport  
from .network_transport import NetworkTransport
from .transparent_communication import (
    TransparentCommunication,
    Message,
    CommunicationMode,
    NetworkConfig,
    BaseCommunicationBackend
)

__all__ = [
    "MemoryTransport",
    "ProcessTransport", 
    "NetworkTransport",
    "TransparentCommunication",
    "Message",
    "CommunicationMode", 
    "NetworkConfig",
    "BaseCommunicationBackend"
]