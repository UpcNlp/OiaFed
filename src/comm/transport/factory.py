"""
Transport 工厂
"""

from typing import TYPE_CHECKING

from .base import Transport
from .memory import MemoryTransport
from .grpc_transport import GrpcTransport
from ..config import TransportConfig

if TYPE_CHECKING:
    from ..node import Node


def create_transport(config: TransportConfig, node: "Node") -> Transport:
    """
    根据配置创建 Transport 实例
    
    Args:
        config: 传输层配置
        node: 所属节点
        
    Returns:
        Transport 实例
    """
    if config.mode == "memory":
        return MemoryTransport(node, config.memory)
    elif config.mode == "grpc":
        return GrpcTransport(node, config.grpc)
    else:
        raise ValueError(f"Unknown transport mode: {config.mode}")
