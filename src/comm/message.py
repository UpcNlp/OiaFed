"""
Node 通信层消息数据结构
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any


class MessageType(Enum):
    """消息类型"""
    
    REQUEST = 1     # 请求，需要响应
    RESPONSE = 2    # 响应，关联到某个 REQUEST
    NOTIFY = 3      # 通知，不需要响应


class ConnectionStatus(Enum):
    """连接状态"""
    
    CONNECTING = 1
    CONNECTED = 2
    DISCONNECTED = 3


@dataclass
class ErrorInfo:
    """错误信息"""
    
    code: str                                    # 错误码
    message: str                                 # 错误描述
    details: Optional[Dict[str, Any]] = None    # 详细信息
    stack_trace: Optional[str] = None           # 调用栈（调试模式）


@dataclass
class Message:
    """通信消息"""
    
    # 消息标识
    id: str                                      # 唯一标识，UUID
    
    # 消息类型
    type: MessageType                            # REQUEST / RESPONSE / NOTIFY
    
    # 路由信息
    method: str                                  # 方法名，路由依据
    source: str                                  # 发送方 node_id
    target: str                                  # 接收方 node_id
    
    # 负载
    payload: bytes                               # 序列化后的业务数据
    
    # 关联信息
    correlation_id: Optional[str] = None         # 关联的请求 ID（RESPONSE 时使用）
    
    # 元数据
    timestamp: int = 0                           # 发送时间戳（纳秒）
    metadata: Dict[str, str] = field(default_factory=dict)  # 扩展元数据
    
    # 错误信息（仅 RESPONSE）
    error: Optional[ErrorInfo] = None            # 错误详情


@dataclass
class MessageContext:
    """方法处理上下文"""
    
    message: Optional[Message]                   # 原始消息
    source: str                                  # 调用方 node_id
    metadata: Dict[str, str] = field(default_factory=dict)  # 元数据


@dataclass
class ConnectionInfo:
    """连接信息"""

    node_id: str                                 # 对端节点 ID
    address: Optional[str] = None                # 对端地址（gRPC 模式）
    status: ConnectionStatus = ConnectionStatus.DISCONNECTED  # 连接状态
    connected_at: float = 0.0                    # 连接建立时间
    last_active: float = 0.0                     # 最后活跃时间
    metadata: Dict[str, str] = field(default_factory=dict)  # 连接元数据

    # 心跳超时宽限期管理
    grace_period_end: Optional[float] = None     # 宽限期结束时间（None表示未进入宽限期）
    timeout_count: int = 0                       # 连续超时次数
