"""
Node 通信层

轻量级、可扩展的节点通信框架，支持：
- 对等通信：任意节点可以调用任意节点的方法
- 多种传输模式：Memory（调试）、gRPC（生产）无感切换
- 可扩展性：通过 Interceptor 机制支持日志、认证、监控等横切关注点
- 类型安全：统一的消息格式和错误处理

Example:
    from oiafed.comm import Node
    from oiafed.config import NodeCommConfig
    
    async def handle_train(payload, ctx):
        return {"status": "completed"}
    
    config = NodeCommConfig(node_id="my_node")
    node = Node(config)
    node.register("train", handle_train)
    
    async with node:
        result = await node.call("other_node", "train", {"epochs": 10})

配置说明：
    所有配置类已统一到 oiafed.config 模块：
    from oiafed.config import (
        NodeCommConfig,       # 通信节点配置
        TransportConfig,      # 传输层配置
        GrpcConfig,           # gRPC 配置
        MemoryTransportConfig, # Memory 传输配置
        SerializationConfig,  # 序列化配置
        InterceptorConfig,    # 拦截器配置
        HeartbeatConfig,      # 心跳配置
        MethodOptions,        # 方法选项
    )
"""

from .node import Node

# 所有配置类从 oiafed.config 导入
from ..config import (
    # 传输配置
    TlsConfig,
    GrpcConfig,
    MemoryTransportConfig,
    TransportConfig,
    # 通信配置
    SerializationConfig,
    MethodSerializationConfig,
    InterceptorConfig,
    AuthConfig,
    RetryConfig,
    HeartbeatConfig,
    MethodOptions,
    # 节点配置
    NodeCommConfig,
    # 便捷函数
    load_config,
)

from .message import (
    Message,
    MessageType,
    MessageContext,
    ConnectionInfo,
    ConnectionStatus,
    ErrorInfo,
)
from .exceptions import (
    NodeError,
    NodeNotConnectedError,
    NodeDisconnectedError,
    CallTimeoutError,
    RemoteExecutionError,
    SerializationError,
    InterceptorAbort,
    AuthenticationError,
)
from .transport import (
    Transport,
    MemoryTransport,
    GrpcTransport,
    create_transport,
)
from .serialization import (
    Serializer,
    JsonSerializer,
    PickleSerializer,
    SerializerRegistry,
)
from .interceptor import (
    Interceptor,
    InterceptorChain,
    InterceptorContext,
    LoggingInterceptor,
    AuthInterceptor,
)

# 向后兼容别名
NodeConfig = NodeCommConfig  # 已废弃，使用 NodeCommConfig
GrpcTransportConfig = GrpcConfig  # 向后兼容

__version__ = "3.0.0"

__all__ = [
    # 核心
    "Node",
    
    # 传输配置（从 oiafed.config）
    "TlsConfig",
    "GrpcConfig",
    "GrpcTransportConfig",  # 向后兼容别名
    "MemoryTransportConfig",
    "TransportConfig",
    
    # 通信配置（从 oiafed.config）
    "SerializationConfig",
    "MethodSerializationConfig",
    "InterceptorConfig",
    "AuthConfig",
    "RetryConfig",
    "HeartbeatConfig",
    "MethodOptions",
    
    # 节点配置（从 oiafed.config）
    "NodeCommConfig",
    "NodeConfig",  # 向后兼容别名
    
    # 便捷函数
    "load_config",
    
    # 消息
    "Message",
    "MessageType",
    "MessageContext",
    "ConnectionInfo",
    "ConnectionStatus",
    "ErrorInfo",
    
    # 异常
    "NodeError",
    "NodeNotConnectedError",
    "NodeDisconnectedError",
    "CallTimeoutError",
    "RemoteExecutionError",
    "SerializationError",
    "InterceptorAbort",
    "AuthenticationError",
    
    # 传输层
    "Transport",
    "MemoryTransport",
    "GrpcTransport",
    "create_transport",
    
    # 序列化
    "Serializer",
    "JsonSerializer",
    "PickleSerializer",
    "SerializerRegistry",
    
    # 拦截器
    "Interceptor",
    "InterceptorChain",
    "InterceptorContext",
    "LoggingInterceptor",
    "AuthInterceptor",
]
