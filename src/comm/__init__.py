"""
Node 通信层

轻量级、可扩展的节点通信框架，支持：
- 对等通信：任意节点可以调用任意节点的方法
- 多种传输模式：Memory（调试）、gRPC（生产）无感切换
- 可扩展性：通过 Interceptor 机制支持日志、认证、监控等横切关注点
- 类型安全：统一的消息格式和错误处理

Example:
    from node_comm import Node, NodeConfig, load_config
    
    async def handle_train(payload, ctx):
        return {"status": "completed"}
    
    config = load_config("config.yaml")
    node = Node("my_node", config)
    node.register("train", handle_train)
    
    async with node:
        result = await node.call("other_node", "train", {"epochs": 10})

Public Extension APIs:
    # 注册自定义序列化器
    node.register_serializer(my_serializer)
    
    # 添加自定义拦截器
    node.add_interceptor(my_interceptor)
"""

from .node import Node
from .config import (
    NodeConfig,
    TransportConfig,
    MemoryTransportConfig,
    GrpcTransportConfig,
    TlsConfig,
    SerializationConfig,
    MethodSerializationConfig,
    InterceptorConfig,
    AuthConfig,
    RetryConfig,
    HeartbeatConfig,
    MethodOptions,
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

__version__ = "1.1.0"

__all__ = [
    # 核心
    "Node",
    
    # 配置
    "NodeConfig",
    "TransportConfig",
    "MemoryTransportConfig",
    "GrpcTransportConfig",
    "TlsConfig",
    "SerializationConfig",
    "MethodSerializationConfig",
    "InterceptorConfig",
    "AuthConfig",
    "RetryConfig",
    "HeartbeatConfig",
    "MethodOptions",
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
    
    # 工具
]
