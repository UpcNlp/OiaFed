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

Public Extension APIs:
    # 注册自定义序列化器
    node.register_serializer(my_serializer)
    
    # 添加自定义拦截器
    node.add_interceptor(my_interceptor)

Migration:
    # 配置类已移至 oiafed.config
    # 旧代码:
    from oiafed.comm import NodeConfig, load_config
    # 新代码:
    from oiafed.config import NodeCommConfig, load_config
"""

from .node import Node
from .config import (
    # 内部配置类
    CommTransportConfig,
    TransportConfig,  # 别名
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
    # 适配器
    from_node_comm_config,
    # 向后兼容（已废弃）
    NodeConfig,
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
TransportConfig = CommTransportConfig

__version__ = "2.0.0"

__all__ = [
    # 核心
    "Node",
    
    # 配置（内部）
    "CommTransportConfig",
    "TransportConfig",  # 向后兼容别名
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
    
    # 适配器
    "from_node_comm_config",
    
    # 向后兼容（已废弃）
    "NodeConfig",
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