"""
Node 通信层配置定义

注意：
- NodeConfig 已移至 oiafed.config.schema，请使用 NodeCommConfig
- 此文件保留通信层内部使用的配置类
- 外部代码应使用 oiafed.config 中的配置类

Migration:
    # 旧代码
    from oiafed.comm.config import NodeConfig
    
    # 新代码
    from oiafed.config import NodeCommConfig
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
import yaml


# ==================== 通信层内部配置类 ====================
# 这些类仅供通信层内部使用

@dataclass
class TlsConfig:
    """TLS 配置"""
    
    enabled: bool = False
    cert_file: Optional[str] = None
    key_file: Optional[str] = None
    ca_file: Optional[str] = None
    mutual_tls: bool = False


@dataclass
class MemoryTransportConfig:
    """Memory 传输配置"""
    
    zero_copy: bool = True                       # 零拷贝模式
    simulate_latency: bool = False               # 模拟延迟
    latency_ms: float = 0.0                      # 延迟毫秒数


@dataclass
class GrpcTransportConfig:
    """gRPC 传输配置"""

    host: str = "0.0.0.0"
    port: int = 50051
    max_workers: int = 10
    max_message_length: int = 104857600  # 100MB
    tls: TlsConfig = field(default_factory=TlsConfig)

    # 双线程配置
    dual_thread_enabled: bool = True  # 默认启用双线程优化

    # 心跳配置（gRPC 专属）
    heartbeat_enabled: bool = True
    heartbeat_interval: float = 5.0       # 心跳间隔（秒）
    heartbeat_timeout: float = 30.0       # 超时时间（秒）
    heartbeat_check_interval: float = 10.0  # 健康检查间隔（秒）

    # 连接失败自动退出配置
    max_connection_wait_time: float = 300.0  # 最大连接等待时间（秒），默认5分钟
    auto_shutdown_on_failure: bool = True    # 连接失败后是否自动shutdown
    critical_peers: List[str] = field(default_factory=list)  # 关键节点列表（如trainer）


@dataclass
class CommTransportConfig:
    """通信层传输配置（内部使用）"""
    
    mode: str = "memory"                         # memory | grpc
    memory: MemoryTransportConfig = field(default_factory=MemoryTransportConfig)
    grpc: GrpcTransportConfig = field(default_factory=GrpcTransportConfig)


@dataclass
class MethodSerializationConfig:
    """方法序列化配置"""
    
    serializer: str = "json"
    compress: bool = False


@dataclass
class SerializationConfig:
    """序列化配置"""

    default: str = "pickle"                      # 默认序列化器（联邦学习使用pickle处理复杂对象）
    methods: Dict[str, MethodSerializationConfig] = field(default_factory=dict)


@dataclass
class RetryConfig:
    """重试配置"""
    
    enabled: bool = False
    max_retries: int = 3
    backoff: float = 1.0


@dataclass
class AuthConfig:
    """认证配置"""
    
    mode: str = "token"                          # token | mutual_tls | custom
    token: Optional[str] = None                  # 静态 token
    # token_provider: Callable[[], str] 可通过代码设置


@dataclass
class InterceptorConfig:
    """拦截器配置"""
    
    logging: bool = True
    metrics: bool = False
    auth: bool = False
    auth_config: AuthConfig = field(default_factory=AuthConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    custom: List[str] = field(default_factory=list)


@dataclass
class HeartbeatConfig:
    """心跳配置"""
    
    enabled: bool = False
    interval: float = 30.0                       # 心跳间隔（秒）
    timeout: float = 90.0                        # 超时时间（秒）
    critical_peers: List[str] = field(default_factory=list)  # 关键节点


@dataclass
class MethodOptions:
    """方法选项（RPC 调用时使用）"""
    
    serializer: Optional[str] = None             # 指定序列化器
    timeout: Optional[float] = None              # 超时时间
    require_auth: bool = False                   # 是否需要认证


# ==================== 向后兼容 ====================
# 重导出 NodeCommConfig 作为 NodeConfig（已废弃，将来移除）

def _get_node_config_class():
    """延迟导入以避免循环依赖"""
    from oiafed.config import NodeCommConfig
    return NodeCommConfig

# 向后兼容：NodeConfig 别名
# 注意：这是一个属性访问，不是直接的类引用
class _NodeConfigProxy:
    """NodeConfig 代理类，用于向后兼容"""
    
    def __new__(cls, *args, **kwargs):
        import warnings
        warnings.warn(
            "comm.config.NodeConfig 已废弃，请使用 oiafed.config.NodeCommConfig",
            DeprecationWarning,
            stacklevel=2
        )
        NodeCommConfig = _get_node_config_class()
        return NodeCommConfig(*args, **kwargs)

# 保留 NodeConfig 名称用于向后兼容
NodeConfig = _NodeConfigProxy


# ==================== 解析函数（内部使用） ====================

def _parse_tls_config(data: Dict[str, Any]) -> TlsConfig:
    """解析 TLS 配置"""
    if not data:
        return TlsConfig()
    return TlsConfig(
        enabled=data.get("enabled", False),
        cert_file=data.get("cert_file"),
        key_file=data.get("key_file"),
        ca_file=data.get("ca_file"),
        mutual_tls=data.get("mutual_tls", False),
    )


def _parse_memory_transport_config(data: Dict[str, Any]) -> MemoryTransportConfig:
    """解析 Memory 传输配置"""
    if not data:
        return MemoryTransportConfig()
    return MemoryTransportConfig(
        zero_copy=data.get("zero_copy", True),
        simulate_latency=data.get("simulate_latency", False),
        latency_ms=data.get("latency_ms", 0.0),
    )


def _parse_grpc_transport_config(data: Dict[str, Any]) -> GrpcTransportConfig:
    """解析 gRPC 传输配置"""
    if not data:
        return GrpcTransportConfig()

    # 解析双线程配置
    dual_thread = data.get("dual_thread", {})
    dual_thread_enabled = dual_thread.get("enabled", True)  # 默认启用

    # 解析心跳配置
    heartbeat = data.get("heartbeat", {})

    return GrpcTransportConfig(
        host=data.get("host", "0.0.0.0"),
        port=data.get("port", 50051),
        max_workers=data.get("max_workers", 10),
        max_message_length=data.get("max_message_length", 104857600),
        tls=_parse_tls_config(data.get("tls", {})),

        # 双线程配置
        dual_thread_enabled=dual_thread_enabled,

        # 心跳配置
        heartbeat_enabled=heartbeat.get("enabled", True),
        heartbeat_interval=heartbeat.get("interval", 5.0),
        heartbeat_timeout=heartbeat.get("timeout", 30.0),
        heartbeat_check_interval=heartbeat.get("check_interval", 10.0),

        # 连接失败自动退出配置
        max_connection_wait_time=heartbeat.get("max_connection_wait_time", 300.0),
        auto_shutdown_on_failure=heartbeat.get("auto_shutdown_on_failure", True),
        critical_peers=heartbeat.get("critical_peers", []),
    )


def _parse_transport_config(data: Dict[str, Any]) -> CommTransportConfig:
    """解析传输层配置"""
    if not data:
        return CommTransportConfig()
    return CommTransportConfig(
        mode=data.get("mode", "memory"),
        memory=_parse_memory_transport_config(data.get("memory", {})),
        grpc=_parse_grpc_transport_config(data.get("grpc", {})),
    )


def _parse_method_serialization_config(data: Dict[str, Any]) -> MethodSerializationConfig:
    """解析方法序列化配置"""
    return MethodSerializationConfig(
        serializer=data.get("serializer", "json"),
        compress=data.get("compress", False),
    )


def _parse_serialization_config(data: Dict[str, Any]) -> SerializationConfig:
    """解析序列化配置"""
    if not data:
        return SerializationConfig()

    methods = {}
    for method_name, method_data in data.get("methods", {}).items():
        methods[method_name] = _parse_method_serialization_config(method_data)

    return SerializationConfig(
        default=data.get("default", "pickle"),  # 默认使用pickle而不是json
        methods=methods,
    )


def _parse_retry_config(data: Dict[str, Any]) -> RetryConfig:
    """解析重试配置"""
    if not data:
        return RetryConfig()
    return RetryConfig(
        enabled=data.get("enabled", False),
        max_retries=data.get("max_retries", 3),
        backoff=data.get("backoff", 1.0),
    )


def _parse_auth_config(data: Dict[str, Any]) -> AuthConfig:
    """解析认证配置"""
    if not data:
        return AuthConfig()
    return AuthConfig(
        mode=data.get("mode", "token"),
        token=data.get("token"),
    )


def _parse_interceptor_config(data: Dict[str, Any]) -> InterceptorConfig:
    """解析拦截器配置"""
    if not data:
        return InterceptorConfig()
    return InterceptorConfig(
        logging=data.get("logging", True),
        metrics=data.get("metrics", False),
        auth=data.get("auth", False),
        auth_config=_parse_auth_config(data.get("auth_config", {})),
        retry=_parse_retry_config(data.get("retry", {})),
        custom=data.get("custom", []),
    )


def _parse_heartbeat_config(data: Dict[str, Any]) -> HeartbeatConfig:
    """解析心跳配置"""
    if not data:
        return HeartbeatConfig()
    return HeartbeatConfig(
        enabled=data.get("enabled", False),
        interval=data.get("interval", 30.0),
        timeout=data.get("timeout", 90.0),
        critical_peers=data.get("critical_peers", []),
    )


# ==================== 适配器函数 ====================

def from_node_comm_config(comm_config) -> Dict[str, Any]:
    """
    从 NodeCommConfig 提取通信层需要的配置
    
    这是 NodeCommConfig 和通信层之间的适配器。
    
    Args:
        comm_config: NodeCommConfig 实例（来自 oiafed.config）
        
    Returns:
        通信层可用的配置字典
        
    Example:
        from oiafed.config import NodeCommConfig
        from oiafed.comm.config import from_node_comm_config
        
        comm_config = NodeCommConfig(node_id="trainer", ...)
        comm_dict = from_node_comm_config(comm_config)
    """
    result = {
        "node_id": comm_config.node_id,
        "default_timeout": comm_config.default_timeout,
        "debug": getattr(comm_config, "debug", False),
        "advertised_address": getattr(comm_config, "advertised_address", None),
        "listen": comm_config.listen,
    }
    
    # 处理 transport
    if hasattr(comm_config, "transport"):
        transport = comm_config.transport
        if hasattr(transport, "mode"):
            result["transport"] = {
                "mode": transport.mode,
            }
            if hasattr(transport, "grpc"):
                grpc = transport.grpc
                result["transport"]["grpc"] = {
                    "max_message_size": getattr(grpc, "max_message_size", 104857600),
                }
    
    # 处理 serialization
    if hasattr(comm_config, "serialization") and comm_config.serialization:
        if isinstance(comm_config.serialization, dict):
            result["serialization"] = comm_config.serialization
        else:
            result["serialization"] = {"default": "pickle"}
    
    # 处理 heartbeat
    if hasattr(comm_config, "heartbeat") and comm_config.heartbeat:
        if isinstance(comm_config.heartbeat, dict):
            result["heartbeat"] = comm_config.heartbeat
    
    return result


def load_config(path: str):
    """
    从 YAML 文件加载配置
    
    注意：此函数已废弃，建议使用 oiafed.config.load_config()
    
    Returns:
        NodeCommConfig 实例（来自 oiafed.config）
    """
    import warnings
    warnings.warn(
        "comm.config.load_config() 已废弃，请使用 oiafed.config.load_config()",
        DeprecationWarning,
        stacklevel=2
    )
    
    from oiafed.config import load_config as config_load
    full_config = config_load(path)
    return full_config.get_comm_config()


# ==================== 向后兼容别名 ====================
# TransportConfig 别名（用于 transport/factory.py）
TransportConfig = CommTransportConfig

# ==================== 导出 ====================

__all__ = [
    # 内部配置类
    "TlsConfig",
    "MemoryTransportConfig",
    "GrpcTransportConfig",
    "CommTransportConfig",
    "TransportConfig",  # 别名
    "MethodSerializationConfig",
    "SerializationConfig",
    "RetryConfig",
    "AuthConfig",
    "InterceptorConfig",
    "HeartbeatConfig",
    "MethodOptions",
    
    # 适配器
    "from_node_comm_config",
    
    # 向后兼容（已废弃）
    "NodeConfig",
    "load_config",
]