"""
Node 通信层配置定义
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
import yaml


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
class TransportConfig:
    """传输层配置"""
    
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


@dataclass
class MethodOptions:
    """方法选项"""
    
    serializer: Optional[str] = None             # 指定序列化器
    timeout: Optional[float] = None              # 超时时间
    require_auth: bool = False                   # 是否需要认证


@dataclass
class NodeConfig:
    """节点配置"""

    # 基本配置
    node_id: str = ""                            # 节点 ID
    debug: bool = False                          # 调试模式

    # 超时配置
    default_timeout: float = 300.0                # 默认调用超时（秒）

    # 网络地址
    advertised_address: Optional[str] = None     # 对外公布的地址（NAT/容器环境）
    listen: Optional[Dict[str, Any]] = None      # 监听配置 {"host": "0.0.0.0", "port": 50051}

    # 传输层配置
    transport: TransportConfig = field(default_factory=TransportConfig)

    # 序列化配置
    serialization: SerializationConfig = field(default_factory=SerializationConfig)

    # 拦截器配置
    interceptors: InterceptorConfig = field(default_factory=InterceptorConfig)

    # 心跳配置
    heartbeat: HeartbeatConfig = field(default_factory=HeartbeatConfig)


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


def _parse_transport_config(data: Dict[str, Any]) -> TransportConfig:
    """解析传输层配置"""
    if not data:
        return TransportConfig()
    return TransportConfig(
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
    )


def load_config(path: str) -> NodeConfig:
    """从 YAML 文件加载配置"""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    
    return NodeConfig(
        node_id=data.get("node_id", ""),
        debug=data.get("debug", False),
        default_timeout=data.get("default_timeout", 300.0),
        advertised_address=data.get("advertised_address"),
        transport=_parse_transport_config(data.get("transport", {})),
        serialization=_parse_serialization_config(data.get("serialization", {})),
        interceptors=_parse_interceptor_config(data.get("interceptors", {})),
        heartbeat=_parse_heartbeat_config(data.get("heartbeat", {})),
    )
