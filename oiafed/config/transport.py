"""
传输层配置

包含 gRPC 和 Memory 传输的配置类及解析方法。

结构：
    TransportConfig
    ├── GrpcConfig
    │   └── TlsConfig
    └── MemoryTransportConfig
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ==================== TLS 配置 ====================

@dataclass
class TlsConfig:
    """TLS 配置"""
    enabled: bool = False
    cert_file: Optional[str] = None
    key_file: Optional[str] = None
    ca_file: Optional[str] = None
    mutual_tls: bool = False


def parse_tls_config(data: Optional[Dict[str, Any]]) -> TlsConfig:
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


# ==================== gRPC 配置 ====================

@dataclass
class GrpcConfig:
    """
    gRPC 配置
    
    Attributes:
        host: 绑定地址
        port: 端口
        max_message_size: 最大消息大小（字节）
        max_workers: 最大工作线程数
        tls: TLS 配置
        
        # 心跳配置（gRPC 层）
        heartbeat_enabled: 是否启用心跳
        heartbeat_interval: 心跳间隔（秒）
        heartbeat_timeout: 心跳超时（秒）
        heartbeat_check_interval: 健康检查间隔（秒）
        
        # 连接管理
        max_connection_wait_time: 最大连接等待时间（秒）
        auto_shutdown_on_failure: 连接失败后是否自动关闭
        critical_peers: 关键节点列表
        
        # 双线程优化
        dual_thread_enabled: 是否启用双线程模式
    """
    # 基础配置
    host: str = "0.0.0.0"
    port: int = 50051
    max_message_size: int = 100 * 1024 * 1024  # 100MB
    max_workers: int = 10
    tls: TlsConfig = field(default_factory=TlsConfig)
    
    # 心跳配置
    heartbeat_enabled: bool = True
    heartbeat_interval: float = 5.0
    heartbeat_timeout: float = 30.0
    heartbeat_check_interval: float = 10.0
    
    # 连接管理
    max_connection_wait_time: float = 300.0
    auto_shutdown_on_failure: bool = True
    critical_peers: List[str] = field(default_factory=list)
    
    # 双线程优化
    dual_thread_enabled: bool = True
    
    def __post_init__(self):
        """初始化后处理"""
        if isinstance(self.tls, dict):
            self.tls = TlsConfig(**self.tls)
    
    def get_address(self) -> str:
        """获取完整地址"""
        return f"{self.host}:{self.port}"
    
    # 向后兼容别名
    @property
    def max_message_length(self) -> int:
        """向后兼容属性"""
        return self.max_message_size


def parse_grpc_config(data: Optional[Dict[str, Any]]) -> GrpcConfig:
    """解析 gRPC 配置"""
    if not data:
        return GrpcConfig()
    
    # 解析 TLS 配置
    tls_config = parse_tls_config(data.get("tls"))
    
    # 从 heartbeat 子配置或直接字段获取心跳配置
    heartbeat = data.get("heartbeat", {})
    
    # 从 dual_thread 子配置获取双线程配置
    dual_thread = data.get("dual_thread", {})
    
    return GrpcConfig(
        host=data.get("host", "0.0.0.0"),
        port=data.get("port", 50051),
        max_message_size=data.get("max_message_size", 100 * 1024 * 1024),
        max_workers=data.get("max_workers", 10),
        tls=tls_config,
        # 心跳配置
        heartbeat_enabled=heartbeat.get("enabled", True),
        heartbeat_interval=heartbeat.get("interval", 5.0),
        heartbeat_timeout=heartbeat.get("timeout", 30.0),
        heartbeat_check_interval=heartbeat.get("check_interval", 10.0),
        # 连接管理
        max_connection_wait_time=heartbeat.get("max_connection_wait_time", 300.0),
        auto_shutdown_on_failure=heartbeat.get("auto_shutdown_on_failure", True),
        critical_peers=heartbeat.get("critical_peers", []),
        # 双线程优化
        dual_thread_enabled=dual_thread.get("enabled", True),
    )


# ==================== Memory 传输配置 ====================

@dataclass
class MemoryTransportConfig:
    """Memory 传输配置"""
    zero_copy: bool = True
    simulate_latency: bool = False
    latency_ms: float = 0.0


def parse_memory_config(data: Optional[Dict[str, Any]]) -> MemoryTransportConfig:
    """解析 Memory 传输配置"""
    if not data:
        return MemoryTransportConfig()
    return MemoryTransportConfig(
        zero_copy=data.get("zero_copy", True),
        simulate_latency=data.get("simulate_latency", False),
        latency_ms=data.get("latency_ms", 0.0),
    )


# ==================== 传输层配置 ====================

@dataclass
class TransportConfig:
    """
    传输层配置
    
    Attributes:
        mode: 传输模式（memory 或 grpc）
        grpc: gRPC 配置
        memory: Memory 传输配置
    """
    mode: str = "memory"
    grpc: GrpcConfig = field(default_factory=GrpcConfig)
    memory: MemoryTransportConfig = field(default_factory=MemoryTransportConfig)
    
    def __post_init__(self):
        """初始化后处理"""
        if isinstance(self.grpc, dict):
            self.grpc = GrpcConfig(**self.grpc)
        if isinstance(self.memory, dict):
            self.memory = MemoryTransportConfig(**self.memory)


def parse_transport_config(data: Optional[Dict[str, Any]]) -> TransportConfig:
    """解析传输层配置"""
    if not data:
        return TransportConfig()
    
    return TransportConfig(
        mode=data.get("mode", "memory"),
        grpc=parse_grpc_config(data.get("grpc")),
        memory=parse_memory_config(data.get("memory")),
    )


# ==================== 连接重试配置 ====================

@dataclass
class ConnectionRetryConfig:
    """
    连接重试配置
    
    Attributes:
        enabled: 是否启用重试
        max_retries: 最大重试次数（-1 表示无限重试）
        retry_interval: 重试间隔（秒）
        timeout: 总超时时间（秒）
        backoff: 退避策略
        backoff_factor: 退避因子
    """
    enabled: bool = True
    max_retries: int = 10
    retry_interval: float = 2.0
    timeout: float = 60.0
    backoff: str = "exponential"
    backoff_factor: float = 1.5
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于传递给 Node.connect）"""
        return {
            "enabled": self.enabled,
            "max_retries": self.max_retries,
            "retry_interval": self.retry_interval,
            "timeout": self.timeout,
            "backoff": self.backoff,
            "backoff_factor": self.backoff_factor,
        }


def parse_connection_retry_config(data: Optional[Dict[str, Any]]) -> ConnectionRetryConfig:
    """解析连接重试配置"""
    if not data:
        return ConnectionRetryConfig()
    return ConnectionRetryConfig(
        enabled=data.get("enabled", True),
        max_retries=data.get("max_retries", 10),
        retry_interval=data.get("retry_interval", 2.0),
        timeout=data.get("timeout", 60.0),
        backoff=data.get("backoff", "exponential"),
        backoff_factor=data.get("backoff_factor", 1.5),
    )


# ==================== 导出 ====================

__all__ = [
    # 配置类
    "TlsConfig",
    "GrpcConfig",
    "MemoryTransportConfig",
    "TransportConfig",
    "ConnectionRetryConfig",
    # 解析方法
    "parse_tls_config",
    "parse_grpc_config",
    "parse_memory_config",
    "parse_transport_config",
    "parse_connection_retry_config",
]