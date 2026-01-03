"""
通信层配置

包含序列化、拦截器、心跳等通信相关的配置类及解析方法。

结构：
    SerializationConfig
    └── MethodSerializationConfig
    
    InterceptorConfig
    ├── AuthConfig
    └── RetryConfig
    
    HeartbeatConfig
    
    MethodOptions
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ==================== 序列化配置 ====================

@dataclass
class MethodSerializationConfig:
    """方法级序列化配置"""
    serializer: str = "json"
    compress: bool = False


def parse_method_serialization_config(data: Optional[Dict[str, Any]]) -> MethodSerializationConfig:
    """解析方法序列化配置"""
    if not data:
        return MethodSerializationConfig()
    return MethodSerializationConfig(
        serializer=data.get("serializer", "json"),
        compress=data.get("compress", False),
    )


@dataclass
class SerializationConfig:
    """
    序列化配置
    
    Attributes:
        default: 默认序列化器（pickle 用于复杂对象，json 用于简单数据）
        methods: 方法级别的序列化配置
    """
    default: str = "pickle"
    methods: Dict[str, MethodSerializationConfig] = field(default_factory=dict)


def parse_serialization_config(data: Optional[Dict[str, Any]]) -> SerializationConfig:
    """解析序列化配置"""
    if not data:
        return SerializationConfig()
    
    methods = {}
    for method_name, method_data in data.get("methods", {}).items():
        methods[method_name] = parse_method_serialization_config(method_data)
    
    return SerializationConfig(
        default=data.get("default", "pickle"),
        methods=methods,
    )


# ==================== 认证配置 ====================

@dataclass
class AuthConfig:
    """认证配置"""
    mode: str = "token"  # token | mutual_tls | custom
    token: Optional[str] = None


def parse_auth_config(data: Optional[Dict[str, Any]]) -> AuthConfig:
    """解析认证配置"""
    if not data:
        return AuthConfig()
    return AuthConfig(
        mode=data.get("mode", "token"),
        token=data.get("token"),
    )


# ==================== 重试配置 ====================

@dataclass
class RetryConfig:
    """RPC 重试配置"""
    enabled: bool = False
    max_retries: int = 3
    backoff: float = 1.0


def parse_retry_config(data: Optional[Dict[str, Any]]) -> RetryConfig:
    """解析重试配置"""
    if not data:
        return RetryConfig()
    return RetryConfig(
        enabled=data.get("enabled", False),
        max_retries=data.get("max_retries", 3),
        backoff=data.get("backoff", 1.0),
    )


# ==================== 拦截器配置 ====================

@dataclass
class InterceptorConfig:
    """
    拦截器配置
    
    Attributes:
        logging: 是否启用日志拦截器
        metrics: 是否启用指标拦截器
        auth: 是否启用认证拦截器
        auth_config: 认证配置
        retry: 重试配置
        custom: 自定义拦截器列表
    """
    logging: bool = True
    metrics: bool = False
    auth: bool = False
    auth_config: AuthConfig = field(default_factory=AuthConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    custom: List[str] = field(default_factory=list)


def parse_interceptor_config(data: Optional[Dict[str, Any]]) -> InterceptorConfig:
    """解析拦截器配置"""
    if not data:
        return InterceptorConfig()
    return InterceptorConfig(
        logging=data.get("logging", True),
        metrics=data.get("metrics", False),
        auth=data.get("auth", False),
        auth_config=parse_auth_config(data.get("auth_config")),
        retry=parse_retry_config(data.get("retry")),
        custom=data.get("custom", []),
    )


# ==================== 心跳配置 ====================

@dataclass
class HeartbeatConfig:
    """
    应用层心跳配置
    
    注意：gRPC 层心跳配置在 GrpcConfig 中
    
    Attributes:
        enabled: 是否启用心跳
        interval: 心跳间隔（秒）
        timeout: 心跳超时（秒）
        critical_peers: 关键节点列表（断开时触发 shutdown）
    """
    enabled: bool = False
    interval: float = 30.0
    timeout: float = 90.0
    critical_peers: List[str] = field(default_factory=list)


def parse_heartbeat_config(data: Optional[Dict[str, Any]]) -> HeartbeatConfig:
    """解析心跳配置"""
    if not data:
        return HeartbeatConfig()
    return HeartbeatConfig(
        enabled=data.get("enabled", False),
        interval=data.get("interval", 30.0),
        timeout=data.get("timeout", 90.0),
        critical_peers=data.get("critical_peers", []),
    )


# ==================== 方法选项 ====================

@dataclass
class MethodOptions:
    """
    方法选项（RPC 调用时使用）
    
    Attributes:
        serializer: 指定序列化器
        timeout: 超时时间
        require_auth: 是否需要认证
    """
    serializer: Optional[str] = None
    timeout: Optional[float] = None
    require_auth: bool = False


# ==================== 导出 ====================

__all__ = [
    # 序列化配置
    "MethodSerializationConfig",
    "SerializationConfig",
    "parse_method_serialization_config",
    "parse_serialization_config",
    # 认证配置
    "AuthConfig",
    "parse_auth_config",
    # 重试配置
    "RetryConfig",
    "parse_retry_config",
    # 拦截器配置
    "InterceptorConfig",
    "parse_interceptor_config",
    # 心跳配置
    "HeartbeatConfig",
    "parse_heartbeat_config",
    # 方法选项
    "MethodOptions",
]
