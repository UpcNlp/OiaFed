"""
MOE-FedCL 配置类定义
fedcl/config.py
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional

import yaml


@dataclass
class TransportLayerConfig:
    """传输层配置"""
    port: Optional[int] = None   # 无需指定默认值，默认值会在工厂方法中根据节点类型指定

    # 通用配置
    timeout: float = 30.0
    retry_attempts: int = 3

    # 模式特定配置
    # host 默认值根据模式动态决定：process模式为 127.0.0.1，network模式为 0.0.0.0
    host: Optional[str] = None
    websocket_port: int = 9501

    # Network模式额外配置
    ssl_enabled: bool = False
    connection_timeout: float = 30.0

    # 服务器地址（仅客户端使用）
    server_host: Optional[str] = None  # 服务器主机地址，如 "192.168.31.68"
    server_port: Optional[int] = None  # 服务器端口，如 8000

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "timeout": self.timeout,
            "retry_attempts": self.retry_attempts,
            "specific_config": {
                "host": self.host,
                "port": self.port,
                "websocket_port": self.websocket_port,
                "ssl_enabled": self.ssl_enabled,
                "connection_timeout": self.connection_timeout,
                "server_host": self.server_host,
                "server_port": self.server_port
            }
        }


@dataclass
class CommunicationLayerConfig:
    """通信层配置"""
    heartbeat_interval: float = 30.0
    heartbeat_timeout: float = 90.0
    registration_timeout: float = 60.0
    max_clients: int = 100
    rpc_timeout: float = 120.0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "heartbeat_interval": self.heartbeat_interval,
            "heartbeat_timeout": self.heartbeat_timeout,
            "registration_timeout": self.registration_timeout,
            "max_clients": self.max_clients,
            "rpc_timeout": self.rpc_timeout
        }


@dataclass
class FederationLayerConfig:
    """联邦学习层配置"""
    coordinator_id: str = "fed_coordinator"
    max_rounds: int = 100
    min_clients: int = 2
    client_selection: str = "all"  # all/random/custom
    training_config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "coordinator_id": self.coordinator_id,
            "max_rounds": self.max_rounds,
            "min_clients": self.min_clients,
            "client_selection": self.client_selection,
            "training_config": self.training_config
        }


@dataclass
class StubLayerConfig:
    """Stub层配置（客户端专用）"""
    auto_register: bool = True
    registration_retry_attempts: int = 3
    registration_retry_delay: float = 5.0
    request_timeout: float = 120.0
    max_concurrent_requests: int = 5

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "auto_register": self.auto_register,
            "registration_retry_attempts": self.registration_retry_attempts,
            "registration_retry_delay": self.registration_retry_delay,
            "request_timeout": self.request_timeout,
            "max_concurrent_requests": self.max_concurrent_requests
        }


@dataclass
class ServerConfig:
    """服务端配置类"""

    # 基本配置
    mode: str = "process"  # memory/process/network
    server_id: Optional[str] = None  # 如果为None，自动生成

    # 各层配置
    transport: TransportLayerConfig = field(default_factory=TransportLayerConfig)
    communication: CommunicationLayerConfig = field(default_factory=CommunicationLayerConfig)
    federation: FederationLayerConfig = field(default_factory=FederationLayerConfig)

    def to_dict(self) -> Dict[str, Any]:
        """转换为标准配置字典格式（兼容现有代码）"""
        return {
            "mode": self.mode,
            "transport": self.transport.to_dict(),
            "communication": self.communication.to_dict(),
            "federation": self.federation.to_dict()
        }

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ServerConfig':
        """从YAML文件加载配置"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        # 从 transport 配置中读取
        transport_config = config_dict.get("transport", {})

        return cls(
            mode=config_dict.get("mode", "process"),
            server_id=config_dict.get("server_id"),
            transport=TransportLayerConfig(
                **transport_config
            ),
            communication=CommunicationLayerConfig(
                **config_dict.get("communication", {})
            ),
            federation=FederationLayerConfig(
                **config_dict.get("federation", {})
            )
        )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ServerConfig':
        """从字典创建配置"""
        transport_dict = config_dict.get("transport", {})
        specific_config = transport_dict.get("specific_config", {})

        return cls(
            mode=config_dict.get("mode", "process"),
            server_id=config_dict.get("server_id"),
            transport=TransportLayerConfig(
                timeout=transport_dict.get("timeout", 30.0),
                retry_attempts=transport_dict.get("retry_attempts", 3),
                host=specific_config.get("host", "127.0.0.1"),
                port=specific_config.get("port", 8000),
                websocket_port=specific_config.get("websocket_port", 9501),
                ssl_enabled=specific_config.get("ssl_enabled", False),
                connection_timeout=specific_config.get("connection_timeout", 30.0)
            ),
            communication=CommunicationLayerConfig(
                **config_dict.get("communication", {})
            ),
            federation=FederationLayerConfig(
                **config_dict.get("federation", {})
            )
        )


@dataclass
class ClientConfig:
    """客户端配置类"""

    # 基本配置
    mode: str = "process"  # memory/process/network
    client_id: Optional[str] = None  # 如果为None，自动生成

    # 各层配置
    transport: TransportLayerConfig = field(default_factory=TransportLayerConfig)
    communication: CommunicationLayerConfig = field(default_factory=CommunicationLayerConfig)
    stub: StubLayerConfig = field(default_factory=StubLayerConfig)

    def to_dict(self) -> Dict[str, Any]:
        """转换为标准配置字典格式（兼容现有代码）"""
        return {
            "mode": self.mode,
            "transport": self.transport.to_dict(),
            "communication": self.communication.to_dict(),
            "stub": self.stub.to_dict()
        }

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ClientConfig':
        """从YAML文件加载配置"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        # 从 transport 配置中读取
        transport_config = config_dict.get("transport", {})

        return cls(
            mode=config_dict.get("mode", "process"),
            client_id=config_dict.get("client_id"),
            transport=TransportLayerConfig(
                **transport_config
            ),
            communication=CommunicationLayerConfig(
                **config_dict.get("communication", {})
            ),
            stub=StubLayerConfig(
                **config_dict.get("stub", {})
            )
        )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ClientConfig':
        """从字典创建配置"""
        transport_dict = config_dict.get("transport", {})
        specific_config = transport_dict.get("specific_config", {})

        return cls(
            mode=config_dict.get("mode", "process"),
            client_id=config_dict.get("client_id"),
            transport=TransportLayerConfig(
                timeout=transport_dict.get("timeout", 30.0),
                retry_attempts=transport_dict.get("retry_attempts", 3),
                host=specific_config.get("host", "127.0.0.1"),
                port=specific_config.get("port", 0),
                websocket_port=specific_config.get("websocket_port", 9501),
                ssl_enabled=specific_config.get("ssl_enabled", False),
                connection_timeout=specific_config.get("connection_timeout", 30.0),
                server_host=specific_config.get("server_host"),
                server_port=specific_config.get("server_port")
            ),
            communication=CommunicationLayerConfig(
                **config_dict.get("communication", {})
            ),
            stub=StubLayerConfig(
                **config_dict.get("stub", {})
            )
        )


# 便捷函数

def load_server_config(yaml_path: str) -> ServerConfig:
    """加载服务端配置"""
    return ServerConfig.from_yaml(yaml_path)


def load_client_config(yaml_path: str) -> ClientConfig:
    """加载客户端配置"""
    return ClientConfig.from_yaml(yaml_path)


def create_default_server_config(mode: str = "process", host: str = "127.0.0.1", port: int = 8000) -> ServerConfig:
    """创建默认服务端配置

    Args:
        mode: 通信模式 (memory/process/network)
        host: 服务端主机地址
        port: 服务端监听端口

    Returns:
        ServerConfig: 服务端配置对象
    """
    return ServerConfig(
        mode=mode,
        transport=TransportLayerConfig(host=host, port=port)
    )


def create_default_client_config(
    mode: str = "process",
    client_host: str = "127.0.0.1",
    client_port: int = 0
) -> ClientConfig:
    """创建默认客户端配置

    Args:
        mode: 通信模式 (memory/process/network)
        client_host: 客户端监听主机地址
        client_port: 客户端监听端口（0表示自动分配）

    Returns:
        ClientConfig: 客户端配置对象
    """
    return ClientConfig(
        mode=mode,
        transport=TransportLayerConfig(host=client_host, port=client_port)
    )
