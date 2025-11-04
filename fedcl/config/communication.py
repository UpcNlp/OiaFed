"""
通信配置类定义（简化版）
统一的通信配置，通过 role 字段区分服务端和客户端
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from .base import BaseConfig


@dataclass
class CommunicationConfig(BaseConfig):
    """
    通信配置（统一的配置类）

    通过 role 字段区分：
    - role="server": 服务端配置
    - role="client": 客户端配置
    - role="local": 本地模式（包含服务端和多个客户端）
    """

    # ========== 基本部署信息 ==========
    mode: str = "process"                      # memory / process / network
    role: str = "server"                       # server / client / local
    node_id: Optional[str] = None              # 节点ID（None = 自动生成）

    # ========== 传输层配置（字典形式，灵活）==========
    transport: Dict[str, Any] = field(default_factory=lambda: {
        "type": "websocket",                   # tcp / websocket / grpc / in_memory
        "host": None,                          # 监听地址（服务端）
        "port": None,                          # 监听端口（服务端）
        "timeout": 30.0,
        # 客户端专用字段
        "server": {                            # 服务端地址（客户端使用）
            "host": "127.0.0.1",
            "port": 8000
        },
        "connection_retry": {                  # 连接重试（客户端使用）
            "enabled": True,
            "max_attempts": 10,
            "initial_delay": 1.0,
            "backoff_factor": 2.0,
            "max_delay": 60.0
        }
    })

    # ========== 通信管理配置（字典形式）==========
    communication: Dict[str, Any] = field(default_factory=lambda: {
        "rpc": {
            "timeout": 120.0,
            "retry_attempts": 3,
            "retry_delay": 5.0
        },
        "heartbeat": {
            "interval": 30.0,
            "timeout": 90.0
        },
        "serialization": {
            "format": "pickle",                # pickle / json / protobuf
            "compression": "gzip"              # none / gzip / lz4
        },
        # 客户端专用
        "registration": {
            "auto_register": True,
            "retry_attempts": 3,
            "retry_delay": 5.0,
            "timeout": 60.0
        }
    })

    # ========== 连接管理配置（字典形式）==========
    connection: Dict[str, Any] = field(default_factory=lambda: {
        # 服务端专用
        "max_clients": 100,
        "client_timeout": 300.0,
        "disconnection_policy": {
            "action": "wait",                  # wait / skip / terminate
            "wait_timeout": 60.0
        },
        # 客户端专用
        "keep_alive": True
    })

    # ========== 本地模式专用 ==========
    # 如果 role="local"，这些字段才有意义
    client_count: int = 3                      # 客户端数量
    client_id_pattern: str = "client_{index:03d}"  # 客户端ID模式

    def __post_init__(self):
        """根据 role 设置默认值"""
        if self.role == "server":
            # 服务端必须有监听地址和端口
            if self.transport.get("host") is None:
                self.transport["host"] = "127.0.0.1"  # 默认使用本地地址
            if self.transport.get("port") is None:
                self.transport["port"] = 8000

        elif self.role == "client":
            # 客户端必须有服务端地址
            if "server" not in self.transport:
                self.transport["server"] = {
                    "host": "127.0.0.1",
                    "port": 8000
                }

    def is_server(self) -> bool:
        """是否是服务端配置"""
        return self.role == "server"

    def is_client(self) -> bool:
        """是否是客户端配置"""
        return self.role == "client"

    def is_local(self) -> bool:
        """是否是本地模式"""
        return self.role == "local"
