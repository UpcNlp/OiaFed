"""
MOE-FedCL 基础类型定义
moe_fedcl/types.py
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import uuid


class CommunicationMode(Enum):
    """通信模式枚举"""
    MEMORY = "memory"
    PROCESS = "process" 
    NETWORK = "network"


class ConnectionStatus(Enum):
    """连接状态枚举"""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ACTIVE = "active"
    IDLE = "idle"
    DISCONNECTED = "disconnected"
    ERROR = "error"


class ConnectionType(Enum):
    """连接类型枚举"""
    BUSINESS_RPC = "business_rpc"
    BUSINESS_PUSH = "business_push"
    CONTROL_REGISTER = "control_register"
    CONTROL_HEARTBEAT = "control_heartbeat"
    CONTROL_STATUS = "control_status"


class RegistrationStatus(Enum):
    """注册状态枚举"""
    UNREGISTERED = "unregistered"
    REGISTERING = "registering"
    REGISTERED = "registered"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"


class TrainingStatus(Enum):
    """训练状态枚举"""
    IDLE = "idle"
    TRAINING = "training"
    EVALUATING = "evaluating"
    BUSY = "busy"


class HealthStatus(Enum):
    """健康状态枚举"""
    HEALTHY = "healthy"
    WARNING = "warning"
    ERROR = "error"
    TIMEOUT = "timeout"


class FederationStatus(Enum):
    """联邦学习状态枚举"""
    INITIALIZING = "initializing"
    WAITING_CLIENTS = "waiting_clients"
    TRAINING = "training"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class ClientInfo:
    """客户端信息"""
    client_id: str
    client_type: str = "learner"
    capabilities: List[str] = field(default_factory=lambda: ["train", "evaluate"])
    metadata: Dict[str, Any] = field(default_factory=dict)
    registration_time: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    status: RegistrationStatus = RegistrationStatus.REGISTERED


@dataclass
class RegistrationRequest:
    """客户端注册请求"""
    client_id: str
    client_type: str = "learner"
    capabilities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TrackerContext:
    """实验跟踪上下文（用于共享 run）

    支持联邦学习场景下 Server 和多个 Clients 共享同一个实验 run。
    适用于 MLflow、WandB、TensorBoard 等多种跟踪系统。

    Attributes:
        enabled: 是否启用实验跟踪
        tracker_type: 跟踪器类型 (mlflow/wandb/tensorboard/none)
        shared_run_id: 共享的 run ID（用于多进程/多机写入同一 run）
        config: 跟踪器配置（如 tracking_uri, experiment_id 等）
        metadata: 额外的元数据
    """
    enabled: bool = False
    tracker_type: str = "none"
    shared_run_id: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RegistrationResponse:
    """注册响应消息"""
    success: bool
    client_id: str
    server_info: Dict[str, Any] = field(default_factory=dict)
    tracker_context: Optional[TrackerContext] = None
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TrainingRequest:
    """训练请求消息"""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    client_id: str = ""
    method_name: str = "train"
    parameters: Dict[str, Any] = field(default_factory=dict)
    timeout: float = 120.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TrainingResponse:
    """训练响应消息"""
    request_id: str
    client_id: str
    success: bool
    result: Any = None
    error_message: Optional[str] = None
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class HeartbeatMessage:
    """心跳消息"""
    client_id: str
    status: str = "alive"
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class EventMessage:
    """事件消息"""
    event_type: str
    source_id: str
    target_id: Optional[str] = None
    data: Any = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Connection:
    """连接信息"""
    connection_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = ""
    target_id: str = ""
    connection_type: ConnectionType = ConnectionType.BUSINESS_RPC
    status: ConnectionStatus = ConnectionStatus.CONNECTING
    created_time: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TransportConfig:
    """传输配置"""
    type: str  # memory/process/network
    node_id: str = ""
    timeout: float = 30.0
    retry_attempts: int = 3
    specific_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CommunicationConfig:
    """通信配置"""
    heartbeat_interval: float = 30.0
    heartbeat_timeout: float = 90.0
    registration_timeout: float = 60.0
    max_clients: int = 100
    rpc_timeout: float = 120.0


@dataclass
class FederationConfig:
    """联邦配置"""
    coordinator_id: str = "fed_coordinator"
    max_rounds: int = 100
    min_clients: int = 2
    client_selection: str = "all"  # all/random/custom
    training_config: Dict[str, Any] = field(default_factory=dict)
    

# 数据类型别名
ModelData = dict[str, Any]
TrainingResult = dict[str, Any]
EvaluationResult = dict[str, Any]
MetricsData = dict[str, Any]
RoundResult = dict[str, Any]