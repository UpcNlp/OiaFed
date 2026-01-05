"""
节点配置（根配置）

包含 GlobalConfig、NodeCommConfig、NodeConfig 等核心配置类。
NodeConfig 是整个配置树的根节点。

结构：
    NodeConfig (根)
    ├── GlobalConfig (共享字段源)
    ├── LogConfig
    ├── TrackerConfig
    ├── TransportConfig
    ├── SerializationConfig
    ├── InterceptorConfig
    ├── HeartbeatConfig
    ├── ConnectionRetryConfig
    ├── ComponentConfig (trainer/learner/aggregator/model)
    ├── DatasetConfig[]
    └── CallbackConfig[]
    
    NodeCommConfig (通信层视图，引用 NodeConfig 的子集)
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

# 导入子配置
from .transport import (
    TransportConfig, GrpcConfig, parse_transport_config, ConnectionRetryConfig, parse_connection_retry_config
)
from .comm import (
    SerializationConfig, InterceptorConfig, HeartbeatConfig,
    parse_serialization_config, parse_interceptor_config, parse_heartbeat_config
)
from .logging_config import LogConfig, parse_log_config
from .tracker import TrackerConfig, parse_tracker_config
from .component import (
    ComponentConfig, DatasetConfig, CallbackConfig,
    parse_component_config, parse_datasets_config, parse_callbacks_config
)
from .defaults import DEFAULT_TIMEOUT


# ==================== 枚举类型 ====================

class LogLevel(str, Enum):
    """日志级别"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class TransportMode(str, Enum):
    """传输模式"""
    MEMORY = "memory"
    GRPC = "grpc"


class BackoffStrategy(str, Enum):
    """退避策略"""
    CONSTANT = "constant"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"


class NodeRole(str, Enum):
    """节点角色"""
    TRAINER = "trainer"
    LEARNER = "learner"
    BOTH = "both"


# ==================== 全局配置 ====================

@dataclass
class GlobalConfig:
    """
    全局共享配置
    
    这些字段会自动同步到子配置（LogConfig、TrackerConfig 等），
    避免在多处重复设置相同的值。
    
    Attributes:
        exp_name: 实验名称，用于组织日志和追踪数据
        run_name: 运行名称，None 则自动生成（基于时间戳）
        log_dir: 日志根目录
    """
    exp_name: str = "default"
    run_name: Optional[str] = None
    log_dir: str = "./logs"
    
    def generate_run_name(self) -> str:
        """生成默认的 run_name（基于时间戳）"""
        if self.run_name:
            return self.run_name
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def get_run_name(self) -> str:
        """获取 run_name，如果为 None 则生成"""
        return self.run_name or self.generate_run_name()


def parse_global_config(data: Optional[Dict[str, Any]]) -> GlobalConfig:
    """解析全局配置"""
    if not data:
        return GlobalConfig()
    return GlobalConfig(
        exp_name=data.get("exp_name", "default"),
        run_name=data.get("run_name"),
        log_dir=data.get("log_dir", "./logs"),
    )


# ==================== 通信节点配置 ====================

@dataclass
class NodeCommConfig:
    """
    通信节点配置（给 comm.Node 使用）
    
    这是 NodeConfig 的子集，只包含通信相关的配置。
    共享字段（如 node_id）从 NodeConfig 获取，不重复存储。
    
    Attributes:
        node_id: 节点唯一标识符（引用自 NodeConfig）
        default_timeout: 默认超时时间（秒）
        debug: 是否开启调试模式
        advertised_address: 广播地址
        listen: 监听配置
        transport: 传输层配置
        serialization: 序列化配置（SerializationConfig 对象）
        heartbeat: 心跳配置
        interceptors: 拦截器配置
    """
    node_id: str
    default_timeout: float = DEFAULT_TIMEOUT
    debug: bool = False
    advertised_address: Optional[str] = None
    listen: Optional[Dict[str, Any]] = None
    transport: TransportConfig = field(default_factory=TransportConfig)
    serialization: Optional[SerializationConfig] = None
    heartbeat: Optional[Dict[str, Any]] = None
    interceptors: Optional[InterceptorConfig] = None
    
    @property
    def transport_mode(self) -> str:
        """获取传输模式"""
        return self.transport.mode
    
    @property
    def listen_port(self) -> Optional[int]:
        """获取监听端口"""
        if self.listen:
            return self.listen.get("port")
        return None
    
    @property
    def listen_host(self) -> str:
        """获取监听地址"""
        if self.listen:
            return self.listen.get("host", "0.0.0.0")
        return "0.0.0.0"
    
    @property
    def grpc_address(self) -> str:
        """获取 gRPC 地址"""
        return f"{self.transport.grpc.host}:{self.transport.grpc.port}"


# ==================== 节点配置（根配置）====================

@dataclass
class NodeConfig:
    """
    节点配置（根配置）
    
    联邦学习系统中单个节点的完整配置。
    这是配置树的根节点，包含所有子配置。
    
    共享字段说明：
    - node_id: 在此定义，NodeCommConfig 引用
    - exp_name, run_name, log_dir: 在 GlobalConfig 定义，同步到子配置
    
    Attributes:
        node_id: 节点唯一标识符
        role: 节点角色（trainer, learner, both）
        
        # 全局配置（共享字段源）
        global_config: 全局共享配置
        
        # 连接配置
        listen: 监听配置（Trainer 使用）
        connect_to: 连接目标列表（Learner 使用）
        
        # 基础设施配置
        transport: 传输层配置
        connection_retry: 连接重试配置
        logging: 日志配置
        tracker: 追踪配置
        
        # 通信配置
        serialization: 序列化配置
        heartbeat: 心跳配置
        interceptors: 拦截器配置
        
        # 组件配置
        trainer: Trainer 组件配置
        learner: Learner 组件配置
        aggregator: Aggregator 组件配置
        model: 模型配置
        
        # 数据集配置
        datasets: 数据集列表
        
        # 回调配置
        callbacks: 回调列表
        
        # 其他配置
        min_peers: Trainer 等待的最少对等节点数
        default_timeout: 默认超时时间
    """
    
    # ========== 基本信息 ==========
    node_id: str = ""
    role: str = "learner"
    extend: Optional[str] = None  # 继承的基础配置文件路径
    config_path: Optional[str] = None  # 原始配置文件路径（用于 Artifact 上传）
    
    # ========== 全局配置（共享字段源）==========
    global_config: Optional[GlobalConfig] = None
    
    # ========== 连接配置 ==========
    listen: Optional[Dict[str, Any]] = None
    connect_to: Optional[List[str]] = None
    
    # ========== 基础设施配置 ==========
    transport: TransportConfig = field(default_factory=TransportConfig)
    connection_retry: ConnectionRetryConfig = field(default_factory=ConnectionRetryConfig)
    logging: Optional[LogConfig] = None
    tracker: Optional[TrackerConfig] = None
    
    # ========== 通信配置 ==========
    serialization: Optional[Dict[str, Any]] = None
    heartbeat: Optional[Dict[str, Any]] = None
    interceptors: Optional[InterceptorConfig] = None
    
    # ========== 组件配置 ==========
    trainer: Optional[Union[ComponentConfig, Dict[str, Any]]] = None
    learner: Optional[Union[ComponentConfig, Dict[str, Any]]] = None
    aggregator: Optional[Union[ComponentConfig, Dict[str, Any]]] = None
    model: Optional[Union[ComponentConfig, Dict[str, Any]]] = None
    
    # ========== 数据集配置 ==========
    datasets: Optional[List[Union[DatasetConfig, Dict[str, Any]]]] = None
    
    # ========== 回调配置 ==========
    callbacks: Optional[List[Union[CallbackConfig, Dict[str, Any]]]] = None
    
    # ========== 其他配置 ==========
    min_peers: int = 0
    default_timeout: float = DEFAULT_TIMEOUT
    debug: bool = False
    advertised_address: Optional[str] = None
    
    # ========== 共享字段访问（从 GlobalConfig）==========
    
    @property
    def exp_name(self) -> str:
        """实验名称"""
        if self.global_config:
            return self.global_config.exp_name
        return "default"
    
    @property
    def run_name(self) -> Optional[str]:
        """运行名称"""
        if self.global_config:
            return self.global_config.run_name
        return None
    
    @property
    def log_dir(self) -> str:
        """日志目录"""
        if self.global_config:
            return self.global_config.log_dir
        return "./logs"
    
    # ========== 角色判断 ==========
    
    def is_trainer(self) -> bool:
        """是否为 Trainer 角色"""
        return self.role in ("trainer", "both") or self.trainer is not None
    
    def is_learner(self) -> bool:
        """是否为 Learner 角色"""
        return self.role in ("learner", "both") or self.learner is not None
    
    # ========== 获取标准化配置 ==========
    
    def get_trainer_config(self) -> Optional[ComponentConfig]:
        """获取标准化的 Trainer 配置"""
        if self.trainer is None:
            return None
        if isinstance(self.trainer, ComponentConfig):
            return self.trainer
        return ComponentConfig(
            type=self.trainer.get("type", ""),
            args=self.trainer.get("args"),
        )
    
    def get_learner_config(self) -> Optional[ComponentConfig]:
        """获取标准化的 Learner 配置"""
        if self.learner is None:
            return None
        if isinstance(self.learner, ComponentConfig):
            return self.learner
        return ComponentConfig(
            type=self.learner.get("type", ""),
            args=self.learner.get("args"),
        )
    
    def get_aggregator_config(self) -> Optional[ComponentConfig]:
        """获取标准化的 Aggregator 配置"""
        if self.aggregator is None:
            return None
        if isinstance(self.aggregator, ComponentConfig):
            return self.aggregator
        return ComponentConfig(
            type=self.aggregator.get("type", ""),
            args=self.aggregator.get("args"),
        )
    
    def get_model_config(self) -> Optional[ComponentConfig]:
        """获取标准化的 Model 配置"""
        if self.model is None:
            return None
        if isinstance(self.model, ComponentConfig):
            return self.model
        return ComponentConfig(
            type=self.model.get("type", ""),
            args=self.model.get("args"),
        )
    
    def get_datasets(self, split: Optional[str] = None) -> List[DatasetConfig]:
        """
        获取数据集配置
        
        Args:
            split: 过滤条件（None: 全部, "train"/"test"/"valid": 对应类型）
        """
        if not self.datasets:
            return []
        
        result = []
        for ds in self.datasets:
            if isinstance(ds, DatasetConfig):
                ds_config = ds
            elif isinstance(ds, dict):
                ds_config = DatasetConfig(
                    type=ds["type"],
                    split=ds.get("split", "train"),
                    args=ds.get("args"),
                    partition=ds.get("partition"),
                )
            else:
                continue
            
            if split is None or ds_config.split == split:
                result.append(ds_config)
        
        return result
    
    def get_train_datasets(self) -> List[DatasetConfig]:
        return self.get_datasets("train")
    
    def get_test_datasets(self) -> List[DatasetConfig]:
        return self.get_datasets("test")
    
    def get_valid_datasets(self) -> List[DatasetConfig]:
        return self.get_datasets("valid")
    
    def get_callbacks(self) -> List[CallbackConfig]:
        """获取标准化的回调列表"""
        if not self.callbacks:
            return []
        
        result = []
        for cb in self.callbacks:
            if isinstance(cb, CallbackConfig):
                result.append(cb)
            elif isinstance(cb, dict):
                result.append(CallbackConfig(
                    type=cb.get("type", ""),
                    args=cb.get("args"),
                ))
        return result
    
    # ========== 获取通信配置（给 comm.Node 使用）==========
    
    def get_comm_config(self) -> NodeCommConfig:
        """
        获取通信节点配置
        
        提取 comm.Node 需要的配置，创建 NodeCommConfig 实例。
        共享字段（node_id）直接引用。
        
        自动处理：
        - 如果使用 gRPC 且 connect_to 非空，自动将 connect_to 中的节点设为 critical_peers
        - 如果 serialization 是字典，转换为 SerializationConfig 对象
        """
        # 准备 heartbeat 配置（可能需要自动填充 critical_peers）
        heartbeat = self.heartbeat
        
        # 自动设置 critical_peers：如果配置为空且使用 gRPC，将 connect_to 中的节点设为关键节点
        if self.transport.mode == "grpc" and self.connect_to:
            heartbeat_config = heartbeat or {}
            existing_critical_peers = heartbeat_config.get("critical_peers", [])
            
            # 如果 critical_peers 为空，则自动填充
            if not existing_critical_peers:
                # 从 connect_to 提取节点 ID（格式: "trainer@localhost:50051" -> "trainer"）
                critical_peers = []
                for addr in self.connect_to:
                    if '@' in addr:
                        node_id = addr.split('@')[0]
                        critical_peers.append(node_id)
                
                if critical_peers:
                    # 创建新的 heartbeat 配置
                    if heartbeat is None:
                        heartbeat = {}
                    else:
                        heartbeat = dict(heartbeat)  # 复制避免修改原配置
                    heartbeat["critical_peers"] = critical_peers
        
        # 转换 serialization：字典 -> SerializationConfig
        serialization = self.serialization
        if serialization is not None and isinstance(serialization, dict):
            serialization = parse_serialization_config(serialization)
        
        return NodeCommConfig(
            node_id=self.node_id,  # 引用共享字段
            default_timeout=self.default_timeout,
            debug=self.debug,
            advertised_address=self.advertised_address,
            listen=self.listen,
            transport=self.transport,
            serialization=serialization,
            heartbeat=heartbeat,
            interceptors=self.interceptors,
        )
    
    # ========== 获取追踪参数（给 Tracker 使用）==========
    
    def get_tracking_params(self) -> Dict[str, Any]:
        """
        提取配置参数用于记录到 MLflow/Tracker
        
        提取 Trainer/Learner/Aggregator/Model/数据集等完整配置信息。
        
        Returns:
            用于记录到 Tracker 的参数字典
        """
        params: Dict[str, Any] = {
            "node_id": self.node_id,
            "exp_name": self.exp_name,
        }
        
        if self.run_name:
            params["run_name"] = self.run_name
        
        # ===== Trainer 配置 =====
        trainer_config = self.get_trainer_config()
        if trainer_config:
            params["trainer_type"] = trainer_config.type
            trainer_args = trainer_config.get_args()
            # 记录所有 trainer 参数
            for key, value in trainer_args.items():
                if value is not None:
                    params[f"trainer/{key}"] = value
        
        # ===== Learner 配置 =====
        learner_config = self.get_learner_config()
        if learner_config:
            params["learner_type"] = learner_config.type
            learner_args = learner_config.get_args()
            # 记录所有 learner 参数
            for key, value in learner_args.items():
                if value is not None:
                    params[f"learner/{key}"] = value
        
        # ===== Aggregator 配置 =====
        aggregator_config = self.get_aggregator_config()
        if aggregator_config:
            params["aggregator_type"] = aggregator_config.type
            agg_args = aggregator_config.get_args()
            # 记录所有 aggregator 参数
            for key, value in agg_args.items():
                if value is not None:
                    params[f"aggregator/{key}"] = value
        
        # ===== Model 配置 =====
        model_config = self.get_model_config()
        if model_config:
            params["model_type"] = model_config.type
            model_args = model_config.get_args()
            # 记录所有 model 参数
            for key, value in model_args.items():
                if value is not None:
                    params[f"model/{key}"] = value
        
        # ===== 数据集配置 =====
        datasets = self.get_datasets()
        if datasets:
            for ds in datasets:
                split = ds.split or "train"
                params[f"dataset/{split}/type"] = ds.type
                
                # 数据集参数
                ds_args = ds.get_args() if hasattr(ds, 'get_args') else (ds.args or {})
                for key, value in ds_args.items():
                    if value is not None and key != "split":
                        params[f"dataset/{split}/{key}"] = value
                
                # 划分配置
                if ds.partition:
                    partition = ds.partition
                    if isinstance(partition, dict):
                        for key, value in partition.items():
                            if value is not None:
                                params[f"dataset/{split}/partition_{key}"] = value
                    else:
                        # partition 是对象
                        for key in ["strategy", "alpha", "num_partitions", "partition_id",
                                    "num_shards", "min_samples", "seed"]:
                            if hasattr(partition, key):
                                val = getattr(partition, key)
                                if val is not None:
                                    params[f"dataset/{split}/partition_{key}"] = val
        
        return params
    
    # ========== 同步共享字段到子配置 ==========
    
    def sync_global_to_children(self):
        """将 GlobalConfig 的字段同步到子配置"""
        if not self.global_config:
            return
        
        exp_name = self.global_config.exp_name
        run_name = self.global_config.get_run_name()
        log_dir = self.global_config.log_dir
        
        # 同步到 LogConfig
        if self.logging:
            self.logging.sync_from_global(log_dir, exp_name, run_name)
        
        # 同步到 TrackerConfig
        if self.tracker:
            self.tracker.sync_from_global(log_dir, exp_name, run_name)


def parse_node_config(data: Dict[str, Any]) -> NodeConfig:
    """
    解析节点配置
    
    这是配置解析的入口点，会递归调用各子配置的解析方法。
    """
    # 解析全局配置
    global_config = parse_global_config(data.get("global") or data.get("global_config"))
    
    # 解析子配置
    transport = parse_transport_config(data.get("transport"))
    connection_retry = parse_connection_retry_config(data.get("connection_retry"))
    logging = parse_log_config(data.get("logging"))
    tracker = parse_tracker_config(data.get("tracker"))
    
    # 解析组件配置
    trainer = parse_component_config(data.get("trainer"))
    learner = parse_component_config(data.get("learner"))
    aggregator = parse_component_config(data.get("aggregator"))
    model = parse_component_config(data.get("model"))
    
    # 解析数据集和回调
    datasets = parse_datasets_config(data.get("datasets"))
    callbacks = parse_callbacks_config(data.get("callbacks"))
    
    # 创建 NodeConfig
    config = NodeConfig(
        node_id=data.get("node_id", ""),
        role=data.get("role", "learner"),
        extend=data.get("extend"),
        global_config=global_config,
        listen=data.get("listen"),
        connect_to=data.get("connect_to"),
        transport=transport,
        connection_retry=connection_retry,
        logging=logging,
        tracker=tracker,
        serialization=data.get("serialization"),
        heartbeat=data.get("heartbeat"),
        trainer=trainer,
        learner=learner,
        aggregator=aggregator,
        model=model,
        datasets=datasets if datasets else None,
        callbacks=callbacks if callbacks else None,
        min_peers=data.get("min_peers", 0),
        default_timeout=data.get("default_timeout", DEFAULT_TIMEOUT),
        debug=data.get("debug", False),
        advertised_address=data.get("advertised_address"),
    )
    
    # 同步共享字段
    config.sync_global_to_children()
    
    return config


# ==================== 导出 ====================

__all__ = [
    # 枚举
    "LogLevel",
    "TransportMode",
    "BackoffStrategy",
    "NodeRole",
    # 全局配置
    "GlobalConfig",
    "parse_global_config",
    # 通信节点配置
    "NodeCommConfig",
    # 节点配置（根）
    "NodeConfig",
    "parse_node_config",
]