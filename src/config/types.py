"""
配置类型定义

纯数据类，只定义配置的结构，不包含加载逻辑。
所有配置类都是不可变的 dataclass，便于序列化和测试。

架构设计：
    GlobalConfig          # 全局共享配置（exp_name, run_name 等）
    ├── LogConfig         # 日志配置
    ├── TrackerConfig     # 追踪配置
    ├── TransportConfig   # 传输层配置
    └── NodeConfig        # 节点配置（包含以上所有）
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum


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
        
    Example:
        global_config = GlobalConfig(
            exp_name="fedavg_mnist",
            run_name="run_001",
            log_dir="./logs",
        )
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


# ==================== 日志配置 ====================

@dataclass
class LogConfig:
    """
    日志配置
    
    基于 loguru 的日志配置，支持控制台和文件输出。
    
    Attributes:
        level: 文件日志级别
        console: 是否输出到控制台
        console_level: 控制台日志级别
        rotation: 日志轮转大小（如 "10 MB"）
        retention: 日志保留时间（如 "30 days"）
        compression: 压缩格式（如 "zip"）
        format: 日志格式字符串
        diagnose: 是否显示详细诊断信息
        log_dir: 日志目录
        exp_name: 实验名称
        
    Example:
        log_config = LogConfig(
            level="DEBUG",
            console=True,
            console_level="INFO",
            log_dir="./logs",
            exp_name="my_experiment",
        )
    """
    # 基础配置
    level: str = "INFO"
    console: bool = True
    console_level: str = "INFO"
    
    # 文件输出配置
    rotation: str = "10 MB"
    retention: str = "30 days"
    compression: str = "zip"
    
    # 格式配置
    format: str = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level: <8} | "
        "{extra[node_id]} | "
        "{name}:{function}:{line} - "
        "{message}"
    )
    
    # 调试配置
    diagnose: bool = False
    
    # ===== 路径和实验信息 =====
    # 这些字段可以直接初始化，也可以由 ConfigManager 同步设置
    log_dir: Optional[str] = field(default="./logs")
    exp_name: Optional[str] = field(default=None)
    run_name: Optional[str] = field(default=None)
    
    # ===== 向后兼容的私有字段别名 =====
    # 保留这些属性以兼容旧代码
    @property
    def _exp_name(self) -> Optional[str]:
        return self.exp_name
    
    @_exp_name.setter
    def _exp_name(self, value: Optional[str]):
        object.__setattr__(self, 'exp_name', value)
    
    @property
    def _log_dir(self) -> Optional[str]:
        return self.log_dir
    
    @_log_dir.setter
    def _log_dir(self, value: Optional[str]):
        object.__setattr__(self, 'log_dir', value)
    
    @property
    def _run_name(self) -> Optional[str]:
        return self.run_name
    
    @_run_name.setter
    def _run_name(self, value: Optional[str]):
        object.__setattr__(self, 'run_name', value)
    
    def get_log_path(self) -> str:
        """
        获取完整的日志路径
        
        Returns:
            格式：{log_dir}/{exp_name}/{run_name}/
        """
        parts = [self.log_dir]
        if self._exp_name:
            parts.append(self._exp_name)
        if self._run_name:
            parts.append(self._run_name)
        return "/".join(parts)
    
    def __post_init__(self):
        """初始化后处理"""
        # 标准化日志级别为大写
        self.level = self.level.upper()
        self.console_level = self.console_level.upper()


# ==================== 追踪配置 ====================

@dataclass
class TrackerBackendConfig:
    """
    单个 Tracker Backend 配置
    
    Attributes:
        type: backend 类型（file, mlflow, wandb, tensorboard）
        args: 配置参数字典
        
    Example:
        backend = TrackerBackendConfig(
            type="mlflow",
            args={"tracking_uri": "http://localhost:5000"},
        )
    """
    type: str
    args: Optional[Dict[str, Any]] = None
    
    def get_args(self) -> Dict[str, Any]:
        """获取参数字典"""
        return self.args or {}


@dataclass
class MLflowConfig:
    """MLflow 追踪配置"""
    tracking_uri: str = "http://localhost:5000"
    experiment_name: Optional[str] = None  # None 则使用 exp_name
    run_name: Optional[str] = None         # None 则使用全局 run_name
    
    # 同步字段
    _exp_name: Optional[str] = field(default=None, repr=False, compare=False)
    _run_name: Optional[str] = field(default=None, repr=False, compare=False)
    
    def get_experiment_name(self) -> str:
        """获取实验名称，优先使用显式设置的值"""
        return self.experiment_name or self._exp_name or "default"
    
    def get_run_name(self) -> Optional[str]:
        """获取运行名称，优先使用显式设置的值"""
        return self.run_name or self._run_name


@dataclass
class WandbConfig:
    """Weights & Biases 追踪配置"""
    project: str = "federated-learning"
    entity: Optional[str] = None
    name: Optional[str] = None             # None 则使用全局 run_name
    tags: List[str] = field(default_factory=list)
    
    # 同步字段
    _exp_name: Optional[str] = field(default=None, repr=False, compare=False)
    _run_name: Optional[str] = field(default=None, repr=False, compare=False)
    
    def get_name(self) -> Optional[str]:
        """获取运行名称"""
        return self.name or self._run_name


@dataclass
class TensorBoardConfig:
    """TensorBoard 追踪配置"""
    log_dir: Optional[str] = None          # None 则自动生成
    
    # 同步字段
    _log_dir: Optional[str] = field(default=None, repr=False, compare=False)
    _exp_name: Optional[str] = field(default=None, repr=False, compare=False)
    _run_name: Optional[str] = field(default=None, repr=False, compare=False)
    
    def get_log_dir(self) -> str:
        """获取日志目录"""
        if self.log_dir:
            return self.log_dir
        # 自动生成：{log_dir}/{exp_name}/{run_name}/tensorboard
        parts = [self._log_dir or "./logs"]
        if self._exp_name:
            parts.append(self._exp_name)
        if self._run_name:
            parts.append(self._run_name)
        parts.append("tensorboard")
        return "/".join(parts)


@dataclass
class TrackerConfig:
    """
    训练追踪配置
    
    Attributes:
        enabled: 是否启用追踪
        tracking_dir: 追踪目录（相对于日志目录）
        backends: backend 配置列表
        
    Example:
        tracker = TrackerConfig(
            enabled=True,
            backends=[
                TrackerBackendConfig(type="file"),
                TrackerBackendConfig(
                    type="mlflow",
                    args={"tracking_uri": "./mlruns"},
                ),
            ],
        )
    """
    enabled: bool = True
    tracking_dir: str = "tracking"
    backends: Optional[List[Union[Dict[str, Any], TrackerBackendConfig]]] = None
    
    # 同步字段
    _exp_name: Optional[str] = field(default=None, repr=False, compare=False)
    _run_name: Optional[str] = field(default=None, repr=False, compare=False)
    _log_dir: Optional[str] = field(default=None, repr=False, compare=False)
    
    @property
    def exp_name(self) -> Optional[str]:
        """实验名称（从 GlobalConfig 同步）"""
        return self._exp_name
    
    @property
    def run_name(self) -> Optional[str]:
        """运行名称（从 GlobalConfig 同步）"""
        return self._run_name
    
    @property
    def log_dir(self) -> Optional[str]:
        """日志目录（从 GlobalConfig 同步）"""
        return self._log_dir
    
    def get_tracking_path(self) -> str:
        """
        获取完整的追踪路径
        
        Returns:
            格式：{log_dir}/{exp_name}/{run_name}/{tracking_dir}/
        """
        parts = [self._log_dir or "./logs"]
        if self._exp_name:
            parts.append(self._exp_name)
        if self._run_name:
            parts.append(self._run_name)
        parts.append(self.tracking_dir)
        return "/".join(parts)
    
    def get_backends(self) -> List[TrackerBackendConfig]:
        """获取标准化的 backend 列表"""
        if not self.backends:
            return []
        
        result = []
        for backend in self.backends:
            if isinstance(backend, TrackerBackendConfig):
                result.append(backend)
            elif isinstance(backend, dict):
                result.append(TrackerBackendConfig(
                    type=backend.get("type", "file"),
                    args=backend.get("args"),
                ))
        return result


# ==================== 传输配置 ====================

@dataclass
class GrpcConfig:
    """gRPC 配置"""
    host: str = "0.0.0.0"
    port: int = 50051
    max_message_size: int = 100 * 1024 * 1024  # 100MB
    
    def get_address(self) -> str:
        """获取完整地址"""
        return f"{self.host}:{self.port}"


@dataclass
class TransportConfig:
    """
    传输层配置
    
    Attributes:
        mode: 传输模式（memory 或 grpc）
        grpc: gRPC 配置
        
    Example:
        transport = TransportConfig(
            mode="grpc",
            grpc=GrpcConfig(host="0.0.0.0", port=50051),
        )
    """
    mode: str = "memory"
    grpc: GrpcConfig = field(default_factory=GrpcConfig)
    
    def __post_init__(self):
        """初始化后处理"""
        # 如果 grpc 是字典，转换为 GrpcConfig
        if isinstance(self.grpc, dict):
            self.grpc = GrpcConfig(**self.grpc)


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


# ==================== 组件配置 ====================

@dataclass
class ComponentConfig:
    """
    通用组件配置
    
    用于 trainer, learner, aggregator, model 等组件。
    
    Attributes:
        type: 组件类型（如 "federated.trainer.fedavg"）
        args: 组件参数
        
    Example:
        trainer = ComponentConfig(
            type="federated.trainer.fedavg",
            args={"max_rounds": 100},
        )
    """
    type: str
    args: Optional[Dict[str, Any]] = None
    
    def get_args(self) -> Dict[str, Any]:
        """获取参数字典"""
        return self.args or {}


@dataclass
class DatasetConfig:
    """数据集配置"""
    type: str
    split: str = "train"
    args: Optional[Dict[str, Any]] = None
    partition: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        # 自动将 split 注入 args
        if self.args is None:
            self.args = {}
        if "split" not in self.args:
            self.args["split"] = self.split
    
    def get_args(self) -> Dict[str, Any]:
        return self.args.copy() if self.args else {}


# ==================== 回调配置 ====================

@dataclass
class CallbackConfig:
    """
    回调配置
    
    Attributes:
        type: 回调类型
        args: 回调参数
        
    Example:
        callback = CallbackConfig(
            type="callbacks.early_stopping",
            args={"patience": 5},
        )
    """
    type: str
    args: Optional[Dict[str, Any]] = None
    
    def get_args(self) -> Dict[str, Any]:
        """获取参数字典"""
        return self.args or {}


# ==================== 通信节点配置 ====================

@dataclass
class NodeCommConfig:
    """
    通信节点配置（给 Node 使用）
    
    这是 NodeConfig 的子集，只包含通信相关的配置。
    Node 类只需要这些配置来初始化 node_comm。
    
    Attributes:
        node_id: 节点唯一标识符
        default_timeout: 默认超时时间（秒）
        debug: 是否开启调试模式
        advertised_address: 广播地址
        listen: 监听配置
        transport: 传输层配置
        serialization: 序列化配置
        heartbeat: 心跳配置
        
    Example:
        comm_config = NodeCommConfig(
            node_id="trainer",
            listen={"port": 50051},
            transport=TransportConfig(mode="grpc"),
        )
        node = Node(comm_config)
    """
    node_id: str
    default_timeout: float = 30.0
    debug: bool = False
    advertised_address: Optional[str] = None
    listen: Optional[Dict[str, Any]] = None
    transport: TransportConfig = field(default_factory=TransportConfig)
    serialization: Optional[Dict[str, Any]] = None
    heartbeat: Optional[Dict[str, Any]] = None
    
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


# ==================== 节点配置 ====================

@dataclass
class NodeConfig:
    """
    节点配置
    
    联邦学习系统中单个节点的完整配置。
    
    Attributes:
        node_id: 节点唯一标识符
        role: 节点角色（trainer, learner, both）
        
        # 全局配置
        global_config: 全局共享配置
        
        # 连接配置
        listen: 监听配置（Trainer 使用）
        connect_to: 连接目标列表（Learner 使用）
        
        # 基础设施配置
        transport: 传输层配置
        connection_retry: 连接重试配置
        logging: 日志配置
        tracker: 追踪配置
        
        # 组件配置
        trainer: Trainer 组件配置
        learner: Learner 组件配置
        aggregator: Aggregator 组件配置
        model: 模型配置
        
        # 数据集配置
        datasets: 训练数据集列表
        test_datasets: 测试数据集列表
        dataset: 单个数据集（向后兼容）
        
        # 回调配置
        callbacks: 回调列表
        
        # 其他配置
        serialization: 序列化配置
        min_peers: Trainer 等待的最少对等节点数
        default_timeout: 默认超时时间
        heartbeat: 心跳配置
        
    Example:
        config = NodeConfig(
            node_id="trainer",
            role="trainer",
            global_config=GlobalConfig(exp_name="my_exp"),
            listen={"port": 50051},
            trainer=ComponentConfig(type="federated.trainer.fedavg"),
            aggregator=ComponentConfig(type="federated.aggregator.fedavg"),
        )
    """
    
    # ========== 基本信息 ==========
    node_id: str = ""
    role: str = "learner"
    extend: Optional[str] = None  # 继承的基础配置文件路径
    
    # ========== 全局配置 ==========
    global_config: Optional[GlobalConfig] = None
    
    # ========== 连接配置 ==========
    listen: Optional[Dict[str, Any]] = None
    connect_to: Optional[List[str]] = None
    
    # ========== 基础设施配置 ==========
    transport: TransportConfig = field(default_factory=TransportConfig)
    connection_retry: ConnectionRetryConfig = field(default_factory=ConnectionRetryConfig)
    logging: Optional[LogConfig] = None
    tracker: Optional[TrackerConfig] = None
    
    # ========== 组件配置 ==========
    trainer: Optional[Union[ComponentConfig, Dict[str, Any]]] = None
    learner: Optional[Union[ComponentConfig, Dict[str, Any]]] = None
    aggregator: Optional[Union[ComponentConfig, Dict[str, Any]]] = None
    model: Optional[Union[ComponentConfig, Dict[str, Any]]] = None
    
    # ========== 数据集配置 ==========
    datasets: Optional[List[Union[DatasetConfig, Dict[str, Any]]]] = None

    
    # ========== 回调配置 ==========
    callbacks: Optional[List[Union[CallbackConfig, Dict[str, Any]]]] = None
    
    # ========== 序列化配置 ==========
    serialization: Optional[Dict[str, Any]] = None
    
    # ========== 其他配置 ==========
    min_peers: int = 0
    default_timeout: float = 30.0
    heartbeat: Optional[Dict[str, Any]] = None
    
    # ========== 便捷属性 ==========
    
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
    
    def is_trainer(self) -> bool:
        """是否为 Trainer 角色"""
        return self.role in ("trainer", "both") or self.trainer is not None
    
    def is_learner(self) -> bool:
        """是否为 Learner 角色"""
        return self.role in ("learner", "both") or self.learner is not None
    
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
            split: 过滤条件
                - None: 返回全部
                - "train" / "test" / "valid": 返回对应类型
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
    
    def get_comm_config(self) -> "NodeCommConfig":
        """
        获取通信节点配置
        
        提取 Node 通信层需要的配置，创建 NodeCommConfig 实例。
        
        Returns:
            NodeCommConfig 实例
            
        Example:
            config = load_config("trainer.yaml")
            comm_config = config.get_comm_config()
            node = Node(comm_config)
        """
        return NodeCommConfig(
            node_id=self.node_id,
            default_timeout=self.default_timeout,
            debug=False,
            advertised_address=None,
            listen=self.listen,
            transport=self.transport,
            serialization=self.serialization,
            heartbeat=self.heartbeat,
        )


# ==================== 导出 ====================

__all__ = [
    # 枚举
    "LogLevel",
    "TransportMode",
    "BackoffStrategy",
    "NodeRole",
    
    # 全局配置
    "GlobalConfig",
    
    # 日志配置
    "LogConfig",
    
    # 追踪配置
    "TrackerBackendConfig",
    "MLflowConfig",
    "WandbConfig",
    "TensorBoardConfig",
    "TrackerConfig",
    
    # 传输配置
    "GrpcConfig",
    "TransportConfig",
    "ConnectionRetryConfig",
    
    # 组件配置
    "ComponentConfig",
    "DatasetConfig",
    "CallbackConfig",
    
    # 通信节点配置
    "NodeCommConfig",
    
    # 节点配置
    "NodeConfig",
]