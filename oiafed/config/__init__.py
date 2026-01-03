"""
配置系统 v4.0

葡萄结构：每个配置类与其解析方法在同一模块中，自包含。

结构：
    config/
    ├── transport.py    # 传输层配置
    ├── comm.py         # 通信层配置
    ├── logging_config.py # 日志配置
    ├── tracker.py      # 追踪配置
    ├── component.py    # 组件配置
    ├── node.py         # 节点配置（根）
    ├── manager.py      # 高层管理器
    ├── generator.py    # 配置生成
    └── defaults.py     # 默认值常量

配置树：
    NodeConfig (根)
    ├── GlobalConfig (共享字段源: exp_name, run_name, log_dir)
    ├── LogConfig
    ├── TrackerConfig
    │   └── TrackerBackendConfig
    ├── TransportConfig
    │   ├── GrpcConfig
    │   │   └── TlsConfig
    │   └── MemoryTransportConfig
    ├── SerializationConfig
    ├── InterceptorConfig
    ├── HeartbeatConfig
    ├── ConnectionRetryConfig
    ├── ComponentConfig (trainer/learner/aggregator/model)
    ├── DatasetConfig[]
    └── CallbackConfig[]

Quick Start:
    from oiafed.config import load_config, NodeConfig
    
    # 从文件加载
    config = load_config("configs/trainer.yaml")
    
    # 访问配置
    print(config.node_id)
    print(config.exp_name)  # 从 GlobalConfig 同步
    print(config.transport.grpc.max_message_size)
    
    # 获取通信配置
    comm_config = config.get_comm_config()
"""

# ==================== 传输层配置 ====================

from .transport import (
    # 配置类
    TlsConfig,
    GrpcConfig,
    MemoryTransportConfig,
    TransportConfig,
    ConnectionRetryConfig,
    # 解析方法
    parse_tls_config,
    parse_grpc_config,
    parse_memory_config,
    parse_transport_config,
    parse_connection_retry_config,
)

# ==================== 通信层配置 ====================

from .comm import (
    # 序列化
    MethodSerializationConfig,
    SerializationConfig,
    parse_method_serialization_config,
    parse_serialization_config,
    # 认证
    AuthConfig,
    parse_auth_config,
    # 重试
    RetryConfig,
    parse_retry_config,
    # 拦截器
    InterceptorConfig,
    parse_interceptor_config,
    # 心跳
    HeartbeatConfig,
    parse_heartbeat_config,
    # 方法选项
    MethodOptions,
)

# ==================== 日志配置 ====================

from .logging_config import (
    LogConfig,
    parse_log_config,
)

# ==================== 追踪配置 ====================

from .tracker import (
    MLflowConfig,
    parse_mlflow_config,
    WandbConfig,
    parse_wandb_config,
    TensorBoardConfig,
    parse_tensorboard_config,
    TrackerBackendConfig,
    parse_tracker_backend_config,
    TrackerConfig,
    parse_tracker_config,
)

# ==================== 组件配置 ====================

from .component import (
    ComponentConfig,
    parse_component_config,
    DatasetConfig,
    parse_dataset_config,
    parse_datasets_config,
    CallbackConfig,
    parse_callback_config,
    parse_callbacks_config,
)

# ==================== 节点配置（根）====================

from .node import (
    # 枚举
    LogLevel,
    TransportMode,
    BackoffStrategy,
    NodeRole,
    # 全局配置
    GlobalConfig,
    parse_global_config,
    # 通信节点配置
    NodeCommConfig,
    # 节点配置
    NodeConfig,
    parse_node_config,
)

# ==================== 管理器 ====================

from .manager import (
    # 异常
    ConfigError,
    ConfigValidationError,
    ConfigLoadError,
    # 管理器
    ConfigManager,
    get_default_manager,
    # 便捷函数
    load_config,
    load_config_from_dict,
    save_config,
    validate_config,
    config_to_dict,
    # 向后兼容
    load_node_config,
    deep_merge,
    create_client_config,
)

# ==================== 配置生成器 ====================

from .generator import (
    ConfigGenerator,
    generate_federation,
)

# ==================== 默认值 ====================

from .defaults import (
    DEFAULT_TRAINER_PORT,
    DEFAULT_LEARNER_BASE_PORT,
    DEFAULT_HOST,
    DEFAULT_LOCALHOST,
    DEFAULT_MAX_MESSAGE_SIZE,
    DEFAULT_TIMEOUT,
    DEFAULT_TRAINER_TYPE,
    DEFAULT_LEARNER_TYPE,
    DEFAULT_AGGREGATOR_TYPE,
    DEFAULT_MODEL_TYPE,
    DEFAULT_DATASET_TYPE,
    DEFAULT_PARTITION_STRATEGY,
    DEFAULT_PARTITION_ALPHA,
    DEFAULT_LOG_LEVEL,
    DEFAULT_LOG_DIR,
    DEFAULT_SERIALIZATION,
    DEFAULT_TRANSPORT_MODE,
    DEFAULT_EXP_NAME,
)

# ==================== 向后兼容别名 ====================

# 旧名称映射
GrpcTransportConfig = GrpcConfig  # 向后兼容
FederationConfig = NodeConfig
LoggingConfig = LogConfig

# ==================== 版本信息 ====================

__version__ = "4.0.0"

# ==================== 导出 ====================

__all__ = [
    # 版本
    "__version__",
    
    # 枚举
    "LogLevel",
    "TransportMode",
    "BackoffStrategy",
    "NodeRole",
    
    # 传输配置
    "TlsConfig",
    "GrpcConfig",
    "GrpcTransportConfig",  # 向后兼容
    "MemoryTransportConfig",
    "TransportConfig",
    "ConnectionRetryConfig",
    "parse_tls_config",
    "parse_grpc_config",
    "parse_memory_config",
    "parse_transport_config",
    "parse_connection_retry_config",
    
    # 通信配置
    "MethodSerializationConfig",
    "SerializationConfig",
    "AuthConfig",
    "RetryConfig",
    "InterceptorConfig",
    "HeartbeatConfig",
    "MethodOptions",
    "parse_method_serialization_config",
    "parse_serialization_config",
    "parse_auth_config",
    "parse_retry_config",
    "parse_interceptor_config",
    "parse_heartbeat_config",
    
    # 日志配置
    "LogConfig",
    "parse_log_config",
    
    # 追踪配置
    "MLflowConfig",
    "WandbConfig",
    "TensorBoardConfig",
    "TrackerBackendConfig",
    "TrackerConfig",
    "parse_mlflow_config",
    "parse_wandb_config",
    "parse_tensorboard_config",
    "parse_tracker_backend_config",
    "parse_tracker_config",
    
    # 组件配置
    "ComponentConfig",
    "DatasetConfig",
    "CallbackConfig",
    "parse_component_config",
    "parse_dataset_config",
    "parse_datasets_config",
    "parse_callback_config",
    "parse_callbacks_config",
    
    # 全局配置
    "GlobalConfig",
    "parse_global_config",
    
    # 通信节点配置
    "NodeCommConfig",
    
    # 节点配置（根）
    "NodeConfig",
    "parse_node_config",
    
    # 异常
    "ConfigError",
    "ConfigValidationError",
    "ConfigLoadError",
    
    # 管理器
    "ConfigManager",
    "get_default_manager",
    
    # 生成器
    "ConfigGenerator",
    "generate_federation",
    
    # 便捷函数
    "load_config",
    "load_config_from_dict",
    "save_config",
    "validate_config",
    "config_to_dict",
    
    # 默认值
    "DEFAULT_TRAINER_PORT",
    "DEFAULT_LEARNER_BASE_PORT",
    "DEFAULT_HOST",
    "DEFAULT_LOCALHOST",
    "DEFAULT_MAX_MESSAGE_SIZE",
    "DEFAULT_TIMEOUT",
    "DEFAULT_TRAINER_TYPE",
    "DEFAULT_LEARNER_TYPE",
    "DEFAULT_AGGREGATOR_TYPE",
    "DEFAULT_MODEL_TYPE",
    "DEFAULT_DATASET_TYPE",
    "DEFAULT_PARTITION_STRATEGY",
    "DEFAULT_PARTITION_ALPHA",
    "DEFAULT_LOG_LEVEL",
    "DEFAULT_LOG_DIR",
    "DEFAULT_SERIALIZATION",
    "DEFAULT_TRANSPORT_MODE",
    "DEFAULT_EXP_NAME",
    
    # 向后兼容
    "load_node_config",
    "deep_merge",
    "create_client_config",
    "FederationConfig",
    "LoggingConfig",
]
