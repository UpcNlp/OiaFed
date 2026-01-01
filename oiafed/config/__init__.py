"""
配置系统 v3.0

重构设计：
- schema.py: 配置 Schema 定义（纯数据类）
- manager.py: 配置管理器（加载、保存、合并、验证）
- generator.py: 配置生成器（统一的配置生成入口）
- defaults.py: 默认值定义

主要特性：
1. 层次化配置（GlobalConfig → NodeConfig）
2. 自动同步共享字段（exp_name, run_name, log_dir）
3. 配置继承（extend 字段）
4. 完整的类型提示和验证

Quick Start:
    # 从文件加载
    from config import load_config
    config = load_config("configs/trainer.yaml")
    
    # 从字典创建
    from config import load_config_from_dict
    config = load_config_from_dict({
        "node_id": "trainer",
        "global": {"exp_name": "my_experiment"},
        "listen": {"port": 50051},
        "trainer": {"type": "federated.trainer.fedavg"},
        "aggregator": {"type": "federated.aggregator.fedavg"},
    })
    
    # 访问自动同步的字段
    print(config.exp_name)              # "my_experiment"
    print(config.logging.exp_name)      # "my_experiment" (自动同步)
    
    # 使用管理器
    from config import ConfigManager
    manager = ConfigManager()
    config = manager.load("config.yaml")
    manager.save(config, "output.yaml")
"""

# ==================== 类型定义 ====================

from .schema import (
    # 枚举
    LogLevel,
    TransportMode,
    BackoffStrategy,
    NodeRole,
    
    # 全局配置
    GlobalConfig,
    
    # 日志配置
    LogConfig,
    
    # 追踪配置
    TrackerBackendConfig,
    MLflowConfig,
    WandbConfig,
    TensorBoardConfig,
    TrackerConfig,
    
    # 传输配置
    GrpcConfig,
    TransportConfig,
    ConnectionRetryConfig,
    
    # 组件配置
    ComponentConfig,
    DatasetConfig,
    CallbackConfig,
    
    # 通信节点配置
    NodeCommConfig,
    
    # 节点配置
    NodeConfig,
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

# 保留旧的类型别名
FederationConfig = NodeConfig
LoggingConfig = LogConfig


# ==================== 版本信息 ====================

__version__ = "3.0.0"


# ==================== 导出 ====================

__all__ = [
    # 版本
    "__version__",
    
    # 枚举
    "LogLevel",
    "TransportMode",
    "BackoffStrategy",
    "NodeRole",
    
    # 配置类
    "GlobalConfig",
    "LogConfig",
    "TrackerBackendConfig",
    "MLflowConfig",
    "WandbConfig",
    "TensorBoardConfig",
    "TrackerConfig",
    "GrpcConfig",
    "TransportConfig",
    "ConnectionRetryConfig",
    "ComponentConfig",
    "DatasetConfig",
    "CallbackConfig",
    "NodeCommConfig",
    "NodeConfig",
    
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