"""
配置系统 v3.0

重构设计：
- types.py: 纯数据类定义
- manager.py: 配置管理器（加载、合并、同步、验证）

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

from .types import (
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
    
    # 便捷函数
    "load_config",
    "load_config_from_dict",
    "save_config",
    "validate_config",
    "config_to_dict",
    
    # 向后兼容
    "load_node_config",
    "deep_merge",
    "create_client_config",
    "FederationConfig",
    "LoggingConfig",
]
