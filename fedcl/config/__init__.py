"""
MOE-FedCL 配置管理模块
fedcl/config/__init__.py
"""

# ========== 旧配置系统（保持兼容性）==========
from .config import (
    ServerConfig,
    ClientConfig,
    TransportLayerConfig,
    load_server_config,
    load_client_config,
    create_default_server_config,
    create_default_client_config
)

# ========== 新配置系统（简化版）==========
# 基础配置类
from .base import BaseConfig

# 通信配置（统一的）
from .communication import CommunicationConfig

# 训练配置（统一的）
from .training import TrainingConfig

# 配置加载器（简化版）
from .loader import ConfigLoader, ConfigLoadError

# 配置验证器
from .validator import (
    ConfigValidator,
    ConfigValidationError,
    ValidationRule,
    ConditionalValidationRule,
    create_communication_validator,
    create_training_validator,
)


__all__ = [
    # ========== 旧配置系统 ==========
    "ServerConfig",
    "ClientConfig",
    "TransportLayerConfig",
    "load_server_config",
    "load_client_config",
    "create_default_server_config",
    "create_default_client_config",

    # ========== 新配置系统（简化版）==========
    # 基础
    "BaseConfig",

    # 配置类（只有2个核心类）
    "CommunicationConfig",  # 统一的通信配置
    "TrainingConfig",       # 统一的训练配置

    # 加载和验证
    "ConfigLoader",
    "ConfigLoadError",
    "ConfigValidator",
    "ConfigValidationError",
    "ValidationRule",
    "ConditionalValidationRule",
    "create_communication_validator",
    "create_training_validator",
]
