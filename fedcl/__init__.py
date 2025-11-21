"""
MOE-FedCL 联邦通信系统主包
moe_fedcl/__init__.py

提供统一的API接口，用户只需要导入这个包就能使用所有功能。
"""

# ==================== 配置全局日志（必须在最前面） ====================
# 移除loguru的默认handler，设置控制台为INFO级别
# 这样其他模块import loguru时就已经是正确的配置了
from loguru import logger
import sys
import os

# 移除默认的stderr handler
logger.remove()

# 从环境变量读取console日志级别（支持批量实验时设置为ERROR）
console_level = os.getenv("FEDCL_CONSOLE_LOG_LEVEL", "INFO")

# 添加配置好的控制台handler
logger.add(
    sys.stderr,
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    level=console_level,  # 从环境变量读取，默认INFO
    colorize=True
)

# 版本信息
__version__ = "0.1.0"
__author__ = "MOE-FedCL Team"
__description__ = "A unified federated learning communication system supporting Memory/Process/Network modes"

# ==================== 核心抽象类 ====================
# 用户需要继承的基类
from .learner.base_learner import BaseLearner
from .trainer.trainer import BaseTrainer

# ==================== 配置系统 ====================
from .config.config import (
    # 配置类
    ServerConfig, ClientConfig,
    TransportLayerConfig, CommunicationLayerConfig,
    FederationLayerConfig, StubLayerConfig,

    # 便捷函数
    load_server_config, load_client_config,
    create_default_server_config, create_default_client_config
)


# ==================== 统一入口 ====================
from .federated_learning import FederatedLearning

# ==================== 配置类型 ====================
from .types import (
    # 核心数据类型
    ModelData, TrainingResult, EvaluationResult, RoundResult, MetricsData,

    # 配置类型
    TransportConfig, CommunicationConfig, FederationConfig,

    # 枚举类型
    CommunicationMode, ConnectionStatus, RegistrationStatus,
    FederationStatus, TrainingStatus, HealthStatus,

    # 请求响应类型
    TrainingRequest, TrainingResponse, RegistrationRequest, RegistrationResponse,
    HeartbeatMessage, EventMessage, ClientInfo
)

# ==================== 核心组件 ====================
# 联邦学习统一入口 - 延迟导入避免循环依赖
# from .federation.coordinator import FederationCoordinator, FederationResult

# 代理和存根
from .learner.proxy import LearnerProxy, ProxyConfig
from .learner.stub import LearnerStub, StubConfig

# 训练器配置
from .trainer.trainer import TrainingConfig

# ==================== 组件工厂 ====================
from .factory.factory import (
    ComponentFactory,
    TrainerComponents, LearnerComponents, StandaloneComponents,

    # 便捷创建函数
    create_memory_system,
    create_process_server, create_process_client,
    create_network_server, create_network_client
)

# ==================== 异常类 ====================
from .exceptions import (
    MOEFedCLError, TransportError, ConnectionError, RegistrationError,
    CommunicationError, TrainingError, TimeoutError, ValidationError,
    ConfigurationError, ClientNotFoundError, FederationError, SerializationError
)

# ==================== 工具函数 ====================
from .utils.serialization import serialize_data, deserialize_data
from .utils.retry import retry_async, RetryConfig

# ==================== 自动注册内置组件 ====================
# 导入 methods 模块，自动注册所有内置组件
# 用户只需要 import fedcl 或 from fedcl import xxx，组件就会自动注册
from . import methods  # noqa: F401


# ==================== 导出API ====================

__all__ = [
    # 版本信息
    "__version__", "__author__", "__description__",

    # 核心抽象类 (用户继承)
    "BaseLearner", "BaseTrainer",

    # ========== 配置系统 ==========
    # 配置类
    "ServerConfig", "ClientConfig",
    "TransportLayerConfig", "CommunicationLayerConfig",
    "FederationLayerConfig", "StubLayerConfig",

    # 配置加载函数
    "load_server_config", "load_client_config",
    "create_default_server_config", "create_default_client_config",

    # # ========== 高层API ==========
    # # API类
    # "ServerAPI", "ClientAPI", "MultiClientAPI", "FederatedSystemAPI",
    #
    # # 便捷启动函数
    # "run_server", "run_client", "run_multi_clients", "run_federated_system",

    # ========== 统一入口 ==========
    "FederatedLearning", "run_federated_learning",

    # ========== 原有系统 ==========
    # 配置类型
    "TransportConfig", "CommunicationConfig", "FederationConfig", "TrainingConfig",
    "ProxyConfig", "StubConfig",

    # 枚举类型
    "CommunicationMode", "ConnectionStatus", "RegistrationStatus",
    "FederationStatus", "TrainingStatus", "HealthStatus",

    # 数据类型
    "ModelData", "TrainingResult", "EvaluationResult", "RoundResult", "MetricsData",

    # 请求响应类型
    "TrainingRequest", "TrainingResponse", "RegistrationRequest", "RegistrationResponse",
    "HeartbeatMessage", "EventMessage", "ClientInfo",

    # 核心组件
    "LearnerProxy", "LearnerStub",

    # 组件工厂
    "ComponentFactory", "TrainerComponents", "LearnerComponents", "StandaloneComponents",

    # 便捷创建函数
    "create_memory_system", "create_process_server", "create_process_client",
    "create_network_server", "create_network_client",

    # 异常类
    "MOEFedCLError", "TransportError", "ConnectionError", "RegistrationError",
    "CommunicationError", "TrainingError", "TimeoutError", "ValidationError",
    "ConfigurationError", "ClientNotFoundError", "FederationError", "SerializationError",

    # 工具函数
    "serialize_data", "deserialize_data", "retry_async", "RetryConfig"
]