"""
配置管理器

负责配置的加载、合并、同步和验证。
将配置的「数据定义」和「加载逻辑」分离。

主要功能：
1. 从 YAML 文件加载配置
2. 支持配置继承（extend 字段）
3. 自动同步共享字段（exp_name, run_name, log_dir）
4. 配置验证
5. 配置序列化

Example:
    # 方式1：使用默认管理器（推荐）
    from config import load_config
    config = load_config("configs/trainer.yaml")
    
    # 方式2：创建管理器实例
    from config import ConfigManager
    manager = ConfigManager()
    config = manager.load("configs/trainer.yaml")
    
    # 方式3：从字典创建
    config = manager.from_dict({
        "node_id": "trainer",
        "global": {"exp_name": "my_experiment"},
    })
"""

import copy
import os
from dataclasses import asdict, fields, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

import yaml

from .types import (
    GlobalConfig,
    LogConfig,
    TrackerConfig,
    TrackerBackendConfig,
    TransportConfig,
    GrpcConfig,
    ConnectionRetryConfig,
    ComponentConfig,
    DatasetConfig,
    CallbackConfig,
    NodeConfig,
    MLflowConfig,
    WandbConfig,
    TensorBoardConfig,
)


T = TypeVar("T")


class ConfigError(Exception):
    """配置错误基类"""
    pass


class ConfigValidationError(ConfigError):
    """配置验证错误"""
    pass


class ConfigLoadError(ConfigError):
    """配置加载错误"""
    pass


class ConfigManager:
    """
    配置管理器
    
    职责：
    1. 加载配置文件（支持 extend 继承）
    2. 同步共享字段（exp_name, run_name, log_dir）
    3. 配置验证
    4. 配置序列化
    
    Example:
        manager = ConfigManager()
        
        # 从文件加载
        config = manager.load("configs/trainer.yaml")
        
        # 从字典创建
        config = manager.from_dict({
            "node_id": "trainer",
            "global": {
                "exp_name": "my_experiment",
                "run_name": "run_001",
            },
            "logging": {"level": "DEBUG"},
            "tracker": {"enabled": True},
        })
        
        # exp_name 自动同步到 logging 和 tracker
        print(config.logging.exp_name)   # "my_experiment"
        print(config.tracker.exp_name)   # "my_experiment"
        
        # 验证配置
        manager.validate(config)
        
        # 序列化为字典
        config_dict = manager.to_dict(config)
    """
    
    def __init__(self, auto_generate_run_name: bool = True):
        """
        初始化配置管理器
        
        Args:
            auto_generate_run_name: 是否自动生成 run_name（当 run_name 为 None 时）
        """
        self.auto_generate_run_name = auto_generate_run_name
        self._generated_run_name: Optional[str] = None
    
    # ==================== 加载方法 ====================
    
    def load(
        self,
        path: str,
        base_dir: Optional[str] = None,
        validate: bool = True,
    ) -> NodeConfig:
        """
        从文件加载配置
        
        Args:
            path: 配置文件路径
            base_dir: 基础目录（用于解析相对路径）
            validate: 是否验证配置
            
        Returns:
            NodeConfig 实例
            
        Raises:
            ConfigLoadError: 文件不存在或解析失败
            ConfigValidationError: 配置验证失败
            
        Example:
            config = manager.load("configs/trainer.yaml")
        """
        path_obj = Path(path)
        
        # 解析相对路径
        if not path_obj.is_absolute() and base_dir:
            path_obj = Path(base_dir) / path_obj
        
        if not path_obj.exists():
            raise ConfigLoadError(f"Config file not found: {path_obj}")
        
        try:
            # 使用 _load_yaml 方法，支持环境变量替换 ${VAR:default}
            data = self._load_yaml(path_obj)
        except yaml.YAMLError as e:
            raise ConfigLoadError(f"Failed to parse YAML: {path_obj}\n{e}")
        
        # 处理 extend 继承
        if "extend" in data:
            extend_path = data.pop("extend")
            
            # 解析相对路径（相对于当前配置文件）
            if not Path(extend_path).is_absolute():
                extend_path = path_obj.parent / extend_path
            
            # 递归加载基础配置
            base_data = self._load_yaml(extend_path)
            
            # 深度合并
            data = self._deep_merge(base_data, data)
        
        # 处理 paper 引用（从论文注册表加载默认配置）
        if "paper" in data:
            paper_id = data.pop("paper")
            paper_defaults = self._load_paper_defaults(paper_id)
            
            if paper_defaults:
                # 论文默认配置作为基础，用户配置覆盖
                data = self._deep_merge(paper_defaults, data)
        
        # 解析为 NodeConfig
        config = self.from_dict(data, validate=False)
        
        # 验证
        if validate:
            self.validate(config)
        
        return config
    
    def from_dict(
        self,
        data: Dict[str, Any],
        validate: bool = True,
    ) -> NodeConfig:
        """
        从字典创建配置
        
        自动同步共享字段到子配置。
        自动为 tracker backends 添加对应的 callbacks。
        
        Args:
            data: 配置字典
            validate: 是否验证配置
            
        Returns:
            NodeConfig 实例
            
        Example:
            config = manager.from_dict({
                "node_id": "trainer",
                "global": {
                    "exp_name": "fedavg_mnist",
                    "run_name": "run_001",
                },
                "logging": {"level": "DEBUG"},
            })
        """
        # 1. 解析全局配置
        global_config = self._parse_global_config(data)
        
        # 2. 处理 run_name 自动生成
        if self.auto_generate_run_name and not global_config.run_name:
            if self._generated_run_name is None:
                self._generated_run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
            # 注意：GlobalConfig 是 dataclass，需要重新创建
            global_config = GlobalConfig(
                exp_name=global_config.exp_name,
                run_name=self._generated_run_name,
                log_dir=global_config.log_dir,
            )
        
        # 3. 解析子配置
        logging_config = self._parse_log_config(data.get("logging", {}))
        tracker_config = self._parse_tracker_config(data.get("tracker"))
        transport_config = self._parse_transport_config(data.get("transport", {}))
        connection_retry_config = self._parse_connection_retry_config(
            data.get("connection_retry", {})
        )
        
        # 4. 同步共享字段
        self._sync_global_fields(
            global_config,
            logging_config,
            tracker_config,
        )
        
        # 5. 自动注入 tracker 对应的 callbacks
        callbacks_data = self._auto_inject_tracker_callbacks(
            data.get("callbacks", []),
            tracker_config,
        )
        
        # 6. 构建 NodeConfig
        config = NodeConfig(
            # 基本信息
            node_id=data.get("node_id", ""),
            role=data.get("role", "learner"),
            extend=data.get("extend"),
            
            # 全局配置
            global_config=global_config,
            
            # 连接配置
            listen=data.get("listen"),
            connect_to=data.get("connect_to"),
            
            # 基础设施配置
            transport=transport_config,
            connection_retry=connection_retry_config,
            logging=logging_config,
            tracker=tracker_config,
            
            # 组件配置（保持原始格式，由 NodeConfig 的方法处理）
            trainer=data.get("trainer"),
            learner=data.get("learner"),
            aggregator=data.get("aggregator"),
            model=data.get("model"),
            
            # 数据集配置
            datasets=data.get("datasets"),
            
            # 回调配置（使用自动注入后的）
            callbacks=callbacks_data,
            
            # 其他配置
            serialization=data.get("serialization"),
            min_peers=data.get("min_peers", 0),
            default_timeout=data.get("default_timeout", 30.0),
            heartbeat=data.get("heartbeat"),
        )
        
        # 验证
        if validate:
            self.validate(config)
        
        return config
    
    # ==================== 序列化方法 ====================
    
    def to_dict(self, config: NodeConfig) -> Dict[str, Any]:
        """
        将 NodeConfig 序列化为字典
        
        Args:
            config: NodeConfig 实例
            
        Returns:
            配置字典（可以保存为 YAML）
            
        Example:
            config_dict = manager.to_dict(config)
            with open("output.yaml", "w") as f:
                yaml.dump(config_dict, f)
        """
        result: Dict[str, Any] = {
            "node_id": config.node_id,
            "role": config.role,
        }
        
        # 全局配置
        if config.global_config:
            result["global"] = {
                "exp_name": config.global_config.exp_name,
                "run_name": config.global_config.run_name,
                "log_dir": config.global_config.log_dir,
            }
        
        # 连接配置
        if config.listen:
            result["listen"] = config.listen
        if config.connect_to:
            result["connect_to"] = config.connect_to
        
        # 传输配置
        result["transport"] = {
            "mode": config.transport.mode,
            "grpc": {
                "host": config.transport.grpc.host,
                "port": config.transport.grpc.port,
                "max_message_size": config.transport.grpc.max_message_size,
            },
        }
        
        # 日志配置
        if config.logging:
            result["logging"] = {
                "level": config.logging.level,
                "console": config.logging.console,
                "console_level": config.logging.console_level,
                "rotation": config.logging.rotation,
                "retention": config.logging.retention,
                "compression": config.logging.compression,
                "format": config.logging.format,
                "diagnose": config.logging.diagnose,
                "log_dir": config.logging.log_dir,
                "exp_name": config.logging.exp_name,
                "run_name": config.logging.run_name,
            }
        
        # 追踪配置
        if config.tracker:
            result["tracker"] = {
                "enabled": config.tracker.enabled,
                "tracking_dir": config.tracker.tracking_dir,
                "backends": config.tracker.backends,
            }
        
        # 连接重试配置
        result["connection_retry"] = {
            "enabled": config.connection_retry.enabled,
            "max_retries": config.connection_retry.max_retries,
            "retry_interval": config.connection_retry.retry_interval,
            "timeout": config.connection_retry.timeout,
            "backoff": config.connection_retry.backoff,
            "backoff_factor": config.connection_retry.backoff_factor,
        }
        
        # 组件配置
        if config.trainer:
            result["trainer"] = self._component_to_dict(config.trainer)
        if config.learner:
            result["learner"] = self._component_to_dict(config.learner)
        if config.aggregator:
            result["aggregator"] = self._component_to_dict(config.aggregator)
        if config.model:
            result["model"] = self._component_to_dict(config.model)
        
        # 数据集配置
        if config.datasets:
            result["datasets"] = [
                self._component_to_dict(ds) for ds in config.datasets
            ]
        if config.test_datasets:
            result["test_datasets"] = [
                self._component_to_dict(ds) for ds in config.test_datasets
            ]
        if config.dataset:
            result["dataset"] = self._component_to_dict(config.dataset)
        
        # 回调配置
        if config.callbacks:
            result["callbacks"] = [
                self._component_to_dict(cb) for cb in config.callbacks
            ]
        
        # 其他配置
        if config.serialization:
            result["serialization"] = config.serialization
        result["min_peers"] = config.min_peers
        result["default_timeout"] = config.default_timeout
        if config.heartbeat:
            result["heartbeat"] = config.heartbeat
        
        return result
    
    def save(self, config: NodeConfig, path: str) -> None:
        """
        保存配置到 YAML 文件
        
        Args:
            config: NodeConfig 实例
            path: 保存路径
        """
        config_dict = self.to_dict(config)
        
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path_obj, "w", encoding="utf-8") as f:
            yaml.dump(
                config_dict,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )
    
    # ==================== 验证方法 ====================
    
    def validate(self, config: NodeConfig) -> None:
        """
        验证配置完整性和一致性
        
        Args:
            config: NodeConfig 实例
            
        Raises:
            ConfigValidationError: 配置无效
        """
        errors: List[str] = []
        
        # 1. 基本信息验证
        if not config.node_id:
            errors.append("node_id is required")
        
        # 2. 角色与组件一致性验证
        if config.trainer and not config.aggregator:
            errors.append(
                f"Node {config.node_id}: aggregator is required when trainer is configured"
            )
        
        # 3. 连接配置验证
        if not config.listen and not config.connect_to:
            errors.append(
                f"Node {config.node_id}: must have either 'listen' or 'connect_to'"
            )
        
        # 4. 至少有一个功能组件
        if not config.trainer and not config.learner:
            errors.append(
                f"Node {config.node_id}: must have at least one of 'trainer' or 'learner'"
            )
        
        # 5. 日志级别验证
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if config.logging:
            if config.logging.level.upper() not in valid_levels:
                errors.append(
                    f"Invalid log level: {config.logging.level}. "
                    f"Valid levels: {valid_levels}"
                )
            if config.logging.console_level.upper() not in valid_levels:
                errors.append(
                    f"Invalid console level: {config.logging.console_level}. "
                    f"Valid levels: {valid_levels}"
                )
        
        # 6. 传输模式验证
        valid_modes = {"memory", "grpc"}
        if config.transport.mode not in valid_modes:
            errors.append(
                f"Invalid transport mode: {config.transport.mode}. "
                f"Valid modes: {valid_modes}"
            )
        
        # 抛出所有错误
        if errors:
            raise ConfigValidationError(
                f"Configuration validation failed:\n" +
                "\n".join(f"  - {e}" for e in errors)
            )
    
    # ==================== 同步方法 ====================
    
    def _sync_global_fields(
        self,
        global_config: GlobalConfig,
        logging_config: Optional[LogConfig],
        tracker_config: Optional[TrackerConfig],
    ) -> None:
        """
        同步全局字段到子配置
        
        将 GlobalConfig 中的共享字段（exp_name, run_name, log_dir）
        同步到 LogConfig 和 TrackerConfig。
        """
        if logging_config:
            logging_config._exp_name = global_config.exp_name
            logging_config._run_name = global_config.run_name
            logging_config._log_dir = global_config.log_dir
        
        if tracker_config:
            tracker_config._exp_name = global_config.exp_name
            tracker_config._run_name = global_config.run_name
            tracker_config._log_dir = global_config.log_dir
    
    def _auto_inject_tracker_callbacks(
        self,
        callbacks: Optional[List[Any]],
        tracker_config: Optional[TrackerConfig],
    ) -> List[Any]:
        """
        自动为 Tracker backends 注入对应的 Callbacks
        
        当用户配置了 tracker.backends 时，自动添加对应的 callback，
        这样用户无需同时配置 tracker 和 callbacks。
        
        规则：
        - tracker.backends: [mlflow] → 自动添加 callbacks: [{type: mlflow}]
        - 如果用户已经配置了对应的 callback，则不重复添加
        
        Args:
            callbacks: 用户配置的 callbacks 列表
            tracker_config: Tracker 配置
            
        Returns:
            合并后的 callbacks 列表
            
        Example:
            # 用户只配置了 tracker
            tracker:
              backends:
                - type: mlflow
                  args: {tracking_uri: ./mlruns}
            
            # 自动添加 mlflow callback，等效于
            callbacks:
              - type: mlflow
        """
        # 初始化 callbacks 列表
        result = list(callbacks) if callbacks else []
        
        # 如果没有 tracker 配置或 tracker 未启用，直接返回
        if not tracker_config or not tracker_config.enabled:
            return result
        
        # 获取已存在的 callback 类型
        existing_types = set()
        for cb in result:
            if isinstance(cb, dict):
                existing_types.add(cb.get("type", ""))
            elif hasattr(cb, "type"):
                existing_types.add(cb.type)
        
        # Tracker backend 类型到 callback 类型的映射
        # 某些 tracker 需要对应的 callback 来记录指标
        tracker_to_callback_map = {
            "mlflow": "mlflow",
            # 可以扩展其他映射
            # "wandb": "wandb",
            # "tensorboard": "tensorboard",
        }
        
        # 遍历 tracker backends，自动添加对应的 callback
        for backend in tracker_config.get_backends():
            backend_type = backend.type
            
            # 检查是否有对应的 callback 需要添加
            if backend_type in tracker_to_callback_map:
                callback_type = tracker_to_callback_map[backend_type]
                
                # 如果用户没有配置这个 callback，自动添加
                if callback_type not in existing_types:
                    result.append({
                        "type": callback_type,
                        "args": {},  # 使用默认参数
                    })
                    existing_types.add(callback_type)
        
        return result
    
    # ==================== 解析方法 ====================
    
    def _parse_global_config(self, data: Dict[str, Any]) -> GlobalConfig:
        """解析全局配置
        
        支持多种配置格式（按优先级）：
        1. global.exp_name / global.run_name
        2. 根级别 exp_name / run_name
        3. logging.exp_name / logging.run_name
        """
        # 支持 "global" 和 "global_" 两种键名（后者用于 Python 关键字规避）
        global_data = data.get("global", data.get("global_", {}))
        logging_data = data.get("logging", {})
        
        # 按优先级获取 exp_name: global > 根级别 > logging
        exp_name = (
            global_data.get("exp_name") or 
            data.get("exp_name") or 
            logging_data.get("exp_name") or 
            "default"
        )
        
        # 按优先级获取 run_name: global > 根级别 > logging
        run_name = (
            global_data.get("run_name") or 
            data.get("run_name") or 
            logging_data.get("run_name")
        )
        
        # 按优先级获取 log_dir: global > logging > 默认
        log_dir = (
            global_data.get("log_dir") or 
            logging_data.get("log_dir") or 
            "./logs"
        )
        
        return GlobalConfig(
            exp_name=exp_name,
            run_name=run_name,
            log_dir=log_dir,
        )
    
    def _parse_log_config(self, data: Dict[str, Any]) -> LogConfig:
        """解析日志配置"""
        if not data:
            return LogConfig()
        
        return LogConfig(
            level=data.get("level", "INFO"),
            console=data.get("console", True),
            console_level=data.get("console_level", "INFO"),
            rotation=data.get("rotation", "10 MB"),
            retention=data.get("retention", "30 days"),
            compression=data.get("compression", "zip"),
            format=data.get("format", LogConfig.format),
            diagnose=data.get("diagnose", False),
            # 添加实验相关字段
            log_dir=data.get("log_dir", "./logs"),
            exp_name=data.get("exp_name"),
            run_name=data.get("run_name"),
        )
    
    def _parse_tracker_config(
        self,
        data: Optional[Dict[str, Any]],
    ) -> Optional[TrackerConfig]:
        """解析追踪配置"""
        if not data:
            return None
        
        return TrackerConfig(
            enabled=data.get("enabled", True),
            tracking_dir=data.get("tracking_dir", "tracking"),
            backends=data.get("backends"),
        )
    
    def _parse_transport_config(self, data: Dict[str, Any]) -> TransportConfig:
        """解析传输配置"""
        if not data:
            return TransportConfig()
        
        grpc_data = data.get("grpc", {})
        grpc_config = GrpcConfig(
            host=grpc_data.get("host", "0.0.0.0"),
            port=grpc_data.get("port", 50051),
            max_message_size=grpc_data.get("max_message_size", 100 * 1024 * 1024),
        )
        
        return TransportConfig(
            mode=data.get("mode", "memory"),
            grpc=grpc_config,
        )
    
    def _parse_connection_retry_config(
        self,
        data: Dict[str, Any],
    ) -> ConnectionRetryConfig:
        """解析连接重试配置"""
        if not data:
            return ConnectionRetryConfig()
        
        return ConnectionRetryConfig(
            enabled=data.get("enabled", True),
            max_retries=data.get("max_retries", 10),
            retry_interval=data.get("retry_interval", 2.0),
            timeout=data.get("timeout", 60.0),
            backoff=data.get("backoff", "exponential"),
            backoff_factor=data.get("backoff_factor", 1.5),
        )
    
    # ==================== 工具方法 ====================
    
    def _load_yaml(self, path: Union[str, Path]) -> Dict[str, Any]:
        """加载 YAML 文件"""
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        # 替换环境变量 ${VAR:default}
        import re
        def replace_env_var(match):
            var_name = match.group(1)
            default_value = match.group(2) if match.group(2) else ""
            return os.environ.get(var_name, default_value)

        # 匹配 ${VAR:default} 或 ${VAR}
        content = re.sub(r'\$\{([^}:]+)(?::([^}]*))?\}', replace_env_var, content)

        return yaml.safe_load(content) or {}
    
    def _load_paper_defaults(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """
        从论文注册表加载默认配置
        
        Args:
            paper_id: 论文ID
            
        Returns:
            论文默认配置字典，如果论文不存在返回None
        """
        try:
            # 尝试导入论文注册表
            try:
                from ..papers import get_registry
            except ImportError:
                try:
                    from oiafed.papers import get_registry
                except ImportError:
                    # 论文模块不可用，静默忽略
                    return None
            
            registry = get_registry()
            paper = registry.get(paper_id)
            
            if not paper:
                import logging
                logging.warning(f"未找到论文: {paper_id}")
                return None
            
            # 构建配置字典
            defaults = registry.get_defaults(paper_id)
            
            # 添加组件类型
            config = {}
            
            if paper.get_component("learner"):
                config["learner"] = {
                    "type": paper.get_component("learner"),
                    "args": defaults.get("learner", {}),
                }
            
            if paper.get_component("aggregator"):
                config["aggregator"] = {
                    "type": paper.get_component("aggregator"),
                    "args": defaults.get("aggregator", {}),
                }
            
            if paper.get_component("trainer"):
                config["trainer"] = {
                    "type": paper.get_component("trainer"),
                    "args": defaults.get("trainer", {}),
                }
            
            if paper.get_component("model"):
                config["model"] = {
                    "type": paper.get_component("model"),
                    "args": defaults.get("model", {}),
                }
            
            return config
            
        except Exception as e:
            import logging
            logging.warning(f"加载论文配置失败: {paper_id}, {e}")
            return None
    
    def _deep_merge(
        self,
        base: Dict[str, Any],
        override: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        深度合并两个字典
        
        策略：
        - 字典：递归合并
        - 列表：完全覆盖
        - 标量：覆盖
        """
        result = copy.deepcopy(base)
        
        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                # 递归合并字典
                result[key] = self._deep_merge(result[key], value)
            else:
                # 覆盖（包括列表和标量）
                result[key] = copy.deepcopy(value)
        
        return result
    
    def _component_to_dict(
        self,
        component: Union[Dict[str, Any], ComponentConfig, DatasetConfig, CallbackConfig, Any],
    ) -> Dict[str, Any]:
        """将组件配置转换为字典"""
        if isinstance(component, dict):
            return component
        if hasattr(component, "type") and hasattr(component, "args"):
            result = {"type": component.type}
            if component.args:
                result["args"] = component.args
            if hasattr(component, "partition") and component.partition:
                result["partition"] = component.partition
            return result
        return {}
    
    # ==================== 工厂方法 ====================
    
    def create_trainer_config(
        self,
        node_id: str,
        exp_name: str = "default",
        run_name: Optional[str] = None,
        port: int = 50051,
        trainer_type: str = "federated.trainer.fedavg",
        aggregator_type: str = "federated.aggregator.fedavg",
        **kwargs,
    ) -> NodeConfig:
        """
        创建 Trainer 配置的便捷方法
        
        Args:
            node_id: 节点 ID
            exp_name: 实验名称
            run_name: 运行名称
            port: 监听端口
            trainer_type: Trainer 类型
            aggregator_type: Aggregator 类型
            **kwargs: 其他配置
            
        Returns:
            NodeConfig 实例
        """
        data = {
            "node_id": node_id,
            "role": "trainer",
            "global": {
                "exp_name": exp_name,
                "run_name": run_name,
            },
            "listen": {"port": port},
            "trainer": {"type": trainer_type},
            "aggregator": {"type": aggregator_type},
            **kwargs,
        }
        return self.from_dict(data)
    
    def create_learner_config(
        self,
        node_id: str,
        connect_to: List[str],
        exp_name: str = "default",
        run_name: Optional[str] = None,
        learner_type: str = "federated.learner.sgd",
        **kwargs,
    ) -> NodeConfig:
        """
        创建 Learner 配置的便捷方法
        
        Args:
            node_id: 节点 ID
            connect_to: 连接目标列表
            exp_name: 实验名称
            run_name: 运行名称
            learner_type: Learner 类型
            **kwargs: 其他配置
            
        Returns:
            NodeConfig 实例
        """
        data = {
            "node_id": node_id,
            "role": "learner",
            "global": {
                "exp_name": exp_name,
                "run_name": run_name,
            },
            "connect_to": connect_to,
            "learner": {"type": learner_type},
            **kwargs,
        }
        return self.from_dict(data)
    
    def clone_config(
        self,
        config: NodeConfig,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> NodeConfig:
        """
        克隆配置并应用覆盖
        
        Args:
            config: 原始配置
            overrides: 覆盖的字段
            
        Returns:
            新的 NodeConfig 实例
        """
        base_dict = self.to_dict(config)
        
        if overrides:
            base_dict = self._deep_merge(base_dict, overrides)
        
        return self.from_dict(base_dict)


# ==================== 默认管理器实例 ====================

_default_manager: Optional[ConfigManager] = None


def get_default_manager() -> ConfigManager:
    """获取默认的配置管理器实例"""
    global _default_manager
    if _default_manager is None:
        _default_manager = ConfigManager()
    return _default_manager


# ==================== 便捷函数 ====================

def load_config(
    path: str,
    base_dir: Optional[str] = None,
    validate: bool = True,
) -> NodeConfig:
    """
    加载配置文件
    
    Args:
        path: 配置文件路径
        base_dir: 基础目录
        validate: 是否验证配置
        
    Returns:
        NodeConfig 实例
        
    Example:
        config = load_config("configs/trainer.yaml")
    """
    return get_default_manager().load(path, base_dir, validate)


def load_config_from_dict(
    data: Dict[str, Any],
    validate: bool = True,
) -> NodeConfig:
    """
    从字典加载配置
    
    Args:
        data: 配置字典
        validate: 是否验证配置
        
    Returns:
        NodeConfig 实例
        
    Example:
        config = load_config_from_dict({
            "node_id": "trainer",
            "global": {"exp_name": "my_exp"},
        })
    """
    return get_default_manager().from_dict(data, validate)


def save_config(config: NodeConfig, path: str) -> None:
    """
    保存配置到文件
    
    Args:
        config: NodeConfig 实例
        path: 保存路径
    """
    get_default_manager().save(config, path)


def validate_config(config: NodeConfig) -> None:
    """
    验证配置
    
    Args:
        config: NodeConfig 实例
        
    Raises:
        ConfigValidationError: 配置无效
    """
    get_default_manager().validate(config)


def config_to_dict(config: NodeConfig) -> Dict[str, Any]:
    """
    将配置转换为字典
    
    Args:
        config: NodeConfig 实例
        
    Returns:
        配置字典
    """
    return get_default_manager().to_dict(config)


# ==================== 向后兼容 ====================

# 保留旧的函数名
def load_node_config(
    path: str,
    base_dir: Optional[str] = None,
) -> NodeConfig:
    """加载节点配置（向后兼容）"""
    return load_config(path, base_dir)


def deep_merge(
    base: Dict[str, Any],
    override: Dict[str, Any],
) -> Dict[str, Any]:
    """深度合并字典（向后兼容）"""
    return get_default_manager()._deep_merge(base, override)


def create_client_config(
    base_config: NodeConfig,
    client_id: str,
    port: int = 0,
) -> NodeConfig:
    """从基础配置创建客户端配置（向后兼容）"""
    manager = get_default_manager()
    
    overrides: Dict[str, Any] = {
        "node_id": client_id,
        "role": "learner",
    }
    
    if port > 0:
        overrides["transport"] = {
            "grpc": {"port": port},
        }
    
    return manager.clone_config(base_config, overrides)


# ==================== 导出 ====================

__all__ = [
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
]