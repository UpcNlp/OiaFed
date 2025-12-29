"""
日志配置类

提供统一的日志配置管理，支持从YAML加载和代码创建
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from pathlib import Path


@dataclass
class LogConfig:
    """
    日志配置类

    统一管理所有日志相关配置，包括：
    - 日志级别
    - 日志目录
    - 控制台输出
    - 文件输出
    - 日志格式
    - 轮转和保留策略

    Examples:
        # 从字典创建
        config = LogConfig.from_dict({
            "level": "DEBUG",
            "log_dir": "./logs",
            "console": True,
        })

        # 直接创建
        config = LogConfig(
            level="DEBUG",
            log_dir="./logs",
            console=True,
        )

        # 转换为字典
        config_dict = config.to_dict()
    """

    # ========== 基础配置 ==========
    level: str = "INFO"                   # 文件日志级别：DEBUG/INFO/WARNING/ERROR
    log_dir: str = "./logs"               # 日志根目录
    exp_name: Optional[str] = None        # 实验名称（用于组织日志目录）

    # ========== 控制台输出配置 ==========
    console: bool = True                  # 是否输出到控制台
    console_level: str = "INFO"          # 控制台日志级别

    # ========== 文件输出配置 ==========
    rotation: str = "10 MB"              # 日志轮转大小
    retention: str = "30 days"           # 日志保留时间
    compression: str = "zip"             # 压缩格式

    # ========== 格式配置 ==========
    # 日志格式（控制台和文件都使用相同格式）
    # 默认格式：简洁的控制台友好格式（来自 logging.py 的原始硬编码格式）
    format: str = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level: <8} | "
        "{extra[node_id]} | "
        "{name}:{function}:{line} - "
        "{message}"
    )

    # ========== 调试配置 ==========
    diagnose: bool = False               # 是否显示详细诊断信息（堆栈跟踪）

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LogConfig":
        """
        从字典创建配置

        Args:
            data: 配置字典

        Returns:
            LogConfig 实例

        Examples:
            config = LogConfig.from_dict({
                "level": "DEBUG",
                "log_dir": "./logs",
                "console": True,
                "console_level": "INFO",
                "diagnose": False,
            })
        """
        # 只提取我们需要的字段
        valid_fields = {
            "level", "log_dir", "exp_name",
            "console", "console_level",
            "rotation", "retention", "compression",
            "format", "diagnose"
        }

        filtered_data = {
            k: v for k, v in data.items()
            if k in valid_fields
        }

        return cls(**filtered_data)

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典

        Returns:
            配置字典
        """
        return {
            "level": self.level,
            "log_dir": self.log_dir,
            "exp_name": self.exp_name,
            "console": self.console,
            "console_level": self.console_level,
            "rotation": self.rotation,
            "retention": self.retention,
            "compression": self.compression,
            "format": self.format,
            "diagnose": self.diagnose,
        }

    def validate(self) -> None:
        """
        验证配置有效性

        Raises:
            ValueError: 配置无效
        """
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

        if self.level.upper() not in valid_levels:
            raise ValueError(
                f"Invalid log level: {self.level}. "
                f"Valid levels: {valid_levels}"
            )

        if self.console_level.upper() not in valid_levels:
            raise ValueError(
                f"Invalid console level: {self.console_level}. "
                f"Valid levels: {valid_levels}"
            )

        # 验证log_dir路径
        if not self.log_dir:
            raise ValueError("log_dir cannot be empty")

    def __post_init__(self):
        """初始化后验证"""
        # 标准化日志级别为大写
        self.level = self.level.upper()
        self.console_level = self.console_level.upper()


# ========== 便捷函数 ==========

def create_default_log_config(**kwargs) -> LogConfig:
    """
    创建默认日志配置

    Args:
        **kwargs: 覆盖的配置项

    Returns:
        LogConfig 实例

    Examples:
        # 使用默认配置
        config = create_default_log_config()

        # 覆盖部分配置
        config = create_default_log_config(
            level="DEBUG",
            console=False,
        )
    """
    return LogConfig(**kwargs)


def load_log_config_from_yaml(yaml_data: Dict[str, Any]) -> LogConfig:
    """
    从YAML数据加载日志配置

    Args:
        yaml_data: YAML加载的字典，包含logging字段

    Returns:
        LogConfig 实例

    Examples:
        import yaml

        with open("config.yaml") as f:
            data = yaml.safe_load(f)

        log_config = load_log_config_from_yaml(data)
    """
    logging_data = yaml_data.get("logging", {})

    if not logging_data:
        return create_default_log_config()

    return LogConfig.from_dict(logging_data)


__all__ = [
    "LogConfig",
    "create_default_log_config",
    "load_log_config_from_yaml",
]
