"""
日志配置

基于 loguru 的日志配置类及解析方法。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


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
        
        # 路径配置（从 GlobalConfig 同步）
        log_dir: 日志目录
        exp_name: 实验名称
        run_name: 运行名称
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
    
    # 路径和实验信息（从 GlobalConfig 同步）
    log_dir: Optional[str] = field(default="./logs")
    exp_name: Optional[str] = field(default=None)
    run_name: Optional[str] = field(default=None)
    
    def __post_init__(self):
        """初始化后处理"""
        # 标准化日志级别为大写
        self.level = self.level.upper()
        self.console_level = self.console_level.upper()
    
    def get_log_path(self) -> str:
        """
        获取完整的日志路径
        
        Returns:
            格式：{log_dir}/{exp_name}/{run_name}/
        """
        parts = [self.log_dir or "./logs"]
        if self.exp_name:
            parts.append(self.exp_name)
        if self.run_name:
            parts.append(self.run_name)
        return "/".join(parts)
    
    def sync_from_global(self, log_dir: str, exp_name: str, run_name: Optional[str]):
        """从全局配置同步路径信息"""
        self.log_dir = log_dir
        self.exp_name = exp_name
        self.run_name = run_name


def parse_log_config(data: Optional[Dict[str, Any]]) -> LogConfig:
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
        log_dir=data.get("log_dir", "./logs"),
        exp_name=data.get("exp_name"),
        run_name=data.get("run_name"),
    )


# ==================== 导出 ====================

__all__ = [
    "LogConfig",
    "parse_log_config",
]
