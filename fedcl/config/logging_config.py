"""
统一的日志配置
fedcl/config/logging_config.py

支持三层日志架构：
- Layer 1: Runtime Logs (Loguru)
- Layer 2: Training Progress (Console)
- Layer 3: Experiment Tracking (MLflow/WandB/TensorBoard)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path


@dataclass
class RuntimeConfig:
    """Layer 1: Runtime Logs 配置"""
    console_enabled: bool = True
    console_level: str = "INFO"
    file_level: str = "DEBUG"
    format: str = "{time} | {level} | {name}:{function}:{line} - {message}"
    rotation: str = "10 MB"
    retention: str = "30 days"
    compression: str = "zip"


@dataclass
class ProgressConfig:
    """Layer 2: Training Progress 配置"""
    enabled: bool = True
    style: str = "rich"  # rich / tqdm / minimal
    refresh_rate: float = 0.5


@dataclass
class TrackerConfig:
    """Layer 3: Experiment Tracking 配置"""
    enabled: bool = True
    type: str = "mlflow"  # mlflow / wandb / tensorboard / none
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoggingConfig:
    """统一的日志配置

    每个节点根据自己的配置文件决定：
    - runtime.console_enabled: 是否输出控制台
    - progress.enabled: 是否显示进度条
    - tracker.enabled: 是否启用实验追踪
    """

    # 基础配置
    base_dir: str = "logs"  # 日志根目录
    experiment_name: Optional[str] = None  # 实验名称

    # Layer 1: Runtime Logs
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    # Layer 2: Training Progress
    progress: ProgressConfig = field(default_factory=ProgressConfig)

    # Layer 3: Experiment Tracking
    tracker: TrackerConfig = field(default_factory=TrackerConfig)

    # 全局选项
    collect_loguru_artifacts: bool = True  # 是否收集 Loguru 日志为 artifacts

    # 保留旧字段用于兼容（废弃）
    save_dir: str = "experiments/results"
    trackers: List[str] = field(default_factory=list)

    def __post_init__(self):
        """验证配置并支持扁平配置自动转换"""
        # 首先将字典转换为嵌套配置对象
        if isinstance(self.runtime, dict):
            self.runtime = RuntimeConfig(**self.runtime)

        if isinstance(self.progress, dict):
            self.progress = ProgressConfig(**self.progress)

        if isinstance(self.tracker, dict):
            self.tracker = TrackerConfig(**self.tracker)

        # 兼容旧的 trackers 列表字段（扁平配置）
        if self.trackers:
            # 从扁平配置转换为嵌套配置
            if 'mlflow' in self.trackers:
                self.tracker.type = 'mlflow'
                self.tracker.enabled = True
            elif 'wandb' in self.trackers:
                self.tracker.type = 'wandb'
                self.tracker.enabled = True
            elif 'tensorboard' in self.trackers:
                self.tracker.type = 'tensorboard'
                self.tracker.enabled = True

        # 如果使用外部追踪系统，必须提供 experiment_name
        external_trackers = ['mlflow', 'wandb', 'tensorboard']
        if self.tracker.enabled and self.tracker.type in external_trackers:
            if not self.experiment_name:
                raise ValueError(
                    f"使用外部追踪系统 {self.tracker.type} 时，必须提供 experiment_name！"
                )

    def get_loguru_config(self) -> Dict[str, Any]:
        """提取 Loguru 配置（用于 AutoLogger）"""
        return {
            'base_dir': self.base_dir,
            'console_enabled': self.runtime.console_enabled,
            'console_level': self.runtime.console_level,
            'file_level': self.runtime.file_level,
            'format': self.runtime.format,
            'rotation': self.runtime.rotation,
            'retention': self.runtime.retention,
            'compression': self.runtime.compression,
        }


def collect_loguru_logs_as_artifacts(loggers: List, loguru_log_dir: Path):
    """收集 Loguru 日志作为 artifacts 上传到外部追踪系统

    Args:
        loggers: Logger 列表
        loguru_log_dir: Loguru 日志目录
    """
    from ..loggers.base_logger import Logger

    # 只上传到外部追踪系统（跳过 JSON Logger）
    external_loggers = [
        logger for logger in loggers
        if isinstance(logger, Logger) and logger.name not in ['json']
    ]

    if not external_loggers:
        return

    # 收集所有 .log 文件
    log_files = list(loguru_log_dir.rglob("*.log"))

    if not log_files:
        return

    # 上传到所有外部追踪系统
    for logger in external_loggers:
        for log_file in log_files:
            try:
                # 保留相对路径结构
                relative_path = log_file.relative_to(loguru_log_dir)
                artifact_path = f"loguru_logs/{relative_path}"
                logger.log_artifact(str(log_file), artifact_path)
            except Exception as e:
                print(f"Warning: Failed to upload {log_file} to {logger.name}: {e}")
