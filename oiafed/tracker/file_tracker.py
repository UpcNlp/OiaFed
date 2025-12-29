"""
File-based Tracker implementation

Uses loguru for file logging
"""

from typing import Dict, Any, Optional
from .base import Tracker
from loguru import logger as base_logger


class FileTracker(Tracker):
    """
    File-based tracker using loguru
    """

    def __init__(
        self,
        node_id: str,
        log_file: str = "./logs/tracker.log",
        level: str = "INFO",
        rotation: str = "10 MB",
        retention: str = "30 days",
        console: bool = False,
        log_format: Optional[str] = None,
    ):
        self.node_id = node_id
        self.log_file = log_file

        # 尝试使用 loguru
        try:
            import sys

            self._use_loguru = True

            # 使用 bind() 创建一个带有特定 context 的 logger
            # 不调用 remove()，避免移除全局 logger 的 handlers
            try:
                from infra.logging import get_logger
                self.logger = get_logger(node_id, "file_tracker")
            except RuntimeError:
                # 在并行模式下，子进程日志系统可能未初始化
                # 先初始化日志系统，然后再获取 logger
                from infra.logging import setup_logging, get_logger
                setup_logging(node_id=node_id)
                self.logger = get_logger(node_id, "file_tracker")

            # 控制台输出（可选）
            if console:
                console_format = (
                    log_format or
                    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                    "<level>{level: <8}</level> | "
                    f"<cyan>{node_id}</cyan> | "
                    "<level>{message}</level>"
                )
                base_logger.add(
                    sink=sys.stderr,
                    level=level,
                    format=console_format,
                    filter=lambda record: record["extra"].get("context") == f"file_tracker_{node_id}",
                )

            # 文件输出
            file_format = (
                log_format or
                "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
                f"{node_id} | {{message}}"
            )
            base_logger.add(
                sink=log_file,
                level=level,
                format=file_format,
                rotation=rotation,
                retention=retention,
                filter=lambda record: record["extra"].get("context") == f"file_tracker_{node_id}",
            )

        except ImportError:
            # 降级到标准 logging
            import logging
            self._use_loguru = False
            self.logger = logging.getLogger(f"file_tracker.{node_id}")
            self.logger.setLevel(level)

            # 文件 handler
            handler = logging.FileHandler(log_file)
            handler.setLevel(level)
            formatter = logging.Formatter(
                f"%(asctime)s | %(levelname)-8s | {node_id} | %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

            # 控制台 handler（可选）
            if console:
                console_handler = logging.StreamHandler(sys.stderr)
                console_handler.setLevel(level)
                console_handler.setFormatter(formatter)
                self.logger.addHandler(console_handler)

    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """记录指标"""
        step_str = f"[Step {step}]" if step is not None else ""
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        self.logger.info(f"{step_str} Metrics: {metrics_str}")

    def log_params(self, params: Dict[str, Any]):
        """记录参数"""
        params_str = ", ".join(f"{k}={v}" for k, v in params.items())
        self.logger.info(f"Params: {params_str}")

    def log_artifact(self, local_path: str, artifact_path: str = None):
        """记录文件"""
        self.logger.info(f"Artifact: {local_path} -> {artifact_path}")

    def set_tags(self, tags: Dict[str, str]):
        """设置标签"""
        tags_str = ", ".join(f"{k}={v}" for k, v in tags.items())
        self.logger.info(f"Tags: {tags_str}")

    def close(self):
        """关闭 tracker"""
        self.logger.info("FileTracker closed")


def create_file_tracker(
    node_id: str,
    log_file: str = "./logs/tracker.log",
    config: Optional[Dict[str, Any]] = None,
) -> FileTracker:
    """
    创建 FileTracker

    Args:
        node_id: 节点 ID
        log_file: 日志文件路径
        config: 配置参数（level, rotation, retention, console, log_format）

    Returns:
        FileTracker 实例
    """
    config = config or {}
    return FileTracker(
        node_id=node_id,
        log_file=log_file,
        level=config.get("level", "INFO"),
        rotation=config.get("rotation", "10 MB"),
        retention=config.get("retention", "30 days"),
        console=config.get("console", False),
        log_format=config.get("log_format"),
    )
