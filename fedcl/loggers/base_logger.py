"""
Logger 抽象基类
fedcl/loggers/base_logger.py

参考 PyTorch Lightning Logger API 设计
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
from pathlib import Path


class Logger(ABC):
    """
    实验记录 Logger 抽象基类

    设计理念（参考 PyTorch Lightning）：
    - Logger 只负责记录，不参与训练流程控制
    - 简单的 API：log_metrics, log_params, log_artifact
    - 支持多个 logger 组合使用
    - 轻量级，零侵入

    使用示例：
        # 单个 logger
        logger = MLflowLogger(experiment_name="my_exp")
        logger.log_metrics({"accuracy": 0.95}, step=10)

        # 多个 logger 组合
        loggers = [
            MLflowLogger(experiment_name="my_exp"),
            JSONLogger(save_dir="results/"),
        ]
        for logger in loggers:
            logger.log_metrics({"accuracy": 0.95}, step=10)
    """

    def __init__(self):
        self._experiment_name: Optional[str] = None
        self._run_name: Optional[str] = None
        self._run_id: Optional[str] = None

    @property
    def name(self) -> str:
        """Logger 名称（如 'mlflow', 'json', 'wandb'）"""
        return self.__class__.__name__.replace('Logger', '').lower()

    @property
    def experiment_name(self) -> Optional[str]:
        """实验名称"""
        return self._experiment_name

    @property
    def run_name(self) -> Optional[str]:
        """Run 名称"""
        return self._run_name

    @property
    def run_id(self) -> Optional[str]:
        """Run ID（唯一标识）"""
        return self._run_id

    @abstractmethod
    def log_metrics(
        self,
        metrics: Dict[str, Union[int, float]],
        step: Optional[int] = None
    ) -> None:
        """
        记录指标

        Args:
            metrics: 指标字典，如 {"accuracy": 0.95, "loss": 0.3}
            step: 当前步数/轮数（可选）
        """
        pass

    @abstractmethod
    def log_params(self, params: Dict[str, Any]) -> None:
        """
        记录超参数/配置

        Args:
            params: 参数字典，如 {"lr": 0.01, "batch_size": 32}
        """
        pass

    @abstractmethod
    def log_artifact(
        self,
        local_path: str,
        artifact_path: Optional[str] = None
    ) -> None:
        """
        上传文件/模型

        Args:
            local_path: 本地文件路径
            artifact_path: 远程存储路径（可选）
        """
        pass

    def log_dict(
        self,
        dictionary: Dict[str, Any],
        filename: str
    ) -> None:
        """
        保存字典为文件（通常为 JSON）

        Args:
            dictionary: 要保存的字典
            filename: 文件名（如 "config.json"）

        默认实现：保存为临时 JSON 文件，然后上传
        子类可以覆盖以提供更高效的实现
        """
        import json
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as f:
            json.dump(dictionary, f, indent=2)
            temp_path = f.name

        try:
            self.log_artifact(temp_path, filename)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def finalize(self, status: str = "success") -> None:
        """
        结束记录（可选）

        Args:
            status: 运行状态（"success", "failed", "interrupted"）

        注意：不是所有 logger 都需要显式结束
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"


class LoggerCollection:
    """
    Logger 集合（参考 PyTorch Lightning 的多 logger 支持）

    使用示例：
        loggers = LoggerCollection([
            MLflowLogger(experiment_name="exp1"),
            JSONLogger(save_dir="results/"),
        ])

        # 自动分发到所有 logger
        loggers.log_metrics({"accuracy": 0.95}, step=10)
    """

    def __init__(self, loggers: list[Logger]):
        """
        Args:
            loggers: Logger 列表
        """
        self.loggers = loggers

    def log_metrics(
        self,
        metrics: Dict[str, Union[int, float]],
        step: Optional[int] = None
    ) -> None:
        """记录指标到所有 logger"""
        for logger in self.loggers:
            try:
                logger.log_metrics(metrics, step=step)
            except Exception as e:
                print(f"Warning: {logger.name} failed to log metrics: {e}")

    def log_params(self, params: Dict[str, Any]) -> None:
        """记录参数到所有 logger"""
        for logger in self.loggers:
            try:
                logger.log_params(params)
            except Exception as e:
                print(f"Warning: {logger.name} failed to log params: {e}")

    def log_artifact(
        self,
        local_path: str,
        artifact_path: Optional[str] = None
    ) -> None:
        """上传文件到所有 logger"""
        for logger in self.loggers:
            try:
                logger.log_artifact(local_path, artifact_path)
            except Exception as e:
                print(f"Warning: {logger.name} failed to log artifact: {e}")

    def log_dict(self, dictionary: Dict[str, Any], filename: str) -> None:
        """保存字典到所有 logger"""
        for logger in self.loggers:
            try:
                logger.log_dict(dictionary, filename)
            except Exception as e:
                print(f"Warning: {logger.name} failed to log dict: {e}")

    def finalize(self, status: str = "success") -> None:
        """结束所有 logger"""
        for logger in self.loggers:
            try:
                logger.finalize(status)
            except Exception as e:
                print(f"Warning: {logger.name} failed to finalize: {e}")

    def __repr__(self) -> str:
        return f"LoggerCollection({[l.name for l in self.loggers]})"
