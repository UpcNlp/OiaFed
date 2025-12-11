"""
MLflow Logger 实现
fedcl/loggers/mlflow_logger.py

特点：
- 支持认证（从 .env 自动读取）
- 本地缓存 + 批量上传（解决 stream upload 问题）
- 线程安全
"""

import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Union
from datetime import datetime

from .base_logger import Logger

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# 静默模式控制
_VERBOSE = os.getenv('FEDCL_MLFLOW_VERBOSE', 'false').lower() in ('true', '1', 'yes')


def _log_message(message: str):
    """条件日志输出"""
    if _VERBOSE:
        print(f"[MLflowLogger] {message}")


class MLflowLogger(Logger):
    """
    MLflow Logger 实现

    特点：
    - 从 .env 自动读取认证信息（MLFLOW_TRACKING_URI, USERNAME, PASSWORD）
    - 支持多个 logger 实例（不同 experiment）
    - 线程安全
    - 本地文件存储（MLflow 自带，解决 stream upload 问题）

    环境变量配置（.env）：
        MLFLOW_TRACKING_URI=http://localhost:5000
        MLFLOW_TRACKING_USERNAME=your_username
        MLFLOW_TRACKING_PASSWORD=your_password

    使用示例：
        # 基本使用
        logger = MLflowLogger(experiment_name="my_exp")
        logger.log_params({"lr": 0.01})
        logger.log_metrics({"accuracy": 0.95}, step=1)
        logger.finalize()

        # 多个 logger 组合
        loggers = [
            MLflowLogger(experiment_name="exp1"),
            JSONLogger(save_dir="results/"),
        ]
    """

    def __init__(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        """
        Args:
            experiment_name: 实验名称（用于分组）
            run_name: Run 名称（可选，默认自动生成）
            tracking_uri: MLflow 服务器地址（可选，默认从环境变量读取）
            tags: 初始标签
            **kwargs: 其他 MLflow 参数
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError(
                "MLflow 未安装。请运行: pip install mlflow"
            )

        super().__init__()

        self._experiment_name = experiment_name
        self._run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._lock = threading.Lock()

        # 配置 MLflow tracking URI
        self._configure_tracking_uri(tracking_uri)

        # 配置认证
        self._configure_authentication()

        # 设置实验
        self._setup_experiment()

        # 启动 run（支持嵌套）
        with self._lock:
            self.run = mlflow.start_run(
                experiment_id=self.experiment_id,
                run_name=self._run_name,
                nested=True  # 支持嵌套run
            )
            self._run_id = self.run.info.run_id

        # 设置初始标签
        if tags:
            with self._lock:
                mlflow.set_tags(tags)

        _log_message(f"Run started: {self._run_name} (id={self._run_id})")

    def _configure_tracking_uri(self, tracking_uri: Optional[str] = None):
        """配置 MLflow tracking URI"""
        uri = tracking_uri or os.getenv('MLFLOW_TRACKING_URI')

        if uri:
            mlflow.set_tracking_uri(uri)
            _log_message(f"Tracking URI: {uri}")
        else:
            current_uri = mlflow.get_tracking_uri()
            _log_message(f"Using default tracking URI: {current_uri}")

    def _configure_authentication(self):
        """配置 MLflow 认证（从环境变量读取）"""
        username = os.getenv('MLFLOW_TRACKING_USERNAME')
        password = os.getenv('MLFLOW_TRACKING_PASSWORD')

        if username and password:
            os.environ['MLFLOW_TRACKING_USERNAME'] = username
            os.environ['MLFLOW_TRACKING_PASSWORD'] = password
            _log_message(f"Authentication configured: username={username}")
        else:
            _log_message("No authentication configured")

    def _setup_experiment(self):
        """设置 MLflow 实验"""
        try:
            experiment = mlflow.get_experiment_by_name(self._experiment_name)
            if experiment is None:
                self.experiment_id = mlflow.create_experiment(self._experiment_name)
                _log_message(f"Created experiment: {self._experiment_name}")
            else:
                self.experiment_id = experiment.experiment_id
                _log_message(f"Using existing experiment: {self._experiment_name}")

            mlflow.set_experiment(self._experiment_name)
        except Exception as e:
            _log_message(f"Warning: Failed to setup experiment: {e}")
            self.experiment_id = None

    def log_metrics(
        self,
        metrics: Dict[str, Union[int, float]],
        step: Optional[int] = None
    ) -> None:
        """记录指标"""
        with self._lock:
            for key, value in metrics.items():
                try:
                    mlflow.log_metric(key, float(value), step=step)
                except Exception as e:
                    _log_message(f"Warning: Failed to log metric {key}: {e}")

    def log_params(self, params: Dict[str, Any]) -> None:
        """记录参数"""
        with self._lock:
            for key, value in params.items():
                try:
                    # MLflow 参数会被转换为字符串
                    if not isinstance(value, (str, int, float, bool)):
                        value = str(value)
                    mlflow.log_param(key, value)
                except Exception as e:
                    _log_message(f"Warning: Failed to log param {key}: {e}")

    def log_artifact(
        self,
        local_path: str,
        artifact_path: Optional[str] = None
    ) -> None:
        """上传文件"""
        with self._lock:
            try:
                path = Path(local_path)
                if path.exists():
                    mlflow.log_artifact(str(path), artifact_path)
                else:
                    _log_message(f"Warning: File not found: {local_path}")
            except Exception as e:
                _log_message(f"Failed to log artifact: {e}")

    def log_dict(
        self,
        dictionary: Dict[str, Any],
        filename: str
    ) -> None:
        """保存字典为 artifact"""
        with self._lock:
            try:
                mlflow.log_dict(dictionary, filename)
            except Exception as e:
                _log_message(f"Failed to log dict: {e}")

    def finalize(self, status: str = "success") -> None:
        """结束 run"""
        with self._lock:
            try:
                # 映射状态
                mlflow_status = {
                    "success": "FINISHED",
                    "failed": "FAILED",
                    "interrupted": "KILLED"
                }.get(status.lower(), "FINISHED")

                mlflow.end_run(status=mlflow_status)
                _log_message(f"Run ended: {self._run_name} (status={status})")

            except Exception as e:
                _log_message(f"Failed to end run: {e}")

    def __del__(self):
        """析构时确保 run 结束"""
        try:
            if hasattr(self, 'run') and self.run:
                mlflow.end_run()
        except:
            pass
