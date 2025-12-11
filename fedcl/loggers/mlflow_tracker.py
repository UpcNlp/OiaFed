"""
MLflow Tracker 实现
fedcl/loggers/mlflow_tracker.py

特点：
- 继承 ExperimentTracker 抽象基类
- 支持认证（从 .env 自动读取）
- 支持嵌套 runs（分布式场景）
- 本地缓存 + 远程同步
- 线程安全
"""

import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Union
from datetime import datetime

# 尝试加载 .env 文件
try:
    from dotenv import load_dotenv
    load_dotenv()  # 自动查找并加载 .env 文件
except ImportError:
    pass  # python-dotenv 未安装，跳过

from .base_tracker import ExperimentTracker, TrackerRole

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
        print(f"[MLflowTracker] {message}")


class MLflowTracker(ExperimentTracker):
    """
    MLflow Tracker 实现

    特点：
    - 从 .env 自动读取认证信息（MLFLOW_TRACKING_URI, USERNAME, PASSWORD）
    - 支持嵌套 runs（分布式场景）
    - 支持角色（AGGREGATOR/WORKER/STANDALONE）
    - 线程安全
    - 本地文件存储（MLflow 自带）

    环境变量配置（.env）：
        MLFLOW_TRACKING_URI=http://localhost:5000
        MLFLOW_TRACKING_USERNAME=your_username
        MLFLOW_TRACKING_PASSWORD=your_password

    使用示例：
        # Server (AGGREGATOR)
        tracker = MLflowTracker(
            experiment_name="my_exp",
            run_name="server",
            role=TrackerRole.AGGREGATOR
        )

        # Client (WORKER)
        client_tracker = tracker.create_nested_run("client_0")
    """

    def __init__(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        role: TrackerRole = TrackerRole.STANDALONE,
        parent_run_id: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        auto_local_backup: bool = True,
        **kwargs
    ):
        """
        Args:
            experiment_name: 实验名称（用于分组）
            run_name: Run 名称（可选，默认自动生成）
            role: 节点角色
            parent_run_id: 父 run ID（嵌套 run 时使用）
            tracking_uri: MLflow 服务器地址（可选，默认从环境变量读取）
            tags: 初始标签
            auto_local_backup: 是否自动本地备份
            **kwargs: 其他 MLflow 参数
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError(
                "MLflow 未安装。请运行: pip install mlflow"
            )

        super().__init__(
            experiment_name=experiment_name,
            run_name=run_name,
            role=role,
            parent_run_id=parent_run_id,
            auto_local_backup=auto_local_backup
        )

        self._lock = threading.Lock()
        self._tracking_uri = tracking_uri
        self._tags = tags or {}
        self._run_id = None
        self.run = None

        # 配置 MLflow
        self._configure_tracking_uri(tracking_uri)
        self._configure_authentication()
        self._setup_experiment()

        # 自动启动 run
        self.start()

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
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                self.experiment_id = mlflow.create_experiment(self.experiment_name)
                _log_message(f"Created experiment: {self.experiment_name}")
            else:
                self.experiment_id = experiment.experiment_id
                _log_message(f"Using existing experiment: {self.experiment_name}")

            mlflow.set_experiment(self.experiment_name)
        except Exception as e:
            _log_message(f"Warning: Failed to setup experiment: {e}")
            self.experiment_id = None

    def start(self) -> str:
        """启动 run"""
        if self._started:
            return self._run_id

        with self._lock:
            # 生成 run_name
            if not self.run_name:
                self.run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # 启动 run（支持嵌套）
            self.run = mlflow.start_run(
                experiment_id=self.experiment_id,
                run_name=self.run_name,
                nested=True  # 总是支持嵌套
            )
            self._run_id = self.run.info.run_id
            self._started = True

            # 兼容字符串和枚举类型的 role
            role_str = self.role.value if hasattr(self.role, 'value') else str(self.role)

            # 设置标签
            tags = {
                "role": role_str,
                **self._tags
            }
            if self.parent_run_id:
                tags["parent_run_id"] = self.parent_run_id

            mlflow.set_tags(tags)

            _log_message(f"Run started: {self.run_name} (id={self._run_id}, role={role_str})")

        return self._run_id

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

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        timestamp: Optional[int] = None
    ) -> None:
        """记录指标（使用 Client API 明确指定 run_id）"""
        with self._lock:
            # 使用 MlflowClient 并明确指定 run_id
            client = MlflowClient()
            for key, value in metrics.items():
                try:
                    client.log_metric(
                        run_id=self._run_id,
                        key=key,
                        value=float(value),
                        step=step if step is not None else 0,
                        timestamp=timestamp
                    )
                except Exception as e:
                    _log_message(f"Warning: Failed to log metric {key}: {e}")

    def log_artifact(
        self,
        local_path: str,
        artifact_path: Optional[str] = None
    ) -> None:
        """上传文件

        Args:
            local_path: 本地文件或目录路径
            artifact_path: MLflow 中的存储路径（可选）
        """
        with self._lock:
            try:
                path = Path(local_path)
                if path.exists():
                    if path.is_dir():
                        # 上传整个目录
                        mlflow.log_artifacts(str(path), artifact_path)
                    else:
                        # 上传单个文件
                        mlflow.log_artifact(str(path), artifact_path)
                else:
                    _log_message(f"Warning: File not found: {local_path}")
            except Exception as e:
                _log_message(f"Failed to log artifact: {e}")

    def log_model(
        self,
        model: Any,
        artifact_path: str,
        **kwargs
    ) -> None:
        """上传模型"""
        with self._lock:
            try:
                # 根据模型类型选择合适的 log 方法
                if hasattr(model, 'state_dict'):  # PyTorch
                    mlflow.pytorch.log_model(model, artifact_path, **kwargs)
                else:
                    _log_message(f"Warning: Unsupported model type: {type(model)}")
            except Exception as e:
                _log_message(f"Failed to log model: {e}")

    def set_tags(self, tags: Dict[str, str]) -> None:
        """设置标签"""
        with self._lock:
            try:
                mlflow.set_tags(tags)
            except Exception as e:
                _log_message(f"Failed to set tags: {e}")

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
                _log_message(f"Run ended: {self.run_name} (status={status})")

            except Exception as e:
                _log_message(f"Failed to end run: {e}")

    def create_nested_run(
        self,
        run_name: str,
        **kwargs
    ) -> 'MLflowTracker':
        """创建嵌套 run（用于 Client）"""
        return MLflowTracker(
            experiment_name=self.experiment_name,
            run_name=run_name,
            role=TrackerRole.WORKER,
            parent_run_id=self.get_run_id(),
            tracking_uri=self._tracking_uri,
            **kwargs
        )

    def get_run_id(self) -> str:
        """获取当前 run ID"""
        return self._run_id

    def get_run_url(self) -> str:
        """获取 run 的 Web UI URL"""
        tracking_uri = mlflow.get_tracking_uri()
        return f"{tracking_uri}/#/experiments/{self.experiment_id}/runs/{self._run_id}"

    def save_local_backup(self) -> None:
        """保存本地备份"""
        # MLflow 自动管理本地缓存，无需额外实现
        pass

    def load_local_backup(self, backup_path: str) -> None:
        """从本地备份恢复"""
        # MLflow 自动管理本地缓存，无需额外实现
        pass

    def __del__(self):
        """析构时确保 run 结束"""
        try:
            if hasattr(self, 'run') and self.run:
                mlflow.end_run()
        except:
            pass


# 保持向后兼容：MLflowLogger 作为 MLflowTracker 的别名
MLflowLogger = MLflowTracker
