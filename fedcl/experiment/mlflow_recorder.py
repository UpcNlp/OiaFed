"""
MLflow-based实验记录器
fedcl/experiment/mlflow_recorder.py

功能：
- 完全兼容原有 Recorder API
- 使用 MLflow 作为后端存储
- 提供强大的可视化和实验管理能力
"""

import os
import threading
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    _log_message("[Warning] MLflow not installed. Please run: pip install mlflow")

# 静默模式控制（通过环境变量）
_VERBOSE = os.getenv('FEDCL_MLFLOW_VERBOSE', 'false').lower() in ('true', '1', 'yes')


def _log_message(message: str):
    """
    条件日志输出

    只在 FEDCL_MLFLOW_VERBOSE=true 时输出
    """
    if _VERBOSE:
        print(message)


class MLflowRecorder:
    """
    MLflow-based 实验记录器
    与原有 Recorder API 完全兼容

    使用方法：
        # 方式1：单例模式
        recorder = MLflowRecorder.initialize("my_exp", "server", "server_0")
        recorder.start_run({"mode": "memory"})
        recorder.log_scalar("accuracy", 0.95, step=1)
        recorder.finish()

        # 方式2：上下文管理器
        with MLflowRecorder("my_exp", "server", "server_0") as recorder:
            recorder.start_run({"mode": "memory"})
            recorder.log_scalar("accuracy", 0.95, step=1)
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self, experiment_name: str, role: str, node_id: str,
                 base_dir: str = "experiments/mlruns"):
        """
        Args:
            experiment_name: 实验名称
            role: "server" 或 "client"
            node_id: 节点ID（如 "server_1", "client_0"）
            base_dir: MLflow 存储目录
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow is not installed. Run: pip install mlflow")

        self.experiment_name = experiment_name
        self.role = role
        self.node_id = node_id
        self.base_dir = base_dir

        # 🎯 关键修复：优先使用已设置的 tracking URI（避免覆盖外部设置）
        current_uri = mlflow.get_tracking_uri()

        # 如果当前 URI 是默认值（空或mlruns），则使用提供的 base_dir
        if not current_uri or current_uri in ["", "file:///mlruns", "mlruns"]:
            tracking_uri = f"file:{Path(base_dir).absolute()}"
            mlflow.set_tracking_uri(tracking_uri)
            _log_message(f"[MLflowRecorder] 使用默认tracking URI: {tracking_uri}")
        else:
            # 已经设置了tracking URI（比如 reproduce_table3_experiments.py 中设置的）
            # 不覆盖，直接使用
            tracking_uri = current_uri
            _log_message(f"[MLflowRecorder] 使用已设置的tracking URI: {tracking_uri}")

        # 创建或获取实验
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
            else:
                experiment_id = experiment.experiment_id
        except Exception as e:
            _log_message(f"[MLflowRecorder] Warning: {e}")
            experiment_id = None

        mlflow.set_experiment(experiment_name)

        # 启动 MLflow run
        run_name = f"{role}_{node_id}"
        self.run = mlflow.start_run(run_name=run_name)
        self.run_id = self.run.info.run_id

        # 设置标签
        mlflow.set_tags({
            "role": role,
            "node_id": node_id,
            "run_type": "federated_learning"
        })

        # MLflow client (用于更细粒度的控制)
        self.client = MlflowClient()

        # 本地状态
        self.start_time = None
        self.is_finished = False

        _log_message(f"[MLflowRecorder] {role}_{node_id}: Run started (run_id={self.run_id})")

    @classmethod
    def get_instance(cls) -> Optional['MLflowRecorder']:
        """获取当前实例（如果已初始化）"""
        return cls._instance

    @classmethod
    def initialize(cls, experiment_name: str, role: str, node_id: str,
                   base_dir: str = "experiments/mlruns") -> 'MLflowRecorder':
        """初始化全局实例"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(experiment_name, role, node_id, base_dir)
        return cls._instance

    @classmethod
    def reset(cls):
        """重置全局实例"""
        with cls._lock:
            if cls._instance is not None and not cls._instance.is_finished:
                try:
                    cls._instance.finish(status="INTERRUPTED")
                except:
                    pass
            cls._instance = None

    def start_run(self, config: dict):
        """
        开始一次运行

        Args:
            config: 运行配置
        """
        try:
            self.start_time = datetime.now()

            # 记录配置参数
            # MLflow限制参数名长度，所以加个前缀
            for key, value in config.items():
                try:
                    # MLflow参数值必须是字符串、数字或布尔值
                    if isinstance(value, (str, int, float, bool)):
                        mlflow.log_param(f"config_{key}", value)
                    else:
                        mlflow.log_param(f"config_{key}", str(value))
                except Exception as e:
                    _log_message(f"[MLflowRecorder] Warning: Failed to log param {key}: {e}")

            # 记录开始时间
            mlflow.set_tag("start_time", self.start_time.isoformat())

        except Exception as e:
            _log_message(f"[MLflowRecorder] Failed to start run: {e}")

    def log_scalar(self, name: str, value: float, step: int = None):
        """
        记录标量指标

        Args:
            name: 指标名称
            value: 指标值
            step: 步骤/轮次编号
        """
        try:
            # MLflow的log_metric会自动处理时间戳
            if step is not None:
                mlflow.log_metric(name, float(value), step=step)
            else:
                mlflow.log_metric(name, float(value))
        except Exception as e:
            _log_message(f"[MLflowRecorder] Failed to log scalar {name}: {e}")

    def log_info(self, key: str, value):
        """
        记录实验信息

        Args:
            key: 信息键
            value: 信息值
        """
        try:
            # 使用 param 或 tag 存储信息
            if isinstance(value, (str, int, float, bool)):
                # 简单类型用 param
                mlflow.log_param(f"info_{key}", value)
            else:
                # 复杂类型转字符串用 tag
                mlflow.set_tag(f"info_{key}", str(value))
        except Exception as e:
            _log_message(f"[MLflowRecorder] Failed to log info {key}: {e}")

    def add_artifact(self, filepath: str, name: str = None):
        """
        添加附件文件

        Args:
            filepath: 文件路径
            name: 附件名称（可选）
        """
        try:
            file_path = Path(filepath)
            if file_path.exists():
                mlflow.log_artifact(str(file_path))
            else:
                _log_message(f"[MLflowRecorder] Warning: Artifact file not found: {filepath}")
        except Exception as e:
            _log_message(f"[MLflowRecorder] Failed to add artifact {filepath}: {e}")

    def finish(self, status: str = "COMPLETED"):
        """
        结束实验并保存结果

        Args:
            status: 实验状态（COMPLETED, FAILED等）
        """
        if self.is_finished:
            return

        try:
            # 记录结束时间和状态
            end_time = datetime.now()
            mlflow.set_tag("end_time", end_time.isoformat())
            mlflow.set_tag("final_status", status)

            # 记录持续时间
            if self.start_time:
                duration = (end_time - self.start_time).total_seconds()
                mlflow.log_metric("duration_seconds", duration)

            # 结束 MLflow run
            mlflow.end_run()

            self.is_finished = True

            _log_message(f"[MLflowRecorder] {self.role}_{self.node_id}: Results saved (status={status})")
            _log_message(f"[MLflowRecorder] View results: mlflow ui --backend-store-uri {self.base_dir}")

        except Exception as e:
            _log_message(f"[MLflowRecorder] Failed to finish run: {e}")
            try:
                mlflow.end_run(status="FAILED")
            except:
                pass

    def __enter__(self):
        """支持上下文管理器"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文时自动结束"""
        if exc_type is not None:
            self.finish(status="FAILED")
        else:
            if not self.is_finished:
                self.finish(status="COMPLETED")
        return False  # 不抑制异常
