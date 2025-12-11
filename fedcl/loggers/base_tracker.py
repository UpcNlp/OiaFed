"""
ExperimentTracker 抽象基类
fedcl/loggers/base_tracker.py

设计原则：
1. 统一接口，支持 MLflow/WandB/TensorBoard
2. 支持嵌套 runs（分布式场景）
3. 本地缓存 + 远程同步
4. 支持用户手动记录指标
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, Optional, List, Union
from pathlib import Path


class TrackerRole(Enum):
    """追踪器角色"""
    AGGREGATOR = "aggregator"  # Server 聚合节点
    WORKER = "worker"          # Client 工作节点
    STANDALONE = "standalone"  # 单机模式


class ExperimentTracker(ABC):
    """
    实验追踪器抽象基类

    设计原则：
    1. 统一接口，支持 MLflow/WandB/TensorBoard
    2. 支持嵌套 runs（分布式场景）
    3. 本地缓存 + 远程同步
    4. 线程安全
    5. 支持用户手动记录指标
    """

    def __init__(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        role: TrackerRole = TrackerRole.STANDALONE,
        parent_run_id: Optional[str] = None,
        auto_local_backup: bool = True,
        **kwargs
    ):
        """
        Args:
            experiment_name: 实验名称（用于分组）
            run_name: Run 名称（可选，默认自动生成）
            role: 节点角色（聚合器/工作节点/单机）
            parent_run_id: 父 run ID（嵌套 run 时使用）
            auto_local_backup: 是否自动本地备份
        """
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.role = role
        self.parent_run_id = parent_run_id
        self.auto_local_backup = auto_local_backup
        self._started = False

    # ========== Core Methods ==========

    @abstractmethod
    def start(self) -> str:
        """
        启动 run

        Returns:
            run_id: 唯一标识符
        """
        pass

    @abstractmethod
    def log_params(self, params: Dict[str, Any]) -> None:
        """记录超参数（只能记录一次）"""
        pass

    @abstractmethod
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        timestamp: Optional[int] = None
    ) -> None:
        """记录指标（可多次记录，形成时间序列）"""
        pass

    @abstractmethod
    def log_artifact(
        self,
        local_path: str,
        artifact_path: Optional[str] = None
    ) -> None:
        """上传文件 artifact"""
        pass

    @abstractmethod
    def log_model(
        self,
        model: Any,
        artifact_path: str,
        **kwargs
    ) -> None:
        """上传模型"""
        pass

    @abstractmethod
    def set_tags(self, tags: Dict[str, str]) -> None:
        """设置标签（用于过滤和搜索）"""
        pass

    @abstractmethod
    def finalize(self, status: str = "success") -> None:
        """
        结束 run

        Args:
            status: "success", "failed", "interrupted"
        """
        pass

    # ========== Distributed Support ==========

    @abstractmethod
    def create_nested_run(
        self,
        run_name: str,
        **kwargs
    ) -> 'ExperimentTracker':
        """
        创建嵌套 run（分布式场景）

        用途：
        - Server 创建主 run
        - Clients 创建嵌套 run

        Returns:
            新的 tracker 实例（嵌套 run）
        """
        pass

    @abstractmethod
    def get_run_id(self) -> str:
        """获取当前 run ID"""
        pass

    @abstractmethod
    def get_run_url(self) -> str:
        """获取 run 的 Web UI URL"""
        pass

    # ========== Aggregation Support ==========

    def log_aggregated_metrics(
        self,
        client_metrics: List[Dict[str, float]],
        step: int,
        aggregation_fn: str = "mean"
    ) -> None:
        """
        记录聚合后的指标

        Args:
            client_metrics: 各 Client 的指标列表
            step: 轮次
            aggregation_fn: 聚合函数（mean/std/min/max/all）

        Example:
            client_metrics = [
                {"accuracy": 0.85, "loss": 0.32},  # client_0
                {"accuracy": 0.82, "loss": 0.38},  # client_1
            ]

            记录聚合指标：
            {
                "aggregated/accuracy_mean": 0.835,
                "aggregated/accuracy_std": 0.015,
                "aggregated/loss_mean": 0.35,
                "aggregated/loss_std": 0.03
            }
        """
        import numpy as np

        aggregated = {}

        # 提取所有 metric 的 keys
        metric_keys = set()
        for metrics in client_metrics:
            metric_keys.update(metrics.keys())

        # 对每个 metric 计算聚合值
        for key in metric_keys:
            values = [m[key] for m in client_metrics if key in m]

            if aggregation_fn == "mean":
                aggregated[f"aggregated/{key}_mean"] = float(np.mean(values))
                aggregated[f"aggregated/{key}_std"] = float(np.std(values))
            elif aggregation_fn == "all":
                aggregated[f"aggregated/{key}_mean"] = float(np.mean(values))
                aggregated[f"aggregated/{key}_std"] = float(np.std(values))
                aggregated[f"aggregated/{key}_min"] = float(np.min(values))
                aggregated[f"aggregated/{key}_max"] = float(np.max(values))

        self.log_metrics(aggregated, step=step)

    # ========== Local Backup ==========

    @abstractmethod
    def save_local_backup(self) -> None:
        """保存本地备份（JSON 格式）"""
        pass

    @abstractmethod
    def load_local_backup(self, backup_path: str) -> None:
        """从本地备份恢复（断点续训）"""
        pass

    # ========== User API (手动记录指标) ==========

    def log(self, metrics: Dict[str, Union[int, float]], step: Optional[int] = None):
        """用户API：记录指标

        这是一个便捷方法，用户可以直接调用来记录自定义指标

        Args:
            metrics: 指标字典
            step: 步骤/轮次（可选）

        Example:
            tracker.log({"custom_metric": 0.95}, step=10)
        """
        float_metrics = {k: float(v) for k, v in metrics.items()}
        self.log_metrics(float_metrics, step=step)

    @property
    def name(self) -> str:
        """Tracker 名称"""
        return self.__class__.__name__.replace('Tracker', '').lower()
