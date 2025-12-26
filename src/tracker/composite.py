"""
组合 Tracker - 统一管理多个 Tracker 后端

职责：
1. 从配置创建 Tracker 实例
2. 处理后端特定逻辑（如 MLflow 的 auto_start_run）
3. 将操作分发到所有后端
"""

from typing import List, Dict, Any, Optional, Union
from .base import Tracker
from ..infra import get_module_logger
from ..core.types import TrainResult, EvalResult, ClientUpdate, RoundResult, RoundMetrics


logger = get_module_logger(__name__)


class CompositeTracker(Tracker):
    """
    组合 Tracker - 统一的 Tracker 创建和管理

    设计目标：
    - 从配置文件创建 Tracker 实例
    - 处理后端特定的逻辑（如 MLflow auto_start_run、node_id 注入）
    - 将操作分发到所有后端

    使用方式1：直接传入 Tracker 实例列表（向后兼容）
        trackers = [mlflow_tracker, tensorboard_tracker]
        composite = CompositeTracker(trackers)

    使用方式2：从 TrackerConfig 创建（推荐）
        from config import TrackerConfig
        tracker_config = TrackerConfig(
            enabled=True,
            backends=[
                {"type": "mlflow", "args": {"tracking_uri": "..."}},
            ]
        )
        composite = CompositeTracker.from_config(
            tracker_config,
            node_id="trainer",
            is_trainer=True
        )
    """

    def __init__(self, trackers: List[Tracker]):
        """
        初始化 CompositeTracker

        Args:
            trackers: Tracker 实例列表
        """
        self.trackers = trackers
        logger.info(f"CompositeTracker initialized with {len(trackers)} backends")

    @classmethod
    def from_config(
        cls,
        tracker_config: Union["TrackerConfig", Dict[str, Any]],
        node_id: str,
        is_trainer: bool = False,
    ) -> Optional["CompositeTracker"]:
        """
        从配置创建 CompositeTracker

        Args:
            tracker_config: TrackerConfig 实例或配置字典（向后兼容）
            node_id: 节点 ID
            is_trainer: 是否是 Trainer 节点

        Returns:
            CompositeTracker 实例或 None

        Example:
            # 方式1：使用 TrackerConfig（推荐）
            from config import TrackerConfig
            tracker_config = TrackerConfig(enabled=True, backends=[...])
            tracker = CompositeTracker.from_config(tracker_config, "trainer", is_trainer=True)
            
            # 方式2：使用字典（向后兼容）
            config = {"enabled": True, "backends": [...]}
            tracker = CompositeTracker.from_config(config, "trainer", is_trainer=True)
        """
        # 处理 TrackerConfig 类型
        if hasattr(tracker_config, 'enabled'):
            # TrackerConfig 对象
            enabled = tracker_config.enabled
            backends = tracker_config.get_backends() if hasattr(tracker_config, 'get_backends') else []
        else:
            # 字典格式（向后兼容）
            enabled = tracker_config.get("enabled", True)
            backends = tracker_config.get("backends", [{"type": "file"}])
        
        # 检查是否启用
        if not enabled:
            logger.debug("Tracker is disabled")
            return None

        logger.debug(f"Tracker backends: {backends}")

        # 创建所有后端
        trackers = []
        for backend_config in backends:
            tracker = cls._create_backend(backend_config, node_id, is_trainer)
            if tracker:
                trackers.append(tracker)

        # 返回 CompositeTracker 或 None
        if len(trackers) == 0:
            logger.debug("No trackers created")
            return None
        else:
            logger.debug(f"Created {len(trackers)} tracker backend(s)")
            return cls(trackers)

    @staticmethod
    def _create_backend(
        backend_config: Dict[str, Any],
        node_id: str,
        is_trainer: bool,
    ) -> Optional[Tracker]:
        """
        创建单个 Tracker 后端

        Args:
            backend_config: 后端配置，支持两种格式：
                - dict: {"type": "mlflow", "args": {...}}
                - TrackerBackendConfig 对象
            node_id: 节点 ID
            is_trainer: 是否是 Trainer 节点

        Returns:
            Tracker 实例或 None
        """
        from ..registry import registry

        # 支持两种格式
        if isinstance(backend_config, dict):
            backend_type = backend_config["type"]
            backend_args = backend_config.get("args", {})
        else:
            # TrackerBackendConfig 对象
            backend_type = backend_config.type
            backend_args = backend_config.get_args()

        logger.debug(f"Creating tracker backend: {backend_type}")

        # 跳过已废弃的 file tracker
        if backend_type == "file":
            logger.debug("Skipping file tracker (deprecated)")
            return None

        # 应用后端特定的配置逻辑
        backend_args = CompositeTracker._apply_backend_specific_config(
            backend_type, backend_args, node_id, is_trainer
        )

        # 使用注册表创建 Tracker
        try:
            tracker = registry.create(
                namespace=f"federated.tracker.{backend_type}",
                **backend_args
            )
            logger.debug(f"Created Tracker backend: {backend_type}")
            return tracker
        except Exception as e:
            logger.error(f"Failed to create tracker backend {backend_type}: {e}")
            return None

    @staticmethod
    def _apply_backend_specific_config(
        backend_type: str,
        backend_args: Dict[str, Any],
        node_id: str,
        is_trainer: bool,
    ) -> Dict[str, Any]:
        """
        应用后端特定的配置逻辑

        Args:
            backend_type: 后端类型（如 "mlflow"）
            backend_args: 原始参数字典
            node_id: 节点 ID
            is_trainer: 是否是 Trainer 节点

        Returns:
            处理后的参数字典
        """
        # 复制参数，避免修改原始字典
        args = backend_args.copy()

        # MLflow 特定逻辑
        if backend_type == "mlflow":
            # 1. 设置 auto_start_run
            if "auto_start_run" not in args:
                args["auto_start_run"] = is_trainer
                logger.debug(
                    f"MLflow auto_start_run set to {is_trainer} (is_trainer={is_trainer})"
                )

            # 2. 注入 node_id
            if "node_id" not in args:
                args["node_id"] = node_id
                logger.debug(f"MLflow node_id set to {node_id}")

        # 可以在这里添加其他后端的特定逻辑
        # elif backend_type == "tensorboard":
        #     args = CompositeTracker._apply_tensorboard_config(args, node_id)

        return args

    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """记录到所有后端"""
        if isinstance(metrics, RoundMetrics):
            metrics = metrics.to_dict()
        else:
            metrics = metrics
            
        for tracker in self.trackers:
            try:
                tracker.log_metrics(metrics, step)
            except Exception as e:
                logger.error(f"Tracker error [{tracker.__class__.__name__}]: {e}")

    def log_params(self, params: Dict[str, Any]):
        """记录到所有后端"""
        for tracker in self.trackers:
            try:
                tracker.log_params(params)
            except Exception as e:
                logger.error(f"Tracker error [{tracker.__class__.__name__}]: {e}")

    def log_artifact(self, local_path: str, artifact_path: str = None):
        """记录到所有后端"""
        for tracker in self.trackers:
            try:
                tracker.log_artifact(local_path, artifact_path)
            except Exception as e:
                logger.error(f"Tracker error [{tracker.__class__.__name__}]: {e}")

    def set_tags(self, tags: Dict[str, str]):
        """记录到所有后端"""
        for tracker in self.trackers:
            try:
                tracker.set_tags(tags)
            except Exception as e:
                logger.error(f"Tracker error [{tracker.__class__.__name__}]: {e}")

    def close(self):
        """关闭所有后端"""
        for tracker in self.trackers:
            try:
                tracker.close()
            except Exception as e:
                logger.error(f"Tracker close error [{tracker.__class__.__name__}]: {e}")

    def __len__(self) -> int:
        """返回 Tracker 数量"""
        return len(self.trackers)

    def __repr__(self) -> str:
        return f"CompositeTracker({len(self)} backends)"