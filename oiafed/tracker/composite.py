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
            is_trainer=True,
            exp_name="my_experiment"
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
        exp_name: str = None,  # 新增：实验名称，用于 MLflow experiment_name
    ) -> Optional["CompositeTracker"]:
        """
        从配置创建 CompositeTracker

        Args:
            tracker_config: TrackerConfig 实例或配置字典（向后兼容）
            node_id: 节点 ID
            is_trainer: 是否是 Trainer 节点
            exp_name: 实验名称（用于 MLflow 的 experiment_name）

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
        print(f"[CompositeTracker-DEBUG] from_config called with tracker_config type: {type(tracker_config)}")
        logger.info(f"[DEBUG] from_config called with tracker_config type: {type(tracker_config)}")
        if hasattr(tracker_config, 'enabled'):
            # TrackerConfig 对象
            enabled = tracker_config.enabled
            backends = tracker_config.get_backends() if hasattr(tracker_config, 'get_backends') else []
            print(f"[CompositeTracker-DEBUG] TrackerConfig: enabled={enabled}, backends count={len(backends)}")
            logger.info(f"[DEBUG] TrackerConfig: enabled={enabled}, backends={backends}")
        else:
            # 字典格式（向后兼容）
            enabled = tracker_config.get("enabled", True)
            backends = tracker_config.get("backends", [{"type": "file"}])
            print(f"[CompositeTracker-DEBUG] Dict config: enabled={enabled}, backends count={len(backends)}")
            logger.info(f"[DEBUG] Dict config: enabled={enabled}, backends={backends}")
        
        # 检查是否启用
        if not enabled:
            logger.debug("Tracker is disabled")
            return None

        print(f"[CompositeTracker-DEBUG] backends type: {type(backends)}")
        print(f"[CompositeTracker-DEBUG] backends count: {len(backends)}")
        print(f"[CompositeTracker-DEBUG] exp_name: {exp_name}")
        for i, b in enumerate(backends):
            print(f"[CompositeTracker-DEBUG] backend[{i}] type: {type(b)}, value: {b}")
        logger.debug(f"Tracker backends: {backends}")

        # 创建所有后端
        trackers = []
        for backend_config in backends:
            tracker = cls._create_backend(backend_config, node_id, is_trainer, exp_name)
            if tracker:
                trackers.append(tracker)

        print(f"[CompositeTracker-DEBUG] Created {len(trackers)} trackers")
        
        # 返回 CompositeTracker 或 None
        if len(trackers) == 0:
            print(f"[CompositeTracker-DEBUG] No trackers created, returning None")
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
        exp_name: str = None,  # 新增
    ) -> Optional[Tracker]:
        """
        创建单个 Tracker 后端

        Args:
            backend_config: 后端配置，支持两种格式：
                - dict: {"type": "mlflow", "args": {...}}
                - TrackerBackendConfig 对象
            node_id: 节点 ID
            is_trainer: 是否是 Trainer 节点
            exp_name: 实验名称

        Returns:
            Tracker 实例或 None
        """
        from ..registry import registry

        print(f"[_create_backend-DEBUG] backend_config type: {type(backend_config)}")
        print(f"[_create_backend-DEBUG] backend_config: {backend_config}")

        # 支持两种格式
        if isinstance(backend_config, dict):
            backend_type = backend_config.get("type")
            backend_args = backend_config.get("args", {})
            print(f"[_create_backend-DEBUG] Dict format: type={backend_type}, args={backend_args}")
        else:
            # TrackerBackendConfig 对象
            backend_type = backend_config.type
            backend_args = backend_config.get_args()
            print(f"[_create_backend-DEBUG] TrackerBackendConfig: type={backend_type}, args={backend_args}")

        if not backend_type:
            print(f"[_create_backend-DEBUG] ERROR: backend_type is None or empty!")
            return None

        logger.debug(f"Creating tracker backend: {backend_type}")

        # 跳过已废弃的 file tracker
        if backend_type == "file":
            logger.debug("Skipping file tracker (deprecated)")
            return None

        # 应用后端特定的配置逻辑
        backend_args = CompositeTracker._apply_backend_specific_config(
            backend_type, backend_args, node_id, is_trainer, exp_name
        )
        print(f"[_create_backend-DEBUG] After apply config: args={backend_args}")

        # 使用注册表创建 Tracker
        namespace = f"federated.tracker.{backend_type}"
        print(f"[_create_backend-DEBUG] Creating with namespace: {namespace}")
        
        try:
            tracker = registry.create(
                namespace=namespace,
                **backend_args
            )
            print(f"[_create_backend-DEBUG] SUCCESS: Created tracker {tracker}")
            logger.debug(f"Created Tracker backend: {backend_type}")
            return tracker
        except Exception as e:
            print(f"[_create_backend-DEBUG] FAILED: {e}")
            import traceback
            print(traceback.format_exc())
            logger.error(f"Failed to create tracker backend {backend_type}: {e}")
            return None

    @staticmethod
    def _apply_backend_specific_config(
        backend_type: str,
        backend_args: Dict[str, Any],
        node_id: str,
        is_trainer: bool,
        exp_name: str = None,  # 新增
    ) -> Dict[str, Any]:
        """
        应用后端特定的配置逻辑

        Args:
            backend_type: 后端类型（如 "mlflow"）
            backend_args: 原始参数字典
            node_id: 节点 ID
            is_trainer: 是否是 Trainer 节点
            exp_name: 实验名称

        Returns:
            处理后的参数字典
        """
        # 复制参数，避免修改原始字典
        args = backend_args.copy()

        # MLflow 特定逻辑
        if backend_type == "mlflow":
            # 1. 设置 experiment_name（如果未指定则使用 exp_name）
            if "experiment_name" not in args and exp_name:
                args["experiment_name"] = exp_name
                logger.info(f"MLflow experiment_name set to {exp_name}")
            
            # 2. 设置 auto_start_run（Trainer 启动 run，Learner 必须等待同步）
            # 对于 Learner，强制设置为 False，避免创建无用的 run
            if is_trainer:
                args["auto_start_run"] = True
            else:
                args["auto_start_run"] = False  # Learner 强制为 False
            logger.info(
                f"MLflow auto_start_run set to {args['auto_start_run']} (is_trainer={is_trainer})"
            )

            # 3. 注入 node_id
            if "node_id" not in args:
                args["node_id"] = node_id
                logger.info(f"MLflow node_id set to {node_id}")

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