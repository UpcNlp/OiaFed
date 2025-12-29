"""
Tracker 同步 Callback

确保所有节点记录到同一个 run/session
"""

from typing import Dict, Any, Optional
from .sync_callback import SyncCallback
from ..registry import register
from ..infra import get_module_logger

logger = get_module_logger(__name__)


@register("federated.callback.tracker_sync")
class TrackerSyncCallback(SyncCallback):
    """
    Tracker 同步 Callback

    功能：
    - Trainer 端：收集 Tracker 的 run_id/session_id
    - Learner 端：加入到相同的 run/session

    这样可以确保所有节点的记录都在同一个运行会话中

    Example:
        # Trainer 端配置
        config = {
            "tracker": {"type": "mlflow", "args": {...}},
            "callbacks": [
                {"type": "tracker_sync", "args": {}}
            ]
        }

        # Learner 端配置
        config = {
            "tracker": {"type": "mlflow", "args": {...}},
            # Learner 端也需要配置 tracker_sync
        }

        # 训练开始前会自动同步
    """

    def __init__(self, tracker=None):
        """
        Args:
            tracker: Tracker 实例（通常由框架自动注入）
        """
        self.tracker = tracker

    async def collect_sync_info(self) -> Dict[str, Any]:
        """
        收集 Tracker 同步信息（Trainer 端）

        遍历所有 Tracker 后端，收集 run_id/session_id
        """
        if self.tracker is None:
            return {}

        sync_info = {}

        # 如果是 CompositeTracker，遍历所有后端
        if hasattr(self.tracker, "trackers"):
            for tracker in self.tracker.trackers:
                if hasattr(tracker, "get_sync_info"):
                    info = tracker.get_sync_info()
                    sync_info.update(info)

        # 单个 Tracker
        elif hasattr(self.tracker, "get_sync_info"):
            sync_info = self.tracker.get_sync_info()

        logger.info(f"Collected tracker sync info: {list(sync_info.keys())}")
        return sync_info

    async def apply_sync_info(self, sync_info: Dict[str, Any]):
        """
        应用 Tracker 同步信息（Learner 端）

        加入到对应的 run/session
        """
        if self.tracker is None or not sync_info:
            return

        # 如果是 CompositeTracker
        if hasattr(self.tracker, "trackers"):
            for tracker in self.tracker.trackers:
                self._apply_to_single_tracker(tracker, sync_info)

        # 单个 Tracker
        else:
            self._apply_to_single_tracker(self.tracker, sync_info)

        logger.info("Tracker synchronized")

    def _apply_to_single_tracker(self, tracker, sync_info: Dict[str, Any]):
        """
        应用到单个 Tracker

        Args:
            tracker: Tracker 实例
            sync_info: 同步信息
        """
        try:
            # MLflow
            if hasattr(tracker, "join_run") and "mlflow_run_id" in sync_info:
                tracker.join_run(sync_info["mlflow_run_id"])
                logger.debug(f"Joined MLflow run: {sync_info['mlflow_run_id']}")

            # Wandb
            if hasattr(tracker, "join_session") and "wandb_session_id" in sync_info:
                tracker.join_session(sync_info["wandb_session_id"])
                logger.debug(f"Joined Wandb session: {sync_info['wandb_session_id']}")

            # Loguru 无需同步

        except Exception as e:
            logger.error(f"Failed to sync tracker {tracker.__class__.__name__}: {e}")
