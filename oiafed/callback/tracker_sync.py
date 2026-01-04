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
    - Learner 端：加入到相同的 run/session，并记录自己的配置参数

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

    def __init__(self, tracker=None, node_config=None):
        """
        Args:
            tracker: Tracker 实例（通常由框架自动注入）
            node_config: NodeConfig 实例（用于记录 Learner 配置）
        """
        self.tracker = tracker
        self.node_config = node_config

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

        加入到对应的 run/session，然后记录 Learner 的配置参数
        """
        print(f"[tracker_sync] apply_sync_info called with sync_info keys: {list(sync_info.keys())}")
        print(f"[tracker_sync] tracker={self.tracker}, node_config={self.node_config}")
        
        if self.tracker is None:
            print("[tracker_sync] WARNING: tracker is None, returning early")
            return
        
        if not sync_info:
            print("[tracker_sync] WARNING: sync_info is empty, returning early")
            return

        # 如果是 CompositeTracker
        if hasattr(self.tracker, "trackers"):
            print(f"[tracker_sync] CompositeTracker with {len(self.tracker.trackers)} backends")
            for tracker in self.tracker.trackers:
                self._apply_to_single_tracker(tracker, sync_info)

        # 单个 Tracker
        else:
            print("[tracker_sync] Single tracker")
            self._apply_to_single_tracker(self.tracker, sync_info)

        print("[tracker_sync] Tracker synchronized, now calling _log_learner_config")
        
        # ===== 记录 Learner 配置参数 =====
        self._log_learner_config()

    def _apply_to_single_tracker(self, tracker, sync_info: Dict[str, Any]):
        """
        应用到单个 Tracker

        Args:
            tracker: Tracker 实例
            sync_info: 同步信息
        """
        print(f"[tracker_sync] _apply_to_single_tracker: tracker={tracker.__class__.__name__}, sync_info keys={list(sync_info.keys())}")
        
        try:
            # MLflow
            if hasattr(tracker, "join_run") and "mlflow_run_id" in sync_info:
                run_id = sync_info["mlflow_run_id"]
                print(f"[tracker_sync] Calling join_run with run_id={run_id}")
                tracker.join_run(run_id)
                print(f"[tracker_sync] join_run completed, tracker.run_id={getattr(tracker, 'run_id', 'N/A')}")

            # Wandb
            if hasattr(tracker, "join_session") and "wandb_session_id" in sync_info:
                tracker.join_session(sync_info["wandb_session_id"])
                print(f"[tracker_sync] Joined Wandb session: {sync_info['wandb_session_id']}")

        except Exception as e:
            print(f"[tracker_sync] ERROR: Failed to sync tracker {tracker.__class__.__name__}: {e}")
            import traceback
            traceback.print_exc()

    def _log_learner_config(self):
        """
        记录 Learner 配置参数到 Tracker
        
        在 join_run 之后调用，确保参数记录到正确的 run 中
        """
        print(f"[tracker_sync] _log_learner_config: tracker={self.tracker}, node_config={self.node_config}")
        
        if self.tracker is None:
            print("[tracker_sync] WARNING: tracker is None, skipping")
            return
        
        if self.node_config is None:
            print("[tracker_sync] WARNING: node_config is None, skipping")
            return
        
        try:
            # 获取配置参数
            params = self.node_config.get_tracking_params()
            print(f"[tracker_sync] Got {len(params)} params: {list(params.keys())}")
            
            # 记录参数
            if hasattr(self.tracker, "log_params"):
                self.tracker.log_params(params)
                print(f"[tracker_sync] Logged {len(params)} params for {self.node_config.node_id}")
            else:
                print("[tracker_sync] WARNING: tracker has no log_params method")
            
            # 设置标签
            if hasattr(self.tracker, "set_tags"):
                self.tracker.set_tags({
                    "node_id": self.node_config.node_id,
                    "role": "learner",
                })
                
        except Exception as e:
            print(f"[tracker_sync] ERROR: Failed to log learner config: {e}")
            import traceback
            traceback.print_exc()