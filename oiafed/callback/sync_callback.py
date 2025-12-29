"""
同步 Callback 基类

用于在训练开始前同步各种信息
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class SyncCallback(ABC):
    """
    同步 Callback 基类

    用于在训练开始前同步训练相关信息：
    - Tracker run_id/session_id 同步
    - 配置参数同步
    - 状态信息同步

    Usage:
        # 1. 实现子类
        class TrackerSyncCallback(SyncCallback):
            async def collect_sync_info(self):
                return {"mlflow_run_id": self.tracker.run_id}

            async def apply_sync_info(self, sync_info):
                self.tracker.join_run(sync_info["mlflow_run_id"])

        # 2. 在 Trainer 中调用
        await trainer._exchange_training_info()  # 自动收集并分发

        # 3. Learner 自动接收
        # Learner.sync_training_info() 被远程调用
    """

    @abstractmethod
    async def collect_sync_info(self) -> Dict[str, Any]:
        """
        收集需要同步的信息（Trainer 端）

        Returns:
            同步信息字典，例如：
            {
                "mlflow_run_id": "abc123",
                "wandb_session_id": "xyz",
                "global_lr": 0.01,
            }
        """
        pass

    @abstractmethod
    async def apply_sync_info(self, sync_info: Dict[str, Any]):
        """
        应用同步信息（Learner 端）

        Args:
            sync_info: 从 Trainer 接收的同步信息
        """
        pass
