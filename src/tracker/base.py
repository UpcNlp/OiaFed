"""
Tracker 基类

定义记录接口
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class Tracker(ABC):
    """
    Tracker 基类

    定义记录接口，用于记录训练指标、参数、文件等
    """

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """
        记录指标

        Args:
            metrics: 指标字典 {"loss": 0.1, "accuracy": 0.95}
            step: 步骤编号（轮次）
        """
        pass

    @abstractmethod
    def log_params(self, params: Dict[str, Any]):
        """
        记录参数

        Args:
            params: 参数字典 {"learning_rate": 0.01, "batch_size": 32}
        """
        pass

    @abstractmethod
    def log_artifact(self, local_path: str, artifact_path: str = None):
        """
        记录文件（如模型）

        Args:
            local_path: 本地文件路径
            artifact_path: 存储路径（可选）
        """
        pass

    @abstractmethod
    def set_tags(self, tags: Dict[str, str]):
        """
        设置标签

        Args:
            tags: 标签字典 {"model": "resnet", "dataset": "mnist"}
        """
        pass

    def close(self):
        """关闭 Tracker（可选）"""
        pass

    # ========== 分布式同步接口（可选实现） ==========

    def get_sync_info(self) -> Dict[str, Any]:
        """
        获取同步信息（可选实现）

        用于分布式环境下的 Tracker 同步，确保所有节点记录到同一个 run/session

        Returns:
            需要同步的信息，如 {"mlflow_run_id": "abc123"}
            如果不需要同步，返回空字典 {}

        Note:
            - Trainer 端：返回 run_id/session_id
            - Learner 端：默认返回 {}
        """
        return {}

    def join_run(self, run_id: str):
        """
        加入已有的 run（可选实现）

        用于 Learner 端加入 Trainer 创建的 run

        Args:
            run_id: 运行 ID（来自 Trainer）

        Note:
            - 仅在支持分布式同步的 Tracker 中实现（如 MLflow）
            - Loguru 等本地 Tracker 无需实现
        """
        pass

    def join_session(self, session_id: str):
        """
        加入已有的 session（可选实现）

        用于 Learner 端加入 Trainer 创建的 session

        Args:
            session_id: 会话 ID（来自 Trainer）

        Note:
            - 用于 Wandb 等云端 Tracker
        """
        pass
