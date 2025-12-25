"""
Callback 基类

定义训练过程中的钩子点
"""

from abc import ABC
from typing import Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.system import FederatedSystem


class Callback(ABC):
    """
    Callback 基类

    定义训练过程中的钩子点

    生命周期：
    - 系统级：on_system_start, on_system_stop
    - 训练级：on_train_begin, on_train_end
    - 轮次级：on_round_begin, on_round_end
    """

    # ========== 系统级钩子 ==========

    async def on_system_start(self, system: "FederatedSystem"):
        """
        系统启动时调用

        Args:
            system: FederatedSystem 实例
        """
        pass

    async def on_system_stop(self, system: "FederatedSystem"):
        """
        系统停止时调用

        Args:
            system: FederatedSystem 实例
        """
        pass

    # ========== 训练级钩子 ==========

    async def on_train_begin(self, trainer: Any, context: Dict[str, Any] = None):
        """
        训练开始时调用

        Args:
            trainer: Trainer 实例
            context: 上下文信息
        """
        pass

    async def on_train_end(self, trainer: Any, context: Dict[str, Any] = None):
        """
        训练结束时调用

        Args:
            trainer: Trainer 实例
            context: 上下文信息（如最终指标）
        """
        pass

    # ========== 轮次级钩子 ==========

    async def on_round_begin(
        self,
        trainer: Any,
        round_num: int,
        context: Dict[str, Any] = None
    ):
        """
        轮次开始时调用

        Args:
            trainer: Trainer 实例
            round_num: 轮次编号
            context: 上下文信息
        """
        pass

    async def on_round_end(
        self,
        trainer: Any,
        round_num: int,
        context: Dict[str, Any] = None
    ):
        """
        轮次结束时调用

        Args:
            trainer: Trainer 实例
            round_num: 轮次编号
            context: 上下文信息（如轮次指标）
        """
        pass

    # ========== Learner 级钩子 (分层训练) ==========

    async def on_fit_start(self, learner: Any, config: Dict[str, Any] = None):
        """
        Learner fit 开始时调用

        Args:
            learner: Learner 实例
            config: 训练配置
        """
        pass

    async def on_fit_end(self, learner: Any, result: Any = None):
        """
        Learner fit 结束时调用

        Args:
            learner: Learner 实例
            result: 训练结果
        """
        pass

    async def on_epoch_start(self, learner: Any, epoch: int):
        """
        Epoch 开始时调用

        Args:
            learner: Learner 实例
            epoch: Epoch 编号
        """
        pass

    async def on_epoch_end(self, learner: Any, epoch: int, metrics: Any = None):
        """
        Epoch 结束时调用

        Args:
            learner: Learner 实例
            epoch: Epoch 编号
            metrics: Epoch 指标
        """
        pass

    async def on_step_start(self, learner: Any, step: int):
        """
        Step 开始时调用

        Args:
            learner: Learner 实例
            step: Step 编号
        """
        pass

    async def on_step_end(self, learner: Any, step: int, metrics: Any = None):
        """
        Step 结束时调用

        Args:
            learner: Learner 实例
            step: Step 编号
            metrics: Step 指标
        """
        pass

    async def on_evaluate_end(self, learner: Any, result: Any = None, context: Dict[str, Any] = None):
        """
        评估结束时调用

        Args:
            learner: Learner 实例
            result: 评估结果 (EvalResult)
            context: 上下文信息 (如 round_num, epoch 等)
        """
        pass
