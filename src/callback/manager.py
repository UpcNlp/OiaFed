"""
Callback 管理器

管理多个 Callback，统一触发，自动处理异常
"""

from typing import List, Dict, Any, Optional
from .base import Callback
from ..infra import get_module_logger

logger = get_module_logger(__name__)


class CallbackManager:
    """
    Callback 管理器

    职责：
    1. 管理多个 Callback 实例
    2. 统一触发各个生命周期钩子
    3. 自动捕获并记录 Callback 异常
    4. 支持自定义事件

    生命周期：
    - 系统级：on_system_start, on_system_stop
    - 训练级（Trainer）：on_train_begin, on_train_end, on_round_begin, on_round_end
    - Learner 级：on_fit_start, on_fit_end, on_epoch_start, on_epoch_end, on_step_start, on_step_end
    """

    def __init__(self, fail_fast: bool = False):
        """
        初始化 CallbackManager

        Args:
            fail_fast: 如果为 True，遇到异常立即抛出；否则记录日志并继续
        """
        self._callbacks: List[Callback] = []
        self._fail_fast = fail_fast

    def __len__(self) -> int:
        """返回回调数量"""
        return len(self._callbacks)

    def __bool__(self) -> bool:
        """判断是否有回调"""
        return len(self._callbacks) > 0

    def add(self, callback: Callback):
        """
        添加 Callback

        Args:
            callback: Callback 实例
        """
        self._callbacks.append(callback)
        logger.debug(f"Added callback: {callback.__class__.__name__}")

    def remove(self, callback: Callback):
        """
        移除 Callback

        Args:
            callback: Callback 实例
        """
        if callback in self._callbacks:
            self._callbacks.remove(callback)
            logger.debug(f"Removed callback: {callback.__class__.__name__}")

    def clear(self):
        """清空所有 Callback"""
        self._callbacks.clear()

    async def _safe_invoke(
        self,
        callback: Callback,
        method_name: str,
        *args,
        **kwargs
    ) -> None:
        """
        安全调用单个 Callback 方法

        自动捕获异常并记录完整堆栈信息

        Args:
            callback: Callback 实例
            method_name: 方法名（如 "on_round_begin"）
            *args: 位置参数
            **kwargs: 关键字参数
        """
        method = getattr(callback, method_name, None)
        if not method:
            return

        try:
            await method(*args, **kwargs)
        except Exception as e:
            logger.error(
                f"Error in {callback.__class__.__name__}.{method_name}: {e}",
                exc_info=True  # 记录完整堆栈
            )
            if self._fail_fast:
                raise

    async def trigger(self, event: str, *args, **kwargs) -> None:
        """
        触发自定义事件

        允许用户触发自己定义的回调事件

        Args:
            event: 事件名（如 "on_custom_event"）
            *args: 传递给回调的位置参数
            **kwargs: 传递给回调的关键字参数

        Example:
            >>> await manager.trigger("on_model_updated", model, metrics)
            >>> # 会调用所有 callback.on_model_updated(model, metrics)
        """
        for callback in self._callbacks:
            await self._safe_invoke(callback, event, *args, **kwargs)

    # ==================== 系统级钩子 ====================

    async def on_system_start(self, system):
        """触发系统启动钩子"""
        for cb in self._callbacks:
            await self._safe_invoke(cb, "on_system_start", system)

    async def on_system_stop(self, system):
        """触发系统停止钩子"""
        for cb in self._callbacks:
            await self._safe_invoke(cb, "on_system_stop", system)

    # ==================== 训练级钩子（Trainer） ====================

    async def on_train_begin(self, trainer, context: Optional[Dict[str, Any]] = None):
        """触发训练开始钩子"""
        context = context or {}
        for cb in self._callbacks:
            await self._safe_invoke(cb, "on_train_begin", trainer, context)

    async def on_train_end(self, trainer, context: Optional[Dict[str, Any]] = None):
        """触发训练结束钩子"""
        context = context or {}
        for cb in self._callbacks:
            await self._safe_invoke(cb, "on_train_end", trainer, context)

    async def on_round_begin(
        self,
        trainer,
        round_num: int,
        context: Optional[Dict[str, Any]] = None
    ):
        """触发轮次开始钩子"""
        context = context or {}
        for cb in self._callbacks:
            await self._safe_invoke(cb, "on_round_begin", trainer, round_num, context)

    async def on_round_end(
        self,
        trainer,
        round_num: int,
        context: Optional[Dict[str, Any]] = None
    ):
        """触发轮次结束钩子"""
        print(f"[CallbackManager-DEBUG] on_round_end called: round={round_num}, num_callbacks={len(self._callbacks)}")
        logger.info(f"[CallbackManager] on_round_end called: round={round_num}, num_callbacks={len(self._callbacks)}")

        context = context or {}
        for i, cb in enumerate(self._callbacks):
            print(f"[CallbackManager-DEBUG] Invoking callback {i}: {cb.__class__.__name__}")
            logger.info(f"[CallbackManager] Invoking callback {i}: {cb.__class__.__name__}")
            await self._safe_invoke(cb, "on_round_end", trainer, round_num, context)

    # ==================== Learner 级钩子 ====================

    async def on_fit_start(self, learner, config: Optional[Dict[str, Any]] = None):
        """触发 fit 开始钩子"""
        config = config or {}
        for cb in self._callbacks:
            await self._safe_invoke(cb, "on_fit_start", learner, config)

    async def on_fit_end(self, learner, result=None):
        """触发 fit 结束钩子"""
        for cb in self._callbacks:
            await self._safe_invoke(cb, "on_fit_end", learner, result)

    async def on_epoch_start(self, learner, epoch: int):
        """触发 epoch 开始钩子"""
        for cb in self._callbacks:
            await self._safe_invoke(cb, "on_epoch_start", learner, epoch)

    async def on_epoch_end(self, learner, epoch: int, metrics=None):
        """触发 epoch 结束钩子"""
        for cb in self._callbacks:
            await self._safe_invoke(cb, "on_epoch_end", learner, epoch, metrics)

    async def on_step_start(self, learner, step: int):
        """触发 step 开始钩子"""
        for cb in self._callbacks:
            await self._safe_invoke(cb, "on_step_start", learner, step)

    async def on_step_end(self, learner, step: int, metrics=None):
        """触发 step 结束钩子"""
        for cb in self._callbacks:
            await self._safe_invoke(cb, "on_step_end", learner, step, metrics)

    async def on_evaluate_end(self, learner, result=None, context: Optional[Dict[str, Any]] = None):
        """触发评估结束钩子"""
        context = context or {}
        for cb in self._callbacks:
            await self._safe_invoke(cb, "on_evaluate_end", learner, result, context)

    # ==================== 辅助方法 ====================

    def __len__(self) -> int:
        """返回 Callback 数量"""
        return len(self._callbacks)

    def __repr__(self) -> str:
        return f"CallbackManager({len(self)} callbacks)"

    def __bool__(self) -> bool:
        """支持 if self.callbacks: 判断"""
        return len(self._callbacks) > 0
