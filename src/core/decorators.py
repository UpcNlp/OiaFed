"""
装饰器模块

提供自动回调、日志记录等装饰器
"""

from functools import wraps
from typing import Callable, Any
from ..infra import get_module_logger

logger = get_module_logger(__name__)


def federated_training(
    auto_log: bool = True,
    auto_sync: bool = True,
) -> Callable:
    """
    联邦训练装饰器

    自动处理：
    - on_training_start/end 回调
    - 训练信息同步（SyncCallback）
    - 可选的日志记录

    Args:
        auto_log: 是否自动记录训练开始/结束
        auto_sync: 是否自动同步训练信息（Tracker等）

    Example:
        @federated_training()
        async def run(self):
            for round_num in range(self.max_rounds):
                await self.train_round(round_num)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            # 训练开始回调
            if hasattr(self, 'callbacks') and self.callbacks:
                await self.callbacks.on_training_start(self)

            if auto_log:
                logger.info(f"Training started: {self.__class__.__name__}")

            # 同步训练信息（Trainer端）
            if auto_sync and hasattr(self, '_sync_training_info'):
                await self._sync_training_info()

            try:
                # 执行训练
                result = await func(self, *args, **kwargs)

                # 训练结束回调
                if hasattr(self, 'callbacks') and self.callbacks:
                    await self.callbacks.on_training_end(self, result)

                if auto_log:
                    logger.info(f"Training completed: {self.__class__.__name__}")

                return result

            except Exception as e:
                logger.error(f"Training failed: {e}")
                if hasattr(self, 'callbacks') and self.callbacks:
                    await self.callbacks.on_training_end(self, {"error": str(e)})
                raise

        return wrapper
    return decorator


def federated_round(
    auto_log_metrics: bool = True,
    log_step: bool = True,
) -> Callable:
    """
    联邦轮次装饰器

    自动处理：
    - on_round_start/end 回调
    - 自动记录轮次指标到 Tracker

    Args:
        auto_log_metrics: 是否自动将返回的指标记录到 Tracker
        log_step: 是否将 round_num 作为 step 参数记录

    Example:
        @federated_round()
        async def train_round(self, round_num: int):
            # 训练逻辑
            return {"accuracy": 0.95, "loss": 0.1}
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            # 提取 round_num（通常是第一个参数）
            round_num = _extract_round_num(func, args, kwargs)

            # 轮次开始回调
            if hasattr(self, 'callbacks') and self.callbacks:
                await self.callbacks.on_round_start(self, round_num, {})

            if auto_log_metrics:
                logger.debug(f"Round {round_num} started")

            # 执行轮次训练
            result = await func(self, *args, **kwargs)

            # 自动记录指标
            if auto_log_metrics and hasattr(self, 'tracker') and self.tracker:
                if isinstance(result, dict):
                    step = round_num if log_step else None
                    self.tracker.log_metrics(result, step=step)

            # 轮次结束回调
            if hasattr(self, 'callbacks') and self.callbacks:
                await self.callbacks.on_round_end(self, round_num, result or {})

            return result

        return wrapper
    return decorator


def federated_fit(
    auto_log_metrics: bool = True,
) -> Callable:
    """
    Learner 训练装饰器

    自动处理：
    - before_fit/after_fit 回调
    - 自动记录训练指标

    Example:
        @federated_fit()
        async def fit(self, model, config):
            # 本地训练
            return TrainResult(...)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            # before_fit 回调
            if hasattr(self, 'callbacks') and self.callbacks:
                model = args[0] if args else kwargs.get('model')
                config = args[1] if len(args) > 1 else kwargs.get('config', {})
                await self.callbacks.before_fit(self, model, config)

            # 执行本地训练
            result = await func(self, *args, **kwargs)

            # 自动记录指标
            if auto_log_metrics and hasattr(self, 'tracker') and self.tracker:
                if hasattr(result, 'metrics') and result.metrics:
                    self.tracker.log_metrics(result.metrics)

            # after_fit 回调
            if hasattr(self, 'callbacks') and self.callbacks:
                await self.callbacks.after_fit(self, result)

            return result

        return wrapper
    return decorator


def federated_evaluate(
    auto_log_metrics: bool = True,
) -> Callable:
    """
    Learner 评估装饰器

    自动记录评估指标

    Example:
        @federated_evaluate()
        async def evaluate(self, model):
            # 评估逻辑
            return EvalResult(...)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            # 执行评估
            result = await func(self, *args, **kwargs)

            # 自动记录指标
            if auto_log_metrics and hasattr(self, 'tracker') and self.tracker:
                if hasattr(result, 'metrics') and result.metrics:
                    eval_metrics = {f"eval_{k}": v for k, v in result.metrics.items()}
                    self.tracker.log_metrics(eval_metrics)

            return result

        return wrapper
    return decorator


def _extract_round_num(func: Callable, args: tuple, kwargs: dict) -> int:
    """
    从参数中提取 round_num

    支持：
    - 第一个位置参数：func(self, round_num, ...)
    - 关键字参数：func(self, round_num=1, ...)
    """
    # 排除 self，取第一个参数
    if len(args) > 1:
        return args[1]

    # 尝试从 kwargs 中获取
    if 'round_num' in kwargs:
        return kwargs['round_num']

    # 默认返回 0
    return 0
