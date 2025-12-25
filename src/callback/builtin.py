"""
内置 Callback 实现

提供常用的 Callback：
- ModelCheckpoint: 模型检查点
- EarlyStopping: 早停
"""

import os
from typing import Dict, Any
from .base import Callback
from ..registry import register
from ..infra.logging import get_module_logger

logger = get_module_logger(__name__)


@register("federated.callback.model_checkpoint")
class ModelCheckpoint(Callback):
    """
    模型检查点 Callback

    定期保存模型
    """

    def __init__(self, save_dir: str = "./checkpoints", save_freq: int = 10):
        """
        Args:
            save_dir: 保存目录
            save_freq: 保存频率（每 N 轮保存一次）
        """
        self.save_dir = save_dir
        self.save_freq = save_freq

        # 创建目录
        os.makedirs(save_dir, exist_ok=True)
        # 使用基础logger输出初始化信息（在日志系统初始化之前）
        logger.info(f"ModelCheckpoint initialized: save_dir={save_dir}, save_freq={save_freq}")

    async def on_round_end(self, trainer: Any, round_num: int, context: Dict[str, Any] = None):
        """每 N 轮保存一次模型"""
        if (round_num + 1) % self.save_freq == 0:
            # 获取模型
            model = getattr(trainer, "model", None)
            # 获取trainer的logger
            train_logger = getattr(trainer, "_train_logger", logger)

            if model and hasattr(model, "save"):
                path = os.path.join(self.save_dir, f"model_round_{round_num + 1}.pt")
                try:
                    model.save(path)
                    train_logger.info(f"Model saved: {path}")
                except Exception as e:
                    train_logger.error(f"Failed to save model: {e}")
            else:
                train_logger.debug(f"Trainer has no saveable model at round {round_num + 1}")


@register("federated.callback.logging")
class LoggingCallback(Callback):
    """
    训练日志 Callback

    在训练过程中打印日志信息
    """

    def __init__(
        self,
        log_epoch: bool = True,
        log_round: bool = True,
        log_fit: bool = False,
    ):
        """
        Args:
            log_epoch: 是否记录 epoch 级别日志
            log_round: 是否记录 round 级别日志
            log_fit: 是否记录 fit 开始/结束日志
        """
        self.log_epoch = log_epoch
        self.log_round = log_round
        self.log_fit = log_fit

        logger.debug(
            f"LoggingCallback initialized: log_epoch={log_epoch}, "
            f"log_round={log_round}, log_fit={log_fit}"
        )

    async def on_fit_start(self, learner: Any, config: Dict[str, Any] = None):
        """Fit 开始时打印日志"""
        if not self.log_fit:
            return

        node_id = getattr(learner, "_node_id", "unknown")
        epochs = config.get("epochs", "?") if config else "?"
        # 获取learner的logger
        train_logger = getattr(learner, "_train_logger", logger)
        train_logger.debug(f"[{node_id}] Starting fit: epochs={epochs}")

    async def on_fit_end(self, learner: Any, result: Any = None):
        """Fit 结束时打印日志"""
        if not self.log_fit:
            return

        node_id = getattr(learner, "_node_id", "unknown")
        # 获取learner的logger
        train_logger = getattr(learner, "_train_logger", logger)
        train_logger.debug(f"[{node_id}] Fit completed")

    async def on_epoch_end(self, learner: Any, epoch: int, metrics: Any = None):
        """Epoch 结束时打印日志"""
        if not self.log_epoch:
            return

        node_id = getattr(learner, "_node_id", "unknown")
        # 获取learner的logger
        train_logger = getattr(learner, "_train_logger", logger)

        if metrics:
            # 从 EpochMetrics 中提取信息
            avg_loss = getattr(metrics, "avg_loss", "N/A")
            total_samples = getattr(metrics, "total_samples", 0)
            extra_metrics = getattr(metrics, "metrics", {})

            # 构建日志消息
            msg_parts = [f"[{node_id}] Epoch {epoch}"]
            msg_parts.append(f"loss={avg_loss:.4f}" if isinstance(avg_loss, float) else f"loss={avg_loss}")
            msg_parts.append(f"samples={total_samples}")

            # 添加额外指标（如 accuracy）
            for key, value in extra_metrics.items():
                if isinstance(value, float):
                    msg_parts.append(f"{key}={value:.4f}")
                else:
                    msg_parts.append(f"{key}={value}")

            train_logger.info(", ".join(msg_parts))
        else:
            train_logger.info(f"[{node_id}] Epoch {epoch} completed")

    async def on_round_end(self, trainer: Any, round_num: int, context: Dict[str, Any] = None):
        """Round 结束时打印日志"""
        if not self.log_round:
            return

        # 获取trainer的logger
        train_logger = getattr(trainer, "_train_logger", logger)

        if context and "metrics" in context:
            metrics = context["metrics"]

            # 从 RoundMetrics 中提取信息
            if hasattr(metrics, "num_clients"):
                num_clients = metrics.num_clients
                total_samples = metrics.total_samples
                metrics_dict = metrics.metrics
            else:
                # 兼容字典格式
                num_clients = metrics.get("num_clients", 0)
                total_samples = metrics.get("total_samples", 0)
                metrics_dict = metrics.get("metrics", {})

            # 构建日志消息
            msg_parts = [f"Round {round_num}"]
            msg_parts.append(f"clients={num_clients}")
            msg_parts.append(f"samples={total_samples}")

            # 添加指标
            for key, value in metrics_dict.items():
                if isinstance(value, float):
                    msg_parts.append(f"{key}={value:.4f}")
                else:
                    msg_parts.append(f"{key}={value}")

            train_logger.debug(", ".join(msg_parts))
        else:
            train_logger.debug(f"Round {round_num} completed")


@register("federated.callback.early_stopping")
class EarlyStopping(Callback):
    """
    早停 Callback

    如果指标不再改善，提前停止训练
    """

    def __init__(
        self,
        monitor: str = "loss",
        patience: int = 5,
        min_delta: float = 0.001,
        mode: str = "min"
    ):
        """
        Args:
            monitor: 监控的指标名称
            patience: 容忍轮次（连续多少轮不改善就停止）
            min_delta: 最小改善量
            mode: 'min' 或 'max'（指标越小越好还是越大越好）
        """
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.best_value = None
        self.wait = 0
        self.stopped = False

        logger.debug(
            f"EarlyStopping initialized: monitor={monitor}, patience={patience}, "
            f"min_delta={min_delta}, mode={mode}"
        )

    async def on_round_end(self, trainer: Any, round_num: int, context: Dict[str, Any] = None):
        """检查是否需要早停"""
        if not context or self.monitor not in context:
            return

        current_value = context[self.monitor]
        # 获取trainer的logger
        train_logger = getattr(trainer, "_train_logger", logger)

        # 第一次
        if self.best_value is None:
            self.best_value = current_value
            train_logger.debug(f"EarlyStopping: initial {self.monitor}={current_value:.4f}")
            return

        # 检查是否改善
        improved = False
        if self.mode == "min":
            if current_value < self.best_value - self.min_delta:
                improved = True
        else:  # mode == "max"
            if current_value > self.best_value + self.min_delta:
                improved = True

        if improved:
            self.best_value = current_value
            self.wait = 0
            train_logger.debug(
                f"EarlyStopping: {self.monitor} improved to {current_value:.4f}, "
                f"reset wait counter"
            )
        else:
            self.wait += 1
            train_logger.debug(
                f"EarlyStopping: {self.monitor}={current_value:.4f} no improvement, "
                f"wait={self.wait}/{self.patience}"
            )

        # 触发早停
        if self.wait >= self.patience:
            train_logger.debug(
                f"Early stopping triggered at round {round_num + 1}: "
                f"{self.monitor} has not improved for {self.patience} rounds"
            )
            self.stopped = True

            # 通知 Trainer 停止（需要 Trainer 支持）
            if hasattr(trainer, "stop_training"):
                trainer.stop_training = True
                train_logger.debug("Set trainer.stop_training = True")


@register("federated.callback.mlflow")
class MLflowCallback(Callback):
    """
    MLflow Callback

    记录训练指标到 MLflow：
    - Learner: 每个 epoch 结束时记录指标
    - Learner: 评估结束时记录评估指标
    - Trainer: 每个 round 结束时记录指标
    """

    def __init__(self, log_epoch: bool = True, log_round: bool = True, log_eval: bool = True):
        """
        Args:
            log_epoch: 是否记录 epoch 级别指标（Learner）
            log_round: 是否记录 round 级别指标（Trainer）
            log_eval: 是否记录评估指标（Learner）
        """
        self.log_epoch = log_epoch
        self.log_round = log_round
        self.log_eval = log_eval

        logger.debug(
            f"MLflowCallback initialized: log_epoch={log_epoch}, log_round={log_round}, log_eval={log_eval}"
        )

    async def on_epoch_end(self, learner: Any, epoch: int, metrics: Any = None):
        """Learner Epoch 结束时记录指标到 MLflow"""
        if not self.log_epoch:
            return

        tracker = getattr(learner, "_tracker", None)
        if not tracker:
            return

        if metrics:
            # 从 EpochMetrics 中提取信息
            avg_loss = getattr(metrics, "avg_loss", None)
            total_samples = getattr(metrics, "total_samples", 0)
            extra_metrics = getattr(metrics, "metrics", {})

            # 构建指标字典
            metrics_dict = {}
            if isinstance(avg_loss, float):
                metrics_dict["loss"] = avg_loss

            # 添加额外指标（如 accuracy）
            metrics_dict.update(extra_metrics)

            # 使用全局 epoch 计数器作为 step（累积的，跨多次 fit 调用）
            global_epoch = getattr(learner, "_global_epoch_counter", epoch)
            tracker.log_metrics(metrics_dict, step=global_epoch)

    async def on_round_end(self, trainer: Any, round_num: int, context: Dict[str, Any] = None):
        """Trainer Round 结束时记录指标到 MLflow"""
        print(f"[MLflowCallback-DEBUG] on_round_end called: round={round_num}, log_round={self.log_round}")

        if not self.log_round:
            print(f"[MLflowCallback-DEBUG] Skipping: log_round is False")
            return

        tracker = getattr(trainer, "_tracker", None)
        print(f"[MLflowCallback-DEBUG] Tracker: {tracker}, type={type(tracker).__name__ if tracker else 'None'}")

        if not tracker:
            print(f"[MLflowCallback-DEBUG] No tracker found on trainer!")
            return

        print(f"[MLflowCallback-DEBUG] Checking context... context={bool(context)}, has_metrics={'metrics' in context if context else False}")

        if context and "metrics" in context:
            metrics = context["metrics"]
            print(f"[MLflowCallback-DEBUG] Metrics found: type={type(metrics).__name__}")

            # 从 RoundMetrics 中提取信息
            if hasattr(metrics, "metrics"):
                # RoundMetrics 对象
                metrics_dict = dict(metrics.metrics)
                print(f"[MLflowCallback-DEBUG] Extracted from RoundMetrics.metrics: {list(metrics_dict.keys())}")
            elif isinstance(metrics, dict):
                # 字典格式（兼容）
                metrics_dict = metrics.get("metrics", {})
                print(f"[MLflowCallback-DEBUG] Extracted from dict: {list(metrics_dict.keys())}")
            else:
                print(f"[MLflowCallback-DEBUG] Cannot extract metrics from {type(metrics)}")
                return

            print(f"[MLflowCallback-DEBUG] Calling tracker.log_metrics() with {len(metrics_dict)} metrics at step={round_num}")
            print(f"[MLflowCallback-DEBUG] Metrics to log: {metrics_dict}")

            # 记录到 tracker，使用 round_num 作为 step
            tracker.log_metrics(metrics_dict, step=round_num)

            print(f"[MLflowCallback-DEBUG] ✓ Successfully logged metrics for round {round_num}")
        else:
            print(f"[MLflowCallback-DEBUG] No metrics in context! context keys: {list(context.keys()) if context else 'None'}")

    async def on_evaluate_end(self, learner: Any, result: Any = None, context: Dict[str, Any] = None):
        """Learner 评估结束时记录评估指标到 MLflow"""
        if not self.log_eval:
            return

        tracker = getattr(learner, "_tracker", None)
        if not tracker:
            return

        if result and hasattr(result, "metrics"):
            # 从 EvalResult 中提取评估指标
            eval_metrics = result.metrics

            # 添加评估类型前缀（如果有上下文信息）
            if context:
                eval_type = context.get("type", "eval")
                # 如果是 post_training 评估，使用 eval_ 前缀
                # 否则保持原有的 post_train_ 前缀（这个会在 learner.py 中添加）
                if eval_type == "post_training":
                    metrics_to_log = {f"eval_{k}": v for k, v in eval_metrics.items()}
                else:
                    metrics_to_log = eval_metrics
            else:
                metrics_to_log = {f"eval_{k}": v for k, v in eval_metrics.items()}

            # 使用全局 epoch 计数器作为 step（累积的，跨多次 fit 调用）
            global_epoch = getattr(learner, "_global_epoch_counter", None)
            tracker.log_metrics(metrics_to_log, step=global_epoch)
