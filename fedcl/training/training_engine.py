# fedcl/training/training_engine.py
"""
TrainingEngine - 训练引擎实现

负责执行具体的训练循环、批次处理、验证和训练环境管理。
支持GPU/CPU自动切换、内存监控、异常恢复等高级功能。
"""

import time
import gc
import psutil
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union, List
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from loguru import logger

from ..core.base_learner import BaseLearner
from ..core.hook_executor import HookExecutor
from ..core.execution_context import ExecutionContext
from ..core.hook import HookPhase
from ..data.results import TaskResults
from ..data.dataloader import DataLoader as FedCLDataLoader


class TrainingEngineError(Exception):
    """TrainingEngine相关异常基类"""
    pass


class TrainingStateError(TrainingEngineError):
    """训练状态相关异常"""
    pass


class ResourceError(TrainingEngineError):
    """资源相关异常"""
    pass


class ValidationError(TrainingEngineError):
    """验证相关异常"""
    pass


class TrainingEngine:
    """
    训练引擎
    
    负责执行具体的训练循环、批次处理、验证和训练环境管理。
    支持GPU/CPU自动切换、内存监控、异常恢复、训练控制等功能。
    
    Attributes:
        hook_executor: 钩子执行器
        context: 执行上下文
        checkpoint_manager: 检查点管理器（可选）
        metrics_logger: 度量记录器（可选）
        training_config: 训练配置参数
        device: 计算设备
        _is_paused: 是否暂停训练
        _should_stop: 是否应该停止训练
        _training_stats: 训练统计信息
    """
    
    def __init__(self, 
                 hook_executor: HookExecutor, 
                 context: ExecutionContext,
                 checkpoint_manager: Optional[Any] = None,
                 metrics_logger: Optional[Any] = None) -> None:
        """
        初始化训练引擎
        
        Args:
            hook_executor: 钩子执行器
            context: 执行上下文
            checkpoint_manager: 检查点管理器（可选）
            metrics_logger: 度量记录器（可选）
            
        Raises:
            TrainingEngineError: 参数无效时抛出
        """
        if not isinstance(hook_executor, HookExecutor):
            raise TrainingEngineError("Invalid hook executor provided")
        if not isinstance(context, ExecutionContext):
            raise TrainingEngineError("Invalid execution context provided")
            
        self.hook_executor = hook_executor
        self.context = context
        self.checkpoint_manager = checkpoint_manager
        self.metrics_logger = metrics_logger
        
        # 从配置获取训练参数
        self.training_config = context.get_config("training", {})
        self.num_epochs = self.training_config.get("num_epochs", 10)
        self.early_stopping = self.training_config.get("early_stopping", {})
        self.optimization_config = self.training_config.get("optimization", {})
        self.gradient_config = self.training_config.get("gradient", {})
        self.validation_config = self.training_config.get("validation", {})
        self.checkpointing_config = self.training_config.get("checkpointing", {})
        
        # 设备配置
        self.device = self._setup_device()
        
        # 训练状态
        self._is_paused = False
        self._should_stop = False
        self._current_epoch = 0
        self._current_batch = 0
        
        # 训练统计
        self._training_stats = {
            "total_training_time": 0.0,
            "total_batches_processed": 0,
            "total_samples_processed": 0,
            "average_batch_time": 0.0,
            "peak_memory_usage": 0.0,
            "device_utilization": 0.0,
            "convergence_history": [],
            "error_count": 0,
            "recovery_count": 0
        }
        
        # 早停相关
        self._best_metric = None
        self._patience_counter = 0
        self._best_model_state = None
        
        logger.info(f"TrainingEngine initialized with device: {self.device}")
        logger.info(f"Training config: {self.training_config}")
    
    def train_task(self, task_id: int, task_data: Union[DataLoader, FedCLDataLoader], 
                   learner: BaseLearner) -> TaskResults:
        """
        训练单个任务
        
        Args:
            task_id: 任务ID
            task_data: 任务数据加载器
            learner: 学习器实例
            
        Returns:
            TaskResults: 任务训练结果
            
        Raises:
            TrainingEngineError: 训练过程中出现错误时抛出
        """
        logger.info(f"Starting training for task {task_id}")
        
        # 重置训练状态
        self._reset_training_state()
        
        # 执行任务前钩子
        self.hook_executor.execute_hooks(
            HookPhase.BEFORE_TASK, 
            self.context, 
            task_id=task_id, 
            task_data=task_data,
            learner=learner
        )
        
        try:
            # 设置训练环境
            self.setup_training_environment(learner)
            
            # 记录开始时间
            start_time = time.time()
            
            # 执行训练循环
            training_metrics = self.execute_training_loop(learner, task_data, self.num_epochs)
            
            # 计算训练时间
            training_time = time.time() - start_time
            training_metrics["training_time"] = training_time
            
            # 获取内存使用情况
            memory_usage = self._get_memory_usage()
            training_metrics["memory_usage"] = memory_usage
            
            # 获取模型大小
            model_size = self._get_model_size(learner)
            training_metrics["model_size"] = model_size
            
            # 获取收敛步数
            convergence_step = training_metrics.get("convergence_step", -1)
            
            # 创建任务结果
            task_results = TaskResults(
                task_id=task_id,
                metrics=training_metrics,
                training_time=training_time,
                memory_usage=memory_usage,
                model_size=model_size,
                convergence_step=convergence_step
            )
            
            # 执行任务后钩子
            self.hook_executor.execute_hooks(
                HookPhase.AFTER_TASK,
                self.context,
                task_id=task_id,
                results=task_results,
                learner=learner
            )
            
            logger.info(f"Task {task_id} training completed in {training_time:.2f}s")
            return task_results
            
        except Exception as e:
            logger.error(f"Training failed for task {task_id}: {str(e)}")
            self._training_stats["error_count"] += 1
            
            # 执行错误处理钩子
            self.hook_executor.execute_hooks(
                HookPhase.ON_ERROR,
                self.context,
                error=e,
                task_id=task_id,
                learner=learner
            )
            
            # 尝试错误恢复
            if self.handle_training_error(e, {"task_id": task_id, "learner": learner}):
                logger.info("Training error handled, retrying...")
                return self.train_task(task_id, task_data, learner)
            else:
                raise TrainingEngineError(f"Failed to train task {task_id}: {str(e)}") from e
                
        finally:
            # 清理训练环境
            self.cleanup_training_environment()
    
    def execute_training_loop(self, learner: BaseLearner, data_loader: Union[DataLoader, FedCLDataLoader], 
                             num_epochs: int) -> Dict[str, float]:
        """
        执行训练循环
        
        Args:
            learner: 学习器实例
            data_loader: 数据加载器
            num_epochs: 训练轮数
            
        Returns:
            Dict[str, float]: 训练度量
            
        Raises:
            TrainingStateError: 训练状态异常时抛出
        """
        logger.info(f"Starting training loop for {num_epochs} epochs")
        
        metrics = defaultdict(list)
        best_loss = float('inf')
        
        for epoch in range(num_epochs):
            if self._should_stop:
                logger.info("Training stopped by user request")
                break
                
            self._current_epoch = epoch
            
            # 等待暂停恢复
            while self._is_paused:
                time.sleep(0.1)
                
            # 执行轮次前钩子
            self.hook_executor.execute_hooks(
                HookPhase.BEFORE_EPOCH,
                self.context,
                epoch=epoch,
                learner=learner
            )
            
            try:
                # 训练一个轮次
                epoch_start_time = time.time()
                epoch_metrics = self._train_epoch(learner, data_loader, epoch)
                epoch_time = time.time() - epoch_start_time
                
                # 记录轮次度量
                epoch_metrics["epoch_time"] = epoch_time
                for metric_name, metric_value in epoch_metrics.items():
                    metrics[metric_name].append(metric_value)
                    self.context.log_metric(f"epoch_{metric_name}", metric_value, epoch)
                
                # 记录到度量记录器
                if self.metrics_logger:
                    for metric_name, metric_value in epoch_metrics.items():
                        self.metrics_logger.log_metric(f"train_{metric_name}", metric_value, epoch)
                
                # 执行轮次后钩子
                self.hook_executor.execute_hooks(
                    HookPhase.AFTER_EPOCH,
                    self.context,
                    epoch=epoch,
                    metrics=epoch_metrics,
                    learner=learner
                )
                
                # 保存最佳模型
                current_loss = epoch_metrics.get("loss", float('inf'))
                if current_loss < best_loss:
                    best_loss = current_loss
                    self._best_model_state = learner.get_model_state()
                    self._best_metric = current_loss
                    self._patience_counter = 0
                else:
                    self._patience_counter += 1
                
                # 检查点保存
                if self._should_save_checkpoint(epoch):
                    self._save_checkpoint(learner, epoch, epoch_metrics)
                
                # 检查早停条件
                if self._should_early_stop(epoch_metrics):
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
                    
                logger.debug(f"Epoch {epoch} completed: {epoch_metrics}")
                
            except Exception as e:
                logger.error(f"Error in epoch {epoch}: {str(e)}")
                if not self.handle_training_error(e, {"epoch": epoch, "learner": learner}):
                    raise
        
        # 计算最终度量
        final_metrics = {}
        for metric_name, values in metrics.items():
            if values:
                final_metrics[f"avg_{metric_name}"] = sum(values) / len(values)
                final_metrics[f"final_{metric_name}"] = values[-1]
                final_metrics[f"best_{metric_name}"] = min(values) if "loss" in metric_name else max(values)
        
        # 添加收敛信息
        if self._best_metric is not None:
            final_metrics["convergence_step"] = self._find_convergence_step(metrics.get("loss", []))
        
        logger.info(f"Training loop completed with metrics: {final_metrics}")
        return final_metrics
    
    def handle_batch(self, batch_data: Tuple[torch.Tensor, torch.Tensor], 
                    learner: BaseLearner) -> Dict[str, float]:
        """
        处理单个批次
        
        Args:
            batch_data: 批次数据(inputs, targets)
            learner: 学习器实例
            
        Returns:
            Dict[str, float]: 批次训练度量
            
        Raises:
            TrainingStateError: 批次处理异常时抛出
        """
        try:
            inputs, targets = batch_data
            
            # 移动到设备
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # 执行批次前钩子
            self.hook_executor.execute_hooks(
                HookPhase.BEFORE_BATCH,
                self.context,
                batch_data=(inputs, targets),
                learner=learner
            )
            
            # 记录批次开始时间
            batch_start_time = time.time()
            
            # 前向传播
            outputs = learner.forward(inputs)
            
            # 计算损失
            loss = learner.compute_loss(outputs, targets)
            
            # 反向传播
            learner.backward(loss)
            
            # 梯度裁剪
            if self.gradient_config.get("clip_norm"):
                torch.nn.utils.clip_grad_norm_(
                    learner.get_model().parameters(), 
                    self.gradient_config["clip_norm"]
                )
            
            # 优化器步骤
            learner.optimizer_step()
            
            # 记录批次时间
            batch_time = time.time() - batch_start_time
            
            # 计算批次度量
            batch_metrics = {
                "loss": loss.item(),
                "batch_time": batch_time,
                "batch_size": inputs.size(0),
                "learning_rate": learner.get_learning_rate()
            }
            
            # 计算准确率（如果适用）
            if hasattr(learner, 'compute_accuracy'):
                accuracy = learner.compute_accuracy(outputs, targets)
                batch_metrics["accuracy"] = accuracy
            
            # 执行批次后钩子
            self.hook_executor.execute_hooks(
                HookPhase.AFTER_BATCH,
                self.context,
                batch_metrics=batch_metrics,
                learner=learner
            )
            
            # 更新统计
            self._update_batch_stats(batch_metrics)
            
            return batch_metrics
            
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            raise TrainingStateError(f"Batch processing failed: {str(e)}") from e
    
    def validate_model(self, learner: BaseLearner, validation_data: Union[DataLoader, FedCLDataLoader]) -> Dict[str, float]:
        """
        验证模型
        
        Args:
            learner: 学习器实例
            validation_data: 验证数据加载器
            
        Returns:
            Dict[str, float]: 验证度量
            
        Raises:
            ValidationError: 验证过程异常时抛出
        """
        logger.debug("Starting model validation")
        
        try:
            # 设置评估模式
            learner.set_eval_mode()
            
            validation_metrics = defaultdict(list)
            total_samples = 0
            
            with torch.no_grad():
                for batch_data in validation_data:
                    inputs, targets = batch_data
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    # 前向传播
                    outputs = learner.forward(inputs)
                    
                    # 计算损失
                    loss = learner.compute_loss(outputs, targets)
                    validation_metrics["loss"].append(loss.item())
                    
                    # 计算准确率
                    if hasattr(learner, 'compute_accuracy'):
                        accuracy = learner.compute_accuracy(outputs, targets)
                        validation_metrics["accuracy"].append(accuracy)
                    
                    total_samples += inputs.size(0)
            
            # 计算平均度量
            final_metrics = {}
            for metric_name, values in validation_metrics.items():
                if values:
                    final_metrics[f"val_{metric_name}"] = sum(values) / len(values)
            
            final_metrics["val_samples"] = total_samples
            
            # 恢复训练模式
            learner.set_train_mode()
            
            # 执行评估钩子
            self.hook_executor.execute_hooks(
                HookPhase.ON_EVALUATION,
                self.context,
                validation_metrics=final_metrics,
                learner=learner
            )
            
            logger.debug(f"Validation completed: {final_metrics}")
            return final_metrics
            
        except Exception as e:
            # 确保恢复训练模式
            learner.set_train_mode()
            logger.error(f"Validation failed: {str(e)}")
            raise ValidationError(f"Model validation failed: {str(e)}") from e
    
    def setup_training_environment(self, learner: BaseLearner) -> None:
        """
        设置训练环境
        
        Args:
            learner: 学习器实例
        """
        logger.debug("Setting up training environment")
        
        try:
            # 设置设备
            learner.to(self.device)
            
            # 设置训练模式
            learner.set_train_mode()
            
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 设置随机种子
            seed = self.context.get_config("seed", 42)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            
            # 配置优化器
            if hasattr(learner, 'configure_optimizer'):
                learner.configure_optimizer(self.optimization_config)
            
            logger.debug("Training environment setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup training environment: {str(e)}")
            raise TrainingEngineError(f"Environment setup failed: {str(e)}") from e
    
    def cleanup_training_environment(self) -> None:
        """清理训练环境"""
        logger.debug("Cleaning up training environment")
        
        try:
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 垃圾回收
            gc.collect()
            
            # 重置训练状态
            self._is_paused = False
            self._should_stop = False
            
            logger.debug("Training environment cleanup completed")
            
        except Exception as e:
            logger.warning(f"Failed to cleanup training environment: {str(e)}")
    
    def handle_training_error(self, error: Exception, context: Dict[str, Any]) -> bool:
        """
        处理训练错误
        
        Args:
            error: 发生的异常
            context: 错误上下文信息
            
        Returns:
            bool: 是否成功处理错误（True表示可以重试）
        """
        logger.warning(f"Handling training error: {str(error)}")
        
        self._training_stats["error_count"] += 1
        
        # 内存不足错误
        if "out of memory" in str(error).lower():
            logger.info("Attempting to recover from OOM error")
            
            # 清理缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            # 减少批次大小
            if "learner" in context:
                learner = context["learner"]
                if hasattr(learner, 'reduce_batch_size'):
                    learner.reduce_batch_size()
                    self._training_stats["recovery_count"] += 1
                    return True
        
        # 模型状态错误
        elif isinstance(error, (RuntimeError, AttributeError)):
            logger.info("Attempting to recover from model state error")
            
            # 重置模型状态
            if "learner" in context and self._best_model_state is not None:
                learner = context["learner"]
                learner.load_model_state(self._best_model_state)
                self._training_stats["recovery_count"] += 1
                return True
        
        # 数据加载错误
        elif "DataLoader" in str(error):
            logger.info("Data loading error detected, retrying with smaller batch")
            # 可以在这里实现数据加载重试逻辑
            return False
        
        logger.error(f"Unrecoverable error: {str(error)}")
        return False
    
    def get_training_stats(self) -> Dict[str, Any]:
        """
        获取训练统计
        
        Returns:
            Dict[str, Any]: 训练统计信息
        """
        stats = self._training_stats.copy()
        stats.update({
            "current_epoch": self._current_epoch,
            "current_batch": self._current_batch,
            "is_paused": self._is_paused,
            "should_stop": self._should_stop,
            "device": str(self.device),
            "best_metric": self._best_metric,
            "patience_counter": self._patience_counter
        })
        return stats
    
    def pause_training(self) -> None:
        """暂停训练"""
        logger.info("Pausing training")
        self._is_paused = True
    
    def resume_training(self) -> None:
        """恢复训练"""
        logger.info("Resuming training")
        self._is_paused = False
    
    def stop_training(self) -> None:
        """停止训练"""
        logger.info("Stopping training")
        self._should_stop = True
        self._is_paused = False
    
    def _setup_device(self) -> torch.device:
        """设置计算设备"""
        device_config = self.context.get_config("device", "auto")
        
        if device_config == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
            else:
                device = torch.device("cpu")
                logger.info("Using CPU device")
        else:
            device = torch.device(device_config)
            logger.info(f"Using specified device: {device}")
        
        return device
    
    def _reset_training_state(self) -> None:
        """重置训练状态"""
        self._current_epoch = 0
        self._current_batch = 0
        self._should_stop = False
        self._is_paused = False
        self._best_metric = None
        self._patience_counter = 0
        self._best_model_state = None
    
    def _train_epoch(self, learner: BaseLearner, data_loader: Union[DataLoader, FedCLDataLoader], 
                     epoch: int) -> Dict[str, float]:
        """训练一个轮次"""
        epoch_metrics = defaultdict(list)
        
        for batch_idx, batch_data in enumerate(data_loader):
            self._current_batch = batch_idx
            
            # 检查停止条件
            if self._should_stop:
                break
            
            # 等待暂停恢复
            while self._is_paused:
                time.sleep(0.1)
            
            # 处理批次
            batch_metrics = self.handle_batch(batch_data, learner)
            
            # 记录批次度量
            for metric_name, metric_value in batch_metrics.items():
                epoch_metrics[metric_name].append(metric_value)
        
        # 计算轮次平均度量
        final_metrics = {}
        for metric_name, values in epoch_metrics.items():
            if values:
                final_metrics[metric_name] = sum(values) / len(values)
        
        return final_metrics
    
    def _should_early_stop(self, epoch_metrics: Dict[str, float]) -> bool:
        """检查是否应该早停"""
        if not self.early_stopping.get("enable", False):
            return False
        
        patience = self.early_stopping.get("patience", 5)
        min_delta = self.early_stopping.get("min_delta", 0.001)
        
        return self._patience_counter >= patience
    
    def _should_save_checkpoint(self, epoch: int) -> bool:
        """检查是否应该保存检查点"""
        if not self.checkpoint_manager:
            return False
        
        save_interval = self.checkpointing_config.get("save_interval", 5)
        save_best = self.checkpointing_config.get("save_best", True)
        
        return (epoch + 1) % save_interval == 0 or (save_best and self._patience_counter == 0)
    
    def _save_checkpoint(self, learner: BaseLearner, epoch: int, metrics: Dict[str, float]) -> None:
        """保存检查点"""
        if self.checkpoint_manager:
            try:
                checkpoint_data = {
                    "epoch": epoch,
                    "model_state": learner.get_model_state(),
                    "optimizer_state": learner.get_optimizer_state(),
                    "metrics": metrics,
                    "training_stats": self._training_stats
                }
                
                self.checkpoint_manager.save_checkpoint(checkpoint_data)
                
                # 执行检查点钩子
                self.hook_executor.execute_hooks(
                    HookPhase.ON_CHECKPOINT,
                    self.context,
                    epoch=epoch,
                    checkpoint_data=checkpoint_data
                )
                
            except Exception as e:
                logger.warning(f"Failed to save checkpoint: {str(e)}")
    
    def _get_memory_usage(self) -> float:
        """获取内存使用情况（MB）"""
        try:
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / 1024 / 1024
            else:
                process = psutil.Process()
                return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0
    
    def _get_model_size(self, learner: BaseLearner) -> float:
        """获取模型大小（MB）"""
        try:
            model = learner.get_model()
            param_size = sum(p.numel() for p in model.parameters()) * 4  # 4 bytes per float32
            return param_size / 1024 / 1024
        except Exception:
            return 0.0
    
    def _find_convergence_step(self, loss_history: List[float]) -> int:
        """找到收敛步数"""
        if len(loss_history) < 2:
            return -1
        
        # 简单的收敛检测：连续5个步骤改善小于阈值
        convergence_threshold = 0.001
        stable_steps = 5
        
        if len(loss_history) < stable_steps:
            return -1
        
        for i in range(stable_steps, len(loss_history)):
            recent_losses = loss_history[i-stable_steps:i]
            if all(abs(recent_losses[j] - recent_losses[j+1]) < convergence_threshold 
                   for j in range(len(recent_losses)-1)):
                return i
        
        return -1
    
    def _update_batch_stats(self, batch_metrics: Dict[str, float]) -> None:
        """更新批次统计"""
        self._training_stats["total_batches_processed"] += 1
        self._training_stats["total_samples_processed"] += batch_metrics.get("batch_size", 0)
        
        # 更新平均批次时间
        batch_time = batch_metrics.get("batch_time", 0.0)
        total_batches = self._training_stats["total_batches_processed"]
        current_avg = self._training_stats["average_batch_time"]
        self._training_stats["average_batch_time"] = (current_avg * (total_batches - 1) + batch_time) / total_batches
        
        # 更新峰值内存使用
        current_memory = self._get_memory_usage()
        if current_memory > self._training_stats["peak_memory_usage"]:
            self._training_stats["peak_memory_usage"] = current_memory
