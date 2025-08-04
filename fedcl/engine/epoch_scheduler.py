# fedcl/engine/epoch_scheduler.py
"""
EpochScheduler系统 - 负责epoch级别的训练调度

包含：
- EpochScheduler基类
- 具体的scheduler实现
- ExecutionResult数据结构
- SchedulerManager管理器
"""

import logging
import time
import threading
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, Future

from ..core.execution_context import ExecutionContext
from ..core.hook import HookPhase
from .exceptions import SchedulerError, ExecutionError


class ExecutionMode(Enum):
    """执行模式枚举"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HYBRID = "hybrid"


class SchedulerPriority(Enum):
    """调度器优先级"""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


@dataclass
class ExecutionResult:
    """Epoch调度执行结果"""
    scheduler_id: str
    learner_id: str
    executed_epochs: List[int]
    metrics: Dict[str, List[float]]  # 每个epoch的指标列表
    final_state: Dict[str, Any]
    exported_knowledge: Optional[Dict[str, Any]] = None
    execution_time: float = 0.0
    memory_usage: Dict[str, float] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None
    
    def get_final_metrics(self) -> Dict[str, float]:
        """获取最终指标（最后一个epoch的指标）"""
        if not self.metrics:
            return {}
        
        final_metrics = {}
        for key, values in self.metrics.items():
            if values:
                final_metrics[key] = values[-1]
        return final_metrics
    
    def get_average_metrics(self) -> Dict[str, float]:
        """获取平均指标"""
        if not self.metrics:
            return {}
        
        avg_metrics = {}
        for key, values in self.metrics.items():
            if values:
                avg_metrics[f"avg_{key}"] = sum(values) / len(values)
        return avg_metrics


class BaseEpochScheduler(ABC):
    """
    EpochScheduler基类
    
    负责管理特定learner的epoch级训练调度逻辑
    """
    
    def __init__(self, 
                 scheduler_id: str,
                 config: Optional[Dict[str, Any]] = None,
                 priority: SchedulerPriority = SchedulerPriority.NORMAL):
        """
        初始化epoch调度器
        
        Args:
            scheduler_id: 调度器唯一标识
            config: 调度器配置
            priority: 调度器优先级
        """
        self.scheduler_id = scheduler_id
        self.config = config or {}
        self.priority = priority
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 执行状态
        self._is_running = False
        self._current_epoch = 0
        self._execution_lock = threading.Lock()
        
        # 统计信息
        self._execution_count = 0
        self._total_execution_time = 0.0
        self._last_execution_time = 0.0
        
        self.logger.debug(f"EpochScheduler '{scheduler_id}' initialized")
    
    @abstractmethod
    def execute_epochs(self,
                      learner: Any,
                      dataloader: Any,
                      epoch_range: List[int],
                      inherited_state: Optional[Dict[str, Any]] = None,
                      context: Optional[ExecutionContext] = None) -> ExecutionResult:
        """
        执行epoch范围内的训练
        
        Args:
            learner: 学习器实例
            dataloader: 数据加载器
            epoch_range: 要执行的epoch列表
            inherited_state: 继承的状态
            context: 执行上下文
            
        Returns:
            ExecutionResult: 执行结果
        """
        pass
    
    @abstractmethod
    def get_scheduler_type(self) -> str:
        """获取调度器类型"""
        pass
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        验证配置的有效性
        
        Args:
            config: 配置字典
            
        Returns:
            bool: 配置是否有效
        """
        # 默认实现，子类可以重写
        return True
    
    def prepare_execution(self, 
                         learner: Any, 
                         dataloader: Any, 
                         inherited_state: Optional[Dict[str, Any]] = None) -> None:
        """
        准备执行环境
        
        Args:
            learner: 学习器实例
            dataloader: 数据加载器
            inherited_state: 继承状态
        """
        # 应用继承状态
        if inherited_state and hasattr(learner, 'set_state'):
            try:
                learner.set_state(inherited_state)
                self.logger.debug(f"Applied inherited state to learner")
            except Exception as e:
                self.logger.warning(f"Failed to apply inherited state: {e}")
    
    def finalize_execution(self, learner: Any) -> Dict[str, Any]:
        """
        完成执行后的清理工作
        
        Args:
            learner: 学习器实例
            
        Returns:
            Dict[str, Any]: 导出的状态
        """
        exported_state = {}
        if hasattr(learner, 'get_state'):
            try:
                exported_state = learner.get_state()
                self.logger.debug(f"Exported learner state")
            except Exception as e:
                self.logger.warning(f"Failed to export learner state: {e}")
        
        return exported_state
    
    def handle_epoch_error(self, epoch: int, error: Exception) -> bool:
        """
        处理epoch执行错误
        
        Args:
            epoch: 出错的epoch
            error: 错误信息
            
        Returns:
            bool: 是否继续执行后续epoch
        """
        self.logger.error(f"Epoch {epoch} failed: {error}")
        
        # 默认策略：记录错误但继续执行
        error_tolerance = self.config.get("error_tolerance", "continue")
        
        if error_tolerance == "stop":
            return False
        elif error_tolerance == "retry":
            max_retries = self.config.get("max_retries", 3)
            # 这里可以实现重试逻辑
            return True
        else:  # continue
            return True
    
    def export_state(self) -> Dict[str, Any]:
        """导出调度器状态"""
        return {
            "scheduler_id": self.scheduler_id,
            "scheduler_type": self.get_scheduler_type(),
            "priority": self.priority.value,
            "execution_count": self._execution_count,
            "total_execution_time": self._total_execution_time,
            "last_execution_time": self._last_execution_time,
            "is_running": self._is_running,
            "current_epoch": self._current_epoch
        }
    
    def import_state(self, state: Dict[str, Any]) -> None:
        """导入调度器状态"""
        self._execution_count = state.get("execution_count", 0)
        self._total_execution_time = state.get("total_execution_time", 0.0)
        self._last_execution_time = state.get("last_execution_time", 0.0)
    
    @property
    def is_running(self) -> bool:
        """检查是否正在运行"""
        return self._is_running
    
    @property
    def current_epoch(self) -> int:
        """获取当前epoch"""
        return self._current_epoch
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """获取执行统计信息"""
        return {
            "execution_count": self._execution_count,
            "total_execution_time": self._total_execution_time,
            "average_execution_time": self._total_execution_time / max(1, self._execution_count),
            "last_execution_time": self._last_execution_time
        }


class StandardEpochScheduler(BaseEpochScheduler):
    """
    标准Epoch调度器
    
    适用于常规的连续学习场景
    """
    
    def get_scheduler_type(self) -> str:
        return "StandardEpochScheduler"
    
    def execute_epochs(self,
                      learner: Any,
                      dataloader: Any,
                      epoch_range: List[int],
                      inherited_state: Optional[Dict[str, Any]] = None,
                      context: Optional[ExecutionContext] = None) -> ExecutionResult:
        """执行标准epoch训练"""
        with self._execution_lock:
            if self._is_running:
                raise SchedulerError(f"Scheduler {self.scheduler_id} is already running")
            
            self._is_running = True
            start_time = time.time()
            
            try:
                self.logger.debug(f"Starting standard epoch execution for epochs {epoch_range}")
                
                # 准备执行环境
                self.prepare_execution(learner, dataloader, inherited_state)
                
                # 执行所有epoch
                executed_epochs = []
                metrics_history = {}
                
                for epoch in epoch_range:
                    self._current_epoch = epoch
                    
                    try:
                        # 执行epoch前Hook
                        if context:
                            self._execute_hooks(context, HookPhase.BEFORE_EPOCH.value,
                                              epoch=epoch, learner=learner, scheduler_id=self.scheduler_id)
                        
                        # 执行单个epoch训练
                        epoch_metrics = self._execute_single_epoch(learner, dataloader, epoch, context)
                        
                        # 记录指标
                        for key, value in epoch_metrics.items():
                            if key not in metrics_history:
                                metrics_history[key] = []
                            metrics_history[key].append(value)
                        
                        executed_epochs.append(epoch)
                        
                        # 执行epoch后Hook
                        if context:
                            self._execute_hooks(context, HookPhase.AFTER_EPOCH.value,
                                              epoch=epoch, metrics=epoch_metrics, learner=learner,
                                              scheduler_id=self.scheduler_id)
                        
                        self.logger.debug(f"Completed epoch {epoch}: {epoch_metrics}")
                        
                    except Exception as e:
                        should_continue = self.handle_epoch_error(epoch, e)
                        if not should_continue:
                            break
                
                # 完成执行
                final_state = self.finalize_execution(learner)
                execution_time = time.time() - start_time
                
                # 更新统计信息
                self._execution_count += 1
                self._total_execution_time += execution_time
                self._last_execution_time = execution_time
                
                result = ExecutionResult(
                    scheduler_id=self.scheduler_id,
                    learner_id=getattr(learner, 'learner_id', 'unknown'),
                    executed_epochs=executed_epochs,
                    metrics=metrics_history,
                    final_state=final_state,
                    execution_time=execution_time,
                    memory_usage=self._get_memory_usage(),
                    success=True
                )
                
                self.logger.debug(f"Standard epoch execution completed: {len(executed_epochs)}/{len(epoch_range)} epochs")
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                self.logger.error(f"Standard epoch execution failed: {e}")
                
                return ExecutionResult(
                    scheduler_id=self.scheduler_id,
                    learner_id=getattr(learner, 'learner_id', 'unknown'),
                    executed_epochs=[],
                    metrics={},
                    final_state={},
                    execution_time=execution_time,
                    success=False,
                    error_message=str(e)
                )
            
            finally:
                self._is_running = False
                self._current_epoch = 0
    
    def _execute_single_epoch(self, 
                             learner: Any, 
                             dataloader: Any, 
                             epoch: int,
                             context: Optional[ExecutionContext] = None) -> Dict[str, float]:
        """执行单个epoch训练"""
        if hasattr(learner, 'train_epoch'):
            return learner.train_epoch(dataloader, epoch)
        elif hasattr(learner, 'train_on_batch'):
            return self._execute_batch_training(learner, dataloader, epoch, context)
        else:
            # 模拟训练
            self.logger.warning("Learner doesn't have train_epoch or train_on_batch method")
            return {
                "loss": max(0.1, 1.0 / (epoch + 1)),
                "accuracy": min(0.9, 0.1 + epoch * 0.1)
            }
    
    def _execute_batch_training(self, 
                               learner: Any, 
                               dataloader: Any, 
                               epoch: int,
                               context: Optional[ExecutionContext] = None) -> Dict[str, float]:
        """执行批次级训练"""
        epoch_metrics = {"loss": 0.0, "accuracy": 0.0}
        batch_count = 0
        
        for batch_idx, batch_data in enumerate(dataloader):
            # 执行批次前Hook
            if context:
                self._execute_hooks(context, HookPhase.BEFORE_BATCH.value,
                                  epoch=epoch, batch_idx=batch_idx, batch_data=batch_data,
                                  learner=learner, scheduler_id=self.scheduler_id)
            
            # 训练批次
            batch_metrics = learner.train_on_batch(batch_data)
            
            # 累积指标
            for key, value in batch_metrics.items():
                if key in epoch_metrics:
                    epoch_metrics[key] += value
                else:
                    epoch_metrics[key] = value
            
            batch_count += 1
            
            # 执行批次后Hook
            if context:
                self._execute_hooks(context, HookPhase.AFTER_BATCH.value,
                                  epoch=epoch, batch_idx=batch_idx, batch_metrics=batch_metrics,
                                  learner=learner, scheduler_id=self.scheduler_id)
        
        # 计算平均指标
        if batch_count > 0:
            epoch_metrics = {k: v / batch_count for k, v in epoch_metrics.items()}
        
        return epoch_metrics
    
    def _execute_hooks(self, context: ExecutionContext, phase: str, **kwargs):
        """执行Hook（如果有上下文）"""
        if context and hasattr(context, 'hook_manager'):
            try:
                context.hook_manager.execute_hooks(phase, context, **kwargs)
            except Exception as e:
                self.logger.error(f"Hook execution failed for phase {phase}: {e}")
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """获取内存使用情况"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                "rss": memory_info.rss / 1024 / 1024,  # MB
                "vms": memory_info.vms / 1024 / 1024   # MB
            }
        except ImportError:
            return {}
        except Exception:
            return {}


class AdaptiveEpochScheduler(BaseEpochScheduler):
    """
    自适应Epoch调度器
    
    根据训练表现动态调整训练策略
    """
    
    def get_scheduler_type(self) -> str:
        return "AdaptiveEpochScheduler"
    
    def execute_epochs(self,
                      learner: Any,
                      dataloader: Any,
                      epoch_range: List[int],
                      inherited_state: Optional[Dict[str, Any]] = None,
                      context: Optional[ExecutionContext] = None) -> ExecutionResult:
        """执行自适应epoch训练"""
        with self._execution_lock:
            if self._is_running:
                raise SchedulerError(f"Scheduler {self.scheduler_id} is already running")
            
            self._is_running = True
            start_time = time.time()
            
            try:
                self.logger.info(f"Starting adaptive epoch execution for epochs {epoch_range}")
                
                # 准备执行环境
                self.prepare_execution(learner, dataloader, inherited_state)
                
                # 自适应训练配置
                patience = self.config.get("early_stopping_patience", 5)
                min_delta = self.config.get("min_improvement_delta", 0.001)
                best_metric_value = float('-inf')
                patience_counter = 0
                
                executed_epochs = []
                metrics_history = {}
                
                for epoch in epoch_range:
                    self._current_epoch = epoch
                    
                    try:
                        # 执行单个epoch
                        epoch_metrics = self._execute_single_epoch(learner, dataloader, epoch, context)
                        
                        # 记录指标
                        for key, value in epoch_metrics.items():
                            if key not in metrics_history:
                                metrics_history[key] = []
                            metrics_history[key].append(value)
                        
                        executed_epochs.append(epoch)
                        
                        # 自适应逻辑：早停检查
                        monitor_metric = self.config.get("monitor_metric", "loss")
                        if monitor_metric in epoch_metrics:
                            current_value = epoch_metrics[monitor_metric]
                            
                            # 对于loss，值越小越好；对于accuracy，值越大越好
                            if monitor_metric.lower() in ["loss", "error"]:
                                improvement = best_metric_value - current_value
                                if current_value < best_metric_value:
                                    best_metric_value = current_value
                            else:
                                improvement = current_value - best_metric_value
                                if current_value > best_metric_value:
                                    best_metric_value = current_value
                            
                            if improvement > min_delta:
                                patience_counter = 0
                                self.logger.debug(f"Improvement detected: {improvement:.6f}")
                            else:
                                patience_counter += 1
                                self.logger.debug(f"No improvement for {patience_counter} epochs")
                            
                            # 早停检查
                            if patience_counter >= patience:
                                self.logger.info(f"Early stopping triggered after {patience} epochs without improvement")
                                break
                        
                        # 动态调整学习率（如果learner支持）
                        if hasattr(learner, 'adjust_learning_rate'):
                            learner.adjust_learning_rate(epoch_metrics)
                        
                    except Exception as e:
                        should_continue = self.handle_epoch_error(epoch, e)
                        if not should_continue:
                            break
                
                # 完成执行
                final_state = self.finalize_execution(learner)
                execution_time = time.time() - start_time
                
                result = ExecutionResult(
                    scheduler_id=self.scheduler_id,
                    learner_id=getattr(learner, 'learner_id', 'unknown'),
                    executed_epochs=executed_epochs,
                    metrics=metrics_history,
                    final_state=final_state,
                    exported_knowledge={"best_metric_value": best_metric_value},
                    execution_time=execution_time,
                    memory_usage=self._get_memory_usage(),
                    success=True
                )
                
                self.logger.debug(f"Adaptive epoch execution completed: {len(executed_epochs)}/{len(epoch_range)} epochs")
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                self.logger.error(f"Adaptive epoch execution failed: {e}")
                
                return ExecutionResult(
                    scheduler_id=self.scheduler_id,
                    learner_id=getattr(learner, 'learner_id', 'unknown'),
                    executed_epochs=[],
                    metrics={},
                    final_state={},
                    execution_time=execution_time,
                    success=False,
                    error_message=str(e)
                )
            
            finally:
                self._is_running = False
                self._current_epoch = 0
    
    def _execute_single_epoch(self, 
                             learner: Any, 
                             dataloader: Any, 
                             epoch: int,
                             context: Optional[ExecutionContext] = None) -> Dict[str, float]:
        """执行单个epoch训练（复用StandardEpochScheduler的实现）"""
        if hasattr(learner, 'train_epoch'):
            return learner.train_epoch(dataloader, epoch)
        elif hasattr(learner, 'train_on_batch'):
            return self._execute_batch_training(learner, dataloader, epoch, context)
        else:
            # 模拟训练
            return {
                "loss": max(0.1, 1.0 / (epoch + 1)),
                "accuracy": min(0.9, 0.1 + epoch * 0.1)
            }
    
    def _execute_batch_training(self, learner: Any, dataloader: Any, epoch: int, context: Optional[ExecutionContext] = None) -> Dict[str, float]:
        """执行批次级训练（复用StandardEpochScheduler的实现）"""
        epoch_metrics = {"loss": 0.0, "accuracy": 0.0}
        batch_count = 0
        
        for batch_idx, batch_data in enumerate(dataloader):
            batch_metrics = learner.train_on_batch(batch_data)
            
            for key, value in batch_metrics.items():
                if key in epoch_metrics:
                    epoch_metrics[key] += value
                else:
                    epoch_metrics[key] = value
            
            batch_count += 1
        
        if batch_count > 0:
            epoch_metrics = {k: v / batch_count for k, v in epoch_metrics.items()}
        
        return epoch_metrics
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """获取内存使用情况"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                "rss": memory_info.rss / 1024 / 1024,
                "vms": memory_info.vms / 1024 / 1024
            }
        except ImportError:
            return {}
        except Exception:
            return {}


class SchedulerManager:
    """
    调度器管理器
    
    负责管理和协调多个EpochScheduler的执行
    """
    
    def __init__(self, scheduler_configs: Dict[str, Dict[str, Any]]):
        """
        初始化调度器管理器
        
        Args:
            scheduler_configs: 调度器配置字典
        """
        self.scheduler_configs = scheduler_configs
        self.schedulers: Dict[str, BaseEpochScheduler] = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 并发控制
        self.executor: Optional[ThreadPoolExecutor] = None
        self.running_futures: Dict[str, Future] = {}
        
        self.logger.debug("SchedulerManager initialized")
    
    def create_scheduler(self, scheduler_id: str, config: Dict[str, Any]) -> BaseEpochScheduler:
        """
        创建调度器实例
        
        Args:
            scheduler_id: 调度器ID
            config: 调度器配置
            
        Returns:
            BaseEpochScheduler: 调度器实例
        """
        scheduler_type = config.get("type", "StandardEpochScheduler")
        priority_str = config.get("priority", "NORMAL")
        priority = SchedulerPriority[priority_str.upper()]
        
        # 映射PyTorch调度器类型到内部类型
        if scheduler_type in ["StandardEpochScheduler", "StepLR", "step"]:
            scheduler = StandardEpochScheduler(scheduler_id, config.get("config", {}), priority)
        elif scheduler_type in ["AdaptiveEpochScheduler", "adaptive"]:
            scheduler = AdaptiveEpochScheduler(scheduler_id, config.get("config", {}), priority)
        else:
            # 对于不支持的类型，使用默认的StandardEpochScheduler
            self.logger.warning(f"Unknown scheduler type: {scheduler_type}, using StandardEpochScheduler")
            scheduler = StandardEpochScheduler(scheduler_id, config.get("config", {}), priority)
        
        self.schedulers[scheduler_id] = scheduler
        self.logger.debug(f"Created scheduler '{scheduler_id}' of type '{scheduler_type}'")
        
        return scheduler
    
    def register_scheduler(self, scheduler: BaseEpochScheduler) -> None:
        """
        注册调度器实例
        
        Args:
            scheduler: 调度器实例
        """
        self.schedulers[scheduler.scheduler_id] = scheduler
        self.logger.debug(f"Registered scheduler '{scheduler.scheduler_id}' of type '{scheduler.get_scheduler_type()}'")
    
    def get_scheduler(self, scheduler_id: str) -> BaseEpochScheduler:
        """
        获取调度器实例
        
        Args:
            scheduler_id: 调度器ID
            
        Returns:
            BaseEpochScheduler: 调度器实例
        """
        if scheduler_id not in self.schedulers:
            raise SchedulerError(f"Scheduler '{scheduler_id}' not found")
        
        return self.schedulers[scheduler_id]
    
    def execute_parallel(self, 
                        execution_tasks: List[Tuple[str, Any, Any, List[int], Optional[Dict[str, Any]], Optional[ExecutionContext]]],
                        max_workers: int = 2) -> Dict[str, ExecutionResult]:
        """
        并行执行多个调度器
        
        Args:
            execution_tasks: 执行任务列表 [(scheduler_id, learner, dataloader, epoch_range, inherited_state, context)]
            max_workers: 最大工作线程数
            
        Returns:
            Dict[str, ExecutionResult]: 执行结果字典
        """
        if not execution_tasks:
            return {}
        
        self.logger.info(f"Starting parallel execution of {len(execution_tasks)} schedulers")
        
        if not self.executor:
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # 按优先级排序任务
        sorted_tasks = sorted(execution_tasks, 
                            key=lambda x: self.get_scheduler(x[0]).priority.value)
        
        # 提交任务
        futures = {}
        for task in sorted_tasks:
            scheduler_id, learner, dataloader, epoch_range, inherited_state, context = task
            scheduler = self.get_scheduler(scheduler_id)
            
            future = self.executor.submit(
                scheduler.execute_epochs,
                learner, dataloader, epoch_range, inherited_state, context
            )
            futures[scheduler_id] = future
            self.running_futures[scheduler_id] = future
        
        # 等待所有任务完成
        results = {}
        for scheduler_id, future in futures.items():
            try:
                result = future.result()
                results[scheduler_id] = result
                self.logger.info(f"Scheduler '{scheduler_id}' 成功完成")
            except Exception as e:
                self.logger.error(f"Scheduler '{scheduler_id}' failed: {e}")
                results[scheduler_id] = ExecutionResult(
                    scheduler_id=scheduler_id,
                    learner_id="unknown",
                    executed_epochs=[],
                    metrics={},
                    final_state={},
                    success=False,
                    error_message=str(e)
                )
            finally:
                if scheduler_id in self.running_futures:
                    del self.running_futures[scheduler_id]
        
        self.logger.info(f"Parallel execution completed: {len(results)} results")
        return results
    
    def execute_sequential(self,
                          execution_tasks: List[Tuple[str, Any, Any, List[int], Optional[Dict[str, Any]], Optional[ExecutionContext]]]) -> Dict[str, ExecutionResult]:
        """
        顺序执行多个调度器
        
        Args:
            execution_tasks: 执行任务列表
            
        Returns:
            Dict[str, ExecutionResult]: 执行结果字典
        """
        if not execution_tasks:
            return {}
        
        self.logger.info(f"Starting sequential execution of {len(execution_tasks)} schedulers")
        
        # 按优先级排序任务
        sorted_tasks = sorted(execution_tasks, 
                            key=lambda x: self.get_scheduler(x[0]).priority.value)
        
        results = {}
        for task in sorted_tasks:
            scheduler_id, learner, dataloader, epoch_range, inherited_state, context = task
            scheduler = self.get_scheduler(scheduler_id)
            
            try:
                self.logger.info(f"Executing scheduler '{scheduler_id}'")
                result = scheduler.execute_epochs(learner, dataloader, epoch_range, inherited_state, context)
                results[scheduler_id] = result
                self.logger.info(f"Scheduler '{scheduler_id}' 成功完成")
            except Exception as e:
                self.logger.error(f"Scheduler '{scheduler_id}' failed: {e}")
                results[scheduler_id] = ExecutionResult(
                    scheduler_id=scheduler_id,
                    learner_id="unknown",
                    executed_epochs=[],
                    metrics={},
                    final_state={},
                    success=False,
                    error_message=str(e)
                )
        
        self.logger.info(f"Sequential execution completed: {len(results)} results")
        return results
    
    def get_running_schedulers(self) -> List[str]:
        """获取正在运行的调度器列表"""
        running = []
        for scheduler_id, scheduler in self.schedulers.items():
            if scheduler.is_running:
                running.append(scheduler_id)
        return running
    
    def get_scheduler_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取所有调度器的统计信息"""
        stats = {}
        for scheduler_id, scheduler in self.schedulers.items():
            stats[scheduler_id] = {
                "type": scheduler.get_scheduler_type(),
                "priority": scheduler.priority.name,
                "is_running": scheduler.is_running,
                "current_epoch": scheduler.current_epoch,
                "execution_stats": scheduler.get_execution_stats()
            }
        return stats
    
    def cleanup(self) -> None:
        """清理调度器管理器"""
        try:
            self.logger.info("Cleaning up SchedulerManager")
            
            # 关闭并发执行器
            if self.executor:
                self.executor.shutdown(wait=True)
                self.executor = None
            
            # 清理调度器
            self.schedulers.clear()
            self.running_futures.clear()
            
            self.logger.debug("SchedulerManager cleanup 完成")
            
        except Exception as e:
            self.logger.error(f"Error during SchedulerManager cleanup: {e}")