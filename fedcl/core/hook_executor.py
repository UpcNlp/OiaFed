# fedcl/core/hook_executor.py
"""
HookExecutor - 钩子执行引擎

负责管理和执行系统中注册的所有钩子，提供错误处理、性能监控、
条件执行等高级功能。支持多种错误处理策略和钩子间通信。
"""

import time
import uuid
import traceback
from collections import defaultdict, deque
from threading import Lock, RLock
from typing import Any, Dict, List, Optional, Union, Callable
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass

from loguru import logger
from omegaconf import DictConfig

from .hook import Hook, HookPhase
from .execution_context import ExecutionContext
from ..exceptions import HookExecutionError
from ..registry.component_registry import ComponentRegistry


@dataclass
class HookExecutionResult:
    """钩子执行结果"""
    hook_id: str
    hook_name: str
    phase: str
    result: Any = None
    success: bool = True
    error: Optional[Exception] = None
    execution_time: float = 0.0
    timestamp: float = 0.0


@dataclass
class ExecutionStats:
    """执行统计信息"""
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    total_execution_time: float = 0.0
    average_execution_time: float = 0.0
    hook_stats: Dict[str, Dict] = None
    
    def __post_init__(self):
        if self.hook_stats is None:
            self.hook_stats = defaultdict(lambda: {
                'executions': 0,
                'successes': 0,
                'failures': 0,
                'total_time': 0.0,
                'average_time': 0.0
            })


class HookExecutor:
    """
    钩子执行引擎
    
    负责管理和执行系统中注册的所有钩子。提供错误处理、性能监控、
    条件执行、并行执行等功能。
    
    Features:
    - 按优先级执行钩子
    - 多种错误处理策略
    - 执行性能监控
    - 钩子启用/禁用管理
    - 超时控制
    - 条件执行
    - 钩子间通信支持
    """
    
    def __init__(self, registry: ComponentRegistry, config: Optional[DictConfig] = None):
        """
        初始化HookExecutor
        
        Args:
            registry: 组件注册表
            config: 配置对象
        """
        self.registry = registry
        self.config = config or DictConfig({})
        
        # 线程安全锁
        self._lock = RLock()
        
        # 钩子存储：phase -> List[hook_instance]
        self._hooks: Dict[str, List[Hook]] = defaultdict(list)
        
        # 钩子实例映射：hook_id -> hook_instance
        self._hook_instances: Dict[str, Hook] = {}
        
        # 钩子状态管理
        self._disabled_hooks: set = set()
        
        # 错误处理策略
        self._error_policy = self.config.get('hook_execution', {}).get('error_policy', 'continue')
        
        # 执行统计
        self._stats = ExecutionStats()
        self._execution_history: deque = deque(maxlen=1000)
        
        # 线程池（用于并行执行）
        self._thread_pool = None
        if self.config.get('hook_execution', {}).get('parallel_execution', False):
            max_workers = self.config.get('hook_execution', {}).get('max_workers', 4)
            self._thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        
        # 超时设置
        self._default_timeout = self.config.get('hook_execution', {}).get('timeout', 30.0)
        self._max_execution_time = self.config.get('hook_execution', {}).get(
            'monitoring', {}
        ).get('max_execution_time', 10.0)
        
        logger.debug(f"HookExecutor initialized with error_policy={self._error_policy}")
    
    def execute_hooks(self, phase: str, context: ExecutionContext, **kwargs) -> List[Any]:
        """
        执行指定阶段的钩子
        
        Args:
            phase: 执行阶段
            context: 执行上下文
            **kwargs: 阶段特定的关键字参数
            
        Returns:
            List[Any]: 钩子执行结果列表
            
        Raises:
            HookExecutionError: 当错误策略为'stop'且有钩子执行失败时
        """
        start_time = time.time()
        
        with self._lock:
            hooks = self._get_enabled_hooks(phase)
            
        if not hooks:
            logger.debug(f"No hooks found for phase: {phase}")
            return []
        
        logger.debug(f"Executing {len(hooks)} hooks for phase: {phase}")
        
        results = []
        execution_results = []
        
        # 按优先级排序钩子
        sorted_hooks = sorted(hooks, key=lambda h: h.get_priority())
        
        try:
            if self._thread_pool and self._should_execute_parallel(phase):
                # 并行执行
                results = self._execute_hooks_parallel(sorted_hooks, phase, context, **kwargs)
            else:
                # 串行执行
                results = self._execute_hooks_sequential(sorted_hooks, phase, context, **kwargs)
                
        except Exception as e:
            logger.error(f"Critical error during hook execution for phase {phase}: {e}")
            if self._error_policy == 'stop':
                raise HookExecutionError(f"Hook execution failed for phase {phase}: {e}")
        
        # 更新统计信息
        execution_time = time.time() - start_time
        self._update_stats(phase, len(hooks), execution_time, results)
        
        # 性能监控
        if execution_time > self._max_execution_time:
            logger.warning(
                f"Hook execution for phase {phase} took {execution_time:.3f}s "
                f"(>{self._max_execution_time}s)"
            )
        
        # 过滤掉跳过的钩子结果，只返回实际执行的结果
        return [r.result for r in results if r.success and r.result is not None]
    
    def _execute_hooks_sequential(self, hooks: List[Hook], phase: str, 
                                context: ExecutionContext, **kwargs) -> List[HookExecutionResult]:
        """串行执行钩子"""
        results = []
        
        for hook in hooks:
            try:
                result = self._execute_single_hook(hook, phase, context, **kwargs)
                results.append(result)
                
                # 错误处理
                if not result.success:
                    if self._error_policy == 'stop':
                        logger.error(f"Stopping hook execution due to error in {hook.get_name()}")
                        break
                    elif self._error_policy == 'skip_phase':
                        logger.warning(f"Skipping remaining hooks in phase {phase} due to error")
                        break
                    # 'continue' - 继续执行下一个钩子
                        
            except Exception as e:
                logger.error(f"Unexpected error executing hook {hook.get_name()}: {e}")
                error_result = HookExecutionResult(
                    hook_id=getattr(hook, '_hook_id', str(uuid.uuid4())),
                    hook_name=hook.get_name(),
                    phase=phase,
                    success=False,
                    error=e,
                    timestamp=time.time()
                )
                results.append(error_result)
                
                if self._error_policy in ['stop', 'skip_phase']:
                    break
        
        return results
    
    def _execute_hooks_parallel(self, hooks: List[Hook], phase: str,
                              context: ExecutionContext, **kwargs) -> List[HookExecutionResult]:
        """并行执行钩子"""
        futures = []
        results = []
        
        # 提交所有钩子执行任务
        for hook in hooks:
            future = self._thread_pool.submit(
                self._execute_single_hook, hook, phase, context, **kwargs
            )
            futures.append((hook, future))
        
        # 收集结果
        for hook, future in futures:
            try:
                result = future.result(timeout=self._default_timeout)
                results.append(result)
            except FutureTimeoutError:
                error = TimeoutError(f"Hook {hook.get_name()} execution timeout")
                error_result = HookExecutionResult(
                    hook_id=getattr(hook, '_hook_id', str(uuid.uuid4())),
                    hook_name=hook.get_name(),
                    phase=phase,
                    success=False,
                    error=error,
                    timestamp=time.time()
                )
                results.append(error_result)
                logger.error(f"Hook {hook.get_name()} execution timeout")
            except Exception as e:
                error_result = HookExecutionResult(
                    hook_id=getattr(hook, '_hook_id', str(uuid.uuid4())),
                    hook_name=hook.get_name(),
                    phase=phase,
                    success=False,
                    error=e,
                    timestamp=time.time()
                )
                results.append(error_result)
                logger.error(f"Error in parallel hook execution for {hook.get_name()}: {e}")
        
        return results
    
    def _execute_single_hook(self, hook: Hook, phase: str, 
                           context: ExecutionContext, **kwargs) -> HookExecutionResult:
        """执行单个钩子"""
        start_time = time.time()
        hook_id = getattr(hook, '_hook_id', str(uuid.uuid4()))
        
        try:
            # 检查是否应该执行
            if not hook.should_execute(context, **kwargs):
                logger.debug(f"Hook {hook.get_name()} skipped (should_execute returned False)")
                return HookExecutionResult(
                    hook_id=hook_id,
                    hook_name=hook.get_name(),
                    phase=phase,
                    result=None,
                    success=False,  # 设置为False，表示跳过
                    execution_time=0.0,
                    timestamp=time.time()
                )
            
            # 准备钩子参数
            hook_kwargs = self.prepare_hook_kwargs(hook, phase, context, **kwargs)
            
            # 执行钩子
            logger.debug(f"Executing hook: {hook.get_name()} (phase: {phase})")
            result = hook.execute(context, **hook_kwargs)
            
            execution_time = time.time() - start_time
            
            # 记录执行时间
            hook.total_execution_time += execution_time
            hook.execution_count += 1
            hook._last_execution_time = execution_time
            
            logger.debug(f"Hook {hook.get_name()} executed successfully in {execution_time:.3f}s")
            
            return HookExecutionResult(
                hook_id=hook_id,
                hook_name=hook.get_name(),
                phase=phase,
                result=result,
                success=True,
                execution_time=execution_time,
                timestamp=time.time()
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Hook {hook.get_name()} execution failed: {e}")
            logger.debug(f"Hook error traceback:\n{traceback.format_exc()}")
            
            # 处理钩子错误
            self.handle_hook_error(hook, e, context)
            
            return HookExecutionResult(
                hook_id=hook_id,
                hook_name=hook.get_name(),
                phase=phase,
                success=False,
                error=e,
                execution_time=execution_time,
                timestamp=time.time()
            )
    
    def register_hook(self, hook: Hook) -> str:
        """
        注册钩子
        
        Args:
            hook: 钩子实例
            
        Returns:
            str: 钩子ID
        """
        with self._lock:
            # 生成唯一ID
            hook_id = str(uuid.uuid4())
            hook._hook_id = hook_id
            
            # 添加到相应阶段
            phase = hook.get_phase()
            self._hooks[phase].append(hook)
            self._hook_instances[hook_id] = hook
            
            # 按优先级排序
            self._hooks[phase].sort(key=lambda h: h.get_priority())
            
            logger.debug(f"Registered hook {hook.get_name()} with ID {hook_id} for phase {phase}")
            return hook_id
    
    def unregister_hook(self, hook_id: str) -> bool:
        """
        注销钩子
        
        Args:
            hook_id: 钩子ID
            
        Returns:
            bool: 是否成功注销
        """
        with self._lock:
            if hook_id not in self._hook_instances:
                logger.warning(f"Hook with ID {hook_id} not found")
                return False
            
            hook = self._hook_instances[hook_id]
            phase = hook.get_phase()
            
            # 从钩子列表中移除
            self._hooks[phase] = [h for h in self._hooks[phase] if getattr(h, '_hook_id', None) != hook_id]
            
            # 从实例映射中移除
            del self._hook_instances[hook_id]
            
            # 从禁用列表中移除（如果存在）
            self._disabled_hooks.discard(hook_id)
            
            logger.debug(f"Unregistered hook {hook.get_name()} with ID {hook_id}")
            return True
    
    def get_hooks(self, phase: str) -> List[Hook]:
        """
        获取指定阶段的钩子
        
        Args:
            phase: 执行阶段
            
        Returns:
            List[Hook]: 钩子列表
        """
        with self._lock:
            return list(self._hooks.get(phase, []))
    
    def prepare_hook_kwargs(self, hook: Hook, phase: str, context: ExecutionContext, 
                           **kwargs) -> Dict[str, Any]:
        """
        准备钩子参数
        
        Args:
            hook: 钩子实例
            phase: 执行阶段
            context: 执行上下文
            **kwargs: 额外参数
            
        Returns:
            Dict[str, Any]: 准备好的参数字典
        """
        # 基础参数
        hook_kwargs = kwargs.copy()
        
        # 添加阶段信息
        hook_kwargs['phase'] = phase
        hook_kwargs['hook_name'] = hook.get_name()
        
        # 添加上下文相关信息（安全地获取）
        try:
            if hasattr(context, 'get_current_round') and callable(getattr(context, 'get_current_round')):
                hook_kwargs['current_round'] = context.get_current_round()
        except Exception:
            pass
        
        try:
            if hasattr(context, 'get_current_task') and callable(getattr(context, 'get_current_task')):
                hook_kwargs['current_task'] = context.get_current_task()
        except Exception:
            pass
        
        return hook_kwargs
    
    def handle_hook_error(self, hook: Hook, error: Exception, context: ExecutionContext) -> None:
        """
        处理钩子错误
        
        Args:
            hook: 出错的钩子
            error: 错误信息
            context: 执行上下文
        """
        error_info = {
            'hook_name': hook.get_name(),
            'hook_phase': hook.get_phase(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'timestamp': time.time()
        }
        
        # 记录错误
        logger.error(f"Hook error: {error_info}")
        
        # 将错误信息存储到上下文中
        if hasattr(context, 'set_state'):
            context.set_state(f"hook_errors.{hook.get_name()}", error_info)
        
        # 触发错误处理钩子
        try:
            error_hooks = self.get_hooks(HookPhase.ON_ERROR.value)
            if error_hooks:
                for error_hook in error_hooks:
                    try:
                        error_hook.execute(context, original_error=error, failed_hook=hook)
                    except Exception as e:
                        logger.error(f"Error hook {error_hook.get_name()} failed: {e}")
        except Exception as e:
            logger.error(f"Failed to execute error hooks: {e}")
    
    def set_error_policy(self, policy: str) -> None:
        """
        设置错误处理策略
        
        Args:
            policy: 错误处理策略 ('continue', 'stop', 'skip_phase')
        """
        valid_policies = ['continue', 'stop', 'skip_phase']
        if policy not in valid_policies:
            raise ValueError(f"Invalid error policy: {policy}. Must be one of {valid_policies}")
        
        self._error_policy = policy
        logger.debug(f"Error policy set to: {policy}")
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """
        获取执行统计
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        with self._lock:
            # 计算平均执行时间
            if self._stats.total_executions > 0:
                self._stats.average_execution_time = (
                    self._stats.total_execution_time / self._stats.total_executions
                )
            
            # 计算每个钩子的平均执行时间
            for hook_name, stats in self._stats.hook_stats.items():
                if stats['executions'] > 0:
                    stats['average_time'] = stats['total_time'] / stats['executions']
            
            return {
                'total_executions': self._stats.total_executions,
                'successful_executions': self._stats.successful_executions,
                'failed_executions': self._stats.failed_executions,
                'success_rate': (
                    self._stats.successful_executions / self._stats.total_executions 
                    if self._stats.total_executions > 0 else 0.0
                ),
                'total_execution_time': self._stats.total_execution_time,
                'average_execution_time': self._stats.average_execution_time,
                'hook_stats': dict(self._stats.hook_stats),
                'registered_hooks': len(self._hook_instances),
                'disabled_hooks': len(self._disabled_hooks),
                'error_policy': self._error_policy
            }
    
    def enable_hook(self, hook_id: str) -> None:
        """
        启用钩子
        
        Args:
            hook_id: 钩子ID
        """
        with self._lock:
            if hook_id in self._hook_instances:
                self._disabled_hooks.discard(hook_id)
                self._hook_instances[hook_id].enable()
                logger.debug(f"Enabled hook with ID: {hook_id}")
            else:
                logger.warning(f"Hook with ID {hook_id} not found")
    
    def disable_hook(self, hook_id: str) -> None:
        """
        禁用钩子
        
        Args:
            hook_id: 钩子ID
        """
        with self._lock:
            if hook_id in self._hook_instances:
                self._disabled_hooks.add(hook_id)
                self._hook_instances[hook_id].disable()
                logger.debug(f"Disabled hook with ID: {hook_id}")
            else:
                logger.warning(f"Hook with ID {hook_id} not found")
    
    def clear_hooks(self, phase: Optional[str] = None) -> None:
        """
        清理钩子
        
        Args:
            phase: 指定阶段，为None时清理所有钩子
        """
        with self._lock:
            if phase is None:
                # 清理所有钩子
                self._hooks.clear()
                self._hook_instances.clear()
                self._disabled_hooks.clear()
                logger.debug("Cleared all hooks")
            else:
                # 清理指定阶段的钩子
                if phase in self._hooks:
                    hooks_to_remove = self._hooks[phase]
                    for hook in hooks_to_remove:
                        hook_id = getattr(hook, '_hook_id', None)
                        if hook_id:
                            self._hook_instances.pop(hook_id, None)
                            self._disabled_hooks.discard(hook_id)
                    
                    self._hooks[phase].clear()
                    logger.debug(f"Cleared hooks for phase: {phase}")
    
    def _get_enabled_hooks(self, phase: str) -> List[Hook]:
        """获取启用的钩子"""
        hooks = self._hooks.get(phase, [])
        return [
            hook for hook in hooks 
            if getattr(hook, '_hook_id', None) not in self._disabled_hooks
            and hook.is_enabled()
        ]
    
    def _should_execute_parallel(self, phase: str) -> bool:
        """判断是否应该并行执行钩子"""
        # 某些关键阶段可能需要串行执行以保证顺序
        serial_phases = {
            HookPhase.BEFORE_EXPERIMENT.value,
            HookPhase.AFTER_EXPERIMENT.value,
            HookPhase.ON_ERROR.value
        }
        return phase not in serial_phases
    
    def _update_stats(self, phase: str, hook_count: int, execution_time: float, 
                     results: List[HookExecutionResult]) -> None:
        """更新统计信息"""
        with self._lock:
            # 更新总体统计
            self._stats.total_executions += hook_count
            self._stats.total_execution_time += execution_time
            
            # 更新成功/失败统计
            successful = sum(1 for r in results if r.success)
            failed = len(results) - successful
            
            self._stats.successful_executions += successful
            self._stats.failed_executions += failed
            
            # 更新单个钩子统计
            for result in results:
                hook_stats = self._stats.hook_stats[result.hook_name]
                hook_stats['executions'] += 1
                hook_stats['total_time'] += result.execution_time
                
                if result.success:
                    hook_stats['successes'] += 1
                else:
                    hook_stats['failures'] += 1
            
            # 记录执行历史
            self._execution_history.append({
                'phase': phase,
                'hook_count': hook_count,
                'execution_time': execution_time,
                'successful': successful,
                '失败': failed,
                'timestamp': time.time()
            })
    
    def __del__(self):
        """析构函数，清理资源"""
        if hasattr(self, '_thread_pool') and self._thread_pool:
            self._thread_pool.shutdown(wait=True)
