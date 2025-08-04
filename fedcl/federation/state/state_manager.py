# fedcl/federation/state/state_manager.py
"""
状态管理器模块

基于现有ExecutionContext实现的状态管理功能，提供：
- 状态转换控制
- 状态历史记录
- 状态回调机制
- 状态汇报功能

更新版本：
- 与层级状态管理器兼容
- 支持新的状态枚举
- 增强的错误处理
- 更好的线程安全
"""

import time
import threading
from typing import Dict, Any, Optional, Callable, List, Union, Tuple
from threading import Lock, RLock
from collections import deque
from dataclasses import dataclass
from loguru import logger

from .state_enums import (
    ServerState, ClientState, ClientLifecycleState, TrainingPhaseState, 
    AuxiliaryState, StateTransition
)
from ...core.execution_context import ExecutionContext
from ...core.hook import Hook, HookPhase
from ...exceptions import FedCLError


class StateError(FedCLError):
    """状态管理错误"""
    pass


@dataclass
class StateTransitionRecord:
    """状态转换记录"""
    from_state: Any
    to_state: Any
    timestamp: float
    metadata: Dict[str, Any]
    duration: Optional[float] = None


@dataclass
class StateStatistics:
    """状态统计信息"""
    current_state: Any
    current_state_duration: float
    total_transitions: int
    total_states_visited: int
    uptime: float
    error_count: int
    average_state_duration: float


class StateCallback:
    """状态回调包装器"""
    
    def __init__(self, callback: Callable, trigger_states: Optional[List[Any]] = None, 
                 callback_id: Optional[str] = None):
        """
        初始化状态回调
        
        Args:
            callback: 回调函数
            trigger_states: 触发状态列表，None表示所有状态
            callback_id: 回调唯一标识
        """
        self.callback = callback
        self.trigger_states = trigger_states or []
        self.callback_id = callback_id or f"callback_{id(callback)}"
        self.last_triggered = None
        self.trigger_count = 0
        self.created_at = time.time()
    
    def should_trigger(self, new_state: Any) -> bool:
        """判断是否应该触发回调"""
        if not self.trigger_states:  # 全局回调
            return True
        return new_state in self.trigger_states
    
    def execute(self, old_state: Any, new_state: Any, metadata: Dict[str, Any]):
        """执行回调"""
        try:
            self.callback(old_state, new_state, metadata)
            self.last_triggered = time.time()
            self.trigger_count += 1
        except Exception as e:
            logger.error(f"状态回调执行失败 {self.callback_id}: {e}")
            raise


class StateManager:
    """
    状态管理器
    
    基于现有ExecutionContext实现状态管理，提供状态转换控制、
    历史记录、回调机制等功能。
    
    更新版本特性：
    - 支持所有新的状态枚举类型
    - 增强的线程安全性
    - 更好的错误处理和恢复
    - 详细的状态统计信息
    """
    
    def __init__(self, initial_state: Any, context: ExecutionContext, 
                 component_id: str = "unknown", max_history: int = 1000,
                 enable_validation: bool = True):
        """
        初始化状态管理器
        
        Args:
            initial_state: 初始状态
            context: 执行上下文
            component_id: 组件标识
            max_history: 最大历史记录数
            enable_validation: 是否启用状态转换验证
        """
        self.current_state = initial_state
        self.context = context
        self.component_id = component_id
        self.max_history = max_history
        self.enable_validation = enable_validation
        
        # 状态历史记录
        self.state_history: deque = deque(maxlen=max_history)
        self.state_history.append(StateTransitionRecord(
            from_state=None,
            to_state=initial_state,
            timestamp=time.time(),
            metadata={'component_id': component_id, 'action': 'initialize'}
        ))
        
        # 回调管理
        self.callbacks: Dict[Any, List[StateCallback]] = {}
        self.global_callbacks: List[StateCallback] = []
        self._callback_lock = RLock()
        
        # 状态元数据
        self.state_metadata: Dict[str, Any] = {
            'component_id': component_id,
            'created_at': time.time(),
            'state_changes': 0,
            'error_count': 0
        }
        
        # 线程安全锁
        self._state_lock = RLock()
        self._transition_in_progress = False
        
        # 状态汇报配置
        self.enable_reporting = True
        self.auto_report_states = set()  # 自动汇报的状态
        
        # 状态验证缓存
        self._validation_cache: Dict[Tuple[Any, Any], bool] = {}
        
        logger.debug(f"状态管理器初始化完成: {component_id}, 初始状态: {initial_state}")
    
    def transition_to(self, new_state: Any, metadata: Optional[Dict[str, Any]] = None,
                     force: bool = False) -> bool:
        """
        状态转换
        
        Args:
            new_state: 新状态
            metadata: 转换元数据
            force: 是否强制转换（跳过合法性检查）
            
        Returns:
            bool: 转换是否成功
            
        Raises:
            StateError: 状态转换失败
        """
        with self._state_lock:
            if self._transition_in_progress:
                logger.warning(f"状态转换正在进行中，忽略新的转换请求: {new_state}")
                return False
            
            self._transition_in_progress = True
            
            try:
                old_state = self.current_state
                transition_start = time.time()
                
                # 如果状态相同，直接返回成功
                if old_state == new_state:
                    logger.debug(f"状态未发生变化: {old_state}")
                    return True
                
                # 验证状态转换合法性
                if not force and self.enable_validation:
                    if not self._is_valid_transition(old_state, new_state):
                        valid_states = StateTransition.get_valid_transitions(old_state)
                        error_msg = (
                            f"非法状态转换: {old_state} -> {new_state}, "
                            f"合法转换: {valid_states}"
                        )
                        logger.error(error_msg)
                        self.state_metadata['error_count'] += 1
                        raise StateError(error_msg)
                
                # 准备转换元数据
                transition_metadata = {
                    'component_id': self.component_id,
                    'transition_time': transition_start,
                    'forced': force,
                    'validation_enabled': self.enable_validation,
                    **(metadata or {})
                }
                
                # 执行状态转换
                self.current_state = new_state
                self.state_metadata['state_changes'] += 1
                self.state_metadata['last_transition'] = transition_start
                
                # 记录状态历史
                transition_record = StateTransitionRecord(
                    from_state=old_state,
                    to_state=new_state,
                    timestamp=transition_start,
                    metadata=transition_metadata.copy()
                )
                self.state_history.append(transition_record)
                
                # 更新ExecutionContext
                self._update_execution_context(old_state, new_state, transition_metadata)
                
                # 触发状态回调
                self._trigger_callbacks(old_state, new_state, transition_metadata)
                
                # 状态汇报
                if self.enable_reporting and new_state in self.auto_report_states:
                    self._report_state_change(old_state, new_state, transition_metadata)
                
                # 更新转换记录的持续时间
                transition_duration = time.time() - transition_start
                transition_record.duration = transition_duration
                
                logger.debug(f"状态转换成功: {self.component_id} {old_state} -> {new_state} "
                          f"(耗时: {transition_duration:.3f}s)")
                
                return True
                
            except Exception as e:
                self.state_metadata['error_count'] += 1
                logger.error(f"状态转换失败: {self.component_id} {old_state} -> {new_state}: {e}")
                if isinstance(e, StateError):
                    raise
                else:
                    raise StateError(f"状态转换失败: {e}") from e
            
            finally:
                self._transition_in_progress = False
    
    def register_callback(self, callback: Callable, 
                         trigger_states: Optional[List[Any]] = None,
                         callback_id: Optional[str] = None) -> str:
        """
        注册状态回调
        
        Args:
            callback: 回调函数，签名为 (old_state, new_state, metadata)
            trigger_states: 触发状态列表，None表示全局回调
            callback_id: 回调标识
            
        Returns:
            str: 回调标识
        """
        with self._callback_lock:
            state_callback = StateCallback(callback, trigger_states, callback_id)
            
            if trigger_states:
                # 注册到特定状态
                for state in trigger_states:
                    if state not in self.callbacks:
                        self.callbacks[state] = []
                    self.callbacks[state].append(state_callback)
            else:
                # 注册为全局回调
                self.global_callbacks.append(state_callback)
            
            logger.debug(f"注册状态回调: {state_callback.callback_id}")
            return state_callback.callback_id
    
    def unregister_callback(self, callback_id: str) -> bool:
        """
        注销状态回调
        
        Args:
            callback_id: 回调标识
            
        Returns:
            bool: 是否成功注销
        """
        with self._callback_lock:
            removed_count = 0
            
            # 从特定状态回调中移除
            for state_callbacks in self.callbacks.values():
                original_length = len(state_callbacks)
                state_callbacks[:] = [cb for cb in state_callbacks if cb.callback_id != callback_id]
                removed_count += original_length - len(state_callbacks)
            
            # 从全局回调中移除
            original_length = len(self.global_callbacks)
            self.global_callbacks[:] = [cb for cb in self.global_callbacks if cb.callback_id != callback_id]
            removed_count += original_length - len(self.global_callbacks)
            
            success = removed_count > 0
            if success:
                logger.debug(f"注销状态回调成功: {callback_id} (移除 {removed_count} 个)")
            else:
                logger.warning(f"未找到要注销的回调: {callback_id}")
            
            return success
    
    def register_hook(self, hook: Hook, trigger_states: Optional[List[Any]] = None) -> str:
        """
        注册Hook作为状态回调
        
        Args:
            hook: Hook对象
            trigger_states: 触发状态列表
            
        Returns:
            str: 回调标识
        """
        def hook_callback(old_state, new_state, metadata):
            """Hook回调包装器"""
            try:
                # 将状态信息添加到执行上下文
                self.context.update_state({
                    'old_state': old_state,
                    'new_state': new_state,
                    'state_metadata': metadata
                })
                # 执行Hook
                hook.execute(self.context, HookPhase.STATE_TRANSITION)
            except Exception as e:
                logger.error(f"Hook执行失败: {hook.__class__.__name__}: {e}")
        
        return self.register_callback(
            hook_callback, 
            trigger_states, 
            f"hook_{hook.__class__.__name__}_{id(hook)}"
        )
    
    def enable_auto_reporting(self, states: List[Any]):
        """
        启用特定状态的自动汇报
        
        Args:
            states: 需要自动汇报的状态列表
        """
        self.auto_report_states.update(states)
        logger.debug(f"启用自动汇报状态: {[str(s) for s in states]}")
    
    def disable_auto_reporting(self, states: Optional[List[Any]] = None):
        """
        禁用状态自动汇报
        
        Args:
            states: 要禁用的状态列表，None表示禁用所有
        """
        if states is None:
            self.auto_report_states.clear()
            logger.debug("禁用所有自动汇报状态")
        else:
            self.auto_report_states.difference_update(states)
            logger.debug(f"禁用自动汇报状态: {[str(s) for s in states]}")
    
    def get_current_state(self) -> Any:
        """获取当前状态"""
        return self.current_state
    
    def get_state_duration(self) -> float:
        """获取当前状态持续时间（秒）"""
        if self.state_history:
            return time.time() - self.state_history[-1].timestamp
        return 0.0
    
    def get_state_history(self, limit: Optional[int] = None) -> List[StateTransitionRecord]:
        """
        获取状态历史记录
        
        Args:
            limit: 返回记录数限制
            
        Returns:
            List[StateTransitionRecord]: 状态转换记录列表
        """
        history = list(self.state_history)
        if limit:
            return history[-limit:]
        return history
    
    def get_state_statistics(self) -> StateStatistics:
        """
        获取状态统计信息
        
        Returns:
            StateStatistics: 状态统计信息
        """
        try:
            total_transitions = len(self.state_history) - 1  # 减去初始状态
            
            # 统计各状态停留时间
            state_durations = []
            states_visited = set()
            
            for i in range(len(self.state_history) - 1):
                current_record = self.state_history[i]
                next_record = self.state_history[i + 1]
                duration = next_record.timestamp - current_record.timestamp
                state_durations.append(duration)
                states_visited.add(current_record.to_state)
            
            # 添加当前状态
            states_visited.add(self.current_state)
            
            # 计算平均停留时间
            avg_duration = sum(state_durations) / len(state_durations) if state_durations else 0.0
            
            return StateStatistics(
                current_state=self.current_state,
                current_state_duration=self.get_state_duration(),
                total_transitions=total_transitions,
                total_states_visited=len(states_visited),
                uptime=time.time() - self.state_metadata['created_at'],
                error_count=self.state_metadata['error_count'],
                average_state_duration=avg_duration
            )
            
        except Exception as e:
            logger.error(f"获取状态统计信息失败: {e}")
            return StateStatistics(
                current_state=self.current_state,
                current_state_duration=0.0,
                total_transitions=0,
                total_states_visited=1,
                uptime=0.0,
                error_count=self.state_metadata.get('error_count', 0),
                average_state_duration=0.0
            )
    
    def _is_valid_transition(self, from_state: Any, to_state: Any) -> bool:
        """检查状态转换是否合法（带缓存）"""
        # 使用缓存提高性能
        cache_key = (from_state, to_state)
        if cache_key in self._validation_cache:
            return self._validation_cache[cache_key]
        
        # 执行验证
        is_valid = StateTransition.is_valid_transition(from_state, to_state)
        
        # 缓存结果
        self._validation_cache[cache_key] = is_valid
        
        return is_valid
    
    def _update_execution_context(self, old_state: Any, new_state: Any, metadata: Dict[str, Any]):
        """更新ExecutionContext中的状态信息"""
        try:
            self.context.update_state({
                'current_state': new_state,
                'previous_state': old_state,
                'state_transition_time': metadata['transition_time'],
                'component_id': self.component_id,
                'state_metadata': metadata
            })
        except Exception as e:
            logger.warning(f"更新ExecutionContext失败: {e}")
    
    def _trigger_callbacks(self, old_state: Any, new_state: Any, metadata: Dict[str, Any]):
        """触发状态回调"""
        triggered_callbacks = []
        
        with self._callback_lock:
            # 触发特定状态回调
            for callback in self.callbacks.get(new_state, []):
                if callback.should_trigger(new_state):
                    triggered_callbacks.append(callback)
            
            # 触发全局回调
            for callback in self.global_callbacks:
                if callback.should_trigger(new_state):
                    triggered_callbacks.append(callback)
        
        # 执行回调（在锁外执行避免死锁）
        for callback in triggered_callbacks:
            try:
                callback.execute(old_state, new_state, metadata)
            except Exception as e:
                logger.error(f"状态回调执行失败: {callback.callback_id}: {e}")
    
    def _report_state_change(self, old_state: Any, new_state: Any, metadata: Dict[str, Any]):
        """汇报状态变化"""
        try:
            logger.debug(f"状态汇报: {self.component_id} {old_state} -> {new_state}")
            
            # 通过ExecutionContext发送事件
            self.context.emit_event('state_changed', {
                'component_id': self.component_id,
                'old_state': str(old_state),
                'new_state': str(new_state),
                'metadata': metadata
            })
            
        except Exception as e:
            logger.warning(f"状态汇报失败: {e}")
    
    def cleanup(self):
        """清理资源"""
        try:
            logger.debug(f"清理状态管理器: {self.component_id}")
            
            # 清理回调
            with self._callback_lock:
                self.callbacks.clear()
                self.global_callbacks.clear()
            
            # 清理历史记录
            self.state_history.clear()
            
            # 清理缓存
            self._validation_cache.clear()
            
            # 重置元数据
            self.state_metadata['state_changes'] = 0
            self.state_metadata['error_count'] = 0
            
            logger.debug(f"状态管理器清理完成: {self.component_id}")
            
        except Exception as e:
            logger.error(f"状态管理器清理失败: {e}")


def create_state_manager(initial_state: Any, 
                        context: ExecutionContext,
                        component_id: str = "unknown",
                        max_history: int = 1000,
                        enable_validation: bool = True) -> StateManager:
    """
    创建状态管理器的工厂函数
    
    Args:
        initial_state: 初始状态
        context: 执行上下文
        component_id: 组件标识
        max_history: 最大历史记录数
        enable_validation: 是否启用状态转换验证
        
    Returns:
        StateManager: 状态管理器实例
    """
    try:
        return StateManager(
            initial_state=initial_state,
            context=context,
            component_id=component_id,
            max_history=max_history,
            enable_validation=enable_validation
        )
    except Exception as e:
        logger.error(f"创建状态管理器失败: {e}")
        raise StateError(f"Failed to create state manager: {e}")


# 向后兼容的类和函数
class ComponentStateManager:
    """
    组件状态管理器（保持向后兼容）
    
    管理多个组件的状态，提供统一的状态查询和监控接口。
    """
    
    def __init__(self):
        self.state_managers: Dict[str, StateManager] = {}
        self._lock = RLock()
    
    def register_component(self, component_id: str, state_manager: StateManager):
        """注册组件状态管理器"""
        with self._lock:
            self.state_managers[component_id] = state_manager
            logger.debug(f"注册组件状态管理器: {component_id}")
    
    def unregister_component(self, component_id: str):
        """注销组件状态管理器"""
        with self._lock:
            if component_id in self.state_managers:
                # 清理状态管理器
                try:
                    self.state_managers[component_id].cleanup()
                except Exception as e:
                    logger.warning(f"清理组件状态管理器失败 {component_id}: {e}")
                
                del self.state_managers[component_id]
                logger.debug(f"注销组件状态管理器: {component_id}")
    
    def get_component_state(self, component_id: str) -> Optional[Any]:
        """获取组件当前状态"""
        with self._lock:
            if component_id in self.state_managers:
                return self.state_managers[component_id].get_current_state()
            return None
    
    def get_all_states(self) -> Dict[str, Any]:
        """获取所有组件状态"""
        with self._lock:
            return {
                component_id: manager.get_current_state()
                for component_id, manager in self.state_managers.items()
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统整体状态"""
        with self._lock:
            total_components = len(self.state_managers)
            error_components = 0
            running_components = 0
            
            component_details = {}
            
            for component_id, manager in self.state_managers.items():
                try:
                    state = manager.get_current_state()
                    stats = manager.get_state_statistics()
                    
                    component_details[component_id] = {
                        'current_state': str(state),
                        'state_duration': manager.get_state_duration(),
                        'total_transitions': stats.total_transitions,
                        'error_count': stats.error_count
                    }
                    
                    # 统计错误和运行状态
                    state_name = getattr(state, 'name', str(state)).upper()
                    if 'ERROR' in state_name or 'FAILED' in state_name:
                        error_components += 1
                    elif any(keyword in state_name for keyword in ['TRAINING', 'RUNNING', 'READY', 'WAITING']):
                        running_components += 1
                        
                except Exception as e:
                    logger.warning(f"获取组件状态失败 {component_id}: {e}")
                    component_details[component_id] = {'error': str(e)}
                    error_components += 1
            
            return {
                'total_components': total_components,
                'error_components': error_components,
                'running_components': running_components,
                'idle_components': total_components - error_components - running_components,
                'component_details': component_details,
                'timestamp': time.time()
            }
    
    def cleanup_all(self):
        """清理所有组件状态管理器"""
        with self._lock:
            component_ids = list(self.state_managers.keys())
            for component_id in component_ids:
                self.unregister_component(component_id)
            
            logger.debug("所有组件状态管理器已清理")


# 全局组件状态管理器实例
global_component_state_manager = ComponentStateManager()