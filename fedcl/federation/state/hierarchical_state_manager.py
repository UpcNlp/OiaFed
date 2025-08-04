# fedcl/federation/state/hierarchical_state_manager.py
"""
层级状态管理器实现 - 修正版

修正问题：
1. 状态同步逻辑的死锁问题
2. 状态转换验证的一致性
3. 错误处理和恢复机制
4. 回调执行的异常安全
"""

import time
import threading
from typing import Dict, Any, Optional, Callable, List, Union
from dataclasses import dataclass
from threading import Lock, RLock
from collections import deque
from loguru import logger

from .state_enums import ClientLifecycleState, TrainingPhaseState, StateTransition
from .state_manager import StateManager, create_state_manager
from ...core.execution_context import ExecutionContext
from ...exceptions import FedCLError


class StateError(FedCLError):
    """状态管理错误"""
    pass


@dataclass
class StateTransitionRule:
    """状态转换规则"""
    from_state: Any
    to_state: Any
    condition: Optional[Callable] = None
    metadata_requirements: Optional[Dict[str, Any]] = None


@dataclass
class StateSyncResult:
    """状态同步结果"""
    success: bool
    triggered_transitions: List[Dict[str, Any]]
    errors: List[str]
    timestamp: float


class HierarchicalStateManager:
    """
    层级状态管理器 - 修正版
    
    修正内容：
    1. 解决状态同步时的死锁问题
    2. 改进错误处理和恢复机制
    3. 增强状态转换验证
    4. 优化回调执行的异常安全
    """
    
    def __init__(self, execution_context: ExecutionContext, component_id: str,
                 max_history: int = 1000, enable_validation: bool = True):
        """
        初始化层级状态管理器
        
        Args:
            execution_context: 执行上下文
            component_id: 组件标识
            max_history: 最大历史记录数
            enable_validation: 是否启用状态转换验证
        """
        self.execution_context = execution_context
        self.component_id = component_id
        self.max_history = max_history
        self.enable_validation = enable_validation
        self.logger = logger.bind(component=f"HierarchicalStateManager:{component_id}")
        
        # 修正：使用单独的锁避免死锁
        self._coordination_lock = RLock()
        self._control_lock = RLock()
        self._sync_lock = RLock()
        
        # 创建协调层状态管理器
        self.coordination_state_manager = create_state_manager(
            initial_state=ClientLifecycleState.INITIALIZING,
            context=execution_context,
            component_id=f"{component_id}_coordination",
            max_history=max_history,
            enable_validation=enable_validation
        )
        
        # 创建控制层状态管理器
        self.control_state_manager = create_state_manager(
            initial_state=TrainingPhaseState.UNINITIALIZED,
            context=execution_context,
            component_id=f"{component_id}_control",
            max_history=max_history,
            enable_validation=enable_validation
        )
        
        # 状态同步配置
        self.enable_auto_sync = True
        self.sync_in_progress = False
        self._sync_timeout = 5.0  # 同步超时时间
        
        # 修正：改进的状态同步规则
        self.coordination_to_control_rules = self._setup_coordination_to_control_rules()
        self.control_to_coordination_rules = self._setup_control_to_coordination_rules()
        
        # 同步历史记录
        self.sync_history: deque = deque(maxlen=max_history)
        
        # 自定义回调
        self.custom_callbacks: Dict[str, List[Callable]] = {
            'coordination_change': [],
            'control_change': [],
            'sync_success': [],
            'sync_failure': []
        }
        
        # 回调执行锁
        self._callback_lock = RLock()
        
        # 建立状态同步机制
        self._setup_state_synchronization()
        
        # 监控统计
        self.stats = {
            'total_coordination_transitions': 0,
            'total_control_transitions': 0,
            'successful_syncs': 0,
            'failed_syncs': 0,
            'sync_conflicts': 0,
            'sync_timeouts': 0,
            'created_at': time.time()
        }
        
        self.logger.debug(f"层级状态管理器初始化完成: {component_id}")
    
    def _setup_coordination_to_control_rules(self) -> Dict[ClientLifecycleState, Dict[TrainingPhaseState, TrainingPhaseState]]:
        """设置协调层到控制层的状态转换规则 - 修正版"""
        return {
            # 当协调层进入TRAINING状态时，控制层的对应转换
            ClientLifecycleState.TRAINING: {
                TrainingPhaseState.UNINITIALIZED: TrainingPhaseState.INITIALIZING,
                TrainingPhaseState.FINISHED: TrainingPhaseState.PREPARING,
                TrainingPhaseState.FAILED: TrainingPhaseState.PREPARING,
                TrainingPhaseState.PAUSED: TrainingPhaseState.RUNNING
            },
            
            # 当协调层进入ERROR状态时，控制层的对应转换
            ClientLifecycleState.ERROR: {
                TrainingPhaseState.RUNNING: TrainingPhaseState.FAILED,
                TrainingPhaseState.PHASE_TRANSITION: TrainingPhaseState.FAILED,
                TrainingPhaseState.EPOCH_EXECUTING: TrainingPhaseState.FAILED,
                TrainingPhaseState.EVALUATING: TrainingPhaseState.FAILED,
                TrainingPhaseState.AGGREGATING: TrainingPhaseState.FAILED,
                TrainingPhaseState.PREPARING: TrainingPhaseState.FAILED,
                TrainingPhaseState.INITIALIZING: TrainingPhaseState.FAILED
            },
            
            # 当协调层进入READY状态时，控制层的对应转换
            ClientLifecycleState.READY: {
                TrainingPhaseState.RUNNING: TrainingPhaseState.PAUSED,
                TrainingPhaseState.EPOCH_EXECUTING: TrainingPhaseState.PAUSED,
                TrainingPhaseState.PHASE_TRANSITION: TrainingPhaseState.PAUSED
            },
            
            # 当协调层进入COMPLETED状态时，控制层的对应转换
            ClientLifecycleState.COMPLETED: {
                TrainingPhaseState.RUNNING: TrainingPhaseState.FINISHED,
                TrainingPhaseState.EVALUATING: TrainingPhaseState.FINISHED,
                TrainingPhaseState.AGGREGATING: TrainingPhaseState.FINISHED
            }
        }
    
    def _setup_control_to_coordination_rules(self) -> Dict[TrainingPhaseState, Dict[ClientLifecycleState, ClientLifecycleState]]:
        """设置控制层到协调层的状态转换规则 - 修正版"""
        return {
            # 当控制层进入FINISHED状态时，协调层的对应转换
            TrainingPhaseState.FINISHED: {
                ClientLifecycleState.TRAINING: ClientLifecycleState.READY,
                ClientLifecycleState.CONNECTED: ClientLifecycleState.READY
            },
            
            # 当控制层进入FAILED状态时，协调层的对应转换
            TrainingPhaseState.FAILED: {
                ClientLifecycleState.TRAINING: ClientLifecycleState.ERROR,
                ClientLifecycleState.READY: ClientLifecycleState.ERROR,
                ClientLifecycleState.CONNECTED: ClientLifecycleState.ERROR
            },
            
            # 当控制层进入RUNNING状态时，协调层的对应转换
            TrainingPhaseState.RUNNING: {
                ClientLifecycleState.READY: ClientLifecycleState.TRAINING,
                ClientLifecycleState.CONNECTED: ClientLifecycleState.TRAINING
            },
            
            # 当控制层进入EPOCH_EXECUTING状态时，确保协调层在TRAINING状态
            TrainingPhaseState.EPOCH_EXECUTING: {
                ClientLifecycleState.READY: ClientLifecycleState.TRAINING,
                ClientLifecycleState.CONNECTED: ClientLifecycleState.TRAINING
            }
        }
    
    def _setup_state_synchronization(self):
        """建立状态同步机制 - 修正版"""
        try:
            # 协调层状态变化回调
            self.coordination_state_manager.register_callback(
                self._on_coordination_state_change_safe,
                callback_id="coordination_sync"
            )
            
            # 控制层状态变化回调
            self.control_state_manager.register_callback(
                self._on_control_state_change_safe,
                callback_id="control_sync"
            )
            
            self.logger.debug("状态同步机制已建立")
            
        except Exception as e:
            self.logger.error(f"建立状态同步机制失败: {e}")
            raise StateError(f"Failed to setup state synchronization: {e}")
    
    def _on_coordination_state_change_safe(self, old_state: ClientLifecycleState, 
                                         new_state: ClientLifecycleState, 
                                         metadata: Dict[str, Any]):
        """协调层状态变化回调 - 安全版本"""
        try:
            # 使用单独的线程处理状态变化，避免死锁
            threading.Thread(
                target=self._handle_coordination_state_change,
                args=(old_state, new_state, metadata),
                daemon=True
            ).start()
        except Exception as e:
            self.logger.error(f"启动协调层状态变化处理线程失败: {e}")
    
    def _handle_coordination_state_change(self, old_state: ClientLifecycleState, 
                                        new_state: ClientLifecycleState, 
                                        metadata: Dict[str, Any]):
        """处理协调层状态变化"""
        try:
            self.stats['total_coordination_transitions'] += 1
            
            self.logger.debug(
                f"协调层状态变化: {old_state.name} -> {new_state.name} "
                f"(metadata: {metadata.get('action', 'unknown')})"
            )
            
            # 执行自定义回调
            self._execute_custom_callbacks_safe('coordination_change', old_state, new_state, metadata)
            
            # 如果启用自动同步，尝试同步控制层状态
            if self.enable_auto_sync:
                sync_result = self._sync_control_state_from_coordination_safe(old_state, new_state, metadata)
                if sync_result.success:
                    self.stats['successful_syncs'] += 1
                    self._execute_custom_callbacks_safe('sync_success', 'coordination_to_control', sync_result)
                else:
                    self.stats['failed_syncs'] += 1
                    self._execute_custom_callbacks_safe('sync_failure', 'coordination_to_control', sync_result)
            
            # 发布协调层状态变化事件
            self.execution_context.publish_event("coordination_state_changed", {
                "component_id": self.component_id,
                "old_state": old_state.name,
                "new_state": new_state.name,
                "metadata": metadata,
                "sync_enabled": self.enable_auto_sync
            })
            
        except Exception as e:
            self.logger.error(f"协调层状态变化处理失败: {e}")
    
    def _on_control_state_change_safe(self, old_state: TrainingPhaseState,
                                    new_state: TrainingPhaseState,
                                    metadata: Dict[str, Any]):
        """控制层状态变化回调 - 安全版本"""
        try:
            # 使用单独的线程处理状态变化，避免死锁
            threading.Thread(
                target=self._handle_control_state_change,
                args=(old_state, new_state, metadata),
                daemon=True
            ).start()
        except Exception as e:
            self.logger.error(f"启动控制层状态变化处理线程失败: {e}")
    
    def _handle_control_state_change(self, old_state: TrainingPhaseState,
                                   new_state: TrainingPhaseState,
                                   metadata: Dict[str, Any]):
        """处理控制层状态变化"""
        try:
            self.stats['total_control_transitions'] += 1
            
            self.logger.debug(
                f"控制层状态变化: {old_state.name} -> {new_state.name} "
                f"(metadata: {metadata.get('action', 'unknown')})"
            )
            
            # 执行自定义回调
            self._execute_custom_callbacks_safe('control_change', old_state, new_state, metadata)
            
            # 如果启用自动同步，尝试同步协调层状态
            if self.enable_auto_sync:
                sync_result = self._sync_coordination_state_from_control_safe(old_state, new_state, metadata)
                if sync_result.success:
                    self.stats['successful_syncs'] += 1
                    self._execute_custom_callbacks_safe('sync_success', 'control_to_coordination', sync_result)
                else:
                    self.stats['failed_syncs'] += 1
                    self._execute_custom_callbacks_safe('sync_failure', 'control_to_coordination', sync_result)
            
            # 发布控制层状态变化事件
            self.execution_context.publish_event("control_state_changed", {
                "component_id": self.component_id,
                "old_state": old_state.name,
                "new_state": new_state.name,
                "metadata": metadata,
                "sync_enabled": self.enable_auto_sync
            })
            
        except Exception as e:
            self.logger.error(f"控制层状态变化处理失败: {e}")
    
    def _sync_control_state_from_coordination_safe(self, old_coordination_state: ClientLifecycleState,
                                                 new_coordination_state: ClientLifecycleState,
                                                 metadata: Dict[str, Any]) -> StateSyncResult:
        """根据协调层状态变化同步控制层状态 - 安全版本"""
        sync_result = StateSyncResult(
            success=True,
            triggered_transitions=[],
            errors=[],
            timestamp=time.time()
        )
        
        # 修正：使用超时机制避免死锁
        try:
            acquired = self._sync_lock.acquire(timeout=self._sync_timeout)
            if not acquired:
                self.stats['sync_timeouts'] += 1
                sync_result.success = False
                sync_result.errors.append("Sync lock timeout")
                return sync_result
            
            try:
                if self.sync_in_progress:
                    sync_result.success = False
                    sync_result.errors.append("Sync already in progress")
                    return sync_result
                
                self.sync_in_progress = True
                
                current_control_state = self.control_state_manager.get_current_state()
                
                # 检查是否需要同步控制层状态
                if new_coordination_state in self.coordination_to_control_rules:
                    control_transitions = self.coordination_to_control_rules[new_coordination_state]
                    
                    if current_control_state in control_transitions:
                        target_control_state = control_transitions[current_control_state]
                        
                        self.logger.debug(
                            f"同步控制层状态: {current_control_state.name} -> {target_control_state.name} "
                            f"(triggered by coordination: {old_coordination_state.name} -> {new_coordination_state.name})"
                        )
                        
                        # 构建同步元数据
                        sync_metadata = {
                            "triggered_by": "coordination_layer",
                            "coordination_transition": f"{old_coordination_state.name} -> {new_coordination_state.name}",
                            "sync_timestamp": time.time(),
                            "sync_reason": "auto_sync"
                        }
                        sync_metadata.update(metadata)
                        
                        # 修正：使用控制层锁保护状态转换
                        with self._control_lock:
                            success = self.control_state_manager.transition_to(target_control_state, sync_metadata)
                        
                        if success:
                            sync_result.triggered_transitions.append({
                                "from_state": current_control_state.name,
                                "to_state": target_control_state.name,
                                "layer": "control",
                                "metadata": sync_metadata
                            })
                        else:
                            sync_result.success = False
                            sync_result.errors.append(f"Failed to transition control state to {target_control_state.name}")
                    else:
                        self.logger.debug(
                            f"无需同步控制层状态: 当前状态 {current_control_state.name} "
                            f"不在协调层状态 {new_coordination_state.name} 的同步规则中"
                        )
                
                # 记录同步历史
                self.sync_history.append({
                    "type": "coordination_to_control",
                    "coordination_transition": f"{old_coordination_state.name} -> {new_coordination_state.name}",
                    "result": sync_result,
                    "timestamp": sync_result.timestamp
                })
                
            finally:
                self.sync_in_progress = False
                
        finally:
            if acquired:
                self._sync_lock.release()
        
        return sync_result
    
    def _sync_coordination_state_from_control_safe(self, old_control_state: TrainingPhaseState,
                                                 new_control_state: TrainingPhaseState,
                                                 metadata: Dict[str, Any]) -> StateSyncResult:
        """根据控制层状态变化同步协调层状态 - 安全版本"""
        sync_result = StateSyncResult(
            success=True,
            triggered_transitions=[],
            errors=[],
            timestamp=time.time()
        )
        
        # 修正：使用超时机制避免死锁
        try:
            acquired = self._sync_lock.acquire(timeout=self._sync_timeout)
            if not acquired:
                self.stats['sync_timeouts'] += 1
                sync_result.success = False
                sync_result.errors.append("Sync lock timeout")
                return sync_result
            
            try:
                if self.sync_in_progress:
                    sync_result.success = False
                    sync_result.errors.append("Sync already in progress")
                    return sync_result
                
                self.sync_in_progress = True
                
                current_coordination_state = self.coordination_state_manager.get_current_state()
                
                # 检查是否需要同步协调层状态
                if new_control_state in self.control_to_coordination_rules:
                    coordination_transitions = self.control_to_coordination_rules[new_control_state]
                    
                    if current_coordination_state in coordination_transitions:
                        target_coordination_state = coordination_transitions[current_coordination_state]
                        
                        self.logger.debug(
                            f"同步协调层状态: {current_coordination_state.name} -> {target_coordination_state.name} "
                            f"(triggered by control: {old_control_state.name} -> {new_control_state.name})"
                        )
                        
                        # 构建同步元数据
                        sync_metadata = {
                            "triggered_by": "control_layer",
                            "control_transition": f"{old_control_state.name} -> {new_control_state.name}",
                            "sync_timestamp": time.time(),
                            "sync_reason": "auto_sync"
                        }
                        sync_metadata.update(metadata)
                        
                        # 修正：使用协调层锁保护状态转换
                        with self._coordination_lock:
                            success = self.coordination_state_manager.transition_to(target_coordination_state, sync_metadata)
                        
                        if success:
                            sync_result.triggered_transitions.append({
                                "from_state": current_coordination_state.name,
                                "to_state": target_coordination_state.name,
                                "layer": "coordination",
                                "metadata": sync_metadata
                            })
                        else:
                            sync_result.success = False
                            sync_result.errors.append(f"Failed to transition coordination state to {target_coordination_state.name}")
                    else:
                        self.logger.debug(
                            f"无需同步协调层状态: 当前状态 {current_coordination_state.name} "
                            f"不在控制层状态 {new_control_state.name} 的同步规则中"
                        )
                
                # 记录同步历史
                self.sync_history.append({
                    "type": "control_to_coordination",
                    "control_transition": f"{old_control_state.name} -> {new_control_state.name}",
                    "result": sync_result,
                    "timestamp": sync_result.timestamp
                })
                
            finally:
                self.sync_in_progress = False
                
        finally:
            if acquired:
                self._sync_lock.release()
        
        return sync_result
    
    def _execute_custom_callbacks_safe(self, callback_type: str, *args, **kwargs):
        """安全执行自定义回调"""
        try:
            with self._callback_lock:
                callbacks = self.custom_callbacks.get(callback_type, [])
                
            # 在锁外执行回调，避免死锁
            for callback in callbacks:
                try:
                    callback(*args, **kwargs)
                except Exception as e:
                    self.logger.warning(f"自定义回调执行失败 ({callback_type}): {e}")
        except Exception as e:
            self.logger.error(f"执行自定义回调失败: {e}")
    
    # ===== 对外接口 - 修正版 =====
    
    def transition_coordination_state(self, new_state: ClientLifecycleState,
                                    metadata: Optional[Dict[str, Any]] = None,
                                    force: bool = False) -> bool:
        """
        转换协调层状态 - 修正版
        
        Args:
            new_state: 新状态
            metadata: 转换元数据
            force: 是否强制转换
            
        Returns:
            bool: 转换是否成功
        """
        try:
            if metadata is None:
                metadata = {}
            
            metadata.update({
                "transition_time": time.time(),
                "initiated_by": "user",
                "component_id": self.component_id
            })
            
            with self._coordination_lock:
                return self.coordination_state_manager.transition_to(new_state, metadata, force)
            
        except Exception as e:
            self.logger.error(f"协调层状态转换失败: {e}")
            return False
    
    def transition_control_state(self, new_state: TrainingPhaseState,
                               metadata: Optional[Dict[str, Any]] = None,
                               force: bool = False) -> bool:
        """
        转换控制层状态 - 修正版
        
        Args:
            new_state: 新状态
            metadata: 转换元数据
            force: 是否强制转换
            
        Returns:
            bool: 转换是否成功
        """
        try:
            if metadata is None:
                metadata = {}
            
            metadata.update({
                "transition_time": time.time(),
                "initiated_by": "user",
                "component_id": self.component_id
            })
            
            with self._control_lock:
                return self.control_state_manager.transition_to(new_state, metadata, force)
            
        except Exception as e:
            self.logger.error(f"控制层状态转换失败: {e}")
            return False
    
    def get_coordination_state(self) -> ClientLifecycleState:
        """获取协调层状态"""
        return self.coordination_state_manager.get_current_state()
    
    def get_control_state(self) -> TrainingPhaseState:
        """获取控制层状态"""
        return self.control_state_manager.get_current_state()
    
    def get_coordination_state_duration(self) -> float:
        """获取协调层当前状态持续时间"""
        return self.coordination_state_manager.get_state_duration()
    
    def get_control_state_duration(self) -> float:
        """获取控制层当前状态持续时间"""
        return self.control_state_manager.get_state_duration()
    
    def get_overall_status(self) -> Dict[str, Any]:
        """获取整体状态信息 - 修正版"""
        try:
            coordination_stats = self.coordination_state_manager.get_state_statistics()
            control_stats = self.control_state_manager.get_state_statistics()
            sync_status = self._check_state_synchronization()
            
            return {
                "component_id": self.component_id,
                "coordination": {
                    "current_state": self.get_coordination_state().name,
                    "state_duration": self.get_coordination_state_duration(),
                    "statistics": coordination_stats
                },
                "control": {
                    "current_state": self.get_control_state().name,
                    "state_duration": self.get_control_state_duration(),
                    "statistics": control_stats
                },
                "synchronization": sync_status,
                "stats": self.stats,
                "auto_sync_enabled": self.enable_auto_sync,
                "timestamp": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"获取整体状态信息失败: {e}")
            return {
                "component_id": self.component_id,
                "error": str(e),
                "timestamp": time.time()
            }
    
    def _check_state_synchronization(self) -> Dict[str, Any]:
        """检查状态同步情况 - 修正版"""
        try:
            coordination_state = self.get_coordination_state()
            control_state = self.get_control_state()
            
            # 检查状态是否匹配
            is_synchronized = True
            sync_issues = []
            
            # 定义期望的状态匹配关系
            expected_matches = {
                ClientLifecycleState.TRAINING: [
                    TrainingPhaseState.PREPARING,
                    TrainingPhaseState.RUNNING,
                    TrainingPhaseState.PHASE_TRANSITION,
                    TrainingPhaseState.EPOCH_EXECUTING,
                    TrainingPhaseState.EVALUATING,
                    TrainingPhaseState.AGGREGATING
                ],
                ClientLifecycleState.ERROR: [TrainingPhaseState.FAILED],
                ClientLifecycleState.COMPLETED: [TrainingPhaseState.FINISHED],
                ClientLifecycleState.READY: [
                    TrainingPhaseState.PAUSED,
                    TrainingPhaseState.FINISHED,
                    TrainingPhaseState.UNINITIALIZED
                ]
            }
            
            if coordination_state in expected_matches:
                if control_state not in expected_matches[coordination_state]:
                    is_synchronized = False
                    sync_issues.append(
                        f"协调层状态 {coordination_state.name} 与控制层状态 {control_state.name} 不匹配"
                    )
            
            # 检查最近的同步历史
            recent_sync_failures = []
            current_time = time.time()
            for sync_record in list(self.sync_history)[-10:]:  # 检查最近10次同步
                if not sync_record["result"].success:
                    if current_time - sync_record["timestamp"] < 300:  # 5分钟内
                        recent_sync_failures.append(sync_record)
            
            return {
                "is_synchronized": is_synchronized,
                "sync_issues": sync_issues,
                "recent_sync_failures": len(recent_sync_failures),
                "last_sync_check": current_time,
                "sync_in_progress": self.sync_in_progress,
                "total_sync_records": len(self.sync_history),
                "sync_timeout_count": self.stats.get('sync_timeouts', 0)
            }
            
        except Exception as e:
            self.logger.error(f"检查状态同步失败: {e}")
            return {
                "error": str(e),
                "last_sync_check": time.time()
            }
    
    def force_state_sync(self) -> Dict[str, Any]:
        """
        强制状态同步 - 修正版
        
        Returns:
            Dict[str, Any]: 同步结果
        """
        try:
            self.logger.debug("执行强制状态同步")
            
            # 修正：使用超时机制避免死锁
            acquired = self._sync_lock.acquire(timeout=self._sync_timeout)
            if not acquired:
                return {
                    "success": False,
                    "error": "Sync lock timeout",
                    "timestamp": time.time()
                }
            
            try:
                coordination_state = self.get_coordination_state()
                control_state = self.get_control_state()
                
                sync_results = []
                
                # 尝试根据协调层状态同步控制层
                if coordination_state in self.coordination_to_control_rules:
                    control_transitions = self.coordination_to_control_rules[coordination_state]
                    if control_state in control_transitions:
                        target_control_state = control_transitions[control_state]
                        
                        with self._control_lock:
                            success = self.control_state_manager.transition_to(
                                target_control_state,
                                {
                                    "action": "force_sync",
                                    "triggered_by": "manual",
                                    "timestamp": time.time()
                                },
                                force=True
                            )
                        
                        sync_results.append({
                            "type": "coordination_to_control",
                            "from_state": control_state.name,
                            "to_state": target_control_state.name,
                            "success": success
                        })
                
                # 尝试根据控制层状态同步协调层
                if control_state in self.control_to_coordination_rules:
                    coordination_transitions = self.control_to_coordination_rules[control_state]
                    if coordination_state in coordination_transitions:
                        target_coordination_state = coordination_transitions[coordination_state]
                        
                        with self._coordination_lock:
                            success = self.coordination_state_manager.transition_to(
                                target_coordination_state,
                                {
                                    "action": "force_sync",
                                    "triggered_by": "manual",
                                    "timestamp": time.time()
                                },
                                force=True
                            )
                        
                        sync_results.append({
                            "type": "control_to_coordination",
                            "from_state": coordination_state.name,
                            "to_state": target_coordination_state.name,
                            "success": success
                        })
                
                return {
                    "success": all(result["success"] for result in sync_results),
                    "sync_results": sync_results,
                    "timestamp": time.time()
                }
                
            finally:
                self._sync_lock.release()
            
        except Exception as e:
            self.logger.error(f"强制状态同步失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            }
    
    def enable_auto_synchronization(self, enable: bool = True):
        """启用或禁用自动状态同步"""
        self.enable_auto_sync = enable
        self.logger.debug(f"自动状态同步已{'启用' if enable else '禁用'}")
    
    def register_custom_callback(self, callback_type: str, callback: Callable) -> bool:
        """
        注册自定义回调 - 修正版
        
        Args:
            callback_type: 回调类型 ('coordination_change', 'control_change', 'sync_success', 'sync_failure')
            callback: 回调函数
            
        Returns:
            bool: 注册是否成功
        """
        try:
            with self._callback_lock:
                if callback_type in self.custom_callbacks:
                    self.custom_callbacks[callback_type].append(callback)
                    self.logger.debug(f"注册自定义回调: {callback_type}")
                    return True
                else:
                    self.logger.error(f"未知的回调类型: {callback_type}")
                    return False
        except Exception as e:
            self.logger.error(f"注册自定义回调失败: {e}")
            return False
    
    def register_coordination_callback(self, callback: Callable, 
                                     callback_id: Optional[str] = None) -> str:
        """注册协调层状态回调"""
        return self.coordination_state_manager.register_callback(callback, callback_id=callback_id)
    
    def register_control_callback(self, callback: Callable, 
                                callback_id: Optional[str] = None) -> str:
        """注册控制层状态回调"""
        return self.control_state_manager.register_callback(callback, callback_id=callback_id)
    
    def cleanup(self):
        """清理资源 - 修正版"""
        try:
            self.logger.debug("开始清理层级状态管理器")
            
            # 禁用自动同步
            self.enable_auto_sync = False
            
            # 等待同步完成
            timeout = 5.0
            start_time = time.time()
            while self.sync_in_progress and (time.time() - start_time) < timeout:
                time.sleep(0.1)
            
            # 注销回调
            try:
                self.coordination_state_manager.unregister_callback("coordination_sync")
                self.control_state_manager.unregister_callback("control_sync")
            except Exception as e:
                self.logger.warning(f"注销回调失败: {e}")
            
            # 清理状态管理器
            if hasattr(self.coordination_state_manager, 'cleanup'):
                self.coordination_state_manager.cleanup()
            
            if hasattr(self.control_state_manager, 'cleanup'):
                self.control_state_manager.cleanup()
            
            # 清理历史记录和回调
            self.sync_history.clear()
            with self._callback_lock:
                self.custom_callbacks.clear()
            
            self.logger.debug("层级状态管理器清理完成")
            
        except Exception as e:
            self.logger.error(f"层级状态管理器清理失败: {e}")


def create_hierarchical_state_manager(execution_context: ExecutionContext, 
                                     component_id: str,
                                     max_history: int = 1000,
                                     enable_validation: bool = True) -> HierarchicalStateManager:
    """
    创建层级状态管理器的工厂函数 - 修正版
    
    Args:
        execution_context: 执行上下文
        component_id: 组件标识
        max_history: 最大历史记录数
        enable_validation: 是否启用状态转换验证
        
    Returns:
        HierarchicalStateManager: 层级状态管理器实例
    """
    try:
        return HierarchicalStateManager(
            execution_context=execution_context,
            component_id=component_id,
            max_history=max_history,
            enable_validation=enable_validation
        )
    except Exception as e:
        logger.error(f"创建层级状态管理器失败: {e}")
        raise StateError(f"Failed to create hierarchical state manager: {e}")


# 向后兼容的别名
HSM = HierarchicalStateManager