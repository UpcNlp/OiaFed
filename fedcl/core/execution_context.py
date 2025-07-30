# fedcl/core/execution_context.py
"""
ExecutionContext - 核心执行上下文管理器

提供跨组件的状态管理、资源管理、通信接口和事件系统支持。
支持线程安全、作用域隔离、状态持久化等核心功能。
"""

import json
import pickle
import threading
import time
import uuid
import weakref
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock, RLock
from typing import Any, Callable, Dict, List, Optional, Union

from loguru import logger
from omegaconf import DictConfig, OmegaConf


class ExecutionContextError(Exception):
    """ExecutionContext相关异常基类"""
    pass


class StateError(ExecutionContextError):
    """状态管理相关异常"""
    pass


class ResourceError(ExecutionContextError):
    """资源管理相关异常"""
    pass


class CommunicationError(ExecutionContextError):
    """通信相关异常"""
    pass


class ConfigurationError(ExecutionContextError):
    """配置相关异常"""
    pass


class MetricValue:
    """度量值包装器，支持时间戳和元数据"""
    
    def __init__(self, value: float, step: Optional[int] = None, timestamp: Optional[float] = None):
        self.value = value
        self.step = step or 0
        self.timestamp = timestamp or time.time()
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'value': self.value,
            'step': self.step,
            'timestamp': self.timestamp
        }


class EventSubscription:
    """事件订阅管理器"""
    
    def __init__(self, subscription_id: str, event: str, callback: Callable, weak_ref: bool = True):
        self.subscription_id = subscription_id
        self.event = event
        if weak_ref:
            try:
                self.callback = weakref.ref(callback)
            except TypeError:
                # 不能创建弱引用的对象（如函数），直接存储
                self.callback = callback
                self.weak_ref = False
            else:
                self.weak_ref = True
        else:
            self.callback = callback
            self.weak_ref = False
    
    def get_callback(self) -> Optional[Callable]:
        if self.weak_ref:
            return self.callback()
        return self.callback
    
    def is_valid(self) -> bool:
        return self.get_callback() is not None


class ExecutionContext:
    """
    执行上下文管理器 - FedCL框架的核心状态管理组件
    
    提供跨组件的状态管理、资源管理、通信接口和事件系统。
    支持线程安全操作、作用域隔离、状态持久化等核心功能。
    
    主要功能：
    - 多作用域状态管理（global, client, task等）
    - 辅助模型注册和管理
    - 度量收集和历史记录
    - 配置访问和动态更新
    - 跨组件通信接口
    - 资源生命周期管理
    - 事件发布订阅系统
    - 状态持久化和恢复
    """
    
    def __init__(
        self,
        config: DictConfig,
        experiment_id: str,
        communication_manager: Optional[Any] = None,
        metrics_logger: Optional[Any] = None
    ):
        """
        初始化执行上下文
        
        Args:
            config: 实验配置
            experiment_id: 实验唯一标识
            communication_manager: 可选的通信管理器
            metrics_logger: 可选的度量记录器
        """
        self.config = config
        self.experiment_id = experiment_id
        self.communication_manager = communication_manager
        self.metrics_logger = metrics_logger
        
        # 线程安全锁
        self._state_lock = RLock()  # 状态管理锁（可重入）
        self._metrics_lock = Lock()  # 度量管理锁
        self._resources_lock = Lock()  # 资源管理锁
        self._events_lock = Lock()  # 事件系统锁
        self._models_lock = Lock()  # 模型管理锁
        
        # 状态存储 - 按作用域组织
        self._global_state: Dict[str, Any] = {}
        self._local_states: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # 模型存储
        self._auxiliary_models: Dict[str, Any] = {}
        
        # 度量存储 - 按作用域和名称组织
        self._metrics: Dict[str, Dict[str, List[MetricValue]]] = defaultdict(lambda: defaultdict(list))
        
        # 资源存储
        self._resources: Dict[str, Any] = {}
        self._resource_cleanup_callbacks: Dict[str, Callable] = {}
        
        # 事件系统
        self._event_subscribers: Dict[str, List[EventSubscription]] = defaultdict(list)
        self._event_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="event-")
        
        # 状态监控
        self._state_change_listeners: Dict[str, List[Callable]] = defaultdict(list)
        self._state_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # 配置缓存
        self._config_cache: Dict[str, Any] = {}
        self._config_cache_lock = Lock()
        
        # 性能监控
        self._operation_times: Dict[str, List[float]] = defaultdict(list)
        self._max_history_length = 1000
        
        # 初始化标志
        self._initialized = False
        self._cleanup_started = False
        
        logger.info(f"ExecutionContext初始化 - 实验ID: {experiment_id}")
        
        # 从配置初始化
        self._init_from_config()
    
    def _init_from_config(self) -> None:
        """从配置初始化上下文设置"""
        try:
            # 设置状态存储配置
            state_config = OmegaConf.select(self.config, 'state_storage', default={})
            self._max_state_size = state_config.get('max_size', 10000) if state_config else 10000
            self._state_cleanup_interval = state_config.get('cleanup_interval', 3600) if state_config else 3600
            
            # 设置事件系统配置
            event_config = OmegaConf.select(self.config, 'event_system', default={})
            self._max_event_queue_size = event_config.get('max_queue_size', 1000) if event_config else 1000
            self._event_timeout = event_config.get('timeout', 30.0) if event_config else 30.0
            
            # 设置度量系统配置
            metrics_config = OmegaConf.select(self.config, 'metrics', default={})
            self._max_metrics_per_name = metrics_config.get('max_per_name', 10000) if metrics_config else 10000
            
            logger.debug("ExecutionContext配置初始化完成")
            
        except Exception as e:
            logger.error(f"配置初始化失败: {e}")
            raise ConfigurationError(f"配置初始化失败: {e}")
    
    def _measure_operation_time(self, operation_name: str):
        """性能监控装饰器上下文管理器"""
        class TimeMeasurer:
            def __init__(self, context: 'ExecutionContext', op_name: str):
                self.context = context
                self.op_name = op_name
                self.start_time = None
            
            def __enter__(self):
                self.start_time = time.perf_counter()
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.start_time:
                    duration = time.perf_counter() - self.start_time
                    times = self.context._operation_times[self.op_name]
                    times.append(duration)
                    if len(times) > self.context._max_history_length:
                        times.pop(0)
        
        return TimeMeasurer(self, operation_name)
    
    # ==================== 状态管理接口 ====================
    
    def get_state(self, key: str, scope: str = "global") -> Any:
        """
        获取指定作用域的状态值
        
        Args:
            key: 状态键名
            scope: 作用域（global, client:{id}, task:{id}等）
            
        Returns:
            状态值，不存在时返回None
            
        Raises:
            StateError: 状态访问异常
        """
        with self._measure_operation_time("get_state"):
            try:
                with self._state_lock:
                    if scope == "global":
                        return self._global_state.get(key)
                    else:
                        return self._local_states[scope].get(key)
            except Exception as e:
                logger.error(f"获取状态失败 - key: {key}, scope: {scope}, error: {e}")
                raise StateError(f"获取状态失败: {e}")
    
    def set_state(self, key: str, value: Any, scope: str = "global") -> None:
        """
        设置指定作用域的状态值
        
        Args:
            key: 状态键名
            value: 状态值
            scope: 作用域（global, client:{id}, task:{id}等）
            
        Raises:
            StateError: 状态设置异常
        """
        with self._measure_operation_time("set_state"):
            try:
                with self._state_lock:
                    # 获取旧值用于记录历史（不调用get_state避免重复计时）
                    if scope == "global":
                        old_value = self._global_state.get(key)
                        self._global_state[key] = value
                    else:
                        old_value = self._local_states[scope].get(key)
                        self._local_states[scope][key] = value
                    
                    # 记录状态变化历史
                    history_key = f"{scope}:{key}"
                    self._state_history[history_key].append({
                        'old_value': old_value,
                        'new_value': value,
                        'timestamp': time.time()
                    })
                    
                    # 触发状态变化监听器
                    self._notify_state_change(key, old_value, value, scope)
                    
                logger.debug(f"状态已设置 - key: {key}, scope: {scope}")
                
            except Exception as e:
                logger.error(f"设置状态失败 - key: {key}, scope: {scope}, error: {e}")
                raise StateError(f"设置状态失败: {e}")
    
    def has_state(self, key: str, scope: str = "global") -> bool:
        """
        检查指定作用域是否存在状态键
        
        Args:
            key: 状态键名
            scope: 作用域
            
        Returns:
            是否存在该状态键
        """
        with self._state_lock:
            if scope == "global":
                return key in self._global_state
            else:
                return key in self._local_states[scope]
    
    def clear_state(self, scope: str = "global") -> None:
        """
        清除指定作用域的所有状态
        
        Args:
            scope: 作用域，"global"表示全局状态，"all"表示所有状态
            
        Raises:
            StateError: 状态清除异常
        """
        try:
            with self._state_lock:
                if scope == "global":
                    self._global_state.clear()
                    logger.info("全局状态已清除")
                elif scope == "all":
                    self._global_state.clear()
                    self._local_states.clear()
                    self._state_history.clear()
                    logger.info("所有状态已清除")
                else:
                    if scope in self._local_states:
                        self._local_states[scope].clear()
                        logger.info(f"作用域 {scope} 状态已清除")
                        
        except Exception as e:
            logger.error(f"清除状态失败 - scope: {scope}, error: {e}")
            raise StateError(f"清除状态失败: {e}")
    
    def get_all_states(self, scope: str = "global") -> Dict[str, Any]:
        """
        获取指定作用域的所有状态
        
        Args:
            scope: 作用域
            
        Returns:
            状态字典的副本
        """
        with self._state_lock:
            if scope == "global":
                return self._global_state.copy()
            else:
                return self._local_states[scope].copy()
    
    def _notify_state_change(self, key: str, old_value: Any, new_value: Any, scope: str) -> None:
        """通知状态变化监听器"""
        listeners = self._state_change_listeners.get(f"{scope}:{key}", [])
        for listener in listeners:
            try:
                listener(key, old_value, new_value, scope)
            except Exception as e:
                logger.warning(f"状态变化监听器执行失败: {e}")
    
    # ==================== 模型管理接口 ====================
    
    def get_model(self, name: str) -> Any:
        """
        获取辅助模型
        
        Args:
            name: 模型名称
            
        Returns:
            模型实例，不存在时返回None
            
        Raises:
            ResourceError: 模型访问异常
        """
        try:
            with self._models_lock:
                return self._auxiliary_models.get(name)
        except Exception as e:
            logger.error(f"获取辅助模型失败 - name: {name}, error: {e}")
            raise ResourceError(f"获取辅助模型失败: {e}")
    
    def register_auxiliary_model(self, name: str, model: Any) -> None:
        """
        注册辅助模型
        
        Args:
            name: 模型名称
            model: 模型实例
            
        Raises:
            ResourceError: 模型注册异常
        """
        try:
            with self._models_lock:
                self._auxiliary_models[name] = model
                logger.info(f"辅助模型已注册: {name}")
                
        except Exception as e:
            logger.error(f"注册辅助模型失败 - name: {name}, error: {e}")
            raise ResourceError(f"注册辅助模型失败: {e}")
    
    def unregister_auxiliary_model(self, name: str) -> None:
        """
        注销辅助模型
        
        Args:
            name: 模型名称
        """
        with self._models_lock:
            if name in self._auxiliary_models:
                del self._auxiliary_models[name]
                logger.info(f"辅助模型已注销: {name}")
    
    def list_auxiliary_models(self) -> List[str]:
        """
        列出所有已注册的辅助模型名称
        
        Returns:
            模型名称列表
        """
        with self._models_lock:
            return list(self._auxiliary_models.keys())
    
    # ==================== 度量管理接口 ====================
    
    def log_metric(
        self,
        name: str,
        value: float,
        step: Optional[int] = None,
        scope: str = "global"
    ) -> None:
        """
        记录度量值
        
        Args:
            name: 度量名称
            value: 度量值
            step: 可选的步骤编号
            scope: 作用域
            
        Raises:
            ValueError: 度量值无效
        """
        if not isinstance(value, (int, float)):
            raise ValueError(f"度量值必须为数字类型，得到: {type(value)}")
            
        try:
            with self._metrics_lock:
                metric_value = MetricValue(float(value), step)
                metrics_for_scope = self._metrics[scope]
                metrics_for_name = metrics_for_scope[name]
                
                metrics_for_name.append(metric_value)
                
                # 限制历史记录长度
                if len(metrics_for_name) > self._max_metrics_per_name:
                    metrics_for_name.pop(0)
                
                # 同时使用外部度量记录器
                if self.metrics_logger is not None:
                    self.metrics_logger.log_scalar(name, value, step or 0)
                    
                logger.debug(f"度量已记录 - name: {name}, value: {value}, scope: {scope}")
                
        except Exception as e:
            logger.error(f"记录度量失败 - name: {name}, error: {e}")
    
    def get_metrics(self, scope: str = "global") -> Dict[str, List[float]]:
        """
        获取指定作用域的所有度量值
        
        Args:
            scope: 作用域
            
        Returns:
            度量名称到值列表的映射
        """
        with self._metrics_lock:
            result = {}
            for name, metrics in self._metrics[scope].items():
                result[name] = [m.value for m in metrics]
            return result
    
    def get_metric_history(self, name: str, scope: str = "global") -> List[float]:
        """
        获取指定度量的历史值
        
        Args:
            name: 度量名称
            scope: 作用域
            
        Returns:
            度量值历史列表
        """
        with self._metrics_lock:
            metrics = self._metrics[scope].get(name, [])
            return [m.value for m in metrics]
    
    # ==================== 配置访问接口 ====================
    
    def get_config(self, path: str, default: Any = None) -> Any:
        """
        获取配置值，支持点分路径
        
        Args:
            path: 配置路径，如 "model.learning_rate"
            default: 默认值
            
        Returns:
            配置值
            
        Raises:
            ConfigurationError: 配置访问异常
        """
        with self._measure_operation_time("get_config"):
            try:
                # 先检查简单缓存（不考虑默认值）
                simple_cache_key = path
                with self._config_cache_lock:
                    if simple_cache_key in self._config_cache:
                        cached_value = self._config_cache[simple_cache_key]
                        return cached_value if cached_value is not None else default
                
                # 从配置中获取
                value = OmegaConf.select(self.config, path)
                if value is None:
                    value = default
                else:
                    # 只缓存非None的配置值
                    with self._config_cache_lock:
                        self._config_cache[simple_cache_key] = value
                
                return value
                
            except Exception as e:
                logger.error(f"获取配置失败 - path: {path}, error: {e}")
                raise ConfigurationError(f"获取配置失败: {e}")
    
    def update_config(self, path: str, value: Any) -> None:
        """
        更新配置值
        
        Args:
            path: 配置路径
            value: 新值
            
        Raises:
            ConfigurationError: 配置更新异常
        """
        try:
            OmegaConf.update(self.config, path, value)
            
            # 清除相关缓存
            with self._config_cache_lock:
                # 清除精确匹配和前缀匹配的缓存
                keys_to_remove = []
                for cache_key in self._config_cache.keys():
                    if cache_key == path or cache_key.startswith(path + "."):
                        keys_to_remove.append(cache_key)
                
                for key in keys_to_remove:
                    del self._config_cache[key]
            
            logger.info(f"配置已更新 - path: {path}")
            
        except Exception as e:
            logger.error(f"更新配置失败 - path: {path}, error: {e}")
            raise ConfigurationError(f"更新配置失败: {e}")
    
    # ==================== 通信接口 ====================
    
    def send_data(self, target: str, data: Any, data_type: str) -> None:
        """
        发送数据到目标
        
        Args:
            target: 目标标识
            data: 要发送的数据
            data_type: 数据类型
            
        Raises:
            CommunicationError: 通信异常
        """
        if self.communication_manager is None:
            raise CommunicationError("CommunicationManager不可用")
        
        try:
            self.communication_manager.send_data(
                source=self.experiment_id,
                target=target,
                data=data,
                data_type=data_type
            )
            logger.debug(f"数据已发送 - target: {target}, type: {data_type}")
            
        except Exception as e:
            logger.error(f"发送数据失败 - target: {target}, error: {e}")
            raise CommunicationError(f"发送数据失败: {e}")
    
    def request_data(self, source: str, data_type: str, timeout: float = 30.0) -> Any:
        """
        从源请求数据
        
        Args:
            source: 数据源标识
            data_type: 数据类型
            timeout: 超时时间（秒）
            
        Returns:
            请求到的数据
            
        Raises:
            CommunicationError: 通信异常
        """
        if self.communication_manager is None:
            raise CommunicationError("CommunicationManager不可用")
        
        try:
            return self.communication_manager.request_data(
                requester=self.experiment_id,
                source=source,
                data_type=data_type,
                timeout=timeout
            )
            
        except Exception as e:
            logger.error(f"请求数据失败 - source: {source}, error: {e}")
            raise CommunicationError(f"请求数据失败: {e}")
    
    def broadcast_data(self, data: Any, data_type: str, targets: List[str]) -> None:
        """
        广播数据到多个目标
        
        Args:
            data: 要广播的数据
            data_type: 数据类型
            targets: 目标列表
            
        Raises:
            CommunicationError: 通信异常
        """
        if self.communication_manager is None:
            raise CommunicationError("CommunicationManager不可用")
        
        try:
            self.communication_manager.broadcast_data(
                source=self.experiment_id,
                data=data,
                data_type=data_type,
                targets=targets
            )
            logger.debug(f"数据已广播 - targets: {len(targets)}, type: {data_type}")
            
        except Exception as e:
            logger.error(f"广播数据失败 - error: {e}")
            raise CommunicationError(f"广播数据失败: {e}")
    
    # ==================== 资源管理接口 ====================
    
    def register_resource(self, name: str, resource: Any) -> None:
        """
        注册资源
        
        Args:
            name: 资源名称
            resource: 资源实例
            
        Raises:
            ResourceError: 资源注册异常
        """
        try:
            with self._resources_lock:
                self._resources[name] = resource
                
                # 按优先级查找清理方法
                cleanup_method = None
                for method_name in ['cleanup', 'close', 'destroy']:
                    if hasattr(resource, method_name):
                        cleanup_method = getattr(resource, method_name)
                        break
                
                if cleanup_method:
                    self._resource_cleanup_callbacks[name] = cleanup_method
                
                logger.info(f"资源已注册: {name}")
                
        except Exception as e:
            logger.error(f"注册资源失败 - name: {name}, error: {e}")
            raise ResourceError(f"注册资源失败: {e}")
    
    def get_resource(self, name: str) -> Any:
        """
        获取资源
        
        Args:
            name: 资源名称
            
        Returns:
            资源实例，不存在时返回None
            
        Raises:
            ResourceError: 资源访问异常
        """
        try:
            with self._resources_lock:
                return self._resources.get(name)
                
        except Exception as e:
            logger.error(f"获取资源失败 - name: {name}, error: {e}")
            raise ResourceError(f"获取资源失败: {e}")
    
    def cleanup_resources(self) -> None:
        """清理所有注册的资源"""
        with self._resources_lock:
            for name, cleanup_callback in self._resource_cleanup_callbacks.items():
                try:
                    cleanup_callback()
                    logger.info(f"资源已清理: {name}")
                except Exception as e:
                    logger.warning(f"清理资源失败 - name: {name}, error: {e}")
            
            self._resources.clear()
            self._resource_cleanup_callbacks.clear()
    
    # ==================== 事件系统接口 ====================
    
    def emit_event(self, event: str, data: Any = None) -> None:
        """
        发布事件
        
        Args:
            event: 事件名称
            data: 事件数据
        """
        with self._events_lock:
            subscribers = self._event_subscribers.get(event, [])
            # 过滤无效的订阅者
            valid_subscribers = [s for s in subscribers if s.is_valid()]
            self._event_subscribers[event] = valid_subscribers
        
        # 异步执行事件回调
        for subscription in valid_subscribers:
            callback = subscription.get_callback()
            if callback:
                self._event_executor.submit(self._execute_event_callback, callback, event, data)
        
        logger.debug(f"事件已发布 - event: {event}, subscribers: {len(valid_subscribers)}")
    
    def _execute_event_callback(self, callback: Callable, event: str, data: Any) -> None:
        """执行事件回调"""
        try:
            callback(event, data)
        except Exception as e:
            logger.warning(f"事件回调执行失败 - event: {event}, error: {e}")
    
    def subscribe_event(self, event: str, callback: Callable) -> str:
        """
        订阅事件
        
        Args:
            event: 事件名称
            callback: 回调函数
            
        Returns:
            订阅ID，用于取消订阅
        """
        subscription_id = str(uuid.uuid4())
        subscription = EventSubscription(subscription_id, event, callback)
        
        with self._events_lock:
            self._event_subscribers[event].append(subscription)
        
        logger.debug(f"事件已订阅 - event: {event}, subscription_id: {subscription_id}")
        return subscription_id
    
    def unsubscribe_event(self, subscription_id: str) -> None:
        """
        取消事件订阅
        
        Args:
            subscription_id: 订阅ID
        """
        with self._events_lock:
            for event, subscribers in self._event_subscribers.items():
                original_length = len(subscribers)
                subscribers[:] = [s for s in subscribers if s.subscription_id != subscription_id]
                if len(subscribers) < original_length:
                    logger.debug(f"事件订阅已取消 - subscription_id: {subscription_id}")
                    break
    
    # ==================== 生命周期管理 ====================
    
    def initialize(self) -> None:
        """初始化执行上下文"""
        if self._initialized:
            logger.warning("ExecutionContext已经初始化")
            return
        
        try:
            # 初始化各个子系统
            logger.info(f"开始初始化ExecutionContext - 实验ID: {self.experiment_id}")
            
            # 设置初始状态
            self.set_state("initialized_at", time.time())
            self.set_state("experiment_id", self.experiment_id)
            
            # 发布初始化事件
            self.emit_event("context_initialized", {"experiment_id": self.experiment_id})
            
            self._initialized = True
            logger.info("ExecutionContext初始化完成")
            
        except Exception as e:
            logger.error(f"ExecutionContext初始化失败: {e}")
            raise ExecutionContextError(f"初始化失败: {e}")
    
    def cleanup(self) -> None:
        """清理执行上下文"""
        if self._cleanup_started:
            logger.warning("ExecutionContext清理已在进行中")
            return
        
        self._cleanup_started = True
        logger.info(f"开始清理ExecutionContext - 实验ID: {self.experiment_id}")
        
        try:
            # 发布清理事件
            self.emit_event("context_cleanup_started", {"experiment_id": self.experiment_id})
            
            # 清理资源
            self.cleanup_resources()
            
            # 关闭事件执行器
            self._event_executor.shutdown(wait=True)
            
            # 清理所有状态
            self.clear_state("all")
            
            # 清理度量数据
            with self._metrics_lock:
                self._metrics.clear()
            
            # 清理事件订阅
            with self._events_lock:
                self._event_subscribers.clear()
            
            # 清理模型
            with self._models_lock:
                self._auxiliary_models.clear()
            
            # 清理配置缓存
            with self._config_cache_lock:
                self._config_cache.clear()
            
            logger.info("ExecutionContext清理完成")
            
        except Exception as e:
            logger.error(f"ExecutionContext清理失败: {e}")
    
    def save_state(self, path: Path) -> None:
        """
        保存执行上下文状态到文件
        
        Args:
            path: 保存路径
            
        Raises:
            StateError: 状态保存异常
        """
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # 收集要保存的状态
            state_data = {
                'experiment_id': self.experiment_id,
                'global_state': self._global_state,
                'local_states': dict(self._local_states),
                'metrics': {
                    scope: {
                        name: [m.to_dict() for m in metrics]
                        for name, metrics in scope_metrics.items()
                    }
                    for scope, scope_metrics in self._metrics.items()
                },
                'auxiliary_models': list(self._auxiliary_models.keys()),  # 只保存名称
                'timestamp': time.time()
            }
            
            # 序列化并保存
            with open(path, 'wb') as f:
                pickle.dump(state_data, f)
            
            logger.info(f"执行上下文状态已保存到: {path}")
            
        except Exception as e:
            logger.error(f"保存状态失败 - path: {path}, error: {e}")
            raise StateError(f"保存状态失败: {e}")
    
    def load_state(self, path: Path) -> None:
        """
        从文件加载执行上下文状态
        
        Args:
            path: 加载路径
            
        Raises:
            StateError: 状态加载异常
        """
        try:
            if not path.exists():
                raise FileNotFoundError(f"状态文件不存在: {path}")
            
            with open(path, 'rb') as f:
                state_data = pickle.load(f)
            
            # 验证实验ID
            if state_data['experiment_id'] != self.experiment_id:
                logger.warning(f"实验ID不匹配: {state_data['experiment_id']} != {self.experiment_id}")
            
            # 恢复状态
            with self._state_lock:
                self._global_state.update(state_data['global_state'])
                for scope, local_state in state_data['local_states'].items():
                    self._local_states[scope].update(local_state)
            
            # 恢复度量数据
            with self._metrics_lock:
                for scope, scope_metrics in state_data['metrics'].items():
                    for name, metric_dicts in scope_metrics.items():
                        metrics = [MetricValue(m['value'], m['step'], m['timestamp']) 
                                 for m in metric_dicts]
                        self._metrics[scope][name].extend(metrics)
            
            logger.info(f"执行上下文状态已从文件加载: {path}")
            
        except Exception as e:
            logger.error(f"加载状态失败 - path: {path}, error: {e}")
            raise StateError(f"加载状态失败: {e}")
    
    # ==================== 监控和调试接口 ====================
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        stats = {}
        for operation, times in self._operation_times.items():
            if times:
                stats[operation] = {
                    'count': len(times),
                    'avg_time': sum(times) / len(times),
                    'max_time': max(times),
                    'min_time': min(times),
                    'total_time': sum(times)
                }
        return stats
    
    def get_memory_usage(self) -> Dict[str, int]:
        """获取内存使用统计"""
        import sys
        
        def get_size(obj):
            size = sys.getsizeof(obj)
            if isinstance(obj, dict):
                size += sum([get_size(v) + get_size(k) for k, v in obj.items()])
            elif isinstance(obj, (list, tuple, set, frozenset)):
                size += sum([get_size(i) for i in obj])
            return size
        
        return {
            'global_state': get_size(self._global_state),
            'local_states': get_size(self._local_states),
            'metrics': get_size(self._metrics),
            'auxiliary_models': get_size(self._auxiliary_models),
            'resources': get_size(self._resources),
            'event_subscribers': get_size(self._event_subscribers)
        }
    
    def __repr__(self) -> str:
        return (f"ExecutionContext(experiment_id='{self.experiment_id}', "
                f"initialized={self._initialized}, "
                f"global_states={len(self._global_state)}, "
                f"local_scopes={len(self._local_states)}, "
                f"auxiliary_models={len(self._auxiliary_models)})")