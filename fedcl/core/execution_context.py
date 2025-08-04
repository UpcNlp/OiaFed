# fedcl/core/execution_context.py (简化版本)
"""
ExecutionContext简化版本 - 专注多Learner支持的核心功能

简化原有复杂的ExecutionContext，专注于：
- 核心状态管理
- Learner间特征共享
- 基础度量记录
- 简单事件系统
"""

import time
import threading
from collections import defaultdict, deque
from threading import Lock, RLock
from typing import Any, Callable, Dict, List, Optional, Union
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf


class ExecutionContextError(Exception):
    """ExecutionContext相关异常基类"""
    pass


class ExecutionContext:
    """
    简化的执行上下文管理器 - 多Learner支持版本
    
    核心功能：
    - 多作用域状态管理
    - Learner间特征共享
    - 基础度量收集
    - 简单事件系统
    - 配置访问
    """
    
    def __init__(self, config: DictConfig, experiment_id: str):
        """
        初始化简化执行上下文
        
        Args:
            config: 实验配置
            experiment_id: 实验唯一标识
        """
        self.config = config
        self.experiment_id = experiment_id
        
        # 线程安全锁
        self._state_lock = RLock()
        self._features_lock = RLock()
        self._metrics_lock = Lock()
        self._events_lock = Lock()
        
        # 状态存储 - 按作用域组织
        self._global_state: Dict[str, Any] = {}
        self._scoped_states: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Learner特征共享
        self._learner_features: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)
        self._feature_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
        
        # 度量存储
        self._metrics: Dict[str, List[float]] = defaultdict(list)
        self._learner_metrics: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        
        # 简单事件系统
        self._event_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        # 配置缓存
        self._config_cache: Dict[str, Any] = {}
        
        logger.debug(f"SimplifiedExecutionContext initialized - ID: {experiment_id}")
    
    # ==================== 状态管理接口 ====================
    
    def set_state(self, key: str, value: Any, scope: str = "global") -> None:
        """
        设置状态值
        
        Args:
            key: 状态键名
            value: 状态值
            scope: 作用域
        """
        with self._state_lock:
            if scope == "global":
                self._global_state[key] = value
            else:
                self._scoped_states[scope][key] = value
        
        logger.debug(f"State set - key: {key}, scope: {scope}")
    
    def get_state(self, key: str, scope: str = "global", default: Any = None) -> Any:
        """
        获取状态值
        
        Args:
            key: 状态键名
            scope: 作用域
            default: 默认值
            
        Returns:
            状态值
        """
        with self._state_lock:
            if scope == "global":
                return self._global_state.get(key, default)
            else:
                return self._scoped_states[scope].get(key, default)
    
    def update_state(self, state_dict: Dict[str, Any], scope: str = "global") -> None:
        """
        批量更新状态
        
        Args:
            state_dict: 状态字典
            scope: 作用域
        """
        with self._state_lock:
            if scope == "global":
                self._global_state.update(state_dict)
            else:
                self._scoped_states[scope].update(state_dict)
        
        logger.debug(f"Batch state update - {len(state_dict)} keys, scope: {scope}")
    
    def clear_state(self, scope: str = "global") -> None:
        """
        清除状态
        
        Args:
            scope: 作用域，"all"表示所有作用域
        """
        with self._state_lock:
            if scope == "global":
                self._global_state.clear()
            elif scope == "all":
                self._global_state.clear()
                self._scoped_states.clear()
            else:
                if scope in self._scoped_states:
                    self._scoped_states[scope].clear()
        
        logger.debug(f"State cleared - scope: {scope}")
    
    def has_state(self, key: str, scope: str = "global") -> bool:
        """
        检查状态是否存在
        
        Args:
            key: 状态键名
            scope: 作用域
            
        Returns:
            是否存在
        """
        with self._state_lock:
            if scope == "global":
                return key in self._global_state
            else:
                return key in self._scoped_states[scope]
    
    def get_all_states(self, scope: str = "global") -> Dict[str, Any]:
        """
        获取指定作用域的所有状态
        
        Args:
            scope: 作用域
            
        Returns:
            状态字典
        """
        with self._state_lock:
            if scope == "global":
                return self._global_state.copy()
            else:
                return self._scoped_states[scope].copy()
    
    # ==================== Learner特征共享接口 ====================
    
    def share_features(self, learner_id: str, features: Dict[str, torch.Tensor]) -> None:
        """
        共享learner特征
        
        Args:
            learner_id: learner ID
            features: 特征字典
        """
        with self._features_lock:
            # 更新当前特征
            self._learner_features[learner_id] = {
                name: feature.clone().detach() for name, feature in features.items()
            }
            
            # 记录特征历史
            self._feature_history[learner_id].append({
                'timestamp': time.time(),
                'features': {name: feature.clone().detach() for name, feature in features.items()}
            })
        
        logger.debug(f"Features shared by learner {learner_id} - {len(features)} features")
    
    def get_shared_features(self, target_learner: str, source_learner: str) -> Dict[str, torch.Tensor]:
        """
        获取共享特征
        
        Args:
            target_learner: 目标learner ID
            source_learner: 源learner ID
            
        Returns:
            特征字典
        """
        with self._features_lock:
            features = self._learner_features.get(source_learner, {})
            if features:
                # 返回特征的副本
                return {name: feature.clone() for name, feature in features.items()}
            else:
                logger.warning(f"No features available from learner {source_learner}")
                return {}
    
    def get_all_shared_features(self, target_learner: str) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        获取所有共享特征
        
        Args:
            target_learner: 目标learner ID
            
        Returns:
            特征字典 {source_learner: {feature_name: feature_tensor}}
        """
        with self._features_lock:
            all_features = {}
            for source_learner, features in self._learner_features.items():
                if source_learner != target_learner:  # 不包括自己
                    all_features[source_learner] = {
                        name: feature.clone() for name, feature in features.items()
                    }
            return all_features
    
    def clear_learner_features(self, learner_id: str = None) -> None:
        """
        清除learner特征
        
        Args:
            learner_id: learner ID，None表示清除所有
        """
        with self._features_lock:
            if learner_id is None:
                self._learner_features.clear()
                self._feature_history.clear()
                logger.debug("All learner features cleared")
            else:
                if learner_id in self._learner_features:
                    del self._learner_features[learner_id]
                if learner_id in self._feature_history:
                    del self._feature_history[learner_id]
                logger.debug(f"Features cleared for learner {learner_id}")
    
    def get_feature_history(self, learner_id: str) -> List[Dict[str, Any]]:
        """
        获取特征历史
        
        Args:
            learner_id: learner ID
            
        Returns:
            特征历史列表
        """
        with self._features_lock:
            return list(self._feature_history.get(learner_id, []))
    
    # ==================== 度量管理接口 ====================
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None, 
                  learner_id: Optional[str] = None) -> None:
        """
        记录度量值
        
        Args:
            name: 度量名称
            value: 度量值
            step: 步骤编号（可选）
            learner_id: learner ID（可选）
        """
        if not isinstance(value, (int, float)):
            raise ValueError(f"Metric value must be numeric, got: {type(value)}")
        
        with self._metrics_lock:
            if learner_id is None:
                # 全局度量
                self._metrics[name].append(float(value))
            else:
                # learner特定度量
                self._learner_metrics[learner_id][name].append(float(value))
        
        logger.debug(f"Metric logged - name: {name}, value: {value}, learner: {learner_id}")
    
    def log_multi_learner_metrics(self, learner_id: str, metrics: Dict[str, float]) -> None:
        """
        记录多learner指标
        
        Args:
            learner_id: learner ID
            metrics: 指标字典
        """
        with self._metrics_lock:
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    self._learner_metrics[learner_id][name].append(float(value))
        
        logger.debug(f"Multi-learner metrics logged for {learner_id} - {len(metrics)} metrics")
    
    def get_metrics(self, name: str = None, learner_id: str = None) -> Union[List[float], Dict[str, List[float]]]:
        """
        获取度量值
        
        Args:
            name: 度量名称，None表示所有度量
            learner_id: learner ID，None表示全局度量
            
        Returns:
            度量值或度量字典
        """
        with self._metrics_lock:
            if learner_id is None:
                # 全局度量
                if name is None:
                    return dict(self._metrics)
                else:
                    return self._metrics.get(name, [])
            else:
                # learner特定度量
                learner_metrics = self._learner_metrics.get(learner_id, {})
                if name is None:
                    return dict(learner_metrics)
                else:
                    return learner_metrics.get(name, [])
    
    def get_all_learner_metrics(self) -> Dict[str, Dict[str, List[float]]]:
        """
        获取所有learner的度量
        
        Returns:
            Dict[learner_id, Dict[metric_name, values]]
        """
        with self._metrics_lock:
            return {
                learner_id: dict(metrics) 
                for learner_id, metrics in self._learner_metrics.items()
            }
    
    def clear_metrics(self, learner_id: str = None) -> None:
        """
        清除度量
        
        Args:
            learner_id: learner ID，None表示清除所有
        """
        with self._metrics_lock:
            if learner_id is None:
                self._metrics.clear()
                self._learner_metrics.clear()
                logger.debug("All metrics cleared")
            else:
                if learner_id in self._learner_metrics:
                    del self._learner_metrics[learner_id]
                logger.debug(f"Metrics cleared for learner {learner_id}")
    
    # ==================== 配置访问接口 ====================
    
    def get_config(self, path: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            path: 配置路径
            default: 默认值
            
        Returns:
            配置值
        """
        # 简单缓存
        if path in self._config_cache:
            return self._config_cache[path]
        
        try:
            value = OmegaConf.select(self.config, path)
            if value is None:
                value = default
            else:
                self._config_cache[path] = value
            return value
        except Exception as e:
            logger.error(f"Failed to get config {path}: {e}")
            return default
    
    def update_config(self, path: str, value: Any) -> None:
        """
        更新配置值
        
        Args:
            path: 配置路径
            value: 新值
        """
        try:
            OmegaConf.update(self.config, path, value)
            # 清除缓存
            self._config_cache.pop(path, None)
            logger.debug(f"Config updated - path: {path}")
        except Exception as e:
            logger.error(f"Failed to update config {path}: {e}")
            raise ExecutionContextError(f"Config update failed: {e}")
    
    # ==================== 简单事件系统 ====================
    
    def emit_event(self, event: str, data: Any = None) -> None:
        """
        发布事件
        
        Args:
            event: 事件名称
            data: 事件数据
        """
        with self._events_lock:
            callbacks = self._event_callbacks.get(event, [])
        
        for callback in callbacks:
            try:
                callback(event, data)
            except Exception as e:
                logger.warning(f"Event callback failed for {event}: {e}")
        
        logger.debug(f"Event emitted - {event}, callbacks: {len(callbacks)}")
    
    def publish_event(self, event: str, data: Any = None) -> None:
        """发布事件（与emit_event相同，保持兼容性）"""
        self.emit_event(event, data)
    
    def subscribe_event(self, event: str, callback: Callable) -> str:
        """
        订阅事件
        
        Args:
            event: 事件名称
            callback: 回调函数
            
        Returns:
            订阅ID
        """
        import uuid
        subscription_id = str(uuid.uuid4())
        
        with self._events_lock:
            self._event_callbacks[event].append(callback)
        
        logger.debug(f"Event subscribed - {event}")
        return subscription_id
    
    def unsubscribe_event(self, event: str, callback: Callable) -> None:
        """
        取消事件订阅
        
        Args:
            event: 事件名称
            callback: 回调函数
        """
        with self._events_lock:
            if event in self._event_callbacks:
                try:
                    self._event_callbacks[event].remove(callback)
                    logger.debug(f"Event unsubscribed - {event}")
                except ValueError:
                    logger.warning(f"Callback not found for event {event}")
    
    # ==================== 多Learner专用接口 ====================
    
    def register_learner(self, learner_id: str, learner_info: Dict[str, Any]) -> None:
        """
        注册learner
        
        Args:
            learner_id: learner ID
            learner_info: learner信息
        """
        self.set_state(f"learner_{learner_id}_info", learner_info, scope="learners")
        self.set_state(f"learner_{learner_id}_status", "registered", scope="learners")
        logger.debug(f"Learner registered: {learner_id}")
    
    def get_learner_info(self, learner_id: str) -> Dict[str, Any]:
        """
        获取learner信息
        
        Args:
            learner_id: learner ID
            
        Returns:
            learner信息
        """
        return self.get_state(f"learner_{learner_id}_info", scope="learners", default={})
    
    def set_learner_status(self, learner_id: str, status: str) -> None:
        """
        设置learner状态
        
        Args:
            learner_id: learner ID
            status: 状态
        """
        self.set_state(f"learner_{learner_id}_status", status, scope="learners")
        logger.debug(f"Learner {learner_id} status: {status}")
    
    def get_learner_status(self, learner_id: str) -> str:
        """
        获取learner状态
        
        Args:
            learner_id: learner ID
            
        Returns:
            状态
        """
        return self.get_state(f"learner_{learner_id}_status", scope="learners", default="unknown")
    
    def get_all_learners(self) -> List[str]:
        """
        获取所有已注册的learner
        
        Returns:
            learner ID列表
        """
        learner_states = self.get_all_states(scope="learners")
        learner_ids = []
        
        for key in learner_states.keys():
            if key.startswith("learner_") and key.endswith("_info"):
                learner_id = key[8:-5]  # 移除 "learner_" 前缀和 "_info" 后缀
                learner_ids.append(learner_id)
        
        return learner_ids
    
    def create_execution_summary(self) -> Dict[str, Any]:
        """
        创建执行摘要
        
        Returns:
            执行摘要
        """
        return {
            'experiment_id': self.experiment_id,
            'registered_learners': len(self.get_all_learners()),
            'global_states': len(self._global_state),
            'shared_features': len(self._learner_features),
            'total_metrics': sum(len(metrics) for metrics in self._metrics.values()),
            'learner_metrics': {
                learner_id: sum(len(metrics) for metrics in learner_metrics.values())
                for learner_id, learner_metrics in self._learner_metrics.items()
            },
            'timestamp': time.time()
        }
    
    # ==================== 生命周期管理 ====================
    
    def cleanup(self) -> None:
        """清理执行上下文"""
        logger.debug(f"Cleaning up ExecutionContext - ID: {self.experiment_id}")
        
        # 清理所有状态
        self.clear_state("all")
        
        # 清理特征
        self.clear_learner_features()
        
        # 清理度量
        self.clear_metrics()
        
        # 清理事件回调
        with self._events_lock:
            self._event_callbacks.clear()
        
        # 清理配置缓存
        self._config_cache.clear()
        
        logger.debug("ExecutionContext cleanup 完成")
    
    def __repr__(self) -> str:
        return (f"SimplifiedExecutionContext("
                f"experiment_id='{self.experiment_id}', "
                f"learners={len(self.get_all_learners())}, "
                f"shared_features={len(self._learner_features)}, "
                f"global_states={len(self._global_state)})")


# 为了兼容性，保持原有类名
ExecutionContext = ExecutionContext