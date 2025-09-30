"""
MOE-FedCL 事件订阅管理
moe_fedcl/learner/subscription.py
"""

import asyncio
import uuid
from typing import Any, Dict, List, Optional, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from ..exceptions import ValidationError, MOEFedCLError


class SubscriptionType(Enum):
    """订阅类型"""
    TRAINING_EVENTS = "training_events"
    MODEL_UPDATES = "model_updates"
    CLIENT_STATUS = "client_status"
    SYSTEM_EVENTS = "system_events"
    CUSTOM = "custom"


class SubscriptionStatus(Enum):
    """订阅状态"""
    ACTIVE = "active"
    PAUSED = "paused"
    EXPIRED = "expired"
    ERROR = "error"


@dataclass
class SubscriptionInfo:
    """订阅信息"""
    subscription_id: str
    subscriber_id: str
    event_type: str
    callback: Callable
    subscription_type: SubscriptionType
    status: SubscriptionStatus
    created_time: datetime
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0
    max_triggers: Optional[int] = None
    ttl: Optional[float] = None  # Time to live in seconds
    filters: Dict[str, Any] = None
    metadata: Dict[str, Any] = None


class EventFilter:
    """事件过滤器"""
    
    def __init__(self, conditions: Dict[str, Any] = None):
        self.conditions = conditions or {}
    
    def match(self, event_data: Dict[str, Any]) -> bool:
        """检查事件数据是否匹配过滤条件"""
        if not self.conditions:
            return True
        
        for key, expected_value in self.conditions.items():
            if key not in event_data:
                return False
            
            actual_value = event_data[key]
            
            # 支持不同类型的匹配
            if isinstance(expected_value, dict):
                # 嵌套条件匹配
                if "$eq" in expected_value:
                    if actual_value != expected_value["$eq"]:
                        return False
                elif "$in" in expected_value:
                    if actual_value not in expected_value["$in"]:
                        return False
                elif "$gt" in expected_value:
                    if not (isinstance(actual_value, (int, float)) and actual_value > expected_value["$gt"]):
                        return False
                elif "$lt" in expected_value:
                    if not (isinstance(actual_value, (int, float)) and actual_value < expected_value["$lt"]):
                        return False
                elif "$contains" in expected_value:
                    if expected_value["$contains"] not in str(actual_value):
                        return False
            else:
                # 直接值匹配
                if actual_value != expected_value:
                    return False
        
        return True


class SubscriptionManager:
    """事件订阅管理器"""
    
    def __init__(self):
        self.subscriptions: Dict[str, SubscriptionInfo] = {}
        self.event_subscriptions: Dict[str, List[str]] = {}  # event_type -> [subscription_ids]
        self.subscriber_subscriptions: Dict[str, List[str]] = {}  # subscriber_id -> [subscription_ids]
        
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        
        # 统计信息
        self.total_events_processed = 0
        self.total_subscriptions_created = 0
        self.total_callbacks_executed = 0
    
    async def start(self):
        """启动订阅管理器"""
        self._running = True
        
        # 启动清理任务
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def stop(self):
        """停止订阅管理器"""
        self._running = False
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
    
    async def subscribe(self,
                       subscriber_id: str,
                       event_type: str,
                       callback: Callable,
                       subscription_type: SubscriptionType = SubscriptionType.CUSTOM,
                       max_triggers: Optional[int] = None,
                       ttl: Optional[float] = None,
                       filters: Dict[str, Any] = None,
                       metadata: Dict[str, Any] = None) -> str:
        """创建订阅
        
        Args:
            subscriber_id: 订阅者ID
            event_type: 事件类型
            callback: 回调函数
            subscription_type: 订阅类型
            max_triggers: 最大触发次数
            ttl: 生存时间（秒）
            filters: 过滤条件
            metadata: 元数据
            
        Returns:
            str: 订阅ID
        """
        subscription_id = str(uuid.uuid4())
        
        async with self._lock:
            # 创建订阅信息
            subscription = SubscriptionInfo(
                subscription_id=subscription_id,
                subscriber_id=subscriber_id,
                event_type=event_type,
                callback=callback,
                subscription_type=subscription_type,
                status=SubscriptionStatus.ACTIVE,
                created_time=datetime.now(),
                max_triggers=max_triggers,
                ttl=ttl,
                filters=filters,
                metadata=metadata or {}
            )
            
            # 存储订阅
            self.subscriptions[subscription_id] = subscription
            
            # 更新索引
            if event_type not in self.event_subscriptions:
                self.event_subscriptions[event_type] = []
            self.event_subscriptions[event_type].append(subscription_id)
            
            if subscriber_id not in self.subscriber_subscriptions:
                self.subscriber_subscriptions[subscriber_id] = []
            self.subscriber_subscriptions[subscriber_id].append(subscription_id)
            
            self.total_subscriptions_created += 1
        
        return subscription_id
    
    async def unsubscribe(self, subscription_id: str) -> bool:
        """取消订阅
        
        Args:
            subscription_id: 订阅ID
            
        Returns:
            bool: 是否成功取消
        """
        async with self._lock:
            if subscription_id not in self.subscriptions:
                return False
            
            subscription = self.subscriptions[subscription_id]
            
            # 从索引中移除
            if subscription.event_type in self.event_subscriptions:
                if subscription_id in self.event_subscriptions[subscription.event_type]:
                    self.event_subscriptions[subscription.event_type].remove(subscription_id)
                
                # 如果该事件类型没有订阅了，删除条目
                if not self.event_subscriptions[subscription.event_type]:
                    del self.event_subscriptions[subscription.event_type]
            
            if subscription.subscriber_id in self.subscriber_subscriptions:
                if subscription_id in self.subscriber_subscriptions[subscription.subscriber_id]:
                    self.subscriber_subscriptions[subscription.subscriber_id].remove(subscription_id)
                
                # 如果该订阅者没有订阅了，删除条目
                if not self.subscriber_subscriptions[subscription.subscriber_id]:
                    del self.subscriber_subscriptions[subscription.subscriber_id]
            
            # 删除订阅
            del self.subscriptions[subscription_id]
            
            return True
    
    async def unsubscribe_all(self, subscriber_id: str) -> int:
        """取消指定订阅者的所有订阅
        
        Args:
            subscriber_id: 订阅者ID
            
        Returns:
            int: 取消的订阅数量
        """
        async with self._lock:
            if subscriber_id not in self.subscriber_subscriptions:
                return 0
            
            subscription_ids = self.subscriber_subscriptions[subscriber_id].copy()
            count = 0
            
            for subscription_id in subscription_ids:
                if await self.unsubscribe(subscription_id):
                    count += 1
            
            return count
    
    async def publish_event(self, event_type: str, source_id: str, data: Any = None) -> int:
        """发布事件
        
        Args:
            event_type: 事件类型
            source_id: 事件源ID
            data: 事件数据
            
        Returns:
            int: 触发的订阅数量
        """
        self.total_events_processed += 1
        
        async with self._lock:
            if event_type not in self.event_subscriptions:
                return 0
            
            subscription_ids = self.event_subscriptions[event_type].copy()
        
        triggered_count = 0
        expired_subscriptions = []
        
        # 准备事件数据
        event_data = {
            "event_type": event_type,
            "source_id": source_id,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        
        for subscription_id in subscription_ids:
            if subscription_id not in self.subscriptions:
                continue
            
            subscription = self.subscriptions[subscription_id]
            
            # 检查订阅状态
            if subscription.status != SubscriptionStatus.ACTIVE:
                continue
            
            # 检查过滤条件
            if subscription.filters:
                event_filter = EventFilter(subscription.filters)
                if not event_filter.match(event_data):
                    continue
            
            # 检查触发次数限制
            if subscription.max_triggers and subscription.trigger_count >= subscription.max_triggers:
                expired_subscriptions.append(subscription_id)
                continue
            
            # 执行回调
            try:
                await self._execute_callback(subscription, event_data)
                
                # 更新触发信息
                subscription.last_triggered = datetime.now()
                subscription.trigger_count += 1
                triggered_count += 1
                self.total_callbacks_executed += 1
                
                # 检查是否需要过期
                if subscription.max_triggers and subscription.trigger_count >= subscription.max_triggers:
                    expired_subscriptions.append(subscription_id)
                
            except Exception as e:
                print(f"Callback execution failed for subscription {subscription_id}: {e}")
                subscription.status = SubscriptionStatus.ERROR
        
        # 清理过期的订阅
        for subscription_id in expired_subscriptions:
            await self.unsubscribe(subscription_id)
        
        return triggered_count
    
    async def _execute_callback(self, subscription: SubscriptionInfo, event_data: Dict[str, Any]):
        """执行回调函数"""
        callback = subscription.callback
        
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(event_data)
            else:
                callback(event_data)
        except Exception as e:
            raise MOEFedCLError(f"Callback execution failed: {str(e)}")
    
    def get_subscription(self, subscription_id: str) -> Optional[SubscriptionInfo]:
        """获取订阅信息"""
        return self.subscriptions.get(subscription_id)
    
    def list_subscriptions(self, 
                          subscriber_id: str = None,
                          event_type: str = None,
                          status: SubscriptionStatus = None) -> List[SubscriptionInfo]:
        """列出订阅"""
        subscriptions = list(self.subscriptions.values())
        
        # 按条件过滤
        if subscriber_id:
            subscriptions = [s for s in subscriptions if s.subscriber_id == subscriber_id]
        
        if event_type:
            subscriptions = [s for s in subscriptions if s.event_type == event_type]
        
        if status:
            subscriptions = [s for s in subscriptions if s.status == status]
        
        return subscriptions
    
    def get_subscription_stats(self) -> Dict[str, Any]:
        """获取订阅统计"""
        active_count = len([s for s in self.subscriptions.values() if s.status == SubscriptionStatus.ACTIVE])
        paused_count = len([s for s in self.subscriptions.values() if s.status == SubscriptionStatus.PAUSED])
        error_count = len([s for s in self.subscriptions.values() if s.status == SubscriptionStatus.ERROR])
        
        # 事件类型分布
        event_type_distribution = {}
        for subscription in self.subscriptions.values():
            event_type = subscription.event_type
            event_type_distribution[event_type] = event_type_distribution.get(event_type, 0) + 1
        
        # 订阅者分布
        subscriber_distribution = {}
        for subscription in self.subscriptions.values():
            subscriber_id = subscription.subscriber_id
            subscriber_distribution[subscriber_id] = subscriber_distribution.get(subscriber_id, 0) + 1
        
        return {
            "total_subscriptions": len(self.subscriptions),
            "active_subscriptions": active_count,
            "paused_subscriptions": paused_count,
            "error_subscriptions": error_count,
            "total_events_processed": self.total_events_processed,
            "total_subscriptions_created": self.total_subscriptions_created,
            "total_callbacks_executed": self.total_callbacks_executed,
            "event_type_distribution": event_type_distribution,
            "subscriber_distribution": subscriber_distribution,
            "unique_event_types": len(self.event_subscriptions),
            "unique_subscribers": len(self.subscriber_subscriptions)
        }
    
    async def pause_subscription(self, subscription_id: str) -> bool:
        """暂停订阅"""
        if subscription_id in self.subscriptions:
            self.subscriptions[subscription_id].status = SubscriptionStatus.PAUSED
            return True
        return False
    
    async def resume_subscription(self, subscription_id: str) -> bool:
        """恢复订阅"""
        if subscription_id in self.subscriptions:
            subscription = self.subscriptions[subscription_id]
            if subscription.status == SubscriptionStatus.PAUSED:
                subscription.status = SubscriptionStatus.ACTIVE
                return True
        return False
    
    async def _cleanup_loop(self):
        """清理循环 - 定期清理过期订阅"""
        while self._running:
            try:
                await self._cleanup_expired_subscriptions()
                await asyncio.sleep(60)  # 每分钟清理一次
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Subscription cleanup error: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_expired_subscriptions(self):
        """清理过期的订阅"""
        current_time = datetime.now()
        expired_subscriptions = []
        
        async with self._lock:
            for subscription_id, subscription in self.subscriptions.items():
                # 检查TTL过期
                if subscription.ttl:
                    elapsed = (current_time - subscription.created_time).total_seconds()
                    if elapsed > subscription.ttl:
                        expired_subscriptions.append(subscription_id)
                        continue
                
                # 检查触发次数过期
                if (subscription.max_triggers and 
                    subscription.trigger_count >= subscription.max_triggers):
                    expired_subscriptions.append(subscription_id)
        
        # 清理过期订阅
        for subscription_id in expired_subscriptions:
            await self.unsubscribe(subscription_id)
            
        if expired_subscriptions:
            print(f"Cleaned up {len(expired_subscriptions)} expired subscriptions")


# ==================== 预定义事件类型 ====================

class CommonEvents:
    """常用事件类型常量"""
    
    # 训练事件
    TRAINING_STARTED = "training_started"
    TRAINING_COMPLETED = "training_completed"
    TRAINING_FAILED = "training_failed"
    TRAINING_PROGRESS = "training_progress"
    
    # 模型事件
    MODEL_UPDATED = "model_updated"
    MODEL_AGGREGATED = "model_aggregated"
    MODEL_EVALUATED = "model_evaluated"
    
    # 客户端事件
    CLIENT_REGISTERED = "client_registered"
    CLIENT_DISCONNECTED = "client_disconnected"
    CLIENT_TIMEOUT = "client_timeout"
    CLIENT_ERROR = "client_error"
    
    # 系统事件
    ROUND_STARTED = "round_started"
    ROUND_COMPLETED = "round_completed"
    FEDERATION_STARTED = "federation_started"
    FEDERATION_COMPLETED = "federation_completed"
    FEDERATION_ERROR = "federation_error"
    
    # 连接事件
    CONNECTION_ESTABLISHED = "connection_established"
    CONNECTION_LOST = "connection_lost"
    HEARTBEAT_RECEIVED = "heartbeat_received"


# ==================== 便捷装饰器 ====================

def event_subscriber(event_type: str, 
                    subscription_type: SubscriptionType = SubscriptionType.CUSTOM,
                    max_triggers: Optional[int] = None,
                    ttl: Optional[float] = None,
                    filters: Dict[str, Any] = None):
    """事件订阅装饰器
    
    Usage:
        @event_subscriber("training_completed", max_triggers=10)
        async def on_training_complete(event_data):
            print(f"Training completed: {event_data}")
    """
    def decorator(func):
        func._event_subscription_info = {
            "event_type": event_type,
            "subscription_type": subscription_type,
            "max_triggers": max_triggers,
            "ttl": ttl,
            "filters": filters
        }
        return func
    
    return decorator


# ==================== 全局订阅管理器 ====================

_global_subscription_manager: Optional[SubscriptionManager] = None


async def get_global_subscription_manager() -> SubscriptionManager:
    """获取全局订阅管理器"""
    global _global_subscription_manager
    
    if _global_subscription_manager is None:
        _global_subscription_manager = SubscriptionManager()
        await _global_subscription_manager.start()
    
    return _global_subscription_manager


async def subscribe_global_event(subscriber_id: str, 
                                event_type: str, 
                                callback: Callable, 
                                **kwargs) -> str:
    """订阅全局事件"""
    manager = await get_global_subscription_manager()
    return await manager.subscribe(subscriber_id, event_type, callback, **kwargs)


async def publish_global_event(event_type: str, source_id: str, data: Any = None) -> int:
    """发布全局事件"""
    manager = await get_global_subscription_manager()
    return await manager.publish_event(event_type, source_id, data)


async def unsubscribe_global_event(subscription_id: str) -> bool:
    """取消全局事件订阅"""
    manager = await get_global_subscription_manager()
    return await manager.unsubscribe(subscription_id)