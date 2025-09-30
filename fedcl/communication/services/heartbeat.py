"""
MOE-FedCL 心跳服务
moe_fedcl/communication/services/heartbeat.py
"""

import asyncio
from typing import Dict, List, Optional, Callable, Set
from datetime import datetime, timedelta
import logging

from ...types import HeartbeatMessage, ClientInfo, EventMessage, HealthStatus
from ...exceptions import CommunicationError
from ...utils.auto_logger import get_logger


class HeartbeatService:
    """心跳服务
    
    负责管理客户端心跳检测和连接保活
    """
    
    def __init__(self, interval: float = 30.0, timeout: float = 90.0):
        """初始化心跳服务
        
        Args:
            interval: 心跳间隔（秒）
            timeout: 心跳超时（秒）
        """
        self.interval = interval
        self.timeout = timeout
        self.logger = get_logger("sys", "heartbeat_service")
        
        # 心跳状态跟踪
        self.heartbeat_status: Dict[str, datetime] = {}
        self.client_metrics: Dict[str, Dict] = {}
        
        # 超时客户端跟踪
        self.timeout_clients: Set[str] = set()
        
        # 事件回调
        self.event_callbacks: List[Callable[[EventMessage], None]] = []
        
        # 异步任务
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._check_task: Optional[asyncio.Task] = None
        
        # 状态控制
        self._running = False
        self._lock = asyncio.Lock()
    
    async def start(self) -> None:
        """启动心跳服务"""
        if self._running:
            self.logger.warning("Heartbeat service is already running")
            return
        
        self._running = True
        
        # 启动心跳检查任务
        self._check_task = asyncio.create_task(self._heartbeat_check_loop())
        
        self.logger.info(f"Heartbeat service started (interval={self.interval}s, timeout={self.timeout}s)")
    
    async def stop(self) -> None:
        """停止心跳服务"""
        if not self._running:
            return
        
        self._running = False
        
        # 取消心跳任务
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None
        
        # 取消检查任务
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
            self._check_task = None
        
        self.logger.info("Heartbeat service stopped")
    
    async def process_heartbeat(self, heartbeat: HeartbeatMessage) -> bool:
        """处理心跳消息
        
        Args:
            heartbeat: 心跳消息
            
        Returns:
            bool: 是否处理成功
        """
        async with self._lock:
            try:
                client_id = heartbeat.client_id
                current_time = datetime.now()
                
                # 更新心跳状态
                self.heartbeat_status[client_id] = current_time
                
                # 更新客户端指标
                self.client_metrics[client_id] = heartbeat.metrics
                
                # 如果之前超时，现在恢复了
                if client_id in self.timeout_clients:
                    self.timeout_clients.remove(client_id)
                    self.logger.info(f"Client {client_id} recovered from timeout")
                    await self._emit_event("CLIENT_RECOVERED", client_id, {
                        "recovery_time": current_time,
                        "status": heartbeat.status,
                        "metrics": heartbeat.metrics
                    })
                
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to process heartbeat from {heartbeat.client_id}: {e}")
                return False
    
    async def register_client(self, client_id: str) -> None:
        """注册客户端到心跳监控
        
        Args:
            client_id: 客户端ID
        """
        async with self._lock:
            current_time = datetime.now()
            self.heartbeat_status[client_id] = current_time
            self.client_metrics[client_id] = {}
            
            # 从超时列表中移除（如果存在）
            self.timeout_clients.discard(client_id)
            
            self.logger.debug(f"Client {client_id} registered for heartbeat monitoring")
    
    async def unregister_client(self, client_id: str) -> None:
        """从心跳监控中注销客户端
        
        Args:
            client_id: 客户端ID
        """
        async with self._lock:
            self.heartbeat_status.pop(client_id, None)
            self.client_metrics.pop(client_id, None)
            self.timeout_clients.discard(client_id)
            
            self.logger.debug(f"Client {client_id} unregistered from heartbeat monitoring")
    
    def is_client_alive(self, client_id: str) -> bool:
        """检查客户端是否活跃
        
        Args:
            client_id: 客户端ID
            
        Returns:
            bool: 是否活跃
        """
        if client_id not in self.heartbeat_status:
            return False
        
        last_heartbeat = self.heartbeat_status[client_id]
        current_time = datetime.now()
        timeout_threshold = timedelta(seconds=self.timeout)
        
        return current_time - last_heartbeat <= timeout_threshold
    
    def get_alive_clients(self) -> List[str]:
        """获取活跃客户端列表
        
        Returns:
            List[str]: 活跃客户端ID列表
        """
        alive_clients = []
        current_time = datetime.now()
        timeout_threshold = timedelta(seconds=self.timeout)
        
        for client_id, last_heartbeat in self.heartbeat_status.items():
            if current_time - last_heartbeat <= timeout_threshold:
                alive_clients.append(client_id)
        
        return alive_clients
    
    def get_timeout_clients(self) -> List[str]:
        """获取超时客户端列表
        
        Returns:
            List[str]: 超时客户端ID列表
        """
        return list(self.timeout_clients)
    
    def get_client_metrics(self, client_id: str) -> Optional[Dict]:
        """获取客户端指标
        
        Args:
            client_id: 客户端ID
            
        Returns:
            Optional[Dict]: 客户端指标，不存在则返回None
        """
        return self.client_metrics.get(client_id)
    
    def get_heartbeat_status(self, client_id: str) -> Optional[datetime]:
        """获取客户端最后心跳时间
        
        Args:
            client_id: 客户端ID
            
        Returns:
            Optional[datetime]: 最后心跳时间，不存在则返回None
        """
        return self.heartbeat_status.get(client_id)
    
    def get_health_status(self) -> HealthStatus:
        """获取整体健康状态
        
        Returns:
            HealthStatus: 健康状态
        """
        total_clients = len(self.heartbeat_status)
        if total_clients == 0:
            return HealthStatus.HEALTHY
        
        timeout_count = len(self.timeout_clients)
        timeout_ratio = timeout_count / total_clients
        
        if timeout_ratio == 0:
            return HealthStatus.HEALTHY
        elif timeout_ratio < 0.2:  # 20%以下超时
            return HealthStatus.WARNING
        else:
            return HealthStatus.ERROR
    
    def get_statistics(self) -> Dict:
        """获取心跳统计信息
        
        Returns:
            Dict: 统计信息
        """
        total_clients = len(self.heartbeat_status)
        alive_clients = len(self.get_alive_clients())
        timeout_clients = len(self.timeout_clients)
        
        return {
            "total_clients": total_clients,
            "alive_clients": alive_clients,
            "timeout_clients": timeout_clients,
            "alive_ratio": alive_clients / total_clients if total_clients > 0 else 0,
            "timeout_ratio": timeout_clients / total_clients if total_clients > 0 else 0,
            "health_status": self.get_health_status().value,
            "service_running": self._running
        }
    
    def register_event_callback(self, callback: Callable[[EventMessage], None]) -> None:
        """注册事件回调
        
        Args:
            callback: 事件回调函数
        """
        if callback not in self.event_callbacks:
            self.event_callbacks.append(callback)
    
    def unregister_event_callback(self, callback: Callable[[EventMessage], None]) -> None:
        """取消注册事件回调
        
        Args:
            callback: 事件回调函数
        """
        if callback in self.event_callbacks:
            self.event_callbacks.remove(callback)
    
    async def _heartbeat_check_loop(self) -> None:
        """心跳检查循环"""
        while self._running:
            try:
                await self._check_heartbeat_timeouts()
                await asyncio.sleep(self.interval / 2)  # 检查频率是心跳间隔的一半
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in heartbeat check loop: {e}")
                await asyncio.sleep(1)  # 错误后短暂休息
    
    async def _check_heartbeat_timeouts(self) -> None:
        """检查心跳超时"""
        async with self._lock:
            current_time = datetime.now()
            timeout_threshold = timedelta(seconds=self.timeout)
            
            newly_timeout_clients = []
            
            for client_id, last_heartbeat in self.heartbeat_status.items():
                if current_time - last_heartbeat > timeout_threshold:
                    if client_id not in self.timeout_clients:
                        # 新的超时客户端
                        self.timeout_clients.add(client_id)
                        newly_timeout_clients.append(client_id)
                        
                        self.logger.warning(f"Client {client_id} heartbeat timeout")
            
            # 为新超时的客户端触发事件
            for client_id in newly_timeout_clients:
                await self._emit_event("CLIENT_TIMEOUT", client_id, {
                    "timeout_time": current_time,
                    "last_heartbeat": self.heartbeat_status.get(client_id),
                    "timeout_duration": self.timeout
                })
    
    async def _emit_event(self, event_type: str, client_id: str, data: any = None) -> None:
        """触发事件
        
        Args:
            event_type: 事件类型
            client_id: 客户端ID
            data: 事件数据
        """
        event = EventMessage(
            event_type=event_type,
            source_id=client_id,
            data=data
        )
        
        # 异步调用所有回调
        for callback in self.event_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                self.logger.error(f"Error in event callback: {e}")
