"""
MOE-FedCL 状态管理服务
moe_fedcl/communication/services/status.py
"""

import asyncio
import platform
import psutil
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import asdict
import json

from ...types import (
    ClientInfo, EventMessage, HealthStatus, 
    RegistrationStatus, TrainingStatus, FederationStatus
)
from ...exceptions import CommunicationError
from ...utils.auto_logger import get_logger


class StatusManagementService:
    """状态管理服务
    
    负责系统状态监控、查询和同步
    """
    
    def __init__(self, collection_interval: float = 60.0):
        """初始化状态管理服务
        
        Args:
            collection_interval: 状态收集间隔（秒）
        """
        self.collection_interval = collection_interval
        self.logger = get_logger("sys", "status_service")
        
        # 状态数据存储
        self.node_status: Dict[str, Any] = {}
        self.client_statuses: Dict[str, Dict[str, Any]] = {}
        self.system_metrics: Dict[str, Any] = {}
        self.connection_stats: Dict[str, Any] = {}
        
        # 历史状态数据（保留最近24小时）
        self.status_history: List[Dict[str, Any]] = []
        self.max_history_items = 1440  # 24小时 * 60分钟
        
        # 事件回调
        self.event_callbacks: List[Callable[[EventMessage], None]] = []
        
        # 异步任务
        self._collection_task: Optional[asyncio.Task] = None
        
        # 状态控制
        self._running = False
        self._lock = asyncio.Lock()
        
        # 初始化节点状态
        self._initialize_node_status()
    
    async def start(self) -> None:
        """启动状态管理服务"""
        if self._running:
            self.logger.warning("Status management service is already running")
            return
        
        self._running = True
        
        # 启动状态收集任务
        self._collection_task = asyncio.create_task(self._status_collection_loop())
        
        self.logger.info(f"Status management service started (collection_interval={self.collection_interval}s)")
    
    async def stop(self) -> None:
        """停止状态管理服务"""
        if not self._running:
            return
        
        self._running = False
        
        # 取消状态收集任务
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
            self._collection_task = None
        
        self.logger.info("Status management service stopped")
    
    async def update_client_status(self, client_id: str, status_updates: Dict[str, Any]) -> bool:
        """更新客户端状态
        
        Args:
            client_id: 客户端ID
            status_updates: 状态更新数据
            
        Returns:
            bool: 是否更新成功
        """
        async with self._lock:
            try:
                current_time = datetime.now()
                
                if client_id not in self.client_statuses:
                    self.client_statuses[client_id] = {
                        "client_id": client_id,
                        "created_time": current_time,
                        "last_updated": current_time
                    }
                
                # 更新状态
                client_status = self.client_statuses[client_id]
                old_status = dict(client_status)
                
                client_status.update(status_updates)
                client_status["last_updated"] = current_time
                
                # 检查是否有重要状态变更
                await self._check_status_changes(client_id, old_status, client_status)
                
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to update client status for {client_id}: {e}")
                return False
    
    async def get_client_status(self, client_id: str) -> Optional[Dict[str, Any]]:
        """获取客户端状态
        
        Args:
            client_id: 客户端ID
            
        Returns:
            Optional[Dict[str, Any]]: 客户端状态，不存在则返回None
        """
        return self.client_statuses.get(client_id)
    
    async def list_client_statuses(self, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """列出客户端状态
        
        Args:
            status_filter: 状态过滤器
            
        Returns:
            List[Dict[str, Any]]: 客户端状态列表
        """
        statuses = list(self.client_statuses.values())
        
        if status_filter:
            statuses = [s for s in statuses if s.get("registration_status") == status_filter]
        
        return statuses
    
    async def get_node_status(self) -> Dict[str, Any]:
        """获取节点状态
        
        Returns:
            Dict[str, Any]: 节点状态
        """
        async with self._lock:
            # 更新实时状态
            await self._collect_node_status()
            return dict(self.node_status)
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """获取系统指标
        
        Returns:
            Dict[str, Any]: 系统指标
        """
        async with self._lock:
            # 更新实时指标
            await self._collect_system_metrics()
            return dict(self.system_metrics)
    
    async def get_connection_stats(self) -> Dict[str, Any]:
        """获取连接统计
        
        Returns:
            Dict[str, Any]: 连接统计
        """
        return dict(self.connection_stats)
    
    async def update_connection_stats(self, stats: Dict[str, Any]) -> None:
        """更新连接统计
        
        Args:
            stats: 连接统计数据
        """
        async with self._lock:
            self.connection_stats.update(stats)
            self.connection_stats["last_updated"] = datetime.now()
    
    async def health_check(self) -> Dict[str, Any]:
        """执行健康检查
        
        Returns:
            Dict[str, Any]: 健康检查结果
        """
        try:
            # 系统健康检查
            system_health = await self._check_system_health()
            
            # 客户端健康检查
            client_health = await self._check_client_health()
            
            # 服务健康检查
            service_health = await self._check_service_health()
            
            # 整体健康状态
            overall_status = self._determine_overall_health(
                system_health, client_health, service_health
            )
            
            return {
                "overall_status": overall_status.value,
                "system_health": system_health,
                "client_health": client_health,
                "service_health": service_health,
                "timestamp": datetime.now(),
                "check_duration": 0.0  # 可以计算实际检查时间
            }
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "overall_status": HealthStatus.ERROR.value,
                "error": str(e),
                "timestamp": datetime.now()
            }
    
    async def get_status_history(self, hours: int = 1) -> List[Dict[str, Any]]:
        """获取状态历史
        
        Args:
            hours: 获取最近几小时的历史
            
        Returns:
            List[Dict[str, Any]]: 状态历史列表
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            item for item in self.status_history
            if item["timestamp"] >= cutoff_time
        ]
    
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
    
    def _initialize_node_status(self) -> None:
        """初始化节点状态"""
        self.node_status = {
            "node_id": "",  # 由上层组件设置
            "start_time": datetime.now(),
            "uptime": 0.0,
            "status": "initializing",
            "version": "1.0.0",  # 由配置或版本信息设置
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "python_version": platform.python_version()
            }
        }
    
    async def _status_collection_loop(self) -> None:
        """状态收集循环"""
        while self._running:
            try:
                await self._collect_all_status()
                await asyncio.sleep(self.collection_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in status collection loop: {e}")
                await asyncio.sleep(5)  # 错误后短暂休息
    
    async def _collect_all_status(self) -> None:
        """收集所有状态信息"""
        async with self._lock:
            timestamp = datetime.now()
            
            # 收集节点状态
            await self._collect_node_status()
            
            # 收集系统指标
            await self._collect_system_metrics()
            
            # 保存到历史记录
            history_item = {
                "timestamp": timestamp,
                "node_status": dict(self.node_status),
                "system_metrics": dict(self.system_metrics),
                "client_count": len(self.client_statuses),
                "active_clients": len([
                    c for c in self.client_statuses.values()
                    if c.get("registration_status") == "registered"
                ])
            }
            
            self.status_history.append(history_item)
            
            # 清理旧的历史记录
            if len(self.status_history) > self.max_history_items:
                self.status_history = self.status_history[-self.max_history_items:]
    
    async def _collect_node_status(self) -> None:
        """收集节点状态"""
        current_time = datetime.now()
        start_time = self.node_status.get("start_time", current_time)
        
        self.node_status.update({
            "last_updated": current_time,
            "uptime": (current_time - start_time).total_seconds(),
            "status": "running" if self._running else "stopped"
        })
    
    async def _collect_system_metrics(self) -> None:
        """收集系统指标"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 内存使用情况
            memory = psutil.virtual_memory()
            
            # 磁盘使用情况
            disk = psutil.disk_usage('/')
            
            # 网络统计
            net_io = psutil.net_io_counters()
            
            self.system_metrics = {
                "cpu": {
                    "percent": cpu_percent,
                    "count": psutil.cpu_count(),
                    "count_logical": psutil.cpu_count(logical=True)
                },
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "percent": memory.percent,
                    "used": memory.used,
                    "free": memory.free
                },
                "disk": {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "percent": disk.used / disk.total * 100
                },
                "network": {
                    "bytes_sent": net_io.bytes_sent,
                    "bytes_recv": net_io.bytes_recv,
                    "packets_sent": net_io.packets_sent,
                    "packets_recv": net_io.packets_recv
                },
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
            self.system_metrics = {
                "error": str(e),
                "timestamp": datetime.now()
            }
    
    async def _check_status_changes(self, client_id: str, old_status: Dict, new_status: Dict) -> None:
        """检查状态变更并触发事件"""
        # 检查注册状态变更
        old_reg_status = old_status.get("registration_status")
        new_reg_status = new_status.get("registration_status")
        
        if old_reg_status != new_reg_status:
            await self._emit_event("CLIENT_STATUS_CHANGED", client_id, {
                "old_status": old_reg_status,
                "new_status": new_reg_status,
                "status_type": "registration"
            })
        
        # 检查训练状态变更
        old_train_status = old_status.get("training_status")
        new_train_status = new_status.get("training_status")
        
        if old_train_status != new_train_status:
            await self._emit_event("CLIENT_TRAINING_STATUS_CHANGED", client_id, {
                "old_status": old_train_status,
                "new_status": new_train_status,
                "status_type": "training"
            })
    
    async def _check_system_health(self) -> Dict[str, Any]:
        """检查系统健康状态"""
        metrics = await self.get_system_metrics()
        
        # CPU健康检查
        cpu_status = HealthStatus.HEALTHY
        cpu_percent = metrics.get("cpu", {}).get("percent", 0)
        if cpu_percent > 90:
            cpu_status = HealthStatus.ERROR
        elif cpu_percent > 80:
            cpu_status = HealthStatus.WARNING
        
        # 内存健康检查
        memory_status = HealthStatus.HEALTHY
        memory_percent = metrics.get("memory", {}).get("percent", 0)
        if memory_percent > 95:
            memory_status = HealthStatus.ERROR
        elif memory_percent > 85:
            memory_status = HealthStatus.WARNING
        
        # 磁盘健康检查
        disk_status = HealthStatus.HEALTHY
        disk_percent = metrics.get("disk", {}).get("percent", 0)
        if disk_percent > 95:
            disk_status = HealthStatus.ERROR
        elif disk_percent > 85:
            disk_status = HealthStatus.WARNING
        
        return {
            "cpu_status": cpu_status.value,
            "memory_status": memory_status.value,
            "disk_status": disk_status.value,
            "metrics": metrics
        }
    
    async def _check_client_health(self) -> Dict[str, Any]:
        """检查客户端健康状态"""
        total_clients = len(self.client_statuses)
        active_clients = len([
            c for c in self.client_statuses.values()
            if c.get("registration_status") == "registered"
        ])
        
        if total_clients == 0:
            status = HealthStatus.WARNING
        elif active_clients / total_clients >= 0.8:
            status = HealthStatus.HEALTHY
        elif active_clients / total_clients >= 0.5:
            status = HealthStatus.WARNING
        else:
            status = HealthStatus.ERROR
        
        return {
            "status": status.value,
            "total_clients": total_clients,
            "active_clients": active_clients,
            "active_ratio": active_clients / total_clients if total_clients > 0 else 0
        }
    
    async def _check_service_health(self) -> Dict[str, Any]:
        """检查服务健康状态"""
        service_status = HealthStatus.HEALTHY
        
        if not self._running:
            service_status = HealthStatus.ERROR
        
        return {
            "status": service_status.value,
            "service_running": self._running,
            "collection_task_active": self._collection_task is not None and not self._collection_task.done()
        }
    
    def _determine_overall_health(self, system_health: Dict, client_health: Dict, service_health: Dict) -> HealthStatus:
        """确定整体健康状态"""
        statuses = [
            HealthStatus(system_health.get("cpu_status", "healthy")),
            HealthStatus(system_health.get("memory_status", "healthy")),
            HealthStatus(system_health.get("disk_status", "healthy")),
            HealthStatus(client_health.get("status", "healthy")),
            HealthStatus(service_health.get("status", "healthy"))
        ]
        
        # 如果有任何ERROR状态，整体为ERROR
        if HealthStatus.ERROR in statuses:
            return HealthStatus.ERROR
        
        # 如果有WARNING状态，整体为WARNING
        if HealthStatus.WARNING in statuses:
            return HealthStatus.WARNING
        
        # 否则为HEALTHY
        return HealthStatus.HEALTHY
    
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
