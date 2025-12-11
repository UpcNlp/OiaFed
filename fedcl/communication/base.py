"""
MOE-FedCL 通用通信层抽象基类
moe_fedcl/communication/base.py
"""

import asyncio
from abc import ABC
from datetime import datetime, timedelta
from typing import Any, Dict, List, Callable, Optional

from .services import (
    ClientRegistryService, HeartbeatService,
    StatusManagementService, SecurityService
)
from ..exceptions import CommunicationError
from ..transport.base import TransportBase
from ..types import (
    ClientInfo, RegistrationRequest, RegistrationResponse,
    HeartbeatMessage, CommunicationConfig, HealthStatus
)
from ..utils.auto_logger import get_comm_logger

class CommunicationManagerBase(ABC):
    """通用通信管理器抽象基类"""

    def __init__(self, node_id: str, transport: TransportBase, config: CommunicationConfig, node_role: str = None):
        self.node_id = node_id
        self.transport = transport
        self.config = config
        self.logger = get_comm_logger(node_id)
        # 确定节点角色
        if node_role is not None:
            self.node_role = node_role.lower()
        else:
            # 向后兼容：从 node_id 推断角色
            self.node_role = self._infer_node_role_from_id()

        # 初始化服务组件 - 根据角色决定
        if self.node_role == "server":
            # 只有 Server 端需要 registry_service（管理所有客户端）
            self.registry_service = ClientRegistryService(max_clients=config.max_clients, node_id=node_id)
        else:
            # Client 端不需要 registry_service
            self.registry_service = None

        # 所有节点都需要的服务（传递node_id以合并日志）
        self.heartbeat_service = HeartbeatService(
            interval=config.heartbeat_interval,
            timeout=config.heartbeat_timeout,
            node_id=node_id
        )
        self.status_service = StatusManagementService(node_id=node_id)
        self.security_service = SecurityService(
            secret_key=f"moe_fedcl_{node_id}",  # 简单的密钥生成
            policy=None,  # 使用默认安全策略
            node_id=node_id
        )

        # 客户端注册表（只有 Server 需要）
        self.clients: Dict[str, ClientInfo] = {} if self.node_role == "server" else {}
        
        # 心跳状态（委托给heartbeat_service）
        self.heartbeat_status: Dict[str, datetime] = {}
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._heartbeat_check_task: Optional[asyncio.Task] = None
        
        # 消息处理器
        self.message_handlers: Dict[str, Callable] = {}
        
        # 状态
        self._running = False
        self._lock = asyncio.Lock()
        
        # 注册服务事件回调
        self._setup_service_callbacks()

    def _infer_node_role_from_id(self) -> str:
        """从 node_id 推断节点角色（向后兼容）

        Returns:
            str: "server" 或 "client"
        """
        node_id_lower = self.node_id.lower()
        if "server" in node_id_lower:
            return "server"
        elif "client" in node_id_lower:
            return "client"
        else:
            # 默认为 client
            return "client"

    # ==================== 客户端管理方法 ====================
    
    async def register_client(self, registration: RegistrationRequest) -> RegistrationResponse:
        """注册客户端（基类默认实现 - 服务端行为）

        子类应该重写此方法来实现客户端/服务端的不同行为

        Args:
            registration: 注册请求

        Returns:
            RegistrationResponse: 注册响应

        Raises:
            RegistrationError: 注册失败
        """
        # 如果没有 registry_service（Client 端），抛出错误
        if self.registry_service is None:
            raise RegistrationError(
                "Client nodes should not call base register_client(). "
                "Subclass must override this method to send registration request to server."
            )

        # Server 端：委托给注册服务
        response = await self.registry_service.register_client(registration)

        if response.success:
            # 同步到本地状态
            client_info = await self.registry_service.get_client_info(registration.client_id)
            if client_info:
                async with self._lock:
                    self.clients[registration.client_id] = client_info

                # 注册到心跳服务
                await self.heartbeat_service.register_client(registration.client_id)

        return response
    
    async def unregister_client(self, client_id: str) -> bool:
        """注销客户端

        Args:
            client_id: 客户端ID

        Returns:
            bool: 是否成功注销
        """
        # Client 端不需要注销操作（或由子类实现）
        if self.registry_service is None:
            return False

        # Server 端：委托给注册服务
        success = await self.registry_service.unregister_client(client_id)

        if success:
            # 同步到本地状态
            async with self._lock:
                self.clients.pop(client_id, None)
                self.heartbeat_status.pop(client_id, None)

            # 从心跳服务注销
            await self.heartbeat_service.unregister_client(client_id)

        return success
    
    async def update_client_info(self, client_id: str, updates: Dict[str, Any]) -> bool:
        """更新客户端信息
        
        Args:
            client_id: 客户端ID
            updates: 更新信息
            
        Returns:
            bool: 是否成功更新
        """
        async with self._lock:
            if client_id in self.clients:
                client = self.clients[client_id]
                for key, value in updates.items():
                    if hasattr(client, key):
                        setattr(client, key, value)
                    else:
                        client.metadata[key] = value
                
                client.last_seen = datetime.now()
                return True
            return False
    
    async def get_client_info(self, client_id: str) -> Optional[ClientInfo]:
        """获取客户端信息

        Args:
            client_id: 客户端ID

        Returns:
            Optional[ClientInfo]: 客户端信息，不存在则返回None
        """
        if self.registry_service is None:
            return None
        return await self.registry_service.get_client_info(client_id)

    async def list_clients(self, filters: Dict[str, Any] = None) -> List[ClientInfo]:
        """列出客户端

        Args:
            filters: 过滤条件

        Returns:
            List[ClientInfo]: 客户端列表
        """
        if self.registry_service is None:
            return []
        return await self.registry_service.list_clients()
    
    def get_active_clients(self) -> List[str]:
        """获取活跃客户端列表"""
        return self.heartbeat_service.get_alive_clients()
    
    # ==================== 心跳管理方法 ====================
    
    async def start_heartbeat(self, interval: float = None) -> None:
        """启动心跳机制"""
        interval = interval or self.config.heartbeat_interval
        
        if self._heartbeat_task is None:
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop(interval))
        
        if self._heartbeat_check_task is None:
            self._heartbeat_check_task = asyncio.create_task(self._heartbeat_check_loop())
    
    async def stop_heartbeat(self) -> None:
        """停止心跳机制"""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None
        
        if self._heartbeat_check_task:
            self._heartbeat_check_task.cancel()
            try:
                await self._heartbeat_check_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_check_task = None
    
    async def send_heartbeat(self, target: str = None) -> bool:
        """发送心跳"""
        try:
            heartbeat = HeartbeatMessage(
                client_id=self.node_id,
                status="alive",
                metrics=await self._get_node_metrics()
            )
            
            if target:
                await self.transport.send(self.node_id, target, heartbeat)
            else:
                # 广播心跳
                active_clients = self.get_active_clients()
                if active_clients:
                    await self.transport.broadcast(self.node_id, active_clients, heartbeat)
            
            return True
            
        except Exception as e:
            self.logger.exception(f"Send heartbeat failed: {e}")
            return False
    
    async def handle_heartbeat(self, source: str, heartbeat: HeartbeatMessage) -> bool:
        """处理接收到的心跳"""
        try:
            async with self._lock:
                self.heartbeat_status[source] = datetime.now()
                
                # 更新客户端最后活跃时间
                if source in self.clients:
                    self.clients[source].last_seen = datetime.now()
            
            return True
            
        except Exception as e:
            self.logger.exception(f"Handle heartbeat failed: {e}")
            return False
    
    async def check_client_alive(self, client_id: str) -> bool:
        """检查客户端是否存活"""
        if client_id not in self.heartbeat_status:
            return False
        
        last_heartbeat = self.heartbeat_status[client_id]
        timeout_threshold = timedelta(seconds=self.config.heartbeat_timeout)
        
        return datetime.now() - last_heartbeat <= timeout_threshold
    
    async def _heartbeat_loop(self, interval: float):
        """心跳循环"""
        while self._running:
            try:
                await self.send_heartbeat()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.exception(f"Heartbeat loop error: {e}")
                await asyncio.sleep(interval)
    
    async def _heartbeat_check_loop(self):
        """心跳检查循环"""
        while self._running:
            try:
                await self._check_client_timeouts()
                await asyncio.sleep(self.config.heartbeat_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.exception(f"Heartbeat check loop error: {e}")
                await asyncio.sleep(self.config.heartbeat_interval)
    
    async def _check_client_timeouts(self):
        """检查客户端超时"""
        now = datetime.now()
        timeout_threshold = timedelta(seconds=self.config.heartbeat_timeout)
        
        timeout_clients = []
        for client_id, last_heartbeat in list(self.heartbeat_status.items()):
            if now - last_heartbeat > timeout_threshold:
                timeout_clients.append(client_id)
        
        # 处理超时客户端
        for client_id in timeout_clients:
            await self._handle_client_timeout(client_id)
    
    async def _handle_client_timeout(self, client_id: str):
        """处理客户端超时"""
        async with self._lock:
            # 从心跳状态中移除
            if client_id in self.heartbeat_status:
                del self.heartbeat_status[client_id]
            
            # 更新客户端状态
            if client_id in self.clients:
                # 可以选择移除或标记为超时
                # del self.clients[client_id]  # 移除
                # 或者标记状态
                self.clients[client_id].metadata['timeout'] = True
        
        # 触发客户端断开事件
        await self.transport.push_event(
            self.node_id,
            "system",  # 或其他监听者
            "CLIENT_DISCONNECTED",
            {"client_id": client_id, "reason": "timeout"}
        )
    
    # ==================== 消息路由方法 ====================
    
    async def send_business_message(self, target: str, message_type: str, data: Any) -> Any:
        """发送业务消息"""
        try:
            return await self.transport.send(self.node_id, target, {
                "message_type": message_type,
                "data": data,
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            raise CommunicationError(f"Send business message failed: {str(e)}")
    
    async def send_control_message(self, target: str, message_type: str, data: Any) -> bool:
        """发送控制消息"""
        try:
            await self.transport.push_event(self.node_id, target, message_type, data)
            return True
        except Exception as e:
            self.logger.exception(f"Send control message failed: {e}")
            return False
    
    async def broadcast_message(self, targets: List[str], message_type: str, data: Any) -> Dict[str, bool]:
        """广播消息"""
        results = {}
        for target in targets:
            try:
                await self.send_control_message(target, message_type, data)
                results[target] = True
            except Exception as e:
                self.logger.exception(f"Broadcast to {target} failed: {e}")
                results[target] = False
        
        return results
    
    def register_message_handler(self, message_type: str, handler: Callable) -> str:
        """注册消息处理器"""
        handler_id = f"{message_type}_{len(self.message_handlers)}"
        self.message_handlers[message_type] = handler

        return handler_id
    
    # ==================== 状态管理方法 ====================
    
    async def get_node_status(self) -> Dict[str, Any]:
        """获取节点状态"""
        return {
            "node_id": self.node_id,
            "running": self._running,
            "clients_count": len(self.clients),
            "active_clients": len(self.get_active_clients()),
            "heartbeat_status": len(self.heartbeat_status),
            "transport_running": self.transport.is_running(),
            "timestamp": datetime.now().isoformat()
        }
    
    async def get_connection_stats(self) -> Dict[str, Any]:
        """获取连接统计"""
        return {
            "total_connections": len(self.transport.connections),
            "active_clients": len(self.get_active_clients()),
            "registered_clients": len(self.clients),
            "message_handlers": len(self.message_handlers)
        }
    
    async def _get_node_metrics(self) -> Dict[str, Any]:
        """获取节点指标"""
        return {
            "clients_count": len(self.clients),
            "active_clients_count": len(self.get_active_clients()),
            "connections_count": len(self.transport.connections)
        }
    
    async def health_check(self) -> HealthStatus:
        """健康检查"""
        if not self._running:
            return HealthStatus.ERROR
        
        if not self.transport.is_running():
            return HealthStatus.ERROR
        
        active_ratio = len(self.get_active_clients()) / max(len(self.clients), 1)
        if active_ratio < 0.5:
            return HealthStatus.WARNING
        
        return HealthStatus.HEALTHY
    
    # ==================== 生命周期方法 ====================
    
    async def start(self) -> None:
        """启动通信管理器"""
        self._running = True
        self.logger.debug(f"[CommunicationManager] 开始启动: {self.node_id}")
        
        # 启动传输层
        self.logger.debug(f"[CommunicationManager] 启动传输层...")
        await self.transport.start()
        self.logger.debug(f"[CommunicationManager] 调用start_event_listener: {self.node_id}")
        await self.transport.start_event_listener(self.node_id)
        
        # 启动服务组件
        await self.start_services()
        
        # 启动心跳机制
        await self.start_heartbeat()
    
    async def stop(self) -> None:
        """停止通信管理器"""
        self._running = False

        # 停止心跳机制
        await self.stop_heartbeat()

        # 停止服务组件
        await self.stop_services()

        # 清理客户端注册表（适用于批量实验场景）
        if self.registry_service is not None and hasattr(self.registry_service, 'clients'):
            self.registry_service.clients.clear()
            self.logger.debug("✓ Client registry cleared")

        # 停止传输层
        await self.transport.stop()
    
    async def cleanup(self) -> None:
        """清理通信管理器资源"""
        async with self._lock:
            self.clients.clear()
            self.heartbeat_status.clear()
            self.message_handlers.clear()
        
        await self.transport.cleanup()
    
    def _setup_service_callbacks(self) -> None:
        """设置服务组件回调"""
        # 注册事件回调（只有 Server 端有 registry_service）
        if self.registry_service is not None:
            self.registry_service.register_event_callback(self._handle_registry_event)
        self.heartbeat_service.register_event_callback(self._handle_heartbeat_event)
        self.status_service.register_event_callback(self._handle_status_event)
        self.security_service.register_event_callback(self._handle_security_event)
    
    async def _handle_registry_event(self, event) -> None:
        """处理注册服务事件"""
        self.logger.debug(f"[基类注册事件处理器] 收到事件: {event.event_type}, 源: {event.source_id}")

        # 由于子类可能有自己的注册逻辑，这里我们不做具体处理
        # 只是确保事件被正确记录
        if hasattr(event, 'event_type'):
            self.logger.debug(f"[基类注册事件处理器] 事件类型: {event.event_type}")
        if hasattr(event, 'source_id'):
            self.logger.debug(f"[基类注册事件处理器] 事件源: {event.source_id}")
        if hasattr(event, 'data'):
            self.logger.debug(f"[基类注册事件处理器] 事件数据: {event.data}")

        # 注意：实际的事件处理由 FederationServer 的桥接回调处理
    
    async def _handle_heartbeat_event(self, event) -> None:
        """处理心跳服务事件"""
        # 可以根据需要处理不同的心跳事件
        pass
    
    async def _handle_status_event(self, event) -> None:
        """处理状态服务事件"""
        # 可以根据需要处理不同的状态事件
        pass
    
    async def _handle_security_event(self, event) -> None:
        """处理安全服务事件"""
        # 可以根据需要处理不同的安全事件
        pass
    
    async def start_services(self) -> None:
        """启动所有服务组件"""
        await self.heartbeat_service.start()
        await self.status_service.start()
        await self.security_service.start()
        
        # 更新状态服务的节点信息
        await self.status_service.update_client_status(self.node_id, {
            "node_id": self.node_id,
            "status": "running",
            "transport_type": self.transport.__class__.__name__
        })
    
    async def stop_services(self) -> None:
        """停止所有服务组件"""
        await self.heartbeat_service.stop()
        await self.status_service.stop()
        await self.security_service.stop()