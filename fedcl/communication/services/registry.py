"""
MOE-FedCL 客户端注册服务
moe_fedcl/communication/services/registry.py
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Callable

from ...exceptions import RegistrationError
from ...types import (
    ClientInfo, RegistrationRequest, RegistrationResponse,
    RegistrationStatus, EventMessage
)
from ...utils.auto_logger import get_comm_logger


class ClientRegistryService:
    """客户端注册服务
    
    负责处理客户端的注册、注销和状态管理
    """
    
    def __init__(self, max_clients: int = 100):
        """初始化注册服务
        
        Args:
            max_clients: 最大客户端数量
        """
        self.max_clients = max_clients
        self.logger = get_comm_logger("sys")
        
        # 客户端注册表
        self.clients: Dict[str, ClientInfo] = {}

        
        # 事件回调
        self.event_callbacks: List[Callable[[EventMessage], None]] = []
        
        # 锁保护并发操作
        self._lock = asyncio.Lock()

    async def register_client(self, registration: RegistrationRequest) -> RegistrationResponse:
        """注册客户端（服务端方法）

        Args:
            registration: 注册请求

        Returns:
            RegistrationResponse: 注册响应
        """
        async with self._lock:
            try:
                # 验证注册请求
                await self._validate_registration(registration)

                client_id = registration.client_id

                # 创建客户端信息
                client_info = ClientInfo(
                    client_id=client_id,
                    client_type=registration.client_type,
                    capabilities=registration.capabilities,
                    metadata=registration.metadata,
                    registration_time=datetime.now(),
                    last_seen=datetime.now(),
                    status=RegistrationStatus.REGISTERED
                )

                # 添加到注册表
                self.clients[client_id] = client_info

                # 记录日志
                self.logger.info(f"Client {client_id} registered successfully")

                # 触发注册事件
                await self._emit_event("CLIENT_REGISTERED", client_id, client_info)

                # 返回成功响应
                return RegistrationResponse(
                    success=True,
                    client_id=client_id,
                    server_info={
                        "message": f"Client {client_id} registered successfully",
                        "registration_time": client_info.registration_time.isoformat()
                    }
                )

            except RegistrationError as e:
                # 注册失败
                self.logger.warning(f"Client registration failed: {e}")
                return RegistrationResponse(
                    success=False,
                    client_id=registration.client_id,
                    error_message=str(e)
                )
            except Exception as e:
                # 未预期的错误
                self.logger.error(f"Unexpected error during client registration: {e}")
                return RegistrationResponse(
                    success=False,
                    client_id=registration.client_id,
                    error_message=f"Registration failed: {str(e)}"
                )

    async def unregister_client(self, client_id: str) -> bool:
        """注销客户端
        
        Args:
            client_id: 客户端ID
            
        Returns:
            bool: 是否成功注销
        """
        async with self._lock:
            if client_id not in self.clients:
                self.logger.warning(f"Client {client_id} not found for unregistration")
                return False
            
            try:
                # 获取客户端信息
                client_info = self.clients[client_id]
                
                # 从注册表移除
                del self.clients[client_id]

                
                # 记录日志
                self.logger.info(f"Client {client_id} unregistered successfully")
                
                # 触发注销事件
                await self._emit_event("CLIENT_UNREGISTERED", client_id, client_info)
                
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to unregister client {client_id}: {e}")
                return False
    
    async def update_client_status(self, client_id: str, status: RegistrationStatus) -> bool:
        """更新客户端状态
        
        Args:
            client_id: 客户端ID
            status: 新状态
            
        Returns:
            bool: 是否成功更新
        """
        async with self._lock:
            if client_id not in self.clients:
                return False
            
            old_status = self.clients[client_id].status
            self.clients[client_id].status = status
            self.clients[client_id].last_seen = datetime.now()
            
            if old_status != status:
                self.logger.info(f"Client {client_id} status changed from {old_status} to {status}")
                await self._emit_event("CLIENT_STATUS_CHANGED", client_id, {
                    "old_status": old_status,
                    "new_status": status
                })
            
            return True
    
    async def get_client_info(self, client_id: str) -> Optional[ClientInfo]:
        """获取客户端信息
        
        Args:
            client_id: 客户端ID
            
        Returns:
            Optional[ClientInfo]: 客户端信息，不存在则返回None
        """
        return self.clients.get(client_id)
    
    async def list_clients(self, status_filter: Optional[RegistrationStatus] = None) -> List[ClientInfo]:
        """列出客户端
        
        Args:
            status_filter: 状态过滤器
            
        Returns:
            List[ClientInfo]: 客户端列表
        """
        clients = list(self.clients.values())
        
        if status_filter:
            clients = [c for c in clients if c.status == status_filter]
        
        return clients
    
    async def get_active_clients(self) -> List[str]:
        """获取活跃客户端ID列表
        
        Returns:
            List[str]: 活跃客户端ID列表
        """
        active_statuses = {RegistrationStatus.REGISTERED, RegistrationStatus.ACTIVE}
        return [
            client_id for client_id, client_info in self.clients.items()
            if client_info.status in active_statuses
        ]
    
    def get_client_count(self) -> int:
        """获取客户端总数"""
        return len(self.clients)
    
    def is_full(self) -> bool:
        """检查是否已达到最大客户端数量"""
        return len(self.clients) >= self.max_clients
    
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
    
    async def _validate_registration(self, registration: RegistrationRequest) -> None:
        """验证注册请求
        
        Args:
            registration: 注册请求
            
        Raises:
            RegistrationError: 验证失败
        """
        if not registration.client_id:
            raise RegistrationError("Client ID is required")
        
        if not registration.client_id.strip():
            raise RegistrationError("Client ID cannot be empty")
        
        if len(registration.client_id) > 100:
            raise RegistrationError("Client ID too long (max 100 characters)")
        
        # 检查是否已达到最大客户端数量
        if self.is_full() and registration.client_id not in self.clients:
            raise RegistrationError(f"Maximum number of clients ({self.max_clients}) reached")
        
        # 检查客户端是否已存在
        if registration.client_id in self.clients:
            existing_client = self.clients[registration.client_id]
            if existing_client.status == RegistrationStatus.REGISTERED:
                raise RegistrationError(f"Client {registration.client_id} is already registered")
        
        # 验证客户端类型
        valid_types = {"learner", "trainer", "observer"}
        if registration.client_type not in valid_types:
            raise RegistrationError(f"Invalid client type: {registration.client_type}")
        
        # 验证能力列表
        valid_capabilities = {"train", "evaluate", "aggregate", "observe"}
        for capability in registration.capabilities:
            if capability not in valid_capabilities:
                raise RegistrationError(f"Invalid capability: {capability}")
    
    async def _emit_event(self, event_type: str, client_id: str, data: any = None) -> None:
        """触发事件
        
        Args:
            event_type: 事件类型
            client_id: 客户端ID
            data: 事件数据
        """
        self.logger.debug(f"[RegistryService] 触发事件: {event_type}, 客户端: {client_id}, 回调数量: {len(self.event_callbacks)}")
        
        event = EventMessage(
            event_type=event_type,
            source_id=client_id,
            data=data
        )
        
        # 异步调用所有回调
        for i, callback in enumerate(self.event_callbacks):
            try:
                callback_name = callback.__name__ if hasattr(callback, '__name__') else str(callback)
                self.logger.debug(f"[RegistryService] 调用回调 #{i+1}: {callback_name}")
                self.logger.debug(f"[RegistryService] 回调详情 #{i+1}: {callback}")
                
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
                self.logger.debug(f"[RegistryService] 回调 #{i+1} 执行成功")
            except Exception as e:
                self.logger.exception(f"[RegistryService] 回调 #{i+1} 执行失败: {e}")
