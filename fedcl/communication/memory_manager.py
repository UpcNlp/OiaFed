"""
MOE-FedCL Memory模式通信管理器
moe_fedcl/communication/memory_manager.py
"""

import asyncio
from typing import Any, Dict, List, Optional
from datetime import datetime

from .base import CommunicationManagerBase
from ..transport.memory import MemoryTransport
from ..types import (
    ClientInfo, RegistrationRequest, RegistrationResponse,
    HeartbeatMessage, CommunicationConfig, RegistrationStatus
)
from ..exceptions import RegistrationError


class MemoryCommunicationManager(CommunicationManagerBase):
    """Memory模式通信管理器 - 同进程内共享状态管理"""
    
    # 全局共享的客户端注册表
    _global_client_registry: Dict[str, ClientInfo] = {}
    
    def __init__(self, node_id: str, transport: MemoryTransport, config: CommunicationConfig):
        super().__init__(node_id, transport, config)
        
        # Memory模式下使用全局共享状态
        self.clients = self._global_client_registry
    
    async def register_client(self, registration: RegistrationRequest) -> RegistrationResponse:
        """注册客户端"""
        try:
            client_id = registration.client_id
            
            # 检查是否已注册
            if client_id in self.clients:
                return RegistrationResponse(
                    success=False,
                    client_id=client_id,
                    error_message=f"Client {client_id} already registered"
                )
            
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
            
            # 添加到注册表服务
            response = await self.registry_service.register_client(registration)
            
            # 如果注册成功，添加到本地字典和心跳状态（用于向后兼容）
            if response.success:
                async with self._lock:
                    self.clients[client_id] = client_info
                    self.heartbeat_status[client_id] = datetime.now()
                
                # 触发客户端注册事件
                await self.transport.push_event(
                    self.node_id,
                    "system",
                    "CLIENT_REGISTERED",
                    {"client_id": client_id, "client_info": client_info}
                )
            
            return response
            
        except Exception as e:
            raise RegistrationError(f"Client registration failed: {str(e)}")
    
    async def unregister_client(self, client_id: str) -> bool:
        """注销客户端"""
        try:
            async with self._lock:
                # 检查客户端是否存在
                if client_id not in self.clients:
                    return False
                
                # 移除客户端
                del self.clients[client_id]
                
                # 移除心跳状态
                if client_id in self.heartbeat_status:
                    del self.heartbeat_status[client_id]
            
            # 触发客户端注销事件
            await self.transport.push_event(
                self.node_id,
                "system", 
                "CLIENT_UNREGISTERED",
                {"client_id": client_id, "reason": "manual"}
            )
            
            return True
            
        except Exception as e:
            print(f"Client unregistration failed: {e}")
            return False
    
    async def handle_rpc_request(self, source: str, data: Any) -> Any:
        """处理RPC请求 - Memory模式专用"""
        try:
            message_type = data.get("message_type", "unknown")
            request_data = data.get("data")
            
            # 根据消息类型分发处理
            if message_type == "registration":
                # 处理注册请求
                registration = RegistrationRequest(**request_data)
                response = await self.register_client(registration)
                return response
            
            elif message_type == "heartbeat":
                # 处理心跳
                heartbeat = HeartbeatMessage(**request_data)
                success = await self.handle_heartbeat(source, heartbeat)
                return {"success": success, "timestamp": datetime.now().isoformat()}
            
            elif message_type in self.message_handlers:
                # 调用注册的处理器
                handler = self.message_handlers[message_type]
                if asyncio.iscoroutinefunction(handler):
                    return await handler(source, request_data)
                else:
                    return handler(source, request_data)
            
            else:
                return {"error": f"Unknown message type: {message_type}"}
                
        except Exception as e:
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
    
    async def setup_message_handling(self):
        """设置消息处理"""
        # 注册自身的RPC处理器到Transport
        if hasattr(self.transport, 'register_request_handler'):
            self.transport.register_request_handler(
                self.node_id,
                self.handle_rpc_request
            )
    
    async def send_to_client(self, client_id: str, message_type: str, data: Any) -> Any:
        """向指定客户端发送消息 - Memory模式快速通道"""
        try:
            # 检查客户端是否存在
            if client_id not in self.clients:
                raise ValueError(f"Client {client_id} not found")
            
            # 直接发送消息
            return await self.send_business_message(client_id, message_type, data)
            
        except Exception as e:
            print(f"Send to client {client_id} failed: {e}")
            raise
    
    async def broadcast_to_clients(self, message_type: str, data: Any, client_filter: Dict[str, Any] = None) -> Dict[str, Any]:
        """广播消息到客户端"""
        results = {}
        
        # 获取目标客户端列表
        if client_filter:
            clients = await self.list_clients(client_filter)
            client_ids = [c.client_id for c in clients]
        else:
            client_ids = list(self.clients.keys())
        
        # 并发发送消息
        tasks = []
        for client_id in client_ids:
            task = asyncio.create_task(self.send_to_client(client_id, message_type, data))
            tasks.append((client_id, task))
        
        # 收集结果
        for client_id, task in tasks:
            try:
                result = await task
                results[client_id] = {"success": True, "result": result}
            except Exception as e:
                results[client_id] = {"success": False, "error": str(e)}
        
        return results
    
    def get_shared_state(self) -> Dict[str, Any]:
        """获取共享状态 - Memory模式调试用"""
        return {
            "global_clients": list(self._global_client_registry.keys()),
            "local_clients": list(self.clients.keys()),
            "heartbeat_status": list(self.heartbeat_status.keys()),
            "message_handlers": list(self.message_handlers.keys())
        }
    
    async def simulate_client_activity(self, client_id: str, activity: str, data: Any = None):
        """模拟客户端活动 - Memory模式测试用"""
        if client_id in self.clients:
            async with self._lock:
                self.clients[client_id].last_seen = datetime.now()
                self.clients[client_id].metadata['last_activity'] = activity
                if data:
                    self.clients[client_id].metadata['activity_data'] = data
                
                # 更新心跳状态
                self.heartbeat_status[client_id] = datetime.now()
    
    async def reset_shared_state(self):
        """重置共享状态 - 测试清理用"""
        async with self._lock:
            self._global_client_registry.clear()
            self.heartbeat_status.clear()
            self.message_handlers.clear()
        
        # 清理传输层状态
        if hasattr(self.transport, 'clear_queues'):
            self.transport.clear_queues()
    
    async def start(self) -> None:
        """启动Memory通信管理器"""
        await super().start()
        
        # 设置消息处理
        await self.setup_message_handling()
    
    async def get_debug_info(self) -> Dict[str, Any]:
        """获取调试信息"""
        base_info = await self.get_node_status()
        memory_info = {
            "shared_state": self.get_shared_state(),
            "transport_queues": getattr(self.transport, 'get_queue_status', lambda: {})()
        }
        
        return {**base_info, **memory_info}