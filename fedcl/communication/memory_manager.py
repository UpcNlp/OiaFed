"""
MOE-FedCL Memory模式通信管理器
moe_fedcl/communication/memory_manager.py
"""

import asyncio
from datetime import datetime
from dataclasses import asdict
from typing import Dict, Any

from .base import CommunicationManagerBase
from ..exceptions import RegistrationError
from ..transport.memory import MemoryTransport
from ..types import (
    ClientInfo, RegistrationRequest, RegistrationResponse,
    HeartbeatMessage, CommunicationConfig, RegistrationStatus
)
from ..exceptions import RegistrationError
from ..utils.auto_logger import get_comm_logger

class MemoryCommunicationManager(CommunicationManagerBase):
    """Memory模式通信管理器 - 同进程内共享状态管理"""

    # 全局共享的客户端注册表（只有 Server 使用）
    _global_client_registry: Dict[str, ClientInfo] = {}

    def __init__(self, node_id: str, transport: MemoryTransport, config: CommunicationConfig, node_role: str = None):
        super().__init__(node_id, transport, config, node_role)
        self.logger = get_comm_logger(node_id)
        # Memory模式下使用全局共享状态（只有 Server 端）
        if self.node_role == "server":
            self.clients = self._global_client_registry

    async def register_client(self, registration: RegistrationRequest) -> RegistrationResponse:
        """注册客户端 - Memory模式

        根据节点角色区分行为：
        - Client 端：通过 transport 发送注册请求到 Server
        - Server 端：处理注册请求并保存客户端信息
        """
        try:
            client_id = registration.client_id

            # === Client 端行为：发送注册请求 ===
            if self.node_role == "client":
                # Memory模式下，客户端通过 transport 发送注册请求到服务端
                # 使用通用别名 "server"（服务端会同时注册到实际ID和别名）
                server_target = self._get_server_target()

                self.logger.debug(f"[Client {self.node_id}] 向服务端发送注册请求，目标: {server_target}")

                response_data = await self.transport.send(
                    self.node_id,
                    server_target,
                    {
                        "message_type": "registration",
                        "data": {
                            "client_id": client_id,
                            "client_type": registration.client_type,
                            "capabilities": registration.capabilities,
                            "metadata": registration.metadata
                        }
                    }
                )

                self.logger.debug(f"[Client {self.node_id}] 收到注册响应: {response_data}")

                # 解析响应
                return RegistrationResponse(**response_data) if isinstance(response_data, dict) else RegistrationResponse(success=False, client_id=client_id, error_message="Invalid response")

            # === Server 端行为：处理注册请求 ===
            self.logger.debug(f"[Server {self.node_id}] 处理客户端注册请求: {client_id}")

            # 检查是否已注册
            if client_id in self.clients:
                return RegistrationResponse(
                    success=False,
                    client_id=client_id,
                    error_message=f"Client {client_id} already registered"
                )

            # 委托给注册服务处理
            response = await self.registry_service.register_client(registration)

            # 如果注册成功，同步到本地状态
            if response.success:
                client_info = await self.registry_service.get_client_info(client_id)
                if client_info:
                    async with self._lock:
                        self.clients[client_id] = client_info
                        self.heartbeat_status[client_id] = datetime.now()

                    self.logger.info(f"[Server {self.node_id}] 客户端 {client_id} 注册成功")

                # 触发客户端注册事件
                await self.transport.push_event(
                    self.node_id,
                    "system",
                    "CLIENT_REGISTERED",
                    {"client_id": client_id, "client_info": client_info}
                )

            return response

        except Exception as e:
            self.logger.exception(f"Client registration failed: {e}")
            raise RegistrationError(f"Client registration failed: {str(e)}")

    def _get_server_target(self) -> str:
        """获取服务端目标ID

        Returns:
            str: 服务端ID或别名
        """
        # 可以从配置中读取，或使用默认别名
        if hasattr(self.config, 'server_id') and self.config.server_id:
            return self.config.server_id
        # 默认使用通用别名
        return "server"
    
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
                # 转换为字典返回（Memory模式RPC传输需要dict格式）
                return asdict(response)
            
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
            # 注册到实际的 node_id
            self.transport.register_request_handler(
                self.node_id,
                self.handle_rpc_request
            )

            # 如果是 Server 端，同时注册到通用别名 "server"
            if self.node_role == "server":
                self.logger.info(f"[MemoryCommunicationManager] Server 端注册到别名 'server'")
                self.transport.register_request_handler(
                    "server",
                    self.handle_rpc_request
                )
                self.logger.info(f"[MemoryCommunicationManager] Server 注册完成: {self.node_id} 和 'server'")
    
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

    @classmethod
    def clear_global_state(cls):
        """清理全局共享状态 - 类方法，用于演示或测试开始前的清理

        注意：这会清除所有Memory模式的共享状态，在演示开始前调用
        """
        cls._global_client_registry.clear()
        # 如果需要清理Transport的全局状态，也可以在这里添加
    
    async def start(self) -> None:
        """启动Memory通信管理器"""
        await super().start()

        # 设置消息处理
        await self.setup_message_handling()

    async def stop(self) -> None:
        """停止Memory通信管理器 - 清理全局共享状态"""
        self.logger.debug(f"[MemoryCommunicationManager] 开始停止: {self.node_id}, 角色: {self.node_role}")

        # 调用基类的stop方法
        await super().stop()

        # 🎯 关键修复：Memory模式下，清理全局共享注册表
        if self.node_role == "server":
            self.logger.debug(f"[MemoryCommunicationManager] 清理全局客户端注册表，当前数量: {len(self._global_client_registry)}")
            self._global_client_registry.clear()
            self.logger.info("✓ Global client registry cleared (Memory mode)")

        self.logger.debug(f"[MemoryCommunicationManager] 停止完成: {self.node_id}")

    async def get_debug_info(self) -> Dict[str, Any]:
        """获取调试信息"""
        base_info = await self.get_node_status()
        memory_info = {
            "shared_state": self.get_shared_state(),
            "transport_queues": getattr(self.transport, 'get_queue_status', lambda: {})()
        }
        
        return {**base_info, **memory_info}