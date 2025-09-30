"""
MOE-FedCL 统一通信管理器 (支持Network和Process模式)
moe_fedcl/communication/network_manager.py
"""

import asyncio
import aiohttp
import json
from typing import Any, Dict, List, Optional
from datetime import datetime

from .base import CommunicationManagerBase
from ..transport.network import NetworkTransport
from ..types import (
    ClientInfo, RegistrationRequest, RegistrationResponse,
    HeartbeatMessage, CommunicationConfig, RegistrationStatus
)
from ..exceptions import RegistrationError, CommunicationError


class NetworkCommunicationManager(CommunicationManagerBase):
    """统一通信管理器 - 支持Network模式和Process模式"""
    
    def __init__(self, node_id: str, transport: NetworkTransport, config: CommunicationConfig):
        super().__init__(node_id, transport, config)
        
        # 检测模式
        self.is_process_mode = "process_" in node_id
        self.is_network_mode = "network_" in node_id
        
        # 网络特定配置
        self.server_url = self._get_server_url()
        self._client_session: Optional[aiohttp.ClientSession] = None
        
        # WebSocket连接状态
        self._ws_connection: Optional[aiohttp.ClientWebSocketResponse] = None
        self._ws_listener_task: Optional[asyncio.Task] = None
        
        # Process模式特定初始化
        if self.is_process_mode:
            self._init_process_mode()
    
    def _init_process_mode(self):
        """初始化Process模式特定功能"""
        # 如果是服务器节点，添加"system"目标ID到transport
        if "server" in self.node_id.lower() and hasattr(self.transport, 'add_target_id'):
            self.transport.add_target_id("system")
            print(f"[NetworkCommunicationManager] 服务器节点添加system目标ID (Process模式)")
    
    def _get_server_url(self) -> str:
        """统一的服务器URL解析"""
        # 从节点ID解析服务器地址
        if self.node_id.startswith(("network_server", "process_server")):
            parts = self.node_id.split("_")
            if len(parts) >= 4:
                host = parts[2] if not self.is_process_mode else "127.0.0.1"  # Process模式强制本地
                port = parts[3]
                return f"http://{host}:{port}"
        
        # 默认地址
        return "http://127.0.0.1:8000" if self.is_process_mode else "http://localhost:8000"
    
    def _is_client_node(self) -> bool:
        """统一的客户端节点判断"""
        return self.node_id.startswith(("network_client", "process_client"))
    
    async def register_client(self, registration: RegistrationRequest) -> RegistrationResponse:
        """注册客户端 - 通过HTTP API"""
        try:
            client_id = registration.client_id
            
            # 检查本地是否已注册
            if client_id in self.clients:
                return RegistrationResponse(
                    success=False,
                    client_id=client_id,
                    error_message=f"Client {client_id} already registered locally"
                )
            
            # 向服务端发送注册请求
            if self._is_client_node():
                # 客户端向服务端注册
                success = await self._register_to_server(registration)
                if not success:
                    return RegistrationResponse(
                        success=False,
                        client_id=client_id,
                        error_message="Failed to register with server"
                    )
            
            # 创建本地客户端信息
            client_info = ClientInfo(
                client_id=client_id,
                client_type=registration.client_type,
                capabilities=registration.capabilities,
                metadata=registration.metadata,
                registration_time=datetime.now(),
                last_seen=datetime.now(),
                status=RegistrationStatus.REGISTERED
            )
            
            # 添加到本地注册表
            async with self._lock:
                self.clients[client_id] = client_info
                self.heartbeat_status[client_id] = datetime.now()
            
            # 触发客户端注册事件
            await self.transport.push_event(
                self.node_id,
                "system",
                "CLIENT_REGISTERED",
                {"client_id": client_id, "client_info": client_info.__dict__}
            )
            
            return RegistrationResponse(
                success=True,
                client_id=client_id,
                server_info={
                    "server_id": self.node_id,
                    "server_url": self.server_url,
                    "capabilities": ["train", "evaluate", "aggregate"],
                    "heartbeat_interval": self.config.heartbeat_interval,
                    "network_mode": True
                }
            )
            
        except Exception as e:
            raise RegistrationError(f"Client registration failed: {str(e)}")
    
    async def _register_to_server(self, registration: RegistrationRequest) -> bool:
        """向服务端注册"""
        try:
            if not self._client_session:
                self._client_session = aiohttp.ClientSession()
            
            registration_data = {
                "client_id": registration.client_id,
                "client_type": registration.client_type,
                "capabilities": registration.capabilities,
                "metadata": registration.metadata,
                "timestamp": datetime.now().isoformat()
            }
            
            url = f"{self.server_url}/api/v1/register"
            timeout = aiohttp.ClientTimeout(total=self.config.registration_timeout)
            
            async with self._client_session.post(
                url, 
                json=registration_data, 
                timeout=timeout
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("success", False)
                else:
                    print(f"Registration failed: HTTP {response.status}")
                    return False
                    
        except Exception as e:
            print(f"Register to server failed: {e}")
            return False
    
    async def unregister_client(self, client_id: str) -> bool:
        """注销客户端"""
        try:
            # 从本地注册表移除
            async with self._lock:
                if client_id not in self.clients:
                    return False
                
                del self.clients[client_id]
                
                if client_id in self.heartbeat_status:
                    del self.heartbeat_status[client_id]
            
            # 如果是客户端节点，向服务端发送注销请求
            if self._is_client_node():
                await self._unregister_from_server(client_id)
            
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
    
    async def _unregister_from_server(self, client_id: str) -> bool:
        """从服务端注销"""
        try:
            if not self._client_session:
                return True  # 没有会话，认为已注销
            
            url = f"{self.server_url}/api/v1/unregister"
            data = {
                "client_id": client_id,
                "timestamp": datetime.now().isoformat()
            }
            
            timeout = aiohttp.ClientTimeout(total=10.0)
            async with self._client_session.post(url, json=data, timeout=timeout) as response:
                return response.status == 200
                
        except Exception as e:
            print(f"Unregister from server failed: {e}")
            return False
    
    async def send_heartbeat(self, target: str = None) -> bool:
        """发送心跳 - Network模式通过HTTP"""
        try:
            heartbeat_data = {
                "client_id": self.node_id,
                "status": "alive",
                "metrics": await self._get_node_metrics(),
                "timestamp": datetime.now().isoformat()
            }
            
            if target:
                # 发送到特定目标
                target_url = self._parse_target_url(target)
                if target_url:
                    url = f"{target_url}/api/v1/heartbeat"
                    return await self._send_http_request(url, heartbeat_data)
            else:
                # 发送到服务端
                if self.node_id.startswith("network_client"):
                    url = f"{self.server_url}/api/v1/heartbeat"
                    return await self._send_http_request(url, heartbeat_data)
            
            return True
            
        except Exception as e:
            print(f"Send heartbeat failed: {e}")
            return False
    
    async def _send_http_request(self, url: str, data: Dict[str, Any]) -> bool:
        """发送HTTP请求"""
        try:
            if not self._client_session:
                self._client_session = aiohttp.ClientSession()
            
            timeout = aiohttp.ClientTimeout(total=10.0)
            async with self._client_session.post(url, json=data, timeout=timeout) as response:
                return response.status == 200
                
        except Exception as e:
            print(f"HTTP request failed: {e}")
            return False
    
    def _parse_target_url(self, target: str) -> Optional[str]:
        """解析目标URL"""
        # network_server_192.168.1.100_8000
        if target.startswith("network_"):
            parts = target.split("_")
            if len(parts) >= 4:
                try:
                    host = parts[2]
                    port = int(parts[3])
                    return f"http://{host}:{port}"
                except (ValueError, IndexError):
                    pass
        return None
    
    async def handle_registration_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理注册请求 - 服务端使用"""
        try:
            registration = RegistrationRequest(
                client_id=request_data["client_id"],
                client_type=request_data.get("client_type", "learner"),
                capabilities=request_data.get("capabilities", []),
                metadata=request_data.get("metadata", {})
            )
            
            response = await self.register_client(registration)
            
            return {
                "success": response.success,
                "client_id": response.client_id,
                "server_info": response.server_info,
                "error_message": response.error_message,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error_message": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def handle_heartbeat_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理心跳请求"""
        try:
            heartbeat = HeartbeatMessage(
                client_id=request_data["client_id"],
                status=request_data.get("status", "alive"),
                metrics=request_data.get("metrics", {}),
                timestamp=datetime.now()
            )
            
            success = await self.handle_heartbeat(request_data["client_id"], heartbeat)
            
            return {
                "success": success,
                "server_status": "alive",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def setup_http_handlers(self):
        """设置HTTP处理器"""
        # 注册HTTP处理器到传输层
        if hasattr(self.transport, 'register_request_handler'):
            self.transport.register_request_handler(self.node_id, self.handle_http_request)
    
    async def handle_http_request(self, source: str, data: Any) -> Any:
        """统一HTTP请求处理"""
        try:
            if isinstance(data, dict):
                message_type = data.get("message_type", "unknown")
                request_data = data.get("data", {})
                
                if message_type == "registration":
                    return await self.handle_registration_request(request_data)
                elif message_type == "heartbeat":
                    return await self.handle_heartbeat_request(request_data)
                elif message_type in self.message_handlers:
                    handler = self.message_handlers[message_type]
                    if asyncio.iscoroutinefunction(handler):
                        return await handler(source, request_data)
                    else:
                        return handler(source, request_data)
            
            return {"error": "Invalid request format"}
            
        except Exception as e:
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
    
    async def connect_websocket(self):
        """连接WebSocket"""
        try:
            if self.node_id.startswith("network_client"):
                ws_url = f"ws://{self.server_url.replace('http://', '')}/ws/events"
                
                if not self._client_session:
                    self._client_session = aiohttp.ClientSession()
                
                self._ws_connection = await self._client_session.ws_connect(ws_url)
                
                # 注册WebSocket连接
                await self._ws_connection.send_str(json.dumps({
                    "type": "register",
                    "client_id": self.node_id
                }))
                
                # 启动WebSocket监听器
                self._ws_listener_task = asyncio.create_task(self._ws_listener())
                
                return True
                
        except Exception as e:
            print(f"WebSocket connection failed: {e}")
            return False
    
    async def _ws_listener(self):
        """WebSocket监听器"""
        if not self._ws_connection:
            return
        
        try:
            async for msg in self._ws_connection:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        await self._handle_ws_message(data)
                    except json.JSONDecodeError:
                        print(f"Invalid WebSocket message: {msg.data}")
                        
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    print(f"WebSocket error: {self._ws_connection.exception()}")
                    break
                    
        except Exception as e:
            print(f"WebSocket listener error: {e}")
        finally:
            if self._ws_connection and not self._ws_connection.closed:
                await self._ws_connection.close()
    
    async def _handle_ws_message(self, data: Dict[str, Any]):
        """处理WebSocket消息"""
        try:
            if data.get("type") == "event":
                await self._handle_event(
                    data.get("event_type"),
                    data.get("source"),
                    data.get("data")
                )
                
        except Exception as e:
            print(f"Handle WebSocket message error: {e}")
    
    def get_network_stats(self) -> Dict[str, Any]:
        """获取网络统计信息"""
        return {
            "server_url": self.server_url,
            "session_active": self._client_session is not None,
            "websocket_connected": self._ws_connection is not None and not self._ws_connection.closed,
            "local_clients": len(self.clients),
            "active_clients": len(self.get_active_clients())
        }
    
    async def start(self) -> None:
        """启动Network通信管理器"""
        await super().start()
        
        # 设置HTTP处理器
        await self.setup_http_handlers()
        
        # 如果是客户端，尝试连接WebSocket
        if self._is_client_node() and self.is_network_mode:
            await self.connect_websocket()
    
    async def stop(self) -> None:
        """停止Network通信管理器"""
        await super().stop()
        
        # 关闭WebSocket连接
        if self._ws_listener_task:
            self._ws_listener_task.cancel()
            try:
                await self._ws_listener_task
            except asyncio.CancelledError:
                pass
        
        if self._ws_connection and not self._ws_connection.closed:
            await self._ws_connection.close()
        
        # 关闭HTTP会话
        if self._client_session:
            await self._client_session.close()
    
    async def cleanup(self) -> None:
        """清理Network通信管理器资源"""
        await super().cleanup()