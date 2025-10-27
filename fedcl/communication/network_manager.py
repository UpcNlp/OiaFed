"""
MOE-FedCL 统一通信管理器 (支持Network和Process模式)
moe_fedcl/communication/network_manager.py
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, Optional

import aiohttp

from .base import CommunicationManagerBase
from ..exceptions import RegistrationError
from ..transport.network import NetworkTransport
from ..types import (
    ClientInfo, RegistrationRequest, RegistrationResponse,
    HeartbeatMessage, CommunicationConfig, RegistrationStatus
)
from ..utils.auto_logger import get_sys_logger


class NetworkCommunicationManager(CommunicationManagerBase):
    """统一通信管理器 - 支持Network模式和Process模式"""

    def __init__(self,
                 node_id: str,
                 transport: NetworkTransport,
                 config: CommunicationConfig,
                 node_role: str = None):
        """
        初始化网络通信管理器

        Args:
            node_id: 节点ID
            transport: 网络传输层实例
            config: 通信配置
            node_role: 节点角色 ('server' 或 'client')。如果为 None，则从 node_id 推断
        """
        super().__init__(node_id, transport, config)

        # 显式设置节点角色（更可靠的判断方式）
        if node_role is not None:
            self.node_role = node_role.lower()
        else:
            # 向后兼容：如果没有显式指定，从node_id推断
            self.node_role = "server" if "server" in node_id.lower() else "client"

        # 检测模式（向后兼容）
        self.is_process_mode = "process_" in node_id
        self.is_network_mode = "network_" in node_id

        # 网络特定配置
        # 从 transport 的配置中读取服务器地址
        self.server_url = self._get_server_url_from_config()
        self._client_session: Optional[aiohttp.ClientSession] = None

        # WebSocket连接状态
        self._ws_connection: Optional[aiohttp.ClientWebSocketResponse] = None
        self._ws_listener_task: Optional[asyncio.Task] = None

        # Process模式特定初始化
        if self.is_process_mode:
            self._init_process_mode()

        self.logger = get_sys_logger()
    
    def _init_process_mode(self):
        """初始化Process模式特定功能"""
        # 如果是服务器节点，添加"system"目标ID到transport
        if "server" in self.node_id.lower() and hasattr(self.transport, 'add_target_id'):
            self.transport.add_target_id("system")
            print(f"[NetworkCommunicationManager] 服务器节点添加system目标ID (Process模式)")

    def _get_server_url_from_config(self) -> str:
        """从配置中获取服务器URL

        Returns:
            str: 服务器URL，格式为 http://host:port
        """
        # 如果是服务端，不需要服务器URL
        if self.node_role == "server":
            # 返回本地地址供调试使用
            host = getattr(self.transport, 'host', '127.0.0.1')
            port = getattr(self.transport, 'port', 8000)
            return f"http://{host}:{port}"

        # 客户端：从 transport 配置中读取服务器地址
        server_host = getattr(self.transport, 'server_host', None)
        server_port = getattr(self.transport, 'server_port', None)

        if server_host and server_port:
            return f"http://{server_host}:{server_port}"

        # 如果配置中没有指定，使用默认值
        if self.is_process_mode:
            return "http://127.0.0.1:8000"
        else:
            # Network模式默认值（向后兼容）
            self.logger.warning(
                "客户端配置中未指定 server_host 和 server_port，使用默认值 localhost:8000。"
                "建议在配置文件的 transport 部分添加 server_host 和 server_port。"
            )
            return "http://localhost:8000"

    
    def _is_client_node(self) -> bool:
        """统一的客户端节点判断

        使用显式的 node_role 字段进行判断，不依赖 node_id 的命名规则。

        Returns:
            bool: True 表示客户端，False 表示服务端
        """
        return self.node_role == "client"
    
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

                # 客户端注册成功后，直接返回（不在客户端本地保存）
                # 客户端地址应该在服务端的transport中注册
                return RegistrationResponse(
                    success=True,
                    client_id=client_id,
                    server_info={
                        "server_id": self.node_id,
                        "registration_mode": "client_to_server"
                    }
                )

            # 以下代码只在服务端执行（处理客户端的注册请求）

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

            # 从metadata中提取客户端地址信息并注册到transport（服务端的transport）
            if hasattr(self.transport, 'register_client_address'):
                client_address = registration.metadata.get("client_address")
                if client_address:
                    self.transport.register_client_address(client_id, client_address)
                    self.logger.info(f"已注册客户端地址到服务端transport: {client_id} -> {client_address.get('url')}")
                else:
                    self.logger.warning(f"客户端 {client_id} 注册时未提供地址信息")

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

            # 从transport中注销客户端地址
            if hasattr(self.transport, 'unregister_client_address'):
                self.transport.unregister_client_address(client_id)

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
                # 客户端发送到服务端
                if self._is_client_node():
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
        """解析目标URL

        Args:
            target: 目标ID（通常是客户端ID）

        Returns:
            Optional[str]: 目标的URL地址，如果找不到则返回 None
        """
        # 优先从 transport 层获取客户端地址（服务端已在注册时保存）
        if hasattr(self.transport, 'get_client_address'):
            client_url = self.transport.get_client_address(target)
            if client_url:
                return client_url

        # 向后兼容：从 node_id 格式解析（不推荐）
        # network_client_192.168.1.100_8001
        if target.startswith("network_"):
            parts = target.split("_")
            if len(parts) >= 4:
                try:
                    host = parts[2]
                    port = int(parts[3])
                    self.logger.warning(
                        f"从 node_id 解析客户端地址（不推荐）: {target} -> http://{host}:{port}"
                    )
                    return f"http://{host}:{port}"
                except (ValueError, IndexError):
                    pass

        self.logger.error(f"无法解析目标地址: {target}")
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
            self.logger.debug(f"registering request handler, transport:{self.transport}")
    
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

            return {"error": f"Invalid request format, data:{message_type}, message:{self.message_handlers}, self_node: {self.node_id}"}
            
        except Exception as e:
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
    
    async def connect_websocket(self):
        """连接WebSocket"""
        try:
            # 只有客户端才连接到服务端的WebSocket
            if self._is_client_node():
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