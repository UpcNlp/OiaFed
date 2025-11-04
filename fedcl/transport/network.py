# 工具函数：递归将所有datetime对象转为字符串
def json_compatible(obj):
    if isinstance(obj, dict):
        return {k: json_compatible(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [json_compatible(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(json_compatible(v) for v in obj)
    elif hasattr(obj, 'isoformat') and callable(obj.isoformat):
        # 处理datetime/date等
        return obj.isoformat()
    else:
        return obj
"""
MOE-FedCL Network模式传输实现
moe_fedcl/transport/network.py
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Any, Dict, Callable, Optional

import aiohttp
from aiohttp import web, WSMsgType

from .base import TransportBase
from ..exceptions import TransportError, TimeoutError
from ..types import TransportConfig
from ..utils.auto_logger import get_sys_logger, get_comm_logger
from ..utils.serialization import CustomJSONEncoder, CustomJSONDecoder


class NetworkTransport(TransportBase):
    """网络传输实现 - 基于HTTP/WebSocket通信"""

    def __init__(self, config: TransportConfig):
        super().__init__(config)

        # 服务端配置
        self.host = config.specific_config.get("host", "127.0.0.1")  # 默认使用本地地址
        self.port = config.specific_config.get("port", 8000)

        self.websocket_port = config.specific_config.get("websocket_port", 9501)  # 改为9501避免冲突

        # 节点角色（从配置中获取，如果未指定则为None）
        self.node_role = config.specific_config.get("node_role", None)

        # 服务器地址（客户端使用）
        self.server_host = config.specific_config.get("server_host", None)
        self.server_port = config.specific_config.get("server_port", None)

        # HTTP客户端会话
        self._http_session: Optional[aiohttp.ClientSession] = None
        
        # 服务端组件
        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._ws_app: Optional[web.Application] = None
        self._ws_runner: Optional[web.AppRunner] = None
        
        # WebSocket连接管理
        self._ws_connections: Dict[str, aiohttp.web.WebSocketResponse] = {}
        self._pending_responses: Dict[str, asyncio.Future] = {}
        
        # 请求处理器
        self._request_handler: Optional[Callable] = None
        
        # Process模式兼容性：目标ID管理
        self.target_ids = set()  # 支持多个目标ID

        # 客户端地址缓存：保存客户端注册时提供的地址信息
        # {client_id: {"host": "127.0.0.1", "port": 8001, "url": "http://127.0.0.1:8001"}}
        self._client_addresses: Dict[str, Dict[str, Any]] = {}

        # 本地事件处理器（用于system事件）
        self._local_event_handlers = {}  # {event_type: [handlers]}

        self.logger = get_sys_logger()
        self.comm_logger = get_comm_logger(config.node_id)
        
        # 模拟Memory模式的全局事件监听器（类变量，跨实例共享）
        if not hasattr(NetworkTransport, '_global_event_listeners'):
            NetworkTransport._global_event_listeners = {}

    def _is_server_node(self) -> bool:
        """判断当前节点是否为服务端

        判断逻辑：
        1. 如果显式设置了 node_role，使用 node_role 判断
        2. 否则，向后兼容地从 node_id 推断（如果 node_id 包含 "server"）

        Returns:
            bool: True 表示服务端，False 表示客户端
        """
        if self.node_role is not None:
            # 显式指定了角色，使用显式角色
            return self.node_role.lower() == "server"
        elif hasattr(self, 'node_id') and self.node_id:
            # 向后兼容：从 node_id 推断
            return "server" in self.node_id.lower()
        else:
            # 默认为客户端
            return False
    
    async def send(self, source: str, target: str, data: Any) -> Any:
        """通过HTTP发送请求并等待响应"""
        if not self.validate_node_id(source) or not self.validate_node_id(target):
            raise TransportError(f"Invalid node ID: {source} -> {target}")
        
        # 解析目标地址
        target_url = self._parse_node_address(target)
        if not target_url:
            raise TransportError(f"Cannot parse target address: {target}")
        
        # 生成请求ID
        request_id = str(uuid.uuid4())
        
        request_data = {
            "request_id": request_id,
            "source": source,
            "target": target,
            "data": json_compatible(data),
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            if not self._http_session:
                self._http_session = aiohttp.ClientSession()

            url = f"{target_url}/api/v1/rpc"
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)

            # 使用CustomJSONEncoder序列化数据
            json_data = json.dumps(request_data, cls=CustomJSONEncoder)

            async with self._http_session.post(
                url,
                data=json_data,
                headers={'Content-Type': 'application/json'},
                timeout=timeout
            ) as response:
                if response.status == 200:
                    response_text = await response.text()
                    response_data = json.loads(response_text, cls=CustomJSONDecoder)
                    return response_data.get("result")
                else:
                    error_text = await response.text()
                    raise TransportError(f"HTTP {response.status}: {error_text}")
                    
        except asyncio.TimeoutError:
            raise TimeoutError(f"Send timeout after {self.config.timeout}s from {source} to {target}")
        except Exception as e:
            raise TransportError(f"Send failed from {source} to {target}: {str(e)}")
    
    async def receive(self, target: str, source: str = None, timeout: float = None) -> Any:
        """Network模式下接收通过HTTP服务器处理"""
        # 这个方法在Network模式下主要由HTTP服务器的路由处理
        # 这里提供一个基于Future的等待机制
        
        timeout = timeout or self.config.timeout
        
        # 创建一个Future来等待请求
        future = asyncio.Future()
        request_id = str(uuid.uuid4())
        self._pending_responses[request_id] = future
        
        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            if request_id in self._pending_responses:
                del self._pending_responses[request_id]
            raise TimeoutError(f"Receive timeout after {timeout}s for {target}")
    
    async def push_event(self, source: str, target: str, event_type: str, data: Any) -> bool:
        """通过WebSocket推送事件"""
        try:
            # 特殊处理：使用全局事件监听器（模拟Memory模式）
            self.comm_logger.debug(f"[NetworkTransport] 推送事件: {source} -> {target}, 类型: {event_type}")
            self.comm_logger.debug(f"[NetworkTransport] 当前全局事件监听器: {list(NetworkTransport._global_event_listeners.keys())}")
            
            if target in NetworkTransport._global_event_listeners:
                if event_type in NetworkTransport._global_event_listeners[target]:
                    handlers = NetworkTransport._global_event_listeners[target][event_type]
                    self.comm_logger.debug(f"[NetworkTransport] 找到 {len(handlers)} 个处理器用于 {target}.{event_type}")
                    
                    for i, handler in enumerate(handlers):
                        try:
                            self.comm_logger.debug(f"[NetworkTransport] 调用处理器 #{i+1}: {handler}")
                            if asyncio.iscoroutinefunction(handler):
                                await handler(json_compatible(data))
                            else:
                                handler(json_compatible(data))
                            self.comm_logger.debug(f"[NetworkTransport] 处理器 #{i+1} 执行成功")
                        except Exception as e:
                            self.comm_logger.exception(f"[NetworkTransport] 处理器 #{i+1} 执行失败: {e}")
                    return True
                else:
                    self.comm_logger.warning(f"[NetworkTransport] 目标 {target} 没有 {event_type} 事件监听器")
            else:
                self.comm_logger.warning(f"[NetworkTransport] 目标 {target} 没有注册任何事件监听器")
            
            # 查找目标WebSocket连接
            if target in self._ws_connections:
                ws = self._ws_connections[target]
                if not ws.closed:
                    event_data = {
                        "source": source,
                        "target": target,
                        "event_type": event_type,
                        "data": data,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    await ws.send_str(json.dumps(event_data))
                    return True
            
            # 如果没有WebSocket连接，尝试HTTP推送
            target_url = self._parse_node_address(target)
            if target_url:
                event_data = {
                    "source": source,
                    "event_type": event_type,
                    "data": data,
                    "timestamp": datetime.now().isoformat()
                }
                
                if not self._http_session:
                    self._http_session = aiohttp.ClientSession()
                
                url = f"{target_url}/api/v1/events"
                timeout = aiohttp.ClientTimeout(total=5.0)  # 短超时
                
                async with self._http_session.post(
                    url, 
                    json=event_data, 
                    timeout=timeout
                ) as response:
                    return response.status == 200
            
            return False
            
        except Exception as e:
            self.comm_logger.exception(f"Push event failed: {e}")
            return False
    
    async def start_event_listener(self, node_id: str) -> None:
        """启动事件监听器

        架构设计：
        - 服务器：启动HTTP服务器（处理注册、RPC）+ WebSocket服务器（双向通信）
        - 客户端：HTTP服务器（接收服务端请求）+ WebSocket客户端（双向通信）

        这样设计的优势：
        1. 支持双向通信（服务端可以主动请求客户端）
        2. WebSocket提供高效的双向实时通信
        3. HTTP处理传统的请求-响应操作
        4. 架构清晰，易于维护
        """
        self.node_id = node_id
        self.comm_logger.debug(f"[NetworkTransport] 开始启动事件监听器: {node_id}")

        if self._is_server_node():
            # 服务器节点：启动HTTP和WebSocket服务器
            self.add_target_id("system")
            self.comm_logger.debug(f"[NetworkTransport] 服务器节点自动添加system目标ID")

            try:
                self.comm_logger.debug(f"[NetworkTransport] 正在启动HTTP服务器: {self.host}:{self.port}")
                await self._start_http_server()
                self.comm_logger.debug(f"[NetworkTransport] HTTP服务器启动成功")
            except Exception as e:
                self.comm_logger.exception(f"[NetworkTransport] HTTP服务器启动失败: {e}")

            try:
                self.comm_logger.debug(f"[NetworkTransport] 正在启动WebSocket服务器: {self.host}:{self.websocket_port}")
                await self._start_websocket_server()
                self.comm_logger.debug(f"[NetworkTransport] WebSocket服务器启动成功")
            except Exception as e:
                self.comm_logger.exception(f"[NetworkTransport] WebSocket服务器启动失败: {e}")

            self.comm_logger.debug(f"[NetworkTransport] 服务器事件监听器已启动: {node_id} (HTTP:{self.port}, WS:{self.websocket_port})")
        else:
            # 客户端节点：启动HTTP服务器
            try:
                self.comm_logger.debug(f"[NetworkTransport] 正在启动HTTP服务器: {self.host}:{self.port}")
                await self._start_http_server()
                self.comm_logger.debug(f"[NetworkTransport] HTTP服务器启动成功")
            except Exception as e:
                self.comm_logger.exception(f"[NetworkTransport] HTTP服务器启动失败: {e}")

            # 客户端节点：作为WebSocket客户端连接到服务器，支持双向通信
            self.comm_logger.debug(f"[NetworkTransport] 客户端将通过WebSocket连接到服务器进行双向通信")
            self.comm_logger.debug(f"[NetworkTransport] 客户端事件监听器已启动: {node_id} (客户端模式)")

        self.comm_logger.debug(f"[NetworkTransport] 事件监听器已启动: {node_id}")
    
    async def _start_http_server(self):
        """启动HTTP服务器"""
        self._app = web.Application(client_max_size = 0)
        
        # 注册路由
        self._app.router.add_post("/api/v1/rpc", self._handle_rpc_request)
        self._app.router.add_post("/api/v1/register", self._handle_register_request)
        self._app.router.add_post("/api/v1/heartbeat", self._handle_heartbeat_request)
        self._app.router.add_post("/api/v1/events", self._handle_event_request)
        self._app.router.add_get("/api/v1/status", self._handle_status_request)
        
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self.comm_logger.debug(f"[NetworkTransport] HTTP服务器已设置")
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()

        # 获取实际分配的端口（如果使用了随机端口）
        if self.port == 0 and self._runner and self._runner.sites:
            for site in self._runner.sites:
                try:
                    # 安全地访问server对象
                    server = getattr(site, '_server', None)
                    if server is not None:
                        # 使用更通用的方法获取socket信息
                        socks = getattr(server, 'sockets', None)
                        if socks and len(socks) > 0:
                            # 获取第一个socket的地址信息
                            addr = socks[0].getsockname()
                            if addr and len(addr) >= 2:
                                self.port = addr[1]  # 端口号是地址元组的第二个元素
                                self.logger.info(f"系统分配的实际端口: {self.port}")
                                break
                except Exception as e:
                    self.logger.warning(f"获取实际端口时出错: {e}")
                    continue

        self.comm_logger.debug(f"[NetworkTransport] HTTP server started on {self.host}:{self.port}")

    async def _start_websocket_server(self):
        """启动WebSocket服务器"""
        self._ws_app = web.Application()
        self._ws_app.router.add_get("/ws/events", self._handle_websocket)
        
        self._ws_runner = web.AppRunner(self._ws_app)
        await self._ws_runner.setup()
        
        ws_site = web.TCPSite(self._ws_runner, self.host, self.websocket_port)
        await ws_site.start()
        
        self.comm_logger.debug(f"WebSocket server started on {self.host}:{self.websocket_port}")
    
    async def _handle_rpc_request(self, request: web.Request):
        """处理RPC请求"""
        try:
            # 使用CustomJSONDecoder反序列化请求
            request_text = await request.text()
            request_data = json.loads(request_text, cls=CustomJSONDecoder)

            # 调用注册的请求处理器
            if self._request_handler:
                if asyncio.iscoroutinefunction(self._request_handler):
                    result = await self._request_handler(
                        request_data.get("source"),
                        request_data.get("data")
                    )
                else:
                    result = self._request_handler(
                        request_data.get("source"),
                        request_data.get("data")
                    )

                # 使用CustomJSONEncoder序列化响应
                response_data = {
                    "success": True,
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                }
                response_json = json.dumps(response_data, cls=CustomJSONEncoder)
                return web.Response(
                    text=response_json,
                    content_type='application/json'
                )
            else:
                response_data = {
                    "success": False,
                    "error": "No request handler registered",
                    "timestamp": datetime.now().isoformat()
                }
                response_json = json.dumps(response_data, cls=CustomJSONEncoder)
                return web.Response(
                    text=response_json,
                    content_type='application/json',
                    status=500
                )

        except Exception as e:
            response_data = {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            response_json = json.dumps(response_data, cls=CustomJSONEncoder)
            return web.Response(
                text=response_json,
                content_type='application/json',
                status=500
            )
    
    async def _handle_register_request(self, request: web.Request):
        """处理注册请求 - 转发给通信管理器处理"""
        try:
            data = await request.json()

            # 调用注册的请求处理器（通信管理器）
            if self._request_handler:
                # 包装成标准格式，标记为注册请求
                wrapped_data = {
                    "message_type": "registration",
                    "data": data
                }

                if asyncio.iscoroutinefunction(self._request_handler):
                    result = await self._request_handler("system", wrapped_data)
                else:
                    result = self._request_handler("system", wrapped_data)

                return web.json_response(result)
            else:
                # 如果没有注册处理器，返回默认成功响应
                return web.json_response({
                    "success": True,
                    "message": "Registration successful",
                    "timestamp": datetime.now().isoformat()
                })

        except Exception as e:
            self.logger.error(f"处理注册请求失败: {e}")
            return web.json_response({
                "success": False,
                "error": str(e)
            }, status=400)
    
    async def _handle_heartbeat_request(self, request: web.Request):
        """处理心跳请求 - 转发给通信管理器处理"""
        try:
            data = await request.json()

            # 调用注册的请求处理器（通信管理器）
            if self._request_handler:
                # 包装成标准格式，标记为心跳请求
                wrapped_data = {
                    "message_type": "heartbeat",
                    "data": data
                }

                if asyncio.iscoroutinefunction(self._request_handler):
                    result = await self._request_handler("system", wrapped_data)
                else:
                    result = self._request_handler("system", wrapped_data)

                return web.json_response(result)
            else:
                # 如果没有注册处理器，返回默认存活响应
                return web.json_response({
                    "status": "alive",
                    "timestamp": datetime.now().isoformat()
                })

        except Exception as e:
            self.logger.error(f"处理心跳请求失败: {e}")
            return web.json_response({
                "success": False,
                "error": str(e)
            }, status=500)
    
    async def _handle_event_request(self, request: web.Request):
        """处理事件请求"""
        try:
            event_data = await request.json()
            
            # 处理事件
            await self._handle_event(
                event_data.get("event_type"),
                event_data.get("source"),
                event_data.get("data")
            )
            
            return web.json_response({"success": True})
        except Exception as e:
            return web.json_response({
                "success": False,
                "error": str(e)
            }, status=500)
    
    async def _handle_status_request(self, request: web.Request):
        """处理状态查询请求"""
        return web.json_response({
            "node_id": self.node_id,
            "status": "running" if self._running else "stopped",
            "connections": len(self._ws_connections),
            "timestamp": datetime.now().isoformat()
        })
    
    async def _handle_websocket(self, request: web.Request):
        """处理WebSocket连接"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        client_id = None
        
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    
                    if data.get("type") == "register":
                        client_id = data.get("client_id")
                        if client_id:
                            self._ws_connections[client_id] = ws
                            await ws.send_str(json.dumps({
                                "type": "register_ack",
                                "success": True
                            }))
                    
                except json.JSONDecodeError:
                    await ws.send_str(json.dumps({
                        "type": "error",
                        "message": "Invalid JSON"
                    }))
                    
            elif msg.type == WSMsgType.ERROR:
                self.comm_logger.debug(f"[NetworkTransport] WebSocket error: {ws.exception()}")
                break
        
        # 清理连接
        if client_id and client_id in self._ws_connections:
            del self._ws_connections[client_id]
        
        return ws
    
    def register_request_handler(self, node_id: str, handler: Callable):
        """注册请求处理器"""
        self._request_handler = handler
        self.logger.debug(f"[NetworkTransport] 注册请求处理器: {node_id}")
        self.comm_logger.debug(f"[NetworkTransport] 已注册请求处理器: {node_id}")
    
    def _parse_node_address(self, node_id: str) -> Optional[str]:
        """解析节点地址，兼容network/process模式

        优先级：
        1. 从客户端注册时提供的地址缓存中获取
        2. 从节点ID中解析
        """
        # 首先检查是否有客户端注册时提供的地址
        self.logger.debug(f"[NetworkTransport] 解析节点地址: {node_id}, 检查注册缓存 {self._client_addresses}")
        if node_id in self._client_addresses:
            client_addr = self._client_addresses[node_id]
            url = client_addr.get("url")
            if url:
                self.logger.debug(f"使用客户端注册地址: {node_id} -> {url}")
                return url

        # network_server_192.168.1.100_8000
        # network_client_192.168.1.101_8001_abc123
        # process_client_8001_xxx
        if node_id.startswith("network_"):
            parts = node_id.split("_")
            if len(parts) >= 4:
                try:
                    host = parts[2]
                    port = int(parts[3])
                    return f"http://{host}:{port}"
                except (ValueError, IndexError):
                    return None
        elif node_id.startswith("process_"):
            # 进程模式，所有通信都在本地，host为127.0.0.1，端口从ID中提取
            parts = node_id.split("_")
            if len(parts) >= 4:
                try:
                    process_port = int(parts[2])  # 从process_client_8001_xxx中提取8001
                    self.logger.debug(f"{node_id} Network transport port: {process_port}")
                    return f"http://127.0.0.1:{process_port}"
                except (ValueError, IndexError):
                    # 如果提取失败，回退到配置的端口
                    self.logger.debug(f"{node_id} 无法解析端口，使用默认端口: {self.port}")
                    return f"http://127.0.0.1:{self.port}"
            else:
                # 格式不正确，使用默认端口
                self.logger.debug(f"{node_id} 格式不正确，使用默认端口: {self.port}")
                return f"http://127.0.0.1:{self.port}"
        return None

    def register_client_address(self, client_id: str, address_info: Dict[str, Any]) -> None:
        """注册客户端地址信息（从客户端注册请求中提取）

        Args:
            client_id: 客户端ID
            address_info: 地址信息字典，包含 host, port, url
        """
        if address_info and address_info.get("url"):
            self._client_addresses[client_id] = address_info
            self.logger.info(f"注册客户端地址: {client_id} -> {self._client_addresses[client_id]}")
        else:
            self.logger.warning(f"客户端 {client_id} 未提供有效地址信息")

    def get_client_address(self, client_id: str) -> Optional[str]:
        """获取客户端的URL地址

        Args:
            client_id: 客户端ID

        Returns:
            客户端URL或None
        """
        addr = self._client_addresses.get(client_id)
        return addr.get("url") if addr else None

    def unregister_client_address(self, client_id: str) -> None:
        """注销客户端地址信息

        Args:
            client_id: 客户端ID
        """
        if client_id in self._client_addresses:
            del self._client_addresses[client_id]
            self.logger.info(f"注销客户端地址: {client_id}")

    
    async def initialize(self) -> bool:
        """初始化Network传输"""
        return True
    
    async def start(self) -> None:
        """启动Network传输"""
        await super().start()
    
    async def stop(self) -> None:
        """停止Network传输"""
        await super().stop()
        
        # 关闭WebSocket连接
        for ws in list(self._ws_connections.values()):
            if not ws.closed:
                await ws.close()
        self._ws_connections.clear()
        
        # 关闭HTTP服务器
        if self._runner:
            await self._runner.cleanup()
        
        if self._ws_runner:
            await self._ws_runner.cleanup()
        
        # 关闭HTTP客户端会话
        if self._http_session:
            await self._http_session.close()
    
    async def cleanup(self) -> None:
        """清理Network传输资源"""
        # 清理待处理的响应
        for future in self._pending_responses.values():
            if not future.done():
                future.cancel()
        self._pending_responses.clear()
        
        await super().cleanup()
    
    # ==================== Process模式兼容方法 ====================
    
    def add_target_id(self, target_id: str):
        """添加目标ID - 兼容Process模式"""
        self.target_ids.add(target_id)
        self.comm_logger.debug(f"[NetworkTransport] 添加目标ID: {target_id}")
    
    def remove_target_id(self, target_id: str):
        """移除目标ID"""
        self.target_ids.discard(target_id)
        self.comm_logger.debug(f"[NetworkTransport] 移除目标ID: {target_id}")
    
    def register_event_listener(self, target: str, event_type: str, listener: Callable) -> bool:
        """注册事件监听器 - 兼容Process模式"""
        try:
            # 使用全局事件监听器（跨实例共享）
            if target not in NetworkTransport._global_event_listeners:
                NetworkTransport._global_event_listeners[target] = {}
            if event_type not in NetworkTransport._global_event_listeners[target]:
                NetworkTransport._global_event_listeners[target][event_type] = []
            
            NetworkTransport._global_event_listeners[target][event_type].append(listener)
            self.comm_logger.debug(f"[NetworkTransport] 注册全局事件监听器: {target} -> {event_type}")
            return True
        except Exception as e:
            self.comm_logger.debug(f"[NetworkTransport] 注册事件监听器失败: {e}")
            return False
    
    def unregister_event_listener(self, target: str, event_type: str, listener: Callable) -> bool:
        """注销事件监听器 - 兼容Process模式"""
        try:
            return self.unregister_event_handler(target, event_type, listener)
        except Exception as e:
            self.comm_logger.debug(f"[NetworkTransport] 注销事件监听器失败: {e}")
            return False


    def validate_node_id(self, node_id: str) -> bool:
        """验证节点ID格式，兼容network和process模式"""
        if not node_id or not isinstance(node_id, str):
            return False
        # 允许 network_ 和 process_ 前缀
        return True