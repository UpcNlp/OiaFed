"""
MOE-FedCL Process模式传输实现 - 基于TCP Socket
moe_fedcl/transport/process_new.py
"""

import asyncio
import json
import socket
from typing import Any, Dict, List, Callable, Optional
from datetime import datetime
import uuid

from .base import TransportBase
from ..types import TransportConfig
from ..exceptions import TransportError, TimeoutError


class ProcessTransport(TransportBase):
    """进程间传输实现 - 基于TCP socket通信"""
    
    def __init__(self, config: TransportConfig):
        super().__init__(config)
        self._server_host = getattr(config, 'host', '127.0.0.1')
        self._server_port = getattr(config, 'port', 8000)
        self._node_ports: Dict[str, int] = {}  # 节点ID到端口的映射
        self._server: Optional[asyncio.Server] = None
        self._connections: Dict[str, asyncio.StreamWriter] = {}
        self._request_handlers: Dict[str, Callable] = {}
        self._response_futures: Dict[str, asyncio.Future] = {}
        self._event_listeners: Dict[str, Dict[str, List[Callable]]] = {}
        
        # 自动分配端口
        self._assign_port()
    
    def _assign_port(self):
        """为当前节点分配可用端口"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('127.0.0.1', 0))
        self._server_port = sock.getsockname()[1]
        sock.close()
        print(f"[ProcessTransport] 分配端口: {self._server_port}")
    
    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """处理客户端连接"""
        try:
            while True:
                # 读取消息长度
                length_bytes = await reader.read(4)
                if not length_bytes:
                    break
                
                message_length = int.from_bytes(length_bytes, byteorder='big')
                
                # 读取消息内容
                message_bytes = await reader.read(message_length)
                if not message_bytes:
                    break
                
                try:
                    message = json.loads(message_bytes.decode('utf-8'))
                    await self._process_message(message, writer)
                except Exception as e:
                    print(f"[ProcessTransport] 处理消息时出错: {e}")
                
        except Exception as e:
            print(f"[ProcessTransport] 客户端连接错误: {e}")
        finally:
            writer.close()
            await writer.wait_closed()
    
    async def _process_message(self, message: dict, writer: asyncio.StreamWriter):
        """处理收到的消息"""
        message_type = message.get('type')
        
        if message_type == 'request':
            await self._handle_request(message, writer)
        elif message_type == 'response':
            await self._handle_response(message)
        elif message_type == 'event':
            await self._handle_event(message)
        else:
            print(f"[ProcessTransport] 未知消息类型: {message_type}")
    
    async def _handle_request(self, message: dict, writer: asyncio.StreamWriter):
        """处理请求消息"""
        request_id = message.get('request_id')
        source = message.get('source')
        target = message.get('target')
        data = message.get('data')
        
        # 查找请求处理器
        handler = self._request_handlers.get(target)
        if handler:
            try:
                result = await handler(data)
                
                # 发送响应
                response = {
                    'type': 'response',
                    'request_id': request_id,
                    'source': target,
                    'target': source,
                    'result': result,
                    'timestamp': datetime.now().isoformat()
                }
                
                await self._send_message(writer, response)
                
            except Exception as e:
                # 发送错误响应
                error_response = {
                    'type': 'response',
                    'request_id': request_id,
                    'source': target,
                    'target': source,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                
                await self._send_message(writer, error_response)
        else:
            print(f"[ProcessTransport] 没有找到处理器: {target}")
    
    async def _handle_response(self, message: dict):
        """处理响应消息"""
        request_id = message.get('request_id')
        
        if request_id in self._response_futures:
            future = self._response_futures[request_id]
            
            if 'error' in message:
                future.set_exception(TransportError(message['error']))
            else:
                future.set_result(message.get('result'))
            
            del self._response_futures[request_id]
    
    async def _handle_event(self, message: dict):
        """处理事件消息"""
        target = message.get('target')
        event_type = message.get('event_type')
        data = message.get('data')
        
        if target in self._event_listeners:
            if event_type in self._event_listeners[target]:
                for listener in self._event_listeners[target][event_type]:
                    try:
                        await listener(data)
                    except Exception as e:
                        print(f"[ProcessTransport] 事件监听器执行失败: {e}")
    
    async def _send_message(self, writer: asyncio.StreamWriter, message: dict):
        """发送消息"""
        try:
            message_json = json.dumps(message, ensure_ascii=False)
            message_bytes = message_json.encode('utf-8')
            
            # 发送消息长度
            length_bytes = len(message_bytes).to_bytes(4, byteorder='big')
            writer.write(length_bytes)
            
            # 发送消息内容
            writer.write(message_bytes)
            await writer.drain()
            
        except Exception as e:
            print(f"[ProcessTransport] 发送消息失败: {e}")
    
    async def start(self, node_id: str) -> bool:
        """启动传输层"""
        try:
            self._server = await asyncio.start_server(
                self._handle_client,
                self._server_host,
                self._server_port
            )
            
            print(f"[ProcessTransport] 服务器启动成功: {self._server_host}:{self._server_port}")
            return True
            
        except Exception as e:
            print(f"[ProcessTransport] 启动失败: {e}")
            return False
    
    async def stop(self) -> bool:
        """停止传输层"""
        try:
            if self._server:
                self._server.close()
                await self._server.wait_closed()
            
            # 关闭所有连接
            for writer in self._connections.values():
                writer.close()
                await writer.wait_closed()
            
            self._connections.clear()
            print(f"[ProcessTransport] 传输层已停止")
            return True
            
        except Exception as e:
            print(f"[ProcessTransport] 停止传输层失败: {e}")
            return False
    
    async def send(self, source: str, target: str, data: Any) -> Any:
        """发送请求并等待响应"""
        if not self.validate_node_id(source) or not self.validate_node_id(target):
            raise TransportError(f"Invalid node ID: {source} -> {target}")
        
        # 生成请求ID
        request_id = str(uuid.uuid4())
        
        # 创建响应Future
        response_future = asyncio.Future()
        self._response_futures[request_id] = response_future
        
        try:
            # 连接到目标节点
            target_port = self._node_ports.get(target, 8000)
            reader, writer = await asyncio.open_connection(
                self._server_host, target_port
            )
            
            # 发送请求
            request = {
                'type': 'request',
                'request_id': request_id,
                'source': source,
                'target': target,
                'data': data,
                'timestamp': datetime.now().isoformat()
            }
            
            await self._send_message(writer, request)
            
            # 等待响应
            timeout = self.config.timeout
            response = await asyncio.wait_for(response_future, timeout=timeout)
            
            # 关闭连接
            writer.close()
            await writer.wait_closed()
            
            return response
            
        except asyncio.TimeoutError:
            # 清理Future
            if request_id in self._response_futures:
                del self._response_futures[request_id]
            raise TimeoutError(f"Request timeout: {source} -> {target}")
        
        except Exception as e:
            # 清理Future
            if request_id in self._response_futures:
                del self._response_futures[request_id]
            raise TransportError(f"Send failed: {source} -> {target}, error: {e}")
    
    def register_handler(self, node_id: str, handler: Callable) -> bool:
        """注册请求处理器"""
        try:
            self._request_handlers[node_id] = handler
            print(f"[ProcessTransport] 注册处理器: {node_id}")
            return True
        except Exception as e:
            print(f"[ProcessTransport] 注册处理器失败: {e}")
            return False
    
    def unregister_handler(self, node_id: str) -> bool:
        """注销请求处理器"""
        try:
            if node_id in self._request_handlers:
                del self._request_handlers[node_id]
                print(f"[ProcessTransport] 注销处理器: {node_id}")
            return True
        except Exception as e:
            print(f"[ProcessTransport] 注销处理器失败: {e}")
            return False
    
    async def push_event(self, source: str, target: str, event_type: str, data: Any) -> bool:
        """推送事件"""
        try:
            # 连接到目标节点
            target_port = self._node_ports.get(target, 8000)
            reader, writer = await asyncio.open_connection(
                self._server_host, target_port
            )
            
            # 发送事件
            event = {
                'type': 'event',
                'source': source,
                'target': target,
                'event_type': event_type,
                'data': data,
                'timestamp': datetime.now().isoformat()
            }
            
            await self._send_message(writer, event)
            
            # 关闭连接
            writer.close()
            await writer.wait_closed()
            
            print(f"[ProcessTransport] 事件推送成功: {source} -> {target}.{event_type}")
            return True
            
        except Exception as e:
            print(f"[ProcessTransport] 事件推送失败: {e}")
            return False
    
    def register_event_listener(self, target: str, event_type: str, listener: Callable) -> bool:
        """注册事件监听器"""
        try:
            if target not in self._event_listeners:
                self._event_listeners[target] = {}
            
            if event_type not in self._event_listeners[target]:
                self._event_listeners[target][event_type] = []
            
            self._event_listeners[target][event_type].append(listener)
            print(f"[ProcessTransport] 注册事件监听器: {target}.{event_type}")
            return True
            
        except Exception as e:
            print(f"[ProcessTransport] 注册事件监听器失败: {e}")
            return False
    
    def unregister_event_listener(self, target: str, event_type: str, listener: Callable) -> bool:
        """注销事件监听器"""
        try:
            if (target in self._event_listeners and 
                event_type in self._event_listeners[target] and
                listener in self._event_listeners[target][event_type]):
                
                self._event_listeners[target][event_type].remove(listener)
                print(f"[ProcessTransport] 注销事件监听器: {target}.{event_type}")
            
            return True
            
        except Exception as e:
            print(f"[ProcessTransport] 注销事件监听器失败: {e}")
            return False
    
    def set_node_port(self, node_id: str, port: int):
        """设置节点端口映射"""
        self._node_ports[node_id] = port
        print(f"[ProcessTransport] 设置节点端口: {node_id} -> {port}")
    
    def get_node_port(self, node_id: str) -> int:
        """获取节点端口"""
        return self._node_ports.get(node_id, self._server_port)
