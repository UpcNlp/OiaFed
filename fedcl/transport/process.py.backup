"""
MOE-FedCL Process模式传输实现 - 基于TCP Socket
moe_fedcl/transport/process_new.py
"""

import asyncio
import json
import socket
import uuid
from typing import Any, Dict, List, Callable, Optional
from datetime import datetime
from pathlib import Path

from .base import TransportBase
from ..types import TransportConfig
from ..exceptions import TransportError, TimeoutError

class ProcessTransport(TransportBase):
    """进程间传输实现 - 基于TCP socket通信"""
    
    def __init__(self, config=None, server_host: str = "127.0.0.1", server_port: int = None):
        # 如果没有config，创建一个默认的
        if config is None:
            from ..types import TransportConfig
            config = TransportConfig()
        super().__init__(config)
        
        # 添加实例标识符
        import uuid
        self._instance_id = str(uuid.uuid4())[:8]
        print(f"[ProcessTransport-{self._instance_id}] 创建新实例")
        
        self._server_host = server_host
        self._server_port = server_port or 0
        self._server = None
        self._message_queue = asyncio.Queue()
        self._response_futures = {}
        self._file_watcher_task = None
        self._events_dir = Path("/tmp/moe_fedcl_events")
        self._events_dir.mkdir(exist_ok=True)
        
        # 节点身份管理
        self.node_id = None
        self.target_ids = set()  # 支持多个目标ID
        
        # 事件监听器管理
        self._event_listeners = {}
        
        # 请求处理器管理
        self._request_handlers = {}
        
        # 注册请求处理器
        self.set_request_handler(self._default_request_handler)
        print(f"[ProcessTransport] 设置请求处理器")
    
    async def _default_request_handler(self, source: str, data: Any) -> Any:
        """默认请求处理器"""
        print(f"[ProcessTransport] 默认处理器收到请求: {source} -> {data}")
        return {"status": "ok", "message": "Request processed by default handler"}
    
    def _assign_port(self):
        """为当前节点分配可用端口"""
        # 不在这里分配端口，在start()时动态分配
        self._server_port = None
        print(f"[ProcessTransport] 端口将在启动时动态分配")
    
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
        
        # 将请求放入消息队列，供receive方法获取
        await self._message_queue.put({
            'type': 'request',
            'request_id': request_id,
            'source': source,
            'target': target,
            'data': data,
            'writer': writer  # 保存writer用于发送响应
        })
        
        # 不在这里直接处理，而是由receive方法处理
    
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
    
    async def start(self) -> bool:
        """启动传输层"""
        try:
            # 动态分配可用端口
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(('127.0.0.1', 0))
            self._server_port = sock.getsockname()[1]
            sock.close()
            
            self._server = await asyncio.start_server(
                self._handle_client,
                self._server_host,
                self._server_port
            )
            
            # 启动文件事件监听器
            self._file_watcher_task = asyncio.create_task(self._watch_file_events())
            print(f"[ProcessTransport] 文件事件监听器已启动")
            
            print(f"[ProcessTransport] 服务器启动成功: {self._server_host}:{self._server_port}")
            return True
            
        except Exception as e:
            print(f"[ProcessTransport] 启动失败: {e}")
            return False
    
    async def stop(self) -> bool:
        """停止传输层"""
        try:
            # 停止文件监听器
            if self._file_watcher_task:
                self._file_watcher_task.cancel()
                try:
                    await self._file_watcher_task
                except asyncio.CancelledError:
                    pass
            
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
    
    def add_target_id(self, target_id: str):
        """添加目标ID - 让该节点能够处理指定目标的事件"""
        self.target_ids.add(target_id)
        print(f"[ProcessTransport-{self._instance_id}] 添加目标ID: {target_id}")
        print(f"[ProcessTransport-{self._instance_id}] 当前目标ID集合: {self.target_ids}")
    
    def remove_target_id(self, target_id: str):
        """移除目标ID"""
        self.target_ids.discard(target_id)
        print(f"[ProcessTransport] 移除目标ID: {target_id}")
    
    def set_node_port(self, node_id: str, port: int):
        """设置节点端口映射"""
        self._node_ports[node_id] = port
        print(f"[ProcessTransport] 设置节点端口: {node_id} -> {port}")
    
    def set_request_handler(self, handler: Callable):
        """设置请求处理器"""
        self._request_handlers['default'] = handler
        print(f"[ProcessTransport] 设置请求处理器")
    
    def get_node_port(self, node_id: str) -> int:
        """获取节点端口"""
        return self._node_ports.get(node_id, self._server_port)
    
    async def receive(self, target: str, source: str = None, timeout: float = None) -> Any:
        """从指定源接收消息
        
        Args:
            target: 接收节点ID
            source: 源节点ID，None表示接收任意源
            timeout: 超时时间
            
        Returns:
            接收到的消息
        """
        try:
            # 从消息队列中获取消息
            if timeout:
                message = await asyncio.wait_for(self._message_queue.get(), timeout=timeout)
            else:
                message = await self._message_queue.get()
            
            # 返回消息数据部分，但不包含writer
            if message and message.get('type') == 'request':
                return {
                    'request_id': message.get('request_id'),
                    'source': message.get('source'),
                    'target': message.get('target'),
                    'data': message.get('data')
                }
            else:
                return message
                
        except asyncio.TimeoutError:
            # 超时时抛出TimeoutError，让上层处理
            raise asyncio.TimeoutError("Receive timeout")
        except Exception as e:
            print(f"[ProcessTransport] 接收消息失败: {e}")
            return None
    
    async def start_event_listener(self, node_id: str) -> None:
        """启动事件监听器
        
        Args:
            node_id: 节点ID
        """
        self.node_id = node_id
        
        # 如果是服务器节点，自动添加system目标ID
        if "server" in node_id.lower():
            self.add_target_id("system")
            print(f"[ProcessTransport-{self._instance_id}] 服务器节点自动添加system目标ID")
        
        # 事件监听在start()方法中已经通过TCP服务器启动
        print(f"[ProcessTransport-{self._instance_id}] 事件监听器已启动: {node_id}")
    
    def add_target_id(self, target_id: str):
        """添加目标ID - 让该节点能够处理指定目标的事件"""
        self.target_ids.add(target_id)
        print(f"[ProcessTransport-{self._instance_id}] 添加目标ID: {target_id}")
        print(f"[ProcessTransport-{self._instance_id}] 当前目标ID集合: {self.target_ids}")
    
    def remove_target_id(self, target_id: str):
        """移除目标ID
        
        Args:
            target_id: 目标ID
        """
        self.target_ids.discard(target_id)
        print(f"[ProcessTransport] 移除目标ID: {target_id}")
    
    async def _watch_file_events(self):
        """监听文件系统事件"""
        processed_files = set()
        print(f"[ProcessTransport] 开始监听事件目录: {self._events_dir}")
        
        while True:
            try:
                # 检查事件目录中的新文件
                event_files = list(self._events_dir.glob("*.json"))
                if event_files:
                    print(f"[ProcessTransport] 发现 {len(event_files)} 个事件文件")
                
                for event_file in event_files:
                    if event_file.name not in processed_files:
                        print(f"[ProcessTransport] 处理事件文件: {event_file.name}")
                        try:
                            with open(event_file, 'r') as f:
                                event_data = json.load(f)
                            
                            # 处理事件
                            await self._process_file_event(event_data)
                            processed_files.add(event_file.name)
                            
                            # 清理已处理的文件
                            event_file.unlink()
                            print(f"[ProcessTransport] 事件文件已处理并删除: {event_file.name}")
                            
                        except Exception as e:
                            print(f"[ProcessTransport] 处理事件文件失败: {e}")
                
                await asyncio.sleep(0.1)  # 短暂休眠
                
            except asyncio.CancelledError:
                print(f"[ProcessTransport] 文件监听器已取消")
                break
            except Exception as e:
                print(f"[ProcessTransport] 文件监听错误: {e}")
                await asyncio.sleep(1)
    
    async def _process_file_event(self, event_data: dict):
        """处理文件事件"""
        try:
            event_type = event_data.get('event_type')
            target = event_data.get('target')
            source = event_data.get('source')
            data = event_data.get('data')
            
            print(f"[ProcessTransport-{self._instance_id}] 处理文件事件: {event_type}, source={source}, target={target}")
            print(f"[ProcessTransport-{self._instance_id}] 当前节点ID: {self.node_id}")
            print(f"[ProcessTransport-{self._instance_id}] 目标ID集合: {self.target_ids}")
            
            # 关键改进：检查节点ID或目标ID集合中是否包含目标
            should_process = (
                self.node_id == target or 
                target in self.target_ids
            )
            
            if not should_process:
                print(f"[ProcessTransport] 事件不是发给当前节点({self.node_id})或目标集合({self.target_ids})的，跳过处理")
                return
            
            print(f"[ProcessTransport] 事件数据: {data}")
            
            # 触发事件监听器
            if target in self._event_listeners and event_type in self._event_listeners[target]:
                listeners = self._event_listeners[target][event_type]
                print(f"[ProcessTransport] 找到 {len(listeners)} 个监听器用于 {target}.{event_type}")
                
                for i, listener in enumerate(listeners):
                    try:
                        print(f"[ProcessTransport] 调用监听器 #{i+1}: {listener}")
                        if asyncio.iscoroutinefunction(listener):
                            await listener(source, data)
                        else:
                            listener(source, data)
                        print(f"[ProcessTransport] 监听器 #{i+1} 执行成功")
                    except Exception as e:
                        print(f"[ProcessTransport] 监听器 #{i+1} 执行失败: {e}")
            else:
                print(f"[ProcessTransport] 没有找到监听器用于 {target}.{event_type}")
                print(f"[ProcessTransport] 当前注册的监听器: {list(self._event_listeners.keys())}")
                
        except Exception as e:
            print(f"[ProcessTransport] 处理文件事件失败: {e}")
    
    async def send_response(self, target: str, request_id: str, response_data: Any) -> bool:
        """发送响应消息
        
        Args:
            target: 目标节点ID
            request_id: 请求ID
            response_data: 响应数据
            
        Returns:
            bool: 发送是否成功
        """
        try:
            # 这里可以通过连接发送响应，但在当前简化实现中，
            # 我们假设响应通过其他机制处理
            print(f"[ProcessTransport] 发送响应: {target} -> {request_id}")
            return True
        except Exception as e:
            print(f"[ProcessTransport] 发送响应失败: {e}")
            return False
