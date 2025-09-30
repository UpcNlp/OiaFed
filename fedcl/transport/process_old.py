"""
MOE-FedCL Process模式传输实现
moe_fedcl/transport/process.py
"""

import asyncio
import json
import socket
from typing import Any, Dict, List, Callable, Optional
from datetime import datetime
import uuid
import time

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
        request_key = self._get_queue_key(node_id, "request")
        response_key = self._get_queue_key(node_id, "response")
        event_key = self._get_queue_key(node_id, "event")
        
        if request_key not in self._global_queues:
            self._global_queues[request_key] = mp.Queue()
        
        if response_key not in self._global_queues:
            self._global_queues[response_key] = mp.Queue()
        
        if event_key not in self._global_queues:
            self._global_queues[event_key] = mp.Queue()
    
    async def send(self, source: str, target: str, data: Any) -> Any:
        """发送请求并等待响应"""
        if not self.validate_node_id(source) or not self.validate_node_id(target):
            raise TransportError(f"Invalid node ID: {source} -> {target}")
        
        # 确保目标节点队列存在
        self._ensure_queues(target)
        
        # 生成请求ID
        request_id = str(uuid.uuid4())
        
        # 创建响应Future
        response_future = asyncio.Future()
        self._response_futures[request_id] = response_future
        
        try:
            # 序列化请求数据
            request_data = {
                "request_id": request_id,
                "source": source,
                "target": target,
                "data": data,
                "timestamp": datetime.now().isoformat()
            }
            serialized_request = pickle.dumps(request_data)
            
            # 发送到目标节点的请求队列
            request_key = self._get_queue_key(target, "request")
            target_queue = self._global_queues[request_key]
            target_queue.put(serialized_request)
            
            # 等待响应
            timeout = self.config.timeout
            response = await asyncio.wait_for(response_future, timeout=timeout)
            
            return response
            
        except asyncio.TimeoutError:
            # 清理Future
            if request_id in self._response_futures:
                del self._response_futures[request_id]
            raise TimeoutError(f"Send timeout after {self.config.timeout}s from {source} to {target}")
        
        except Exception as e:
            # 清理Future
            if request_id in self._response_futures:
                del self._response_futures[request_id]
            raise TransportError(f"Send failed from {source} to {target}: {str(e)}")
    
    async def receive(self, target: str, source: str = None, timeout: float = None) -> Any:
        """从队列接收消息"""
        timeout = timeout or self.config.timeout
        self._ensure_queues(target)
        
        request_key = self._get_queue_key(target, "request")
        request_queue = self._global_queues[request_key]
        
        start_time = time.time()
        
        while True:
            try:
                # 非阻塞获取消息
                serialized_data = request_queue.get_nowait()
                request_data = pickle.loads(serialized_data)
                
                # 检查源过滤
                if source is None or request_data["source"] == source:
                    return request_data
                    
                # 如果不匹配，放回队列
                request_queue.put(serialized_data)
                
            except:
                # 队列为空
                pass
            
            # 检查超时
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                raise TimeoutError(f"Receive timeout after {timeout}s for {target}")
            
            # 短暂休眠
            await asyncio.sleep(0.01)
    
    async def send_response(self, target: str, request_id: str, response_data: Any):
        """发送响应"""
        try:
            self._ensure_queues(target)
            
            response = {
                "request_id": request_id,
                "response": response_data,
                "timestamp": datetime.now().isoformat()
            }
            serialized_response = pickle.dumps(response)
            
            response_key = self._get_queue_key(target, "response")
            response_queue = self._global_queues[response_key]
            response_queue.put(serialized_response)
            
        except Exception as e:
            print(f"Send response failed: {e}")
    
    async def push_event(self, source: str, target: str, event_type: str, data: Any) -> bool:
        """推送事件"""
        try:
            self._ensure_queues(target)
            
            event_data = {
                "source": source,
                "target": target,
                "event_type": event_type,
                "data": data,
                "timestamp": datetime.now().isoformat()
            }
            serialized_event = pickle.dumps(event_data)
            
            event_key = self._get_queue_key(target, "event")
            event_queue = self._global_queues[event_key]
            event_queue.put(serialized_event)
            
            return True
            
        except Exception as e:
            print(f"Push event failed: {e}")
            return False
    
    def register_event_listener(self, node_id: str, event_type: str, handler: Callable):
        """注册事件监听器 - 为了兼容MemoryTransport接口"""
        # 在ProcessTransport中，事件监听是通过队列机制处理的
        # 这里我们将处理器存储在全局处理器字典中
        handler_key = f"{node_id}_{event_type}"
        if handler_key not in self._global_handlers:
            self._global_handlers[handler_key] = []
        self._global_handlers[handler_key].append(handler)
        print(f"[ProcessTransport] 注册事件监听器: {node_id}.{event_type}")
    
    async def start_event_listener(self, node_id: str) -> None:
        """启动事件监听器"""
        self.node_id = node_id
        self._ensure_queues(node_id)
        
        # 启动响应监听器
        if self._listener_task is None:
            self._listener_task = asyncio.create_task(self._response_listener())
        
        # 启动事件监听器
        asyncio.create_task(self._event_listener())
    
    async def _response_listener(self):
        """响应监听器"""
        if not self.node_id:
            return
        
        response_key = self._get_queue_key(self.node_id, "response")
        response_queue = self._global_queues[response_key]
        
        while self._running:
            try:
                serialized_response = response_queue.get_nowait()
                response_data = pickle.loads(serialized_response)
                
                request_id = response_data.get("request_id")
                if request_id in self._response_futures:
                    future = self._response_futures.pop(request_id)
                    if not future.done():
                        future.set_result(response_data.get("response"))
                        
            except:
                # 队列为空，短暂休眠
                await asyncio.sleep(0.01)
    
    async def _event_listener(self):
        """事件监听器"""
        if not self.node_id:
            return
        
        event_key = self._get_queue_key(self.node_id, "event")
        event_queue = self._global_queues[event_key]
        
        while self._running:
            try:
                serialized_event = event_queue.get_nowait()
                event_data = pickle.loads(serialized_event)
                
                # 查找并调用注册的事件处理器
                event_type = event_data.get("event_type")
                source = event_data.get("source")
                data = event_data.get("data")
                
                handler_key = f"{self.node_id}_{event_type}"
                if handler_key in self._global_handlers:
                    handlers = self._global_handlers[handler_key]
                    print(f"[ProcessTransport] 找到 {len(handlers)} 个处理器用于 {self.node_id}.{event_type}")
                    for handler in handlers:
                        try:
                            # 调用处理器
                            if asyncio.iscoroutinefunction(handler):
                                await handler(data)
                            else:
                                handler(data)
                            print(f"[ProcessTransport] 处理器执行成功")
                        except Exception as e:
                            print(f"[ProcessTransport] 处理器执行失败: {e}")
                else:
                    print(f"[ProcessTransport] 未找到处理器用于 {self.node_id}.{event_type}")
                    # 回退到默认事件处理
                    await self._handle_event(event_type, source, data)
                
            except:
                # 队列为空，短暂休眠
                await asyncio.sleep(0.01)
    
    def register_request_handler(self, node_id: str, handler: Callable):
        """注册请求处理器"""
        self._global_handlers[node_id] = handler
    
    async def initialize(self) -> bool:
        """初始化Process传输"""
        return True
    
    async def start(self) -> None:
        """启动Process传输"""
        await super().start()
    
    async def stop(self) -> None:
        """停止Process传输"""
        await super().stop()
        
        # 停止监听任务
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
    
    async def cleanup(self) -> None:
        """清理Process传输资源"""
        # 清理响应Future
        for future in self._response_futures.values():
            if not future.done():
                future.cancel()
        self._response_futures.clear()
        
        await super().cleanup()
    
    def validate_node_id(self, node_id: str) -> bool:
        """验证Process模式节点ID格式"""
        return node_id.startswith("process_") and len(node_id) > 8
    
    def get_queue_sizes(self) -> Dict[str, int]:
        """获取队列大小 - 调试用"""
        sizes = {}
        for key, queue in self._global_queues.items():
            try:
                sizes[key] = queue.qsize()
            except:
                sizes[key] = -1  # 无法获取大小
        return sizes