"""
MOE-FedCL Memory模式传输实现
moe_fedcl/transport/memory.py
"""

import asyncio
from collections import defaultdict, deque
from typing import Any, Dict, List, Callable, Optional
from datetime import datetime

from .base import TransportBase
from ..types import TransportConfig
from ..exceptions import TransportError, TimeoutError


class MemoryTransport(TransportBase):
    """内存传输实现 - 同进程内直接函数调用通信"""
    
    # 全局共享的消息队列和处理器注册表
    _global_message_queues: Dict[str, deque] = defaultdict(deque)
    _global_request_handlers: Dict[str, Callable] = {}
    _global_event_listeners: Dict[str, Dict[str, List[Callable]]] = defaultdict(lambda: defaultdict(list))
    
    def __init__(self, config: TransportConfig):
        super().__init__(config)
        self._message_queue_lock = asyncio.Lock()
        
    async def send(self, source: str, target: str, data: Any) -> Any:
        """发送消息并等待响应"""
        if not self.validate_node_id(source) or not self.validate_node_id(target):
            raise TransportError(f"Invalid node ID: {source} -> {target}")
        
        print(f"[MemoryTransport] 发送消息: {source} -> {target}")
        print(f"[MemoryTransport] 当前注册的处理器: {list(self._global_request_handlers.keys())}")
        print(f"[MemoryTransport] 检查目标节点 {target} 是否在处理器中...")
        print(f"[MemoryTransport] target in _global_request_handlers: {target in self._global_request_handlers}")
        
        # 检查目标节点是否存在处理器
        if target not in self._global_request_handlers:
            print(f"[MemoryTransport] ❌ 目标节点 {target} 没有注册处理器")
            raise TransportError(f"Target node {target} not available")
        
        print(f"[MemoryTransport] ✅ 找到目标节点 {target} 的处理器")
        
        try:
            # 直接调用目标节点的处理器
            handler = self._global_request_handlers[target]
            print(f"[MemoryTransport] 开始调用处理器: {handler}")
            
            # 如果是异步处理器
            if asyncio.iscoroutinefunction(handler):
                print(f"[MemoryTransport] 异步调用处理器")
                result = await handler(source, data)
            else:
                print(f"[MemoryTransport] 同步调用处理器")
                result = handler(source, data)
            
            print(f"[MemoryTransport] 处理器调用成功，结果: {type(result)}")
            return result
            
        except Exception as e:
            print(f"[MemoryTransport] ❌ 处理器调用失败: {str(e)}")
            import traceback
            print(f"[MemoryTransport] 错误堆栈: {traceback.format_exc()}")
            raise TransportError(f"Send failed from {source} to {target}: {str(e)}")
    
    async def receive(self, target: str, source: str = None, timeout: float = None) -> Any:
        """从消息队列接收消息"""
        timeout = timeout or self.config.timeout
        queue_key = f"{target}"
        if source:
            queue_key = f"{target}_{source}"
        
        start_time = datetime.now()
        
        while True:
            async with self._message_queue_lock:
                if queue_key in self._global_message_queues and self._global_message_queues[queue_key]:
                    return self._global_message_queues[queue_key].popleft()
            
            # 检查超时
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed >= timeout:
                raise TimeoutError(f"Receive timeout after {timeout}s for {target}")
            
            # 短暂休眠避免过度轮询
            await asyncio.sleep(0.001)
    
    async def push_event(self, source: str, target: str, event_type: str, data: Any) -> bool:
        """推送事件到目标节点"""
        try:
            print(f"[MemoryTransport] 推送事件: {source} -> {target}, 类型: {event_type}")
            print(f"[MemoryTransport] 当前事件监听器: {dict(self._global_event_listeners)}")
            
            # 检查目标节点是否有事件监听器
            if target in self._global_event_listeners:
                if event_type in self._global_event_listeners[target]:
                    handlers = self._global_event_listeners[target][event_type]
                    print(f"[MemoryTransport] 找到 {len(handlers)} 个处理器用于 {target}.{event_type}")
                    
                    # 直接调用所有匹配的事件处理器
                    for i, handler in enumerate(handlers):
                        try:
                            print(f"[MemoryTransport] 调用处理器 #{i+1}: {handler}")
                            if asyncio.iscoroutinefunction(handler):
                                await handler(data)  # 修正参数传递
                            else:
                                handler(data)  # 修正参数传递
                            print(f"[MemoryTransport] 处理器 #{i+1} 执行成功")
                        except Exception as e:
                            print(f"[MemoryTransport] 处理器 #{i+1} 执行失败: {e}")
                else:
                    print(f"[MemoryTransport] 目标 {target} 没有 {event_type} 事件监听器")
            else:
                print(f"[MemoryTransport] 目标 {target} 没有注册任何事件监听器")
            
            return True
            
        except Exception as e:
            print(f"Push event failed: {e}")
            return False
    
    async def start_event_listener(self, node_id: str) -> None:
        """启动事件监听器 - Memory模式下为立即生效"""
        self.node_id = node_id
        
        # 注册消息处理器
        async def default_handler(source: str, data: Any) -> Any:
            # 将消息放入队列供receive方法获取
            queue_key = f"{node_id}_{source}"
            async with self._message_queue_lock:
                self._global_message_queues[queue_key].append(data)
            return {"status": "received", "timestamp": datetime.now().isoformat()}
        
        self._global_request_handlers[node_id] = default_handler
    
    def register_request_handler(self, node_id: str, handler: Callable):
        """注册请求处理器"""
        self._global_request_handlers[node_id] = handler
    
    def register_event_listener(self, node_id: str, event_type: str, handler: Callable):
        """注册事件监听器"""
        self._global_event_listeners[node_id][event_type].append(handler)
    
    async def put_message(self, target: str, source: str, data: Any):
        """向消息队列放入消息 - 供外部调用"""
        queue_key = f"{target}_{source}"
        async with self._message_queue_lock:
            self._global_message_queues[queue_key].append(data)
    
    def get_queue_status(self) -> Dict[str, int]:
        """获取队列状态 - 调试用"""
        return {key: len(queue) for key, queue in self._global_message_queues.items()}
    
    def clear_queues(self):
        """清空所有队列 - 测试用"""
        self._global_message_queues.clear()
        self._global_request_handlers.clear()
        self._global_event_listeners.clear()
    
    async def initialize(self) -> bool:
        """初始化Memory传输"""
        return True
    
    async def start(self) -> None:
        """启动Memory传输"""
        await super().start()
    
    async def stop(self) -> None:
        """停止Memory传输"""
        await super().stop()
    
    async def cleanup(self) -> None:
        """清理Memory传输资源"""
        if self.node_id:
            # 清理该节点的注册信息
            if self.node_id in self._global_request_handlers:
                del self._global_request_handlers[self.node_id]
            
            if self.node_id in self._global_event_listeners:
                del self._global_event_listeners[self.node_id]
        
        await super().cleanup()
    
    def validate_node_id(self, node_id: str) -> bool:
        """验证Memory模式节点ID格式"""
        # 更宽松的验证：接受memory_server或任何包含client的ID
        return (node_id.startswith("memory_server") or 
                "client" in node_id) and len(node_id) > 3