"""
MOE-FedCL 传输抽象层基础接口
moe_fedcl/transport/base.py
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Callable, Optional

from ..types import Connection, TransportConfig, ConnectionStatus
from ..exceptions import TransportError


class TransportBase(ABC):
    """传输抽象基类 - 所有传输实现的统一接口"""
    
    def __init__(self, config: TransportConfig):
        self.config = config
        self.node_id: str = ""
        self.connections: Dict[str, Connection] = {}
        self.event_handlers: Dict[str, List[Callable]] = {}
        self._running = False
        self._lock = asyncio.Lock()
    
    # ==================== 基础传输方法 ====================
    
    @abstractmethod
    async def send(self, source: str, target: str, data: Any) -> Any:
        """发送消息到目标节点并等待响应
        
        Args:
            source: 源节点ID
            target: 目标节点ID  
            data: 要发送的数据
            
        Returns:
            目标节点的响应结果
            
        Raises:
            TransportError: 传输失败
        """
        pass
    
    @abstractmethod
    async def receive(self, target: str, source: str = None, timeout: float = None) -> Any:
        """从指定源接收消息
        
        Args:
            target: 接收节点ID
            source: 源节点ID，None表示接收任意源
            timeout: 超时时间
            
        Returns:
            接收到的数据
            
        Raises:
            TransportError: 接收失败
            TimeoutError: 接收超时
        """
        pass
    
    async def broadcast(self, source: str, targets: List[str], data: Any) -> Dict[str, Any]:
        """广播消息到多个目标节点
        
        Args:
            source: 源节点ID
            targets: 目标节点ID列表
            data: 要广播的数据
            
        Returns:
            Dict[target_id, response] 响应结果字典
        """
        tasks = []
        for target in targets:
            task = self.send(source, target, data)
            tasks.append((target, task))
        
        results = {}
        for target, task in tasks:
            try:
                result = await task
                results[target] = result
            except Exception as e:
                results[target] = e
        
        return results
    
    async def gather(self, target: str, sources: List[str], timeout: float = None) -> Dict[str, Any]:
        """从多个源节点收集消息
        
        Args:
            target: 接收节点ID
            sources: 源节点ID列表
            timeout: 超时时间
            
        Returns:
            Dict[source_id, data] 收集结果字典
        """
        tasks = []
        for source in sources:
            task = self.receive(target, source, timeout)
            tasks.append((source, task))
        
        results = {}
        for source, task in tasks:
            try:
                result = await task
                results[source] = result
            except Exception as e:
                results[source] = e
                
        return results
    
    # ==================== 双向通信方法 ====================
    
    @abstractmethod
    async def push_event(self, source: str, target: str, event_type: str, data: Any) -> bool:
        """推送事件到目标节点
        
        Args:
            source: 源节点ID
            target: 目标节点ID
            event_type: 事件类型
            data: 事件数据
            
        Returns:
            bool: 推送是否成功
        """
        pass
    
    def register_event_handler(self, event_type: str, handler: Callable) -> str:
        """注册事件处理器
        
        Args:
            event_type: 事件类型
            handler: 处理函数
            
        Returns:
            str: 处理器ID
        """
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        
        handler_id = f"{event_type}_{len(self.event_handlers[event_type])}"
        self.event_handlers[event_type].append((handler_id, handler))
        return handler_id
    
    def unregister_event_handler(self, handler_id: str) -> bool:
        """取消注册事件处理器
        
        Args:
            handler_id: 处理器ID
            
        Returns:
            bool: 是否成功取消注册
        """
        for event_type, handlers in self.event_handlers.items():
            for i, (hid, handler) in enumerate(handlers):
                if hid == handler_id:
                    del handlers[i]
                    return True
        return False
    
    async def _handle_event(self, event_type: str, source: str, data: Any):
        """处理接收到的事件"""
        if event_type in self.event_handlers:
            for handler_id, handler in self.event_handlers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(source, data)
                    else:
                        handler(source, data)
                except Exception as e:
                    # 记录错误但不影响其他处理器
                    print(f"Event handler {handler_id} error: {e}")
    
    @abstractmethod
    async def start_event_listener(self, node_id: str) -> None:
        """启动事件监听器
        
        Args:
            node_id: 节点ID
        """
        pass
    
    # ==================== 连接管理方法 ====================
    
    async def create_connection(self, source: str, target: str) -> Connection:
        """创建连接
        
        Args:
            source: 源节点ID
            target: 目标节点ID
            
        Returns:
            Connection: 连接对象
        """
        async with self._lock:
            connection = Connection(source_id=source, target_id=target)
            connection_key = f"{source}->{target}"
            self.connections[connection_key] = connection
            return connection
    
    async def close_connection(self, connection_id: str) -> bool:
        """关闭连接
        
        Args:
            connection_id: 连接ID
            
        Returns:
            bool: 是否成功关闭
        """
        async with self._lock:
            # 查找并关闭连接
            for key, conn in list(self.connections.items()):
                if conn.connection_id == connection_id:
                    conn.status = ConnectionStatus.DISCONNECTED
                    del self.connections[key]
                    return True
            return False
    
    def get_connection_status(self, connection_id: str) -> ConnectionStatus:
        """获取连接状态
        
        Args:
            connection_id: 连接ID
            
        Returns:
            ConnectionStatus: 连接状态
        """
        for conn in self.connections.values():
            if conn.connection_id == connection_id:
                return conn.status
        return ConnectionStatus.DISCONNECTED
    
    def list_connections(self) -> List[Connection]:
        """列出所有连接
        
        Returns:
            List[Connection]: 连接列表
        """
        return list(self.connections.values())
    
    # ==================== 生命周期方法 ====================
    
    async def initialize(self) -> bool:
        """初始化传输层
        
        Returns:
            bool: 初始化是否成功
        """
        return True
    
    async def start(self) -> None:
        """启动传输层"""
        self._running = True
    
    async def stop(self) -> None:
        """停止传输层"""
        self._running = False
    
    async def cleanup(self) -> None:
        """清理传输层资源"""
        async with self._lock:
            self.connections.clear()
            self.event_handlers.clear()
    
    # ==================== 工具方法 ====================
    
    def is_running(self) -> bool:
        """检查传输层是否运行中"""
        return self._running
    
    def validate_node_id(self, node_id: str) -> bool:
        """验证节点ID格式
        
        Args:
            node_id: 节点ID
            
        Returns:
            bool: 是否有效
        """
        if not node_id or not isinstance(node_id, str):
            return False
        
        # 根据模式验证格式
        if self.config.type == "memory":
            return node_id.startswith("memory_")
        elif self.config.type == "process":
            return node_id.startswith("process_")
        elif self.config.type == "network":
            return node_id.startswith("network_")
        
        return True