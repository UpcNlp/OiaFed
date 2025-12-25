"""
拦截器抽象基类和拦截器链
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Awaitable
import asyncio

from ..message import Message


@dataclass
class InterceptorContext:
    """
    拦截器上下文
    
    Attributes:
        node_id: 当前节点 ID
        direction: 消息方向 "inbound" | "outbound"
        attributes: 拦截器间共享数据，约定的键名：
            - trace_id: 分布式追踪 ID (str)
            - auth_token: 认证令牌 (str)
            - auth_identity: 认证后的身份信息 (dict)
            - start_time: 请求开始时间 (float)
            - user_*: 用户自定义键（前缀约定）
    """
    
    node_id: str
    direction: str  # "inbound" | "outbound"
    attributes: Dict[str, Any] = field(default_factory=dict)


class Interceptor(ABC):
    """拦截器基类"""
    
    @abstractmethod
    async def intercept_inbound(
        self,
        message: Message,
        context: InterceptorContext,
        next_handler: Callable[[Message], Awaitable[Message]],
    ) -> Message:
        """
        拦截入站消息
        
        Args:
            message: 接收到的消息
            context: 拦截器上下文
            next_handler: 调用链中的下一个处理器
            
        Returns:
            处理后的消息（可修改）
            
        Raises:
            InterceptorAbort: 中止处理链
        """
        pass
    
    @abstractmethod
    async def intercept_outbound(
        self,
        message: Message,
        context: InterceptorContext,
        next_handler: Callable[[Message], Awaitable[Message]],
    ) -> Message:
        """拦截出站消息"""
        pass


class InterceptorChain:
    """
    拦截器链
    
    执行顺序遵循"洋葱模型"：
    - 添加顺序: [A, B, C]
    - 执行顺序: C.before -> B.before -> A.before -> handler -> A.after -> B.after -> C.after
    
    使用 prepend=True 可将拦截器添加到最外层（最先执行）
    """
    
    def __init__(self, interceptors: Optional[List[Interceptor]] = None):
        self._interceptors = interceptors or []
    
    def add(self, interceptor: Interceptor, prepend: bool = False) -> None:
        """
        添加拦截器
        
        Args:
            interceptor: 拦截器实例
            prepend: 是否添加到最外层（最先执行）
        """
        if prepend:
            self._interceptors.insert(0, interceptor)
        else:
            self._interceptors.append(interceptor)
    
    def remove(self, interceptor: Interceptor) -> bool:
        """移除拦截器，返回是否成功"""
        try:
            self._interceptors.remove(interceptor)
            return True
        except ValueError:
            return False
    
    def clear(self) -> None:
        """清空拦截器"""
        self._interceptors.clear()
    
    async def execute_inbound(self, message: Message, node_id: str) -> Message:
        """执行入站拦截器链"""
        context = InterceptorContext(
            node_id=node_id,
            direction="inbound",
            attributes={},
        )
        
        async def terminal(msg: Message) -> Message:
            return msg
        
        handler = terminal
        for interceptor in reversed(self._interceptors):
            handler = self._wrap_inbound(interceptor, context, handler)
        
        return await handler(message)
    
    async def execute_outbound(self, message: Message, node_id: str) -> Message:
        """执行出站拦截器链"""
        context = InterceptorContext(
            node_id=node_id,
            direction="outbound",
            attributes={},
        )
        
        async def terminal(msg: Message) -> Message:
            return msg
        
        handler = terminal
        for interceptor in reversed(self._interceptors):
            handler = self._wrap_outbound(interceptor, context, handler)
        
        return await handler(message)
    
    def _wrap_inbound(
        self,
        interceptor: Interceptor,
        context: InterceptorContext,
        next_handler: Callable[[Message], Awaitable[Message]],
    ) -> Callable[[Message], Awaitable[Message]]:
        """包装入站拦截器"""
        async def wrapped(message: Message) -> Message:
            return await interceptor.intercept_inbound(message, context, next_handler)
        return wrapped
    
    def _wrap_outbound(
        self,
        interceptor: Interceptor,
        context: InterceptorContext,
        next_handler: Callable[[Message], Awaitable[Message]],
    ) -> Callable[[Message], Awaitable[Message]]:
        """包装出站拦截器"""
        async def wrapped(message: Message) -> Message:
            return await interceptor.intercept_outbound(message, context, next_handler)
        return wrapped
