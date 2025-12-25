"""
日志拦截器
"""

import time
from typing import Callable, Awaitable, Optional

from .base import Interceptor, InterceptorContext
from ..message import Message
from ...infra.logging import get_logger


class LoggingInterceptor(Interceptor):
    """日志拦截器"""

    def __init__(self, node_id: Optional[str] = None):
        """
        初始化日志拦截器

        Args:
            node_id: 节点ID（可选），用于绑定 logger context
        """
        # 使用 loguru 统一日志
        self.logger = get_logger(node_id=node_id, log_type="system")

    async def intercept_inbound(
        self,
        message: Message,
        context: InterceptorContext,
        next_handler: Callable[[Message], Awaitable[Message]],
    ) -> Message:
        self.logger.debug(
            f"[INBOUND] {message.source} -> {message.target} | "
            f"type={message.type.name} method={message.method} id={message.id[:8]}"
        )
        start = time.time()

        result = await next_handler(message)

        elapsed = (time.time() - start) * 1000
        self.logger.debug(
            f"[INBOUND] {message.method} completed in {elapsed:.2f}ms"
        )

        return result

    async def intercept_outbound(
        self,
        message: Message,
        context: InterceptorContext,
        next_handler: Callable[[Message], Awaitable[Message]],
    ) -> Message:
        self.logger.debug(
            f"[OUTBOUND] {message.source} -> {message.target} | "
            f"type={message.type.name} method={message.method} id={message.id[:8]}"
        )
        return await next_handler(message)
