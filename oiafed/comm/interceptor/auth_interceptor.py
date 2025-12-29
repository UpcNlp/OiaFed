"""
认证拦截器

提供 Token-based 认证机制
"""

from typing import Any, Awaitable, Callable, Dict, Optional

from ...infra.logging import get_logger
from .base import Interceptor, InterceptorContext
from ..message import Message, ErrorInfo
from ..exceptions import InterceptorAbort, AuthenticationError



class AuthInterceptor(Interceptor):
    """
    认证拦截器
    
    - 出站：在 message.metadata 中添加 auth_token
    - 入站：校验 auth_token，失败抛出 InterceptorAbort
    
    Example:
        interceptor = AuthInterceptor(
            token="my-secret-token",
            skip_methods=["_handshake", "_heartbeat"],
        )
        node._interceptor_chain.add(interceptor)
    """
    
    # 默认跳过认证的方法
    DEFAULT_SKIP_METHODS = {"_handshake", "_heartbeat"}
    
    def __init__(
        self,
        token: Optional[str] = None,
        token_provider: Optional[Callable[[], str]] = None,
        token_validator: Optional[Callable[[str], bool]] = None,
        skip_methods: Optional[set] = None,
    ):
        """
        初始化认证拦截器
        
        Args:
            token: 静态 token（出站时使用）
            token_provider: 动态 token 提供者（优先于 token）
            token_validator: 自定义 token 验证器（入站时使用）
            skip_methods: 跳过认证的方法列表
        """
        self._token = token
        self._token_provider = token_provider
        self._token_validator = token_validator
        self._skip_methods = skip_methods or self.DEFAULT_SKIP_METHODS
        
        # 用于入站验证的已知 token 集合
        self._valid_tokens: set = set()
        if token:
            self._valid_tokens.add(token)

        # 创建 logger（没有 node_id，使用"auth"作为默认）
        self.logger = get_logger(node_id="auth", log_type="system")
    
    def add_valid_token(self, token: str) -> None:
        """添加有效 token（用于入站验证）"""
        self._valid_tokens.add(token)
    
    def remove_valid_token(self, token: str) -> None:
        """移除有效 token"""
        self._valid_tokens.discard(token)
    
    async def intercept_outbound(
        self,
        message: Message,
        context: InterceptorContext,
        next_handler: Callable[[Message], Awaitable[Message]],
    ) -> Message:
        """出站：添加认证 token 到 metadata"""
        
        # 跳过某些方法
        if message.method in self._skip_methods:
            return await next_handler(message)
        
        # 获取 token
        token = None
        if self._token_provider:
            token = self._token_provider()
        elif self._token:
            token = self._token
        
        if token:
            # 添加到 metadata
            if message.metadata is None:
                message.metadata = {}
            message.metadata["auth_token"] = token
            
            # 同时存储到 context 供后续拦截器使用
            context.attributes["auth_token"] = token
        
        return await next_handler(message)
    
    async def intercept_inbound(
        self,
        message: Message,
        context: InterceptorContext,
        next_handler: Callable[[Message], Awaitable[Message]],
    ) -> Message:
        """入站：验证认证 token"""
        
        # 跳过某些方法
        if message.method in self._skip_methods:
            return await next_handler(message)
        
        # 获取 token
        token = message.metadata.get("auth_token") if message.metadata else None
        
        if not token:
            self.logger.warning(f"Missing auth token for method: {message.method}")
            raise InterceptorAbort(
                ErrorInfo(
                    code="AUTH_MISSING",
                    message="Authentication token is required",
                )
            )
        
        # 验证 token
        valid = False
        
        if self._token_validator:
            # 使用自定义验证器
            valid = self._token_validator(token)
        else:
            # 使用内置验证
            valid = token in self._valid_tokens
        
        if not valid:
            self.logger.warning(f"Invalid auth token for method: {message.method}")
            raise InterceptorAbort(
                ErrorInfo(
                    code="AUTH_INVALID",
                    message="Invalid authentication token",
                )
            )
        
        # 存储认证信息到 context
        context.attributes["auth_token"] = token
        context.attributes["auth_identity"] = {"token": token}
        
        self.logger.debug(f"Auth passed for method: {message.method}")
        return await next_handler(message)
