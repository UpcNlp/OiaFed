"""
MOE-FedCL 安全认证服务
moe_fedcl/communication/services/security.py
"""

import asyncio
import hashlib
import hmac
import secrets
import json
import base64
from typing import Dict, List, Optional, Any, Set, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from ...types import ClientInfo, EventMessage
from ...exceptions import SecurityError, AuthenticationError, AuthorizationError
from ...utils.auto_logger import get_logger


@dataclass
class AuthToken:
    """认证令牌"""
    token: str
    client_id: str
    issued_at: datetime
    expires_at: datetime
    permissions: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityPolicy:
    """安全策略"""
    require_authentication: bool = True
    token_expiration: int = 3600  # 秒
    max_failed_attempts: int = 5
    lockout_duration: int = 300  # 秒
    allowed_capabilities: Set[str] = field(default_factory=lambda: {"train", "evaluate"})
    ip_whitelist: Optional[Set[str]] = None
    rate_limit: Dict[str, int] = field(default_factory=lambda: {"requests_per_minute": 60})


class SecurityService:
    """安全认证服务
    
    负责身份验证、授权和安全策略管理
    """
    
    def __init__(self, secret_key: str, policy: SecurityPolicy = None, node_id: Optional[str] = None):
        """初始化安全认证服务

        Args:
            secret_key: 用于生成和验证令牌的密钥
            policy: 安全策略
            node_id: 节点ID（用于日志归属）
        """
        self.secret_key = secret_key.encode() if isinstance(secret_key, str) else secret_key
        self.policy = policy or SecurityPolicy()
        # 使用节点ID的运行日志，让安全日志合并到节点日志中
        if node_id:
            self.logger = get_logger("runtime", node_id)
        else:
            # 向后兼容：如果没有node_id，使用旧的方式
            self.logger = get_logger("sys", "security_service")
        
        # 令牌存储
        self.active_tokens: Dict[str, AuthToken] = {}
        self.client_tokens: Dict[str, str] = {}  # client_id -> token
        
        # 认证失败跟踪
        self.failed_attempts: Dict[str, List[datetime]] = {}
        self.locked_clients: Dict[str, datetime] = {}
        
        # 权限管理
        self.client_permissions: Dict[str, Set[str]] = {}
        
        # 事件回调
        self.event_callbacks: List[Callable[[EventMessage], None]] = []
        
        # 锁保护并发操作
        self._lock = asyncio.Lock()
        
        # 清理任务
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self) -> None:
        """启动安全服务"""
        if self._running:
            self.logger.warning("Security service is already running")
            return
        
        self._running = True
        
        # 启动令牌清理任务
        self._cleanup_task = asyncio.create_task(self._token_cleanup_loop())
        
        self.logger.debug("Security service started")
    
    async def stop(self) -> None:
        """停止安全服务"""
        if not self._running:
            return
        
        self._running = False
        
        # 取消清理任务
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
        
        self.logger.debug("Security service stopped")
    
    async def authenticate_client(self, client_id: str, credentials: Dict[str, Any]) -> AuthToken:
        """认证客户端
        
        Args:
            client_id: 客户端ID
            credentials: 认证凭据
            
        Returns:
            AuthToken: 认证令牌
            
        Raises:
            AuthenticationError: 认证失败
            SecurityError: 安全错误
        """
        async with self._lock:
            try:
                # 检查客户端是否被锁定
                await self._check_client_lockout(client_id)
                
                # 验证凭据
                if not await self._verify_credentials(client_id, credentials):
                    await self._record_failed_attempt(client_id)
                    raise AuthenticationError(f"Invalid credentials for client {client_id}")
                
                # 清除失败记录
                self.failed_attempts.pop(client_id, None)
                
                # 生成认证令牌
                token = await self._generate_token(client_id, credentials)
                
                # 保存令牌
                self.active_tokens[token.token] = token
                
                # 如果客户端已有令牌，撤销旧令牌
                if client_id in self.client_tokens:
                    old_token = self.client_tokens[client_id]
                    self.active_tokens.pop(old_token, None)
                
                self.client_tokens[client_id] = token.token
                
                self.logger.info(f"Client {client_id} authenticated successfully")
                
                # 触发认证事件
                await self._emit_event("CLIENT_AUTHENTICATED", client_id, {
                    "token_id": token.token[:8] + "...",  # 只显示前8位
                    "permissions": list(token.permissions),
                    "expires_at": token.expires_at
                })
                
                return token
                
            except (AuthenticationError, SecurityError):
                raise
            except Exception as e:
                self.logger.error(f"Authentication error for client {client_id}: {e}")
                raise SecurityError(f"Authentication failed: {e}")
    
    async def validate_token(self, token: str) -> AuthToken:
        """验证令牌
        
        Args:
            token: 认证令牌
            
        Returns:
            AuthToken: 令牌信息
            
        Raises:
            AuthenticationError: 令牌无效
        """
        async with self._lock:
            # 检查令牌是否存在
            if token not in self.active_tokens:
                raise AuthenticationError("Invalid token")
            
            auth_token = self.active_tokens[token]
            
            # 检查令牌是否过期
            if datetime.now() > auth_token.expires_at:
                await self._revoke_token(token)
                raise AuthenticationError("Token expired")
            
            return auth_token
    
    async def authorize_operation(self, token: str, operation: str, resource: str = None) -> bool:
        """授权操作
        
        Args:
            token: 认证令牌
            operation: 操作类型
            resource: 资源（可选）
            
        Returns:
            bool: 是否授权
            
        Raises:
            AuthenticationError: 令牌无效
            AuthorizationError: 授权失败
        """
        try:
            # 验证令牌
            auth_token = await self.validate_token(token)
            
            # 检查权限
            if not await self._check_permission(auth_token, operation, resource):
                await self._emit_event("AUTHORIZATION_DENIED", auth_token.client_id, {
                    "operation": operation,
                    "resource": resource,
                    "reason": "insufficient_permissions"
                })
                raise AuthorizationError(f"Client {auth_token.client_id} not authorized for operation: {operation}")
            
            return True
            
        except (AuthenticationError, AuthorizationError):
            raise
        except Exception as e:
            self.logger.error(f"Authorization error: {e}")
            raise SecurityError(f"Authorization failed: {e}")
    
    async def revoke_token(self, token: str) -> bool:
        """撤销令牌
        
        Args:
            token: 认证令牌
            
        Returns:
            bool: 是否成功撤销
        """
        async with self._lock:
            return await self._revoke_token(token)
    
    async def revoke_client_tokens(self, client_id: str) -> bool:
        """撤销客户端所有令牌
        
        Args:
            client_id: 客户端ID
            
        Returns:
            bool: 是否成功撤销
        """
        async with self._lock:
            if client_id in self.client_tokens:
                token = self.client_tokens[client_id]
                await self._revoke_token(token)
                return True
            return False
    
    async def refresh_token(self, token: str) -> AuthToken:
        """刷新令牌
        
        Args:
            token: 当前令牌
            
        Returns:
            AuthToken: 新令牌
            
        Raises:
            AuthenticationError: 令牌无效
        """
        async with self._lock:
            # 验证当前令牌
            old_token = await self.validate_token(token)
            
            # 生成新令牌
            new_token = AuthToken(
                token=await self._generate_jwt_token(old_token.client_id, old_token.permissions),
                client_id=old_token.client_id,
                issued_at=datetime.now(),
                expires_at=datetime.now() + timedelta(seconds=self.policy.token_expiration),
                permissions=old_token.permissions,
                metadata=old_token.metadata
            )
            
            # 替换令牌
            self.active_tokens[new_token.token] = new_token
            self.client_tokens[old_token.client_id] = new_token.token
            await self._revoke_token(token, log_event=False)
            
            self.logger.debug(f"Token refreshed for client {old_token.client_id}")
            
            return new_token
    
    async def set_client_permissions(self, client_id: str, permissions: Set[str]) -> bool:
        """设置客户端权限
        
        Args:
            client_id: 客户端ID
            permissions: 权限集合
            
        Returns:
            bool: 是否设置成功
        """
        async with self._lock:
            # 验证权限是否有效
            invalid_permissions = permissions - self.policy.allowed_capabilities
            if invalid_permissions:
                self.logger.warning(f"Invalid permissions for {client_id}: {invalid_permissions}")
                return False
            
            self.client_permissions[client_id] = permissions
            
            # 如果客户端当前有活跃令牌，更新权限
            if client_id in self.client_tokens:
                token = self.client_tokens[client_id]
                if token in self.active_tokens:
                    self.active_tokens[token].permissions = permissions
            
            return True
    
    async def get_client_permissions(self, client_id: str) -> Set[str]:
        """获取客户端权限
        
        Args:
            client_id: 客户端ID
            
        Returns:
            Set[str]: 权限集合
        """
        return self.client_permissions.get(client_id, {"train", "evaluate"})
    
    def get_security_statistics(self) -> Dict[str, Any]:
        """获取安全统计信息
        
        Returns:
            Dict[str, Any]: 安全统计
        """
        now = datetime.now()
        active_tokens = len([
            token for token in self.active_tokens.values()
            if token.expires_at > now
        ])
        
        return {
            "active_tokens": active_tokens,
            "total_tokens_issued": len(self.active_tokens),
            "locked_clients": len(self.locked_clients),
            "failed_attempts_tracked": len(self.failed_attempts),
            "policy": {
                "require_authentication": self.policy.require_authentication,
                "token_expiration": self.policy.token_expiration,
                "max_failed_attempts": self.policy.max_failed_attempts,
                "lockout_duration": self.policy.lockout_duration
            }
        }
    
    def register_event_callback(self, callback: Callable[[EventMessage], None]) -> None:
        """注册事件回调
        
        Args:
            callback: 事件回调函数
        """
        if callback not in self.event_callbacks:
            self.event_callbacks.append(callback)
    
    def unregister_event_callback(self, callback: Callable[[EventMessage], None]) -> None:
        """取消注册事件回调
        
        Args:
            callback: 事件回调函数
        """
        if callback in self.event_callbacks:
            self.event_callbacks.remove(callback)
    
    async def _check_client_lockout(self, client_id: str) -> None:
        """检查客户端是否被锁定
        
        Args:
            client_id: 客户端ID
            
        Raises:
            SecurityError: 客户端被锁定
        """
        if client_id in self.locked_clients:
            locked_until = self.locked_clients[client_id]
            if datetime.now() < locked_until:
                raise SecurityError(f"Client {client_id} is locked until {locked_until}")
            else:
                # 锁定已过期，移除锁定
                del self.locked_clients[client_id]
    
    async def _verify_credentials(self, client_id: str, credentials: Dict[str, Any]) -> bool:
        """验证认证凭据
        
        Args:
            client_id: 客户端ID
            credentials: 认证凭据
            
        Returns:
            bool: 是否验证通过
        """
        # 简单的验证逻辑，实际项目中应该对接具体的认证系统
        if "password" in credentials:
            # 密码验证
            expected_hash = self._generate_password_hash(client_id, credentials["password"])
            return hmac.compare_digest(expected_hash, credentials.get("password_hash", ""))
        
        if "api_key" in credentials:
            # API密钥验证
            return await self._verify_api_key(client_id, credentials["api_key"])
        
        # 如果不要求认证，允许通过
        if not self.policy.require_authentication:
            return True
        
        return False
    
    async def _verify_api_key(self, client_id: str, api_key: str) -> bool:
        """验证API密钥
        
        Args:
            client_id: 客户端ID
            api_key: API密钥
            
        Returns:
            bool: 是否验证通过
        """
        # 简单验证逻辑，实际项目中应该从数据库或配置中验证
        expected_key = hashlib.sha256(f"{client_id}:{self.secret_key.decode()}".encode()).hexdigest()
        return hmac.compare_digest(expected_key, api_key)
    
    def _generate_password_hash(self, client_id: str, password: str) -> str:
        """生成密码哈希
        
        Args:
            client_id: 客户端ID
            password: 密码
            
        Returns:
            str: 密码哈希
        """
        salt = f"{client_id}:{self.secret_key.decode()}"
        return hashlib.pbkdf2_hex(password.encode(), salt.encode(), 100000)
    
    async def _generate_token(self, client_id: str, credentials: Dict[str, Any]) -> AuthToken:
        """生成认证令牌
        
        Args:
            client_id: 客户端ID
            credentials: 认证凭据
            
        Returns:
            AuthToken: 认证令牌
        """
        # 获取客户端权限
        permissions = await self.get_client_permissions(client_id)
        
        # 生成JWT令牌
        token_str = await self._generate_jwt_token(client_id, permissions)
        
        return AuthToken(
            token=token_str,
            client_id=client_id,
            issued_at=datetime.now(),
            expires_at=datetime.now() + timedelta(seconds=self.policy.token_expiration),
            permissions=permissions,
            metadata={"auth_method": list(credentials.keys())[0] if credentials else "none"}
        )
    
    async def _generate_jwt_token(self, client_id: str, permissions: Set[str]) -> str:
        """生成简化的JWT风格令牌
        
        Args:
            client_id: 客户端ID
            permissions: 权限集合
            
        Returns:
            str: JWT风格令牌
        """
        now = datetime.now()
        payload = {
            "sub": client_id,
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(seconds=self.policy.token_expiration)).timestamp()),
            "permissions": list(permissions),
            "jti": secrets.token_hex(16)  # JWT ID
        }
        
        # 简化的JWT实现：header.payload.signature
        header = {"alg": "HS256", "typ": "JWT"}
        
        # Base64 编码
        header_b64 = base64.urlsafe_b64encode(
            json.dumps(header, separators=(',', ':')).encode()
        ).decode().rstrip('=')
        
        payload_b64 = base64.urlsafe_b64encode(
            json.dumps(payload, separators=(',', ':')).encode()
        ).decode().rstrip('=')
        
        # 生成签名
        message = f"{header_b64}.{payload_b64}"
        signature = hmac.new(
            self.secret_key,
            message.encode(),
            hashlib.sha256
        ).digest()
        
        signature_b64 = base64.urlsafe_b64encode(signature).decode().rstrip('=')
        
        return f"{message}.{signature_b64}"
    
    async def _record_failed_attempt(self, client_id: str) -> None:
        """记录认证失败尝试
        
        Args:
            client_id: 客户端ID
        """
        now = datetime.now()
        
        if client_id not in self.failed_attempts:
            self.failed_attempts[client_id] = []
        
        self.failed_attempts[client_id].append(now)
        
        # 清理超过5分钟的失败记录
        cutoff = now - timedelta(minutes=5)
        self.failed_attempts[client_id] = [
            attempt for attempt in self.failed_attempts[client_id]
            if attempt > cutoff
        ]
        
        # 检查是否需要锁定客户端
        if len(self.failed_attempts[client_id]) >= self.policy.max_failed_attempts:
            locked_until = now + timedelta(seconds=self.policy.lockout_duration)
            self.locked_clients[client_id] = locked_until
            
            self.logger.warning(f"Client {client_id} locked until {locked_until} due to too many failed attempts")
            
            await self._emit_event("CLIENT_LOCKED", client_id, {
                "locked_until": locked_until,
                "failed_attempts": len(self.failed_attempts[client_id])
            })
    
    async def _check_permission(self, auth_token: AuthToken, operation: str, resource: str = None) -> bool:
        """检查权限
        
        Args:
            auth_token: 认证令牌
            operation: 操作类型
            resource: 资源
            
        Returns:
            bool: 是否有权限
        """
        # 简单权限检查逻辑
        if operation in auth_token.permissions:
            return True
        
        # 特殊权限检查
        if operation == "admin" and "admin" in auth_token.permissions:
            return True
        
        return False
    
    async def _revoke_token(self, token: str, log_event: bool = True) -> bool:
        """撤销令牌
        
        Args:
            token: 令牌
            log_event: 是否记录事件
            
        Returns:
            bool: 是否成功撤销
        """
        if token in self.active_tokens:
            auth_token = self.active_tokens[token]
            del self.active_tokens[token]
            
            # 从客户端令牌映射中移除
            if auth_token.client_id in self.client_tokens:
                if self.client_tokens[auth_token.client_id] == token:
                    del self.client_tokens[auth_token.client_id]
            
            if log_event:
                self.logger.debug(f"Token revoked for client {auth_token.client_id}")
                await self._emit_event("TOKEN_REVOKED", auth_token.client_id, {
                    "token_id": token[:8] + "...",
                    "reason": "manual_revocation"
                })
            
            return True
        
        return False
    
    async def _token_cleanup_loop(self) -> None:
        """令牌清理循环"""
        while self._running:
            try:
                await self._cleanup_expired_tokens()
                await asyncio.sleep(300)  # 每5分钟清理一次
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in token cleanup loop: {e}")
                await asyncio.sleep(60)  # 错误后1分钟再试
    
    async def _cleanup_expired_tokens(self) -> None:
        """清理过期令牌"""
        async with self._lock:
            now = datetime.now()
            expired_tokens = []
            
            for token, auth_token in self.active_tokens.items():
                if auth_token.expires_at <= now:
                    expired_tokens.append(token)
            
            for token in expired_tokens:
                await self._revoke_token(token, log_event=False)
            
            if expired_tokens:
                self.logger.debug(f"Cleaned up {len(expired_tokens)} expired tokens")
    
    async def _emit_event(self, event_type: str, client_id: str, data: any = None) -> None:
        """触发事件
        
        Args:
            event_type: 事件类型
            client_id: 客户端ID
            data: 事件数据
        """
        event = EventMessage(
            event_type=event_type,
            source_id=client_id,
            data=data
        )
        
        # 异步调用所有回调
        for callback in self.event_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                self.logger.error(f"Error in event callback: {e}")
