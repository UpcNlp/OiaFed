"""
Node 通信层异常定义
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class ErrorInfo:
    """错误信息"""
    
    code: str                                    # 错误码
    message: str                                 # 错误描述
    details: Optional[Dict[str, Any]] = None    # 详细信息
    stack_trace: Optional[str] = None           # 调用栈（调试模式）


class NodeError(Exception):
    """Node 层基础异常"""
    pass


class NodeNotConnectedError(NodeError):
    """目标节点未连接"""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        super().__init__(f"Node '{node_id}' is not connected")


class NodeDisconnectedError(NodeError):
    """节点断开连接"""
    
    def __init__(self, node_id: str, reason: str = None):
        self.node_id = node_id
        self.reason = reason
        super().__init__(f"Node '{node_id}' disconnected: {reason}")


class CallTimeoutError(NodeError):
    """调用超时"""
    
    def __init__(
        self,
        message: str = None,
        msg_id: str = None,
        method: str = None,
        target: str = None,
        timeout: float = None,
    ):
        self.msg_id = msg_id
        self.method = method
        self.target = target
        self.timeout = timeout
        
        if message is None:
            message = f"Request {msg_id} to {target}.{method}() timed out after {timeout}s"
        super().__init__(message)


class AuthenticationError(NodeError):
    """认证失败"""
    
    def __init__(self, message: str = "Authentication failed", details: Dict[str, Any] = None):
        self.details = details
        super().__init__(message)


class RemoteExecutionError(NodeError):
    """远程执行错误"""
    
    def __init__(self, code: str, message: str, details: Dict[str, Any] = None):
        self.code = code
        self.details = details
        super().__init__(f"[{code}] {message}")


class SerializationError(NodeError):
    """序列化错误"""
    pass


class InterceptorAbort(NodeError):
    """拦截器中止"""
    
    def __init__(self, error_info: ErrorInfo):
        self.error_info = error_info
        super().__init__(error_info.message)
