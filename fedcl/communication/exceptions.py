# fedcl/communication/exceptions.py
"""
通信模块异常类定义

定义了通信模块中使用的所有自定义异常类，提供详细的错误信息和错误处理机制。
"""

from typing import Optional, Any
from fedcl.exceptions import FedCLError


class CommunicationError(FedCLError):
    """
    通信基础异常类
    
    所有通信相关异常的基类，用于处理通信过程中的各种错误。
    """
    pass


class NetworkError(CommunicationError):
    """
    网络连接异常
    
    当网络连接、断开或传输过程中发生错误时抛出。
    """
    
    def __init__(self, message: str, connection_id: Optional[str] = None,
                 network_details: Optional[dict] = None, **kwargs) -> None:
        """
        初始化网络异常
        
        Args:
            message: 错误消息
            connection_id: 连接ID
            network_details: 网络详细信息
            **kwargs: 其他参数传递给父类
        """
        super().__init__(message, **kwargs)
        self.connection_id = connection_id
        self.network_details = network_details or {}


class SerializationError(CommunicationError):
    """
    序列化/反序列化异常
    
    当数据序列化或反序列化过程中发生错误时抛出。
    """
    
    def __init__(self, message: str, data_type: Optional[str] = None,
                 serialization_details: Optional[dict] = None, **kwargs) -> None:
        """
        初始化序列化异常
        
        Args:
            message: 错误消息
            data_type: 数据类型
            serialization_details: 序列化详细信息
            **kwargs: 其他参数传递给父类
        """
        super().__init__(message, **kwargs)
        self.data_type = data_type
        self.serialization_details = serialization_details or {}


class ProtocolError(CommunicationError):
    """
    协议错误异常
    
    当消息协议验证、格式化或处理过程中发生错误时抛出。
    """
    
    def __init__(self, message: str, message_type: Optional[str] = None,
                 protocol_details: Optional[dict] = None, **kwargs) -> None:
        """
        初始化协议异常
        
        Args:
            message: 错误消息
            message_type: 消息类型
            protocol_details: 协议详细信息
            **kwargs: 其他参数传递给父类
        """
        super().__init__(message, **kwargs)
        self.message_type = message_type
        self.protocol_details = protocol_details or {}


class ConnectionTimeoutError(NetworkError):
    """
    连接超时异常
    
    当网络连接或数据传输超时时抛出。
    """
    
    def __init__(self, message: str, timeout_value: Optional[float] = None, **kwargs) -> None:
        """
        初始化超时异常
        
        Args:
            message: 错误消息
            timeout_value: 超时时间值
            **kwargs: 其他参数传递给父类
        """
        super().__init__(message, **kwargs)
        self.timeout_value = timeout_value


class SecurityError(CommunicationError):
    """
    安全相关异常
    
    当加密、解密或身份验证过程中发生错误时抛出。
    """
    
    def __init__(self, message: str, security_level: Optional[str] = None,
                 security_details: Optional[dict] = None, **kwargs) -> None:
        """
        初始化安全异常
        
        Args:
            message: 错误消息
            security_level: 安全级别
            security_details: 安全详细信息
            **kwargs: 其他参数传递给父类
        """
        super().__init__(message, **kwargs)
        self.security_level = security_level
        self.security_details = security_details or {}
