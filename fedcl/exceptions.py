"""
MOE-FedCL 自定义异常定义
moe_fedcl/exceptions.py
"""


class MOEFedCLError(Exception):
    """MOE-FedCL框架基础异常"""
    
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
    
    def __str__(self):
        base_msg = self.message
        if self.error_code:
            base_msg = f"[{self.error_code}] {base_msg}"
        if self.details:
            base_msg = f"{base_msg} | Details: {self.details}"
        return base_msg


class TransportError(MOEFedCLError):
    """传输层异常"""
    pass


class ConnectionError(MOEFedCLError):
    """连接相关异常"""
    pass


class RegistrationError(MOEFedCLError):
    """客户端注册异常"""
    pass


class CommunicationError(MOEFedCLError):
    """通信过程异常"""
    pass


class TrainingError(MOEFedCLError):
    """训练执行异常"""
    pass


class TimeoutError(MOEFedCLError):
    """超时异常"""
    pass


class ValidationError(MOEFedCLError):
    """参数验证异常"""
    pass


class ConfigurationError(MOEFedCLError):
    """配置异常"""
    pass


class ClientNotFoundError(MOEFedCLError):
    """客户端未找到异常"""
    pass


class FederationError(MOEFedCLError):
    """联邦学习异常"""
    pass


class SerializationError(MOEFedCLError):
    """序列化异常"""
    pass


class SecurityError(MOEFedCLError):
    """安全相关异常"""
    pass


class AuthenticationError(SecurityError):
    """身份认证异常"""
    pass


class AuthorizationError(SecurityError):
    """权限授权异常"""
    pass