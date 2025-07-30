# fedcl/core/exceptions.py
"""
Core模块异常类定义

定义了核心模块中使用的所有自定义异常类，提供详细的错误信息和错误处理机制。
"""

from typing import Optional, Any


class FedCLError(Exception):
    """
    FedCL框架基础异常类
    
    所有FedCL相关异常的基类，提供统一的异常处理接口。
    """
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 details: Optional[dict] = None) -> None:
        """
        初始化异常
        
        Args:
            message: 错误消息
            error_code: 错误代码
            details: 详细错误信息
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        
    def __str__(self) -> str:
        """字符串表示"""
        base_msg = self.message
        if self.error_code:
            base_msg = f"[{self.error_code}] {base_msg}"
        if self.details:
            base_msg += f" Details: {self.details}"
        return base_msg


class LearnerError(FedCLError):
    """
    学习器相关异常
    
    当学习器训练、评估或状态管理过程中发生错误时抛出。
    """
    pass


class TrainingError(FedCLError):
    """
    训练异常
    
    当模型训练过程中发生错误时抛出。
    """
    pass


class ModelStateError(FedCLError):
    """
    模型状态异常
    
    当模型状态不一致、未初始化或操作无效时抛出。
    """
    pass


class ConfigurationError(FedCLError):
    """
    配置异常
    
    当配置参数无效、缺失或冲突时抛出。
    """
    pass


class AggregationError(FedCLError):
    """
    聚合异常
    
    当联邦聚合过程中发生错误时抛出。
    """
    pass


class EvaluationError(FedCLError):
    """
    评估异常
    
    当模型评估过程中发生错误时抛出。
    """
    pass


class HookExecutionError(FedCLError):
    """
    钩子执行异常
    
    当钩子执行过程中发生错误时抛出。
    """
    pass


class ContextError(FedCLError):
    """
    执行上下文异常
    
    当执行上下文操作失败时抛出。
    """
    pass


class DataError(FedCLError):
    """
    数据异常
    
    当数据加载、处理或验证失败时抛出。
    """
    pass


class NetworkError(FedCLError):
    """
    网络通信异常
    
    当分布式通信过程中发生错误时抛出。
    """
    pass


class ResourceError(FedCLError):
    """
    资源异常
    
    当资源分配、管理或释放失败时抛出。
    """
    pass