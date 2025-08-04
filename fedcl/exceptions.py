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


class ModelCreationError(FedCLError):
    """
    模型创建错误
    
    当模型创建过程中出现错误时抛出，包括：
    - 模型类未找到
    - 模型参数无效
    - 工厂函数执行失败
    - 模型初始化失败
    """
    
    def __init__(self, message: str, model_type: Optional[str] = None, 
                 original_error: Optional[Exception] = None):
        """
        初始化模型创建错误
        
        Args:
            message: 错误消息
            model_type: 模型类型（可选）
            original_error: 原始异常（可选）
        """
        super().__init__(message)
        self.model_type = model_type
        self.original_error = original_error
    
    def __str__(self) -> str:
        """返回详细的错误信息"""
        base_msg = super().__str__()
        
        if self.model_type:
            base_msg = f"[{self.model_type}] {base_msg}"
        
        if self.original_error:
            base_msg = f"{base_msg} (caused by: {self.original_error})"
        
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


class EngineError(FedCLError):
    """
    引擎异常
    
    当引擎操作失败时抛出。
    """
    pass


class EngineStateError(EngineError):
    """
    引擎状态异常
    
    当引擎状态转换失败时抛出。
    """
    pass


class ExperimentEngineError(EngineError):
    """
    实验引擎异常
    
    当实验引擎操作失败时抛出。
    """
    pass


class FederationEngineError(EngineError):
    """
    联邦引擎异常
    
    当联邦引擎操作失败时抛出。
    """
    pass


class TrainingEngineError(EngineError):
    """
    训练引擎异常
    
    当训练引擎操作失败时抛出。
    """
    pass


class EvaluationEngineError(EngineError):
    """
    评估引擎异常
    
    当评估引擎操作失败时抛出。
    """
    pass