"""
Engine层异常定义

定义了引擎层特定的异常类型，用于更精确的错误处理和调试。
"""

from typing import Optional, Any, Dict


class EngineError(Exception):
    """引擎层基础异常类"""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.context = context or {}
    
    def __str__(self) -> str:
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} (Context: {context_str})"
        return self.message


class ExperimentEngineError(EngineError):
    """实验引擎异常"""
    pass


class FederationEngineError(EngineError):
    """联邦引擎异常"""
    pass


class TrainingEngineError(EngineError):
    """训练引擎异常"""
    pass


class EvaluationEngineError(EngineError):
    """评估引擎异常"""
    pass


class EngineStateError(EngineError):
    """引擎状态异常 - 当引擎处于不正确状态时抛出"""
    pass


class EngineConfigurationError(EngineError):
    """引擎配置异常 - 当引擎配置无效时抛出"""
    pass


class EngineResourceError(EngineError):
    """引擎资源异常 - 当资源不可用或耗尽时抛出"""
    pass


class SchedulerError(EngineError):
    """调度器异常"""
    pass


class ExecutionError(EngineError):
    """执行异常"""
    pass
