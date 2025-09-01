# fedcl/__init__.py
"""
FedCL: 全新的透明联邦持续学习框架

让真联邦和伪联邦对用户完全透明，专注于算法逻辑而非分布式细节。
核心理念：分布式联邦写代码过程和集中式一样，底层自动处理权重、梯度、特征获取等。
"""

# 导入全新的简洁API
from .api.trainer import FederatedTrainer, TrainingResult, EvaluationResult
from .api.decorators import learner, aggregator, evaluator, list_components, get_component_info, clear_registry
from .api.experiments import train, train_from_config, quick_experiment

# 版本信息
__version__ = "0.2.0"
__author__ = "FedCL Development Team"

# 导出所有新API
__all__ = [
    # 核心类
    "FederatedTrainer",
    "TrainingResult", 
    "EvaluationResult",
    
    # 装饰器
    "learner",
    "aggregator",
    "evaluator",
    
    # 快速启动接口
    "train",
    "train_from_config",
    "quick_experiment",
    
    # 工具函数
    "list_components",
    "get_component_info",
    "clear_registry",
    
    # 版本信息
    "__version__",
    "__author__"
]

# 初始化日志
from loguru import logger
logger.info(f"🚀 FedCL 透明联邦框架 v{__version__} 已加载")
logger.info("💡 使用 fedcl.train() 一行代码启动联邦学习")
logger.info("📚 使用 @fedcl.learner 装饰器定义学习器")