# fedcl/__init__.py
"""
FedCL: Federated Continual Learning Framework

提供简洁的装饰器API和快速实验接口，支持联邦持续学习的快速原型开发。
"""

from typing import Dict, Any, Optional, List, Union, Type, Callable
from pathlib import Path
from loguru import logger

# 导入核心组件
from .registry.component_registry import registry
from .core.execution_context import ExecutionContext
from .config.config_manager import ConfigManager, DictConfig
from .config.schema_validator import SchemaValidator
from .experiment.experiment import FedCLExperiment

# 导入基类
from .core.base_learner import BaseLearner
from .core.base_aggregator import BaseAggregator
from .core.base_evaluator import BaseEvaluator
from .core.hook import Hook

# 版本信息
__version__ = "0.1.0"
__author__ = "FedCL Development Team"

# 初始化全局组件
_config_manager = None
_schema_validator = None

def _ensure_initialized():
    """确保全局组件已初始化"""
    global _config_manager, _schema_validator
    if _config_manager is None:
        _schema_validator = SchemaValidator()
        _config_manager = ConfigManager(_schema_validator)
    return _config_manager, _schema_validator

# ==================== 装饰器API ====================

def loss(name: str, scope: str = "local") -> Callable:
    """
    损失函数装饰器
    
    Args:
        name: 损失函数名称
        scope: 作用域 ("local", "global", "distributed")
        
    Returns:
        装饰器函数
        
    Example:
        @fedcl.loss("custom_kl_loss")
        def kl_divergence_loss(predictions, targets, context):
            return F.kl_div(F.log_softmax(predictions, dim=1), 
                           F.softmax(targets, dim=1), 
                           reduction='batchmean')
    """
    return registry.loss_function(name, scope)

def hook(phase: str, priority: int = 0, enable=False, **metadata) -> Callable:
    """
    钩子函数装饰器
    
    Args:
        phase: 钩子执行阶段
        priority: 优先级（数值越大优先级越高）
        **metadata: 额外元数据
        
    Returns:
        装饰器函数
        
    Example:
        @fedcl.hook("before_task", priority=100)
        class DataAugmentationHook(Hook):
            def execute(self, context, task_data, **kwargs):
                # 数据增强逻辑
                return augmented_data
    """
    return registry.hook(phase, priority,enable, **metadata)

def model(name: str, model_type: str = "auxiliary") -> Callable:
    """
    辅助模型装饰器
    
    Args:
        name: 模型名称
        model_type: 模型类型
    
    Returns:
        装饰器函数
    
    Example:
        @fedcl.model("teacher_model")
        class TeacherModel:
            def __init__(self, config=None, context=None):
                self.config = config or {}
                self.context = context
                
            def create_model(self):
                model = torchvision.models.resnet50(pretrained=True)
                model.eval()
                return {'model': model, 'feature_extractor': model.avgpool}
    """
    return registry.auxiliary_model(name, model_type)

def learner(name: str, **metadata) -> Callable:
    """
    学习器装饰器
    
    Args:
        name: 学习器名称
        **metadata: 额外元数据
        
    Returns:
        装饰器函数
        
    Example:
        @fedcl.learner("ewc_mnist")
        class EWCMNISTLearner(BaseLearner):
            def train_task(self, task_data, task_id):
                # EWC训练逻辑
                pass
    """
    return registry.learner(name, **metadata)

def aggregator(name: str, **metadata) -> Callable:
    """
    聚合器装饰器
    
    Args:
        name: 聚合器名称
        **metadata: 额外元数据
        
    Returns:
        装饰器函数
        
    Example:
        @fedcl.aggregator("custom_fedavg")
        class CustomFedAvgAggregator(BaseAggregator):
            def aggregate(self, client_updates):
                # 自定义聚合逻辑
                pass
    """
    return registry.aggregator(name, **metadata)

def evaluator(name: str, **metadata) -> Callable:
    """
    评估器装饰器
    
    Args:
        name: 评估器名称
        **metadata: 额外元数据
        
    Returns:
        装饰器函数  
        
    Example:
        @fedcl.evaluator("custom_accuracy")
        class CustomAccuracyEvaluator(BaseEvaluator):
            def evaluate(self, model, test_data):
                # 自定义评估逻辑
                pass
    """
    return registry.evaluator(name, **metadata)

# ==================== 实验管理API ====================

def load_config(config_path: Union[str, Path]) -> DictConfig:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置对象
    """
    config_manager, _ = _ensure_initialized()
    return config_manager.load_config(config_path)

def create_experiment(config: Union[str, Path, DictConfig], 
                     working_dir: Optional[Path] = None,
                     seed: Optional[int] = None) -> 'FedCLExperiment':
    """
    创建实验对象
    
    Args:
        config: 配置文件路径或配置对象
        working_dir: 工作目录
        seed: 随机种子
        
    Returns:
        实验对象
    """
    return FedCLExperiment(config, working_dir, seed)

# ==================== 查询API ====================

def list_components(component_type: str) -> List[str]:
    """
    列出已注册的组件
    
    Args:
        component_type: 组件类型
        
    Returns:
        组件名称列表
    """
    return registry.list_components(component_type)

def get_component_info(component_type: str, name: str) -> Dict[str, Any]:
    """
    获取组件信息
    
    Args:
        component_type: 组件类型
        name: 组件名称
        
    Returns:
        组件元数据
    """
    metadata = registry.get_component_metadata(component_type, name)
    return metadata.to_dict()

def get_registry_stats() -> Dict[str, Any]:
    """获取注册表统计信息"""
    return registry.get_registry_stats()

# ==================== 全局设置 ====================

def set_log_level(level: str) -> None:
    """设置日志级别"""
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level=level.upper())

def clear_registry() -> None:
    """清空组件注册表"""
    registry.clear_registry()
    logger.info("Component registry cleared")

# ==================== 导出API ====================

__all__ = [
    # 版本信息
    "__version__", "__author__",
    
    # 装饰器API
    "loss", "hook", "model", "learner", "aggregator", "evaluator",
    
    # 实验API
    "quick_experiment", "create_experiment", "load_config",
    
    # 新的统一API
    "MOEFedCL", "quick_start", "create_federated_experiment", 
    "run_federated_experiment", "register_custom_learner",
    "ScheduleMode",
    
    # 查询API
    "list_components", "get_component_info", "get_registry_stats",
    
    # 基类
    "BaseLearner", "Base聚合器", "BaseEvaluator", "Hook",
    
    # 核心类
    "FedCLExperiment", "ExecutionContext", "DictConfig",
    
    # 工具函数
    "set_log_level", "clear_registry"
]

# 启动时的信息
logger.debug(f"FedCL Framework v{__version__} initialized")
logger.debug("Use fedcl.quick_experiment() for fast prototyping")
logger.debug("Use @fedcl.loss, @fedcl.hook, @fedcl.model decorators for customization")