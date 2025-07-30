# fedcl/registry/__init__.py
"""
FedCL 组件注册系统

提供组件注册、查询、验证和组合等功能，支持插件式的组件管理架构。

主要模块:
- ComponentRegistry: 核心组件注册表
- ComponentComposer: 组件组合器  
- ComponentMetadata: 组件元数据管理

使用示例:
    from fedcl.registry import registry, ComponentComposer
    
    # 注册自定义学习器
    @registry.learner("MyLearner")
    class MyLearner(BaseLearner):
        pass
    
    # 创建实验组件
    composer = ComponentComposer(registry)
    components = composer.compose_experiment(config, context)
"""

from .component_registry import (
    ComponentRegistry,
    ComponentMetadata,
    ComponentRegistrationError,
    ComponentNotFoundError,
    ComponentValidationError,
    ComponentConflictError,
    registry  # 全局注册表实例
)

from .component_composer import (
    ComponentComposer,
    ComponentSpec,
    ExperimentComponents,
    ComponentCompositionError,
    DependencyResolutionError,
    ComponentCreationError,
    CircularDependencyError
)

# 版本信息
__version__ = "1.0.0"
__author__ = "FedCL Team"

# API 导出
__all__ = [
    # 核心类
    'ComponentRegistry',
    'ComponentComposer',
    'ComponentMetadata',
    'ComponentSpec',
    'ExperimentComponents',
    
    # 全局实例
    'registry',
    
    # 异常类
    'ComponentRegistrationError',
    'ComponentNotFoundError', 
    'ComponentValidationError',
    'ComponentConflictError',
    'ComponentCompositionError',
    'DependencyResolutionError',
    'ComponentCreationError',
    'CircularDependencyError',
    
    # 装饰器 (从全局registry实例导出)
    'learner',
    'aggregator', 
    'evaluator',
    'hook',
    'loss_function',
    'auxiliary_model'
]

# 从全局registry导出装饰器方法，方便直接使用
learner = registry.learner
aggregator = registry.aggregator
evaluator = registry.evaluator
hook = registry.hook
loss_function = registry.loss_function
auxiliary_model = registry.auxiliary_model

# 模块级别的便捷函数
def get_component(component_type: str, name: str):
    """
    获取注册的组件
    
    Args:
        component_type: 组件类型
        name: 组件名称
        
    Returns:
        组件类
    """
    return registry.get_component(component_type, name)


def list_components(component_type: str = None):
    """
    列出已注册的组件
    
    Args:
        component_type: 组件类型，None表示列出所有类型
        
    Returns:
        组件列表或组件类型字典
    """
    if component_type is None:
        # 返回所有类型的组件
        result = {}
        for comp_type in ComponentRegistry.COMPONENT_TYPES:
            result[comp_type] = registry.list_components(comp_type)
        return result
    else:
        return registry.list_components(component_type)


def get_registry_stats():
    """
    获取注册表统计信息
    
    Returns:
        统计信息字典
    """
    return registry.get_registry_stats()


def clear_registry():
    """
    清空注册表
    
    警告: 这将清空所有已注册的组件，谨慎使用!
    """
    registry.clear_registry()


# 模块初始化日志
from loguru import logger
logger.debug(f"FedCL Registry module initialized (version {__version__})")