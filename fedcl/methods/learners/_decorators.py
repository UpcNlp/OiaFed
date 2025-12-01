"""
Learner Decorator with Explicit Registration
显式指定命名空间和方法名的装饰器

用户需要明确指定命名空间和方法名,提供最大的灵活性
"""

from typing import Type, Optional
from functools import wraps

from ._registry import register as registry_register


def learner(namespace: str, name: str, description: Optional[str] = None):
    """
    Learner装饰器 - 显式注册

    使用方式:
        @learner('fl', 'FedAvg')
        class FedAvgLearner(BaseLearner):
            pass

        @learner('cl', 'TARGET', description='Federated Class-Continual Learning')
        class TARGETLearner(BaseLearner):
            pass

    Args:
        namespace: 命名空间 (fl/cl/ul)
        name: 方法名 (如 FedAvg, TARGET)
        description: 可选的描述信息 (用于文档生成)

    注册结果:
        learner('fl', 'FedAvg') -> 注册为 'fl.FedAvg'
        learner('cl', 'TARGET') -> 注册为 'cl.TARGET'
    """

    def decorator(cls: Type) -> Type:
        # 验证命名空间
        valid_namespaces = ('fl', 'cl', 'ul')
        if namespace not in valid_namespaces:
            raise ValueError(
                f"Invalid namespace '{namespace}'. "
                f"Must be one of: {', '.join(valid_namespaces)}"
            )

        # 验证方法名
        if not name or not isinstance(name, str):
            raise ValueError(f"Method name must be a non-empty string, got: {name}")

        # 注册到registry
        registry_register(namespace, name, cls)

        # 在类上添加元数据
        cls._learner_namespace = namespace
        cls._learner_method = name
        cls._learner_full_name = f"{namespace}.{name}"
        if description:
            cls._learner_description = description

        return cls

    return decorator


def get_learner_info(cls: Type) -> dict:
    """
    获取learner的注册信息

    Args:
        cls: Learner类

    Returns:
        包含注册信息的字典
    """
    return {
        'namespace': getattr(cls, '_learner_namespace', None),
        'method': getattr(cls, '_learner_method', None),
        'full_name': getattr(cls, '_learner_full_name', None),
        'description': getattr(cls, '_learner_description', None),
        'class_name': cls.__name__,
    }
