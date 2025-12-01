"""
Learner Registry with Namespace Support
联邦学习方法注册中心

命名空间:
- fl: 标准联邦学习 (federated_learning)
- cl: 联邦持续学习 (continual_learning)
- ul: 联邦遗忘学习 (unlearning)

使用方式:
    from fedcl.methods.learners import get_learner

    # 获取learner类
    learner_cls = get_learner('fl.FedAvg')
    learner_cls = get_learner('cl.TARGET')
"""

import importlib
from pathlib import Path
from typing import Type, List, Dict, Any, Optional

# 导入核心组件
from ._registry import get_registry, get as get_learner_from_registry
from ._decorators import learner, get_learner_info

# 获取全局registry
registry = get_registry()


def auto_register_all_learners():
    """
    自动扫描并注册所有命名空间下的learner

    扫描 fl/, cl/, ul/ 目录下的所有.py文件并导入,
    触发@learner装饰器自动注册
    """
    base_path = Path(__file__).parent

    # 扫描所有命名空间目录
    for namespace_dir in ['fl', 'cl', 'ul']:
        namespace_path = base_path / namespace_dir

        if not namespace_path.exists():
            continue

        # 导入该命名空间下的所有模块
        for py_file in namespace_path.glob('*.py'):
            if py_file.name == '__init__.py':
                continue

            module_name = py_file.stem
            try:
                # 动态导入模块,触发@learner装饰器
                importlib.import_module(
                    f'fedcl.methods.learners.{namespace_dir}.{module_name}'
                )
            except Exception as e:
                print(f"Warning: Failed to import {namespace_dir}.{module_name}: {e}")


# 自动注册所有learner
auto_register_all_learners()


# 导出便捷函数
def get_learner(name: str) -> Type:
    """
    获取learner类

    Args:
        name: 完整名称 (如 "fl.FedAvg", "cl.TARGET")

    Returns:
        Learner类

    Example:
        >>> learner_cls = get_learner('fl.FedAvg')
        >>> learner = learner_cls(client_id='client_0', ...)
    """
    return get_learner_from_registry(name)


def list_all_learners() -> List[str]:
    """列出所有已注册的learner (带命名空间)"""
    return sorted(registry.get_all().keys())


def list_learners_by_namespace(namespace: str) -> List[str]:
    """
    列出命名空间下的所有learner

    Args:
        namespace: 命名空间 (fl/cl/ul)

    Returns:
        方法名列表 (不含命名空间前缀)
    """
    return registry.list_learners(namespace)


def get_namespace_info(namespace: str) -> Dict[str, Any]:
    """获取命名空间元数据"""
    return registry.get_namespace_info(namespace)


def search_learner(query: str, **kwargs) -> List[str]:
    """搜索learner"""
    return registry.search(query, **kwargs)


# 向后兼容导出 (如果有其他代码依赖这些)
from .contrastive import ContrastiveLearner
from .personalized import PersonalizedClientLearner
from .meta import MetaLearner
from .continual_base import ContinualLearner

__all__ = [
    # 核心函数
    'get_learner',
    'list_all_learners',
    'list_learners_by_namespace',
    'get_namespace_info',
    'search_learner',
    'registry',
    'learner',

    # 兼容导出
    'ContrastiveLearner',
    'PersonalizedClientLearner',
    'MetaLearner',
    'ContinualLearner',
]
