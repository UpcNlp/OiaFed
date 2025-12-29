"""
组件注册装饰器

提供更丰富的元数据注册方式
"""

from typing import Type, Optional, List, Dict, Any
from ..infra import get_module_logger

logger = get_module_logger(__name__)


def _log_registration(message: str, level: str = 'info'):
    """记录注册日志"""
    if level == 'warning':
        logger.warning(message)
    else:
        logger.debug(message)


def learner(
    name: Optional[str] = None,
    description: Optional[str] = None,
    version: str = "1.0",
    author: Optional[str] = None,
    **metadata
):
    """
    学习器注册装饰器

    使用方式:
        @learner('MyLearner', description='自定义学习器')
        class MyLearner(BaseLearner):
            pass

    Args:
        name: 学习器名称，如果为None则使用类名
        description: 学习器描述
        version: 版本号
        author: 作者
        **metadata: 其他元数据
    """
    def decorator(cls: Type) -> Type:
        from .registry import registry

        # 获取学习器名称
        learner_name = name or cls.__name__

        # 添加元数据到类
        cls._component_metadata = {
            'type': 'learner',
            'name': learner_name,
            'description': description or f"{learner_name} 学习器",
            'version': version,
            'author': author,
            'registered_at': str(id(cls)),
            **metadata
        }

        # 注册到全局注册表（使用新的命名空间格式）
        namespace = f"learner.{learner_name.lower()}"
        registry.register(namespace)(cls)

        # _log_registration(f"已注册学习器: {learner_name} (版本: {version}) -> {namespace}")

        return cls

    return decorator


def trainer(
    name: Optional[str] = None,
    description: Optional[str] = None,
    version: str = "1.0",
    author: Optional[str] = None,
    algorithms: Optional[List[str]] = None,
    **metadata
):
    """
    训练器注册装饰器

    使用方式:
        @trainer('FedAvg', description='联邦平均算法训练器', algorithms=['fedavg'])
        class FedAvgTrainer(BaseTrainer):
            pass

    Args:
        name: 训练器名称，如果为None则使用类名
        description: 训练器描述
        version: 版本号
        author: 作者
        algorithms: 支持的算法列表
        **metadata: 其他元数据
    """
    def decorator(cls: Type) -> Type:
        from .registry import registry

        # 获取训练器名称
        trainer_name = name or cls.__name__

        # 添加元数据到类
        cls._component_metadata = {
            'type': 'trainer',
            'name': trainer_name,
            'description': description or f"{trainer_name} 训练器",
            'version': version,
            'author': author,
            'algorithms': algorithms or [],
            'registered_at': str(id(cls)),
            **metadata
        }

        # 注册到全局注册表
        namespace = f"trainer.{trainer_name.lower()}"
        registry.register(namespace)(cls)

        # _log_registration(
        #     f"已注册训练器: {trainer_name} (版本: {version}, 算法: {algorithms or []}) -> {namespace}"
        # )

        return cls

    return decorator


def aggregator(
    name: Optional[str] = None,
    description: Optional[str] = None,
    version: str = "1.0",
    author: Optional[str] = None,
    weighted: bool = True,
    **metadata
):
    """
    聚合器注册装饰器

    使用方式:
        @aggregator('FedAvg', description='联邦平均聚合器', weighted=True)
        class FedAvgAggregator(BaseAggregator):
            pass

    Args:
        name: 聚合器名称，如果为None则使用类名
        description: 聚合器描述
        version: 版本号
        author: 作者
        weighted: 是否支持加权聚合
        **metadata: 其他元数据
    """
    def decorator(cls: Type) -> Type:
        from .registry import registry

        # 获取聚合器名称
        aggregator_name = name or cls.__name__

        # 添加元数据到类
        cls._component_metadata = {
            'type': 'aggregator',
            'name': aggregator_name,
            'description': description or f"{aggregator_name} 聚合器",
            'version': version,
            'author': author,
            'weighted': weighted,
            'registered_at': str(id(cls)),
            **metadata
        }

        # 注册到全局注册表
        namespace = f"aggregator.{aggregator_name.lower()}"
        registry.register(namespace)(cls)

        # _log_registration(
        #     f"已注册聚合器: {aggregator_name} (版本: {version}, 加权: {weighted}) -> {namespace}"
        # )

        return cls

    return decorator


def dataset(
    name: Optional[str] = None,
    description: Optional[str] = None,
    version: str = "1.0",
    author: Optional[str] = None,
    dataset_type: Optional[str] = None,
    num_classes: Optional[int] = None,
    **metadata
):
    """
    数据集注册装饰器

    使用方式:
        @dataset('MNIST', description='MNIST手写数字数据集',
                 dataset_type='image_classification', num_classes=10)
        class MNISTDataset(BaseDataset):
            pass

    Args:
        name: 数据集名称，如果为None则使用类名
        description: 数据集描述
        version: 版本号
        author: 作者
        dataset_type: 数据集类型（image_classification, text, etc.）
        num_classes: 类别数量
        **metadata: 其他元数据
    """
    def decorator(cls: Type) -> Type:
        from .registry import registry

        # 获取数据集名称
        dataset_name = name or cls.__name__

        # 添加元数据到类
        cls._component_metadata = {
            'type': 'dataset',
            'name': dataset_name,
            'description': description or f"{dataset_name} 数据集",
            'version': version,
            'author': author,
            'dataset_type': dataset_type,
            'num_classes': num_classes,
            'registered_at': str(id(cls)),
            **metadata
        }

        # 注册到全局注册表
        namespace = f"dataset.{dataset_name.lower()}"
        registry.register(namespace)(cls)

        # _log_registration(
        #     f"已注册数据集: {dataset_name} (版本: {version}, 类型: {dataset_type}, "
        #     f"类别数: {num_classes}) -> {namespace}"
        # )

        return cls

    return decorator


def model(
    name: Optional[str] = None,
    description: Optional[str] = None,
    version: str = "1.0",
    author: Optional[str] = None,
    model_type: Optional[str] = None,
    framework: str = "pytorch",
    **metadata
):
    """
    模型注册装饰器

    使用方式:
        @model('SimpleCNN', description='简单CNN模型',
               model_type='cnn', framework='pytorch')
        class SimpleCNN(nn.Module):
            pass

    Args:
        name: 模型名称，如果为None则使用类名
        description: 模型描述
        version: 版本号
        author: 作者
        model_type: 模型类型（cnn, rnn, transformer, etc.）
        framework: 框架（pytorch, tensorflow, etc.）
        **metadata: 其他元数据
    """
    def decorator(cls: Type) -> Type:
        from .registry import registry

        # 获取模型名称
        model_name = name or cls.__name__

        # 添加元数据到类
        cls._component_metadata = {
            'type': 'model',
            'name': model_name,
            'description': description or f"{model_name} 模型",
            'version': version,
            'author': author,
            'model_type': model_type,
            'framework': framework,
            'registered_at': str(id(cls)),
            **metadata
        }

        # 注册到全局注册表
        namespace = f"model.{model_name.lower()}"
        registry.register(namespace)(cls)

        # _log_registration(
        #     f"已注册模型: {model_name} (版本: {version}, 类型: {model_type}, "
        #     f"框架: {framework}) -> {namespace}"
        # )

        return cls

    return decorator


def callback(
    name: Optional[str] = None,
    description: Optional[str] = None,
    version: str = "1.0",
    author: Optional[str] = None,
    **metadata
):
    """
    回调注册装饰器

    使用方式:
        @callback('EarlyStopping', description='早停回调')
        class EarlyStoppingCallback(BaseCallback):
            pass

    Args:
        name: 回调名称，如果为None则使用类名
        description: 回调描述
        version: 版本号
        author: 作者
        **metadata: 其他元数据
    """
    def decorator(cls: Type) -> Type:
        from .registry import registry

        # 获取回调名称
        callback_name = name or cls.__name__

        # 添加元数据到类
        cls._component_metadata = {
            'type': 'callback',
            'name': callback_name,
            'description': description or f"{callback_name} 回调",
            'version': version,
            'author': author,
            'registered_at': str(id(cls)),
            **metadata
        }

        # 注册到全局注册表
        namespace = f"callback.{callback_name.lower()}"
        registry.register(namespace)(cls)

        # _log_registration(f"已注册回调: {callback_name} (版本: {version}) -> {namespace}")

        return cls

    return decorator


def get_component_metadata(cls: Type) -> Optional[Dict[str, Any]]:
    """
    获取组件元数据

    Args:
        cls: 组件类

    Returns:
        元数据字典，如果不存在则返回None
    """
    return getattr(cls, '_component_metadata', None)


def list_components_with_metadata(component_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    列出所有组件及其元数据

    Args:
        component_type: 组件类型过滤（learner, trainer, aggregator, etc.）

    Returns:
        组件元数据列表
    """
    from .registry import registry

    results = []
    prefix = f"{component_type}." if component_type else ""

    for namespace in registry.list(prefix):
        try:
            cls = registry.get(namespace)
            metadata = get_component_metadata(cls)
            if metadata:
                metadata['namespace'] = namespace
                results.append(metadata)
        except Exception as e:
            logger.warning(f"Failed to get metadata for {namespace}: {e}")

    return results
