"""
MOE-FedCL 组件注册装饰器
fedcl/api/decorators.py

提供装饰器自动注册机制，支持学习器、训练器、聚合器、评估器等组件的注册。
"""

import functools
import inspect
from typing import Any, Callable, Optional, Type, Dict
from loguru import logger

from .registry import registry


def learner(name: Optional[str] = None, 
           description: Optional[str] = None,
           version: str = "1.0",
           author: Optional[str] = None,
           **metadata):
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
        # 获取学习器名称
        learner_name = name or cls.__name__
        
        # 验证类是否继承了正确的基类
        from ..learner.base_learner import BaseLearner
        if not issubclass(cls, BaseLearner):
            logger.warning(f"学习器 {learner_name} 未继承 BaseLearner，可能导致兼容性问题")
        
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
        
        # 注册到全局注册表
        registry.register_learner(learner_name, cls)
        
        logger.info(f"已注册学习器: {learner_name} (版本: {version})")
        
        return cls
    
    return decorator


def trainer(name: Optional[str] = None,
           description: Optional[str] = None, 
           version: str = "1.0",
           author: Optional[str] = None,
           algorithms: Optional[list] = None,
           **metadata):
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
        # 获取训练器名称
        trainer_name = name or cls.__name__
        
        # 验证类是否继承了正确的基类
        from ..trainer.trainer import BaseTrainer
        if not issubclass(cls, BaseTrainer):
            logger.warning(f"训练器 {trainer_name} 未继承 BaseTrainer，可能导致兼容性问题")
        
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
        registry.register_trainer(trainer_name, cls)
        
        logger.info(f"已注册训练器: {trainer_name} (版本: {version}, 算法: {algorithms or []})")
        
        return cls
    
    return decorator


def aggregator(name: Optional[str] = None,
              description: Optional[str] = None,
              version: str = "1.0", 
              author: Optional[str] = None,
              algorithm: Optional[str] = None,
              **metadata):
    """
    聚合器注册装饰器
    
    使用方式:
    @aggregator('FedAvg', description='联邦平均聚合器', algorithm='fedavg')
    class FedAvgAggregator(BaseAggregator):
        pass
    
    Args:
        name: 聚合器名称，如果为None则使用类名
        description: 聚合器描述
        version: 版本号
        author: 作者
        algorithm: 聚合算法名称
        **metadata: 其他元数据
    """
    def decorator(cls: Type) -> Type:
        # 获取聚合器名称
        aggregator_name = name or cls.__name__
        
        # 添加元数据到类
        cls._component_metadata = {
            'type': 'aggregator',
            'name': aggregator_name,
            'description': description or f"{aggregator_name} 聚合器",
            'version': version,
            'author': author,
            'algorithm': algorithm,
            'registered_at': str(id(cls)),
            **metadata
        }
        
        # 注册到全局注册表
        registry.register_aggregator(aggregator_name, cls)
        
        logger.info(f"已注册聚合器: {aggregator_name} (版本: {version}, 算法: {algorithm})")
        
        return cls
    
    return decorator


def evaluator(name: Optional[str] = None,
             description: Optional[str] = None,
             version: str = "1.0",
             author: Optional[str] = None,
             metrics: Optional[list] = None,
             **metadata):
    """
    评估器注册装饰器
    
    使用方式:
    @evaluator('Accuracy', description='准确率评估器', metrics=['accuracy', 'loss'])
    class AccuracyEvaluator(BaseEvaluator):
        pass
    
    Args:
        name: 评估器名称，如果为None则使用类名
        description: 评估器描述
        version: 版本号
        author: 作者
        metrics: 支持的指标列表
        **metadata: 其他元数据
    """
    def decorator(cls: Type) -> Type:
        # 获取评估器名称
        evaluator_name = name or cls.__name__
        
        # 添加元数据到类
        cls._component_metadata = {
            'type': 'evaluator',
            'name': evaluator_name,
            'description': description or f"{evaluator_name} 评估器",
            'version': version,
            'author': author,
            'metrics': metrics or [],
            'registered_at': str(id(cls)),
            **metadata
        }
        
        # 注册到全局注册表
        registry.register_evaluator(evaluator_name, cls)
        
        logger.info(f"已注册评估器: {evaluator_name} (版本: {version}, 指标: {metrics or []})")
        
        return cls
    
    return decorator


def component(component_type: str, 
             name: Optional[str] = None,
             description: Optional[str] = None,
             version: str = "1.0",
             author: Optional[str] = None,
             **metadata):
    """
    通用组件注册装饰器
    
    支持自定义组件类型的注册
    
    Args:
        component_type: 组件类型 ('learner', 'trainer', 'aggregator', 'evaluator')
        name: 组件名称
        description: 组件描述
        version: 版本号
        author: 作者
        **metadata: 其他元数据
    """
    # 映射到具体的装饰器
    decorator_map = {
        'learner': learner,
        'trainer': trainer,
        'aggregator': aggregator,
        'evaluator': evaluator
    }
    
    if component_type not in decorator_map:
        raise ValueError(f"不支持的组件类型: {component_type}. 支持的类型: {list(decorator_map.keys())}")
    
    # 调用对应的装饰器
    return decorator_map[component_type](
        name=name,
        description=description,
        version=version,
        author=author,
        **metadata
    )


def dataset(name: Optional[str] = None,
           description: Optional[str] = None,
           version: str = "1.0",
           author: Optional[str] = None,
           dataset_type: Optional[str] = None,
           num_classes: Optional[int] = None,
           **metadata):
    """
    数据集注册装饰器

    使用方式:
    @dataset('mnist_federated', description='MNIST联邦数据集', num_classes=10)
    class MNISTFederated(FederatedDataset):
        pass

    注意：装饰后，registry.get_dataset(name) 返回的是透明工厂类，
    实例化后直接得到 PyTorch Dataset，无需调用 get_pytorch_dataset()

    Args:
        name: 数据集名称，如果为None则使用类名
        description: 数据集描述
        version: 版本号
        author: 作者
        dataset_type: 数据集类型 ('image_classification', 'text', 'tabular')
        num_classes: 类别数量
        **metadata: 其他元数据
    """
    def decorator(cls: Type) -> Type:
        # 获取数据集名称
        dataset_name = name or cls.__name__

        # 验证类是否继承了正确的基类
        try:
            from ..methods.datasets.base import FederatedDataset
            if not issubclass(cls, FederatedDataset):
                logger.warning(f"数据集 {dataset_name} 未继承 FederatedDataset，可能导致兼容性问题")
        except ImportError:
            logger.warning("无法导入 FederatedDataset 进行类型检查")

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

        # 创建透明工厂类
        # 该工厂类在实例化时返回 PyTorch Dataset，而非 FederatedDataset
        class DatasetFactory:
            """
            透明数据集工厂类

            用途：使 Trainer/Learner 无需感知 FederatedDataset 包装器
            实例化时自动返回底层 PyTorch Dataset
            """

            # 保留原始 FederatedDataset 类的引用（用于需要联邦功能的场景）
            _federated_class = cls
            _component_metadata = cls._component_metadata

            def __new__(cls_factory, *args, **kwargs):
                """
                重写 __new__ 实现透明数据集访问

                当实例化时：
                1. 创建 FederatedDataset 实例
                2. 调用 get_pytorch_dataset() 获取底层数据集
                3. 直接返回 PyTorch Dataset（上层无感知）
                """
                # 实例化 FederatedDataset 包装器
                federated_dataset = cls_factory._federated_class(*args, **kwargs)

                # 获取底层 PyTorch Dataset
                pytorch_dataset = federated_dataset.get_pytorch_dataset()

                # 将联邦功能方法附加到 PyTorch Dataset（可选，用于需要时）
                # 这样上层如果需要联邦划分功能，仍可以通过 dataset.partition() 等访问
                pytorch_dataset._federated_dataset = federated_dataset
                pytorch_dataset.partition = federated_dataset.partition
                pytorch_dataset.get_client_partition = federated_dataset.get_client_partition
                pytorch_dataset.get_statistics = federated_dataset.get_statistics
                pytorch_dataset.get_class_distribution = federated_dataset.get_class_distribution

                return pytorch_dataset

        # 注册工厂类到全局注册表（而非原始类）
        registry.register_dataset(dataset_name, DatasetFactory)

        logger.info(f"已注册数据集: {dataset_name} (版本: {version}, 类型: {dataset_type}, 透明访问模式)")

        return cls

    return decorator


def model(name: Optional[str] = None,
         description: Optional[str] = None,
         version: str = "1.0",
         author: Optional[str] = None,
         task: Optional[str] = None,
         input_shape: Optional[tuple] = None,
         output_shape: Optional[tuple] = None,
         **metadata):
    """
    模型注册装饰器

    使用方式:
    @model('simple_cnn', description='简单CNN模型', task='classification')
    class SimpleCNN(nn.Module):
        pass

    Args:
        name: 模型名称，如果为None则使用类名
        description: 模型描述
        version: 版本号
        author: 作者
        task: 任务类型 ('classification', 'regression', 'generation')
        input_shape: 输入形状
        output_shape: 输出形状
        **metadata: 其他元数据
    """
    def decorator(cls: Type) -> Type:
        # 获取模型名称
        model_name = name or cls.__name__

        # 验证类是否继承了 nn.Module
        try:
            import torch.nn as nn
            if not issubclass(cls, nn.Module):
                logger.warning(f"模型 {model_name} 未继承 torch.nn.Module，可能导致兼容性问题")
        except ImportError:
            logger.warning("无法导入 torch.nn.Module 进行类型检查")

        # 添加元数据到类
        cls._component_metadata = {
            'type': 'model',
            'name': model_name,
            'description': description or f"{model_name} 模型",
            'version': version,
            'author': author,
            'task': task,
            'input_shape': input_shape,
            'output_shape': output_shape,
            'registered_at': str(id(cls)),
            **metadata
        }

        # 注册到全局注册表
        registry.register_model(model_name, cls)

        logger.info(f"已注册模型: {model_name} (版本: {version}, 任务: {task})")

        return cls

    return decorator


# 别名装饰器，提供更简洁的语法
register_learner = learner
register_trainer = trainer
register_aggregator = aggregator
register_evaluator = evaluator
register_dataset = dataset
register_model = model
