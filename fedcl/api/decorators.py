"""
MOE-FedCL 组件注册装饰器
fedcl/api/decorators.py

提供装饰器自动注册机制，支持学习器、训练器、聚合器、评估器等组件的注册。
"""

import functools
import inspect
from typing import Any, Callable, Optional, Type, Dict
from loguru import logger

from ..registry import registry


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
        from ..trainer.base_trainer import BaseTrainer
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


# 别名装饰器，提供更简洁的语法
register_learner = learner
register_trainer = trainer
register_aggregator = aggregator
register_evaluator = evaluator
