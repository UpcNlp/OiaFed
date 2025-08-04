# fedcl/models/factory.py
"""
模型工厂模块

提供统一的模型创建接口，支持灵活的模型构建方式。
支持内置模型类型和用户自定义模型创建函数。
"""

from typing import Dict, Any, Optional, Callable, Type
import torch.nn as nn
from loguru import logger

from ..core.execution_context import ExecutionContext
from ..exceptions import ModelCreationError


class ModelFactory:
    """
    模型工厂类
    
    提供统一的模型创建接口，支持多种模型创建方式：
    - 注册的内置模型类型
    - 用户自定义工厂函数
    - 动态类导入
    """
    
    # 注册的模型类型
    _registered_models: Dict[str, Callable] = {}
    
    @classmethod
    def register_model(cls, model_type: str, creator: Callable[[Dict[str, Any], Optional[ExecutionContext]], nn.Module]) -> None:
        """
        注册模型类型
        
        Args:
            model_type: 模型类型名称
            creator: 模型创建函数
        """
        cls._registered_models[model_type] = creator
        logger.debug(f"Registered model type: {model_type}")
    
    @classmethod
    def create_model(cls, model_config: Dict[str, Any], context: Optional[ExecutionContext] = None) -> nn.Module:
        """
        根据配置创建模型
        
        Args:
            model_config: 模型配置字典
            context: 执行上下文（可选）
            
        Returns:
            创建的模型实例
            
        Raises:
            ModelCreationError: 模型创建失败时抛出
        """
        if 'type' not in model_config:
            raise ModelCreationError("Model config must contain 'type' field")
        
        model_type = model_config['type']
        
        # 检查注册的模型类型
        if model_type in cls._registered_models:
            creator = cls._registered_models[model_type]
            try:
                return creator(model_config, context)
            except Exception as e:
                raise ModelCreationError(f"Failed to create model {model_type}: {str(e)}") from e
        
        # 内置模型类型
        if model_type == 'sequential':
            return cls._create_sequential_model(model_config)
        elif model_type == 'custom':
            return cls._create_custom_model(model_config, context)
        else:
            raise ModelCreationError(f"Unknown model type: {model_type}")
    
    @classmethod
    def _create_sequential_model(cls, config: Dict[str, Any]) -> nn.Module:
        """
        创建Sequential模型
        
        Args:
            config: 模型配置
            
        Returns:
            Sequential模型实例
        """
        layers = []
        layer_configs = config.get('layers', [])
        
        for layer_config in layer_configs:
            layer_type = layer_config.get('type')
            params = layer_config.get('params', {})
            
            if layer_type == 'linear':
                layers.append(nn.Linear(**params))
            elif layer_type == 'conv2d':
                layers.append(nn.Conv2d(**params))
            elif layer_type == 'relu':
                layers.append(nn.ReLU())
            elif layer_type == 'maxpool2d':
                layers.append(nn.MaxPool2d(**params))
            elif layer_type == 'flatten':
                layers.append(nn.Flatten())
            elif layer_type == 'dropout':
                layers.append(nn.Dropout(**params))
            else:
                logger.warning(f"Unknown layer type: {layer_type}")
        
        return nn.Sequential(*layers)
    
    @classmethod
    def _create_custom_model(cls, config: Dict[str, Any], context: Optional[ExecutionContext]) -> nn.Module:
        """
        创建自定义模型
        
        Args:
            config: 模型配置
            context: 执行上下文
            
        Returns:
            自定义模型实例
        """
        class_name = config.get('class_name')
        module_name = config.get('module', 'models')
        model_params = config.get('params', {})
        
        if not class_name:
            raise ModelCreationError("Custom model must specify 'class_name'")
        
        try:
            # 动态导入模型类
            module = __import__(module_name, fromlist=[class_name])
            model_class = getattr(module, class_name)
            return model_class(**model_params)
        except Exception as e:
            raise ModelCreationError(f"Failed to create custom model {class_name}: {str(e)}") from e
    
    @classmethod
    def list_registered_models(cls) -> list:
        """
        列出所有注册的模型类型
        
        Returns:
            注册的模型类型列表
        """
        return list(cls._registered_models.keys())


# 注册一些内置模型
def _create_mnist_cnn(config: Dict[str, Any], context: Optional[ExecutionContext]) -> nn.Module:
    """创建MNIST CNN模型"""
    num_classes = config.get('num_classes', 10)
    hidden_size = config.get('hidden_size', 128)
    
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, num_classes)
    )


# 注册内置模型
ModelFactory.register_model('mnist_cnn', _create_mnist_cnn)