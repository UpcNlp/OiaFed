"""
MOE-FedCL 组件构建器
fedcl/api/builder.py

提供Builder模式支持，从配置创建组件实例
"""

import os
import yaml
from typing import Any, Dict, Optional, Union, Type
from pathlib import Path

from .registry import registry
from loguru import logger


class ComponentBuilder:
    """组件构建器 - 解析配置，返回类引用和参数

    统一初始化策略：
    1. Builder只解析配置，不创建实例
    2. 返回包含'class'和'params'的配置字典
    3. 实际创建工作由Trainer/Learner内部完成

    支持三种创建方式：
    1. 从注册表名称：build_dataset("MNIST", ...)
    2. 从类：build_dataset(MNISTDataset, ...)
    3. 从配置字典：parse_config(config_dict)
    """

    def __init__(self, component_registry=None):
        """初始化构建器

        Args:
            component_registry: 组件注册表，默认使用全局注册表
        """
        self.registry = component_registry or registry

    def parse_config(self, config: Union[Dict, str, Path]) -> Dict[str, Any]:
        """
        解析配置文件/字典，返回包含类引用和参数的配置

        Args:
            config: 配置字典或YAML文件路径

        Returns:
            Dict[str, Any]: 解析后的配置，格式为：
                {
                    "trainer": {
                        "class": FedAvgTrainer,
                        "params": {...},
                        "lazy_init": True
                    },
                    "aggregator": {
                        "class": FedAvgAggregator,
                        "params": {...}
                    },
                    ...
                }

        Examples:
            >>> builder = ComponentBuilder()
            >>> # 从配置文件
            >>> config = builder.parse_config("configs/server.yaml")
            >>> # 从字典
            >>> config = builder.parse_config({
            ...     "trainer": {"name": "FedAvgTrainer", "params": {...}}
            ... })
        """
        # 如果是文件路径，先加载
        if isinstance(config, (str, Path)):
            config = self.load_config(config)

        parsed = {}

        # 定义支持的组件类型
        component_types = [
            'trainer', 'aggregator', 'global_model', 'evaluator',
            'learner', 'dataset', 'local_model'
        ]

        # 解析各个组件配置
        for component_type in component_types:
            # 先检查training下的配置（新格式）
            training_config = config.get('training', {})
            component_config = training_config.get(component_type)

            # 如果training下没有，检查顶层（兼容旧格式）
            if component_config is None:
                component_config = config.get(component_type)

            if component_config and isinstance(component_config, dict):
                # 提取name和params
                component_name = component_config.get('name')
                if component_name:
                    # 从注册表获取类
                    component_class = self._get_class_from_registry(
                        component_type,
                        component_name
                    )

                    parsed[component_type] = {
                        "class": component_class,
                        "params": component_config.get('params', {}),
                        "lazy_init": component_config.get('lazy_init', True)
                    }

        logger.debug(f"Parsed config with components: {list(parsed.keys())}")
        return parsed

    def _get_class_from_registry(self, component_type: str, name: str):
        """从注册表获取类引用

        Args:
            component_type: 组件类型
            name: 组件名称

        Returns:
            组件类

        Raises:
            ValueError: 如果组件未注册
        """
        registry_map = {
            'trainer': self.registry.get_trainer,
            'aggregator': self.registry.get_aggregator,
            'learner': self.registry.get_learner,
            'dataset': self.registry.get_dataset,
            'global_model': self.registry.get_model,
            'local_model': self.registry.get_model,
            'evaluator': self.registry.get_evaluator
        }

        getter = registry_map.get(component_type)
        if getter:
            return getter(name)
        else:
            raise ValueError(f"未知的组件类型: {component_type}")

    # ===== 向后兼容：保留旧的build_*方法，但现在返回实例 =====

    def build_dataset(self, name_or_class: Union[str, Type], **kwargs) -> Any:
        """构建数据集

        Args:
            name_or_class: 数据集名称（字符串）或类
            **kwargs: 初始化参数

        Returns:
            数据集实例

        Examples:
            >>> builder = ComponentBuilder()
            >>> # 方式1: 从注册表名称创建
            >>> dataset = builder.build_dataset("MNIST", root="./data", train=True)
            >>> # 方式2: 从类创建
            >>> dataset = builder.build_dataset(MNISTDataset, root="./data", train=True)
        """
        if isinstance(name_or_class, str):
            # 从注册表获取
            dataset_cls = self.registry.get_dataset(name_or_class)
            logger.debug(f"从注册表创建数据集: {name_or_class}")
        else:
            # 直接使用类
            dataset_cls = name_or_class
            logger.debug(f"从类创建数据集: {dataset_cls.__name__}")

        return dataset_cls(**kwargs)

    def build_model(self, name_or_class: Union[str, Type], **kwargs) -> Any:
        """构建模型

        Args:
            name_or_class: 模型名称（字符串）或类
            **kwargs: 初始化参数

        Returns:
            模型实例

        Examples:
            >>> builder = ComponentBuilder()
            >>> model = builder.build_model("SimpleCNN", num_classes=10)
            >>> model = builder.build_model(SimpleCNN, num_classes=10)
        """
        if isinstance(name_or_class, str):
            model_cls = self.registry.get_model(name_or_class)
            logger.debug(f"从注册表创建模型: {name_or_class}")
        else:
            model_cls = name_or_class
            logger.debug(f"从类创建模型: {model_cls.__name__}")

        return model_cls(**kwargs)

    def build_aggregator(self, name_or_class: Union[str, Type], **kwargs) -> Any:
        """构建聚合器

        Args:
            name_or_class: 聚合器名称（字符串）或类
            **kwargs: 初始化参数

        Returns:
            聚合器实例

        Examples:
            >>> builder = ComponentBuilder()
            >>> aggregator = builder.build_aggregator("FedAvg", weighted=True)
        """
        if isinstance(name_or_class, str):
            aggregator_cls = self.registry.get_aggregator(name_or_class)
            logger.debug(f"从注册表创建聚合器: {name_or_class}")
        else:
            aggregator_cls = name_or_class
            logger.debug(f"从类创建聚合器: {aggregator_cls.__name__}")

        return aggregator_cls(**kwargs)

    def build_trainer(self, name_or_class: Union[str, Type], **kwargs) -> Any:
        """构建训练器

        Args:
            name_or_class: 训练器名称（字符串）或类
            **kwargs: 初始化参数

        Returns:
            训练器实例
        """
        if isinstance(name_or_class, str):
            trainer_cls = self.registry.get_trainer(name_or_class)
            logger.debug(f"从注册表创建训练器: {name_or_class}")
        else:
            trainer_cls = name_or_class
            logger.debug(f"从类创建训练器: {trainer_cls.__name__}")

        return trainer_cls(**kwargs)

    def build_learner(self, name_or_class: Union[str, Type], **kwargs) -> Any:
        """构建学习器

        Args:
            name_or_class: 学习器名称（字符串）或类
            **kwargs: 初始化参数

        Returns:
            学习器实例
        """
        if isinstance(name_or_class, str):
            learner_cls = self.registry.get_learner(name_or_class)
            logger.debug(f"从注册表创建学习器: {name_or_class}")
        else:
            learner_cls = name_or_class
            logger.debug(f"从类创建学习器: {learner_cls.__name__}")

        return learner_cls(**kwargs)

    def build_evaluator(self, name_or_class: Union[str, Type], **kwargs) -> Any:
        """构建评估器

        Args:
            name_or_class: 评估器名称（字符串）或类
            **kwargs: 初始化参数

        Returns:
            评估器实例
        """
        if isinstance(name_or_class, str):
            evaluator_cls = self.registry.get_evaluator(name_or_class)
            logger.debug(f"从注册表创建评估器: {name_or_class}")
        else:
            evaluator_cls = name_or_class
            logger.debug(f"从类创建评估器: {evaluator_cls.__name__}")

        return evaluator_cls(**kwargs)

    def build_component(self, component_type: str, name_or_class: Union[str, Type], **kwargs) -> Any:
        """通用组件构建方法

        Args:
            component_type: 组件类型 ('dataset', 'model', 'aggregator', 'trainer', 'learner', 'evaluator')
            name_or_class: 组件名称或类
            **kwargs: 初始化参数

        Returns:
            组件实例

        Examples:
            >>> builder = ComponentBuilder()
            >>> dataset = builder.build_component('dataset', 'MNIST', root='./data')
        """
        builder_map = {
            'dataset': self.build_dataset,
            'model': self.build_model,
            'aggregator': self.build_aggregator,
            'trainer': self.build_trainer,
            'learner': self.build_learner,
            'evaluator': self.build_evaluator
        }

        if component_type not in builder_map:
            raise ValueError(
                f"不支持的组件类型: {component_type}. "
                f"支持的类型: {list(builder_map.keys())}"
            )

        return builder_map[component_type](name_or_class, **kwargs)

    def build_from_config(self, config: Union[Dict[str, Any], str, Path]) -> Dict[str, Any]:
        """从配置创建所有组件（向后兼容方法）

        注意：新代码建议使用parse_config()配合Trainer/Learner的统一初始化

        Args:
            config: 配置字典或YAML文件路径

        Returns:
            Dict[str, Any]: 包含所有创建的组件的字典（为向后兼容）

        Examples:
            >>> builder = ComponentBuilder()
            >>> # 旧方式（仍然支持）
            >>> components = builder.build_from_config("configs/experiment.yaml")
            >>> aggregator = components["aggregator"]
            >>>
            >>> # 新方式（推荐）
            >>> parsed = builder.parse_config("configs/experiment.yaml")
            >>> trainer = TrainerClass(config=parsed)
        """
        # 解析配置
        parsed_config = self.parse_config(config)

        # 为向后兼容，实际创建组件实例
        components = {}

        for component_type, component_config in parsed_config.items():
            if 'class' in component_config:
                component_class = component_config['class']
                component_params = component_config.get('params', {})

                logger.info(f"✅ 创建{component_type}: {component_class.__name__}")
                components[component_type] = component_class(**component_params)

        return components

    def load_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """加载配置文件

        Args:
            config_path: 配置文件路径（支持YAML）

        Returns:
            Dict[str, Any]: 配置字典

        Raises:
            FileNotFoundError: 配置文件不存在
            ValueError: 不支持的配置文件格式
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        # 根据文件扩展名选择加载方式
        if config_path.suffix in ['.yaml', '.yml']:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        elif config_path.suffix == '.json':
            import json
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            raise ValueError(
                f"不支持的配置文件格式: {config_path.suffix}. "
                f"支持的格式: .yaml, .yml, .json"
            )

        logger.info(f"📄 加载配置文件: {config_path}")
        return config

    def save_config(self, config: Dict[str, Any], save_path: Union[str, Path]) -> None:
        """保存配置到文件

        Args:
            config: 配置字典
            save_path: 保存路径

        Examples:
            >>> builder = ComponentBuilder()
            >>> config = {
            ...     "dataset": {"name": "MNIST", "params": {"root": "./data"}},
            ...     "model": {"name": "SimpleCNN", "params": {"num_classes": 10}}
            ... }
            >>> builder.save_config(config, "configs/my_experiment.yaml")
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if save_path.suffix in ['.yaml', '.yml']:
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        elif save_path.suffix == '.json':
            import json
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"不支持的文件格式: {save_path.suffix}")

        logger.info(f"💾 保存配置到: {save_path}")

    def list_available_components(self, component_type: Optional[str] = None) -> Dict[str, Any]:
        """列出可用的组件

        Args:
            component_type: 组件类型，如果为None则列出所有类型

        Returns:
            Dict[str, Any]: 可用组件字典

        Examples:
            >>> builder = ComponentBuilder()
            >>> # 列出所有组件
            >>> all_components = builder.list_available_components()
            >>> # 列出特定类型
            >>> datasets = builder.list_available_components('dataset')
        """
        if component_type is None:
            return self.registry.list_all_components()
        else:
            all_components = self.registry.list_all_components()
            component_key = f"{component_type}s"  # dataset -> datasets
            if component_key in all_components:
                return {component_key: all_components[component_key]}
            else:
                return {}

    def get_component_info(self, component_type: str, name: str) -> Optional[Dict[str, Any]]:
        """获取组件元数据信息

        Args:
            component_type: 组件类型
            name: 组件名称

        Returns:
            Optional[Dict[str, Any]]: 组件元数据，如果不存在则返回None

        Examples:
            >>> builder = ComponentBuilder()
            >>> info = builder.get_component_info('dataset', 'MNIST')
            >>> print(info['description'])
        """
        component_cls = self.registry.get(name, component_type)
        if component_cls is None:
            return None

        # 获取元数据
        if hasattr(component_cls, '_component_metadata'):
            return component_cls._component_metadata
        else:
            return {
                'name': name,
                'type': component_type,
                'class': component_cls.__name__
            }


# 全局构建器实例
_global_builder = None


def get_builder() -> ComponentBuilder:
    """获取全局构建器实例

    Returns:
        ComponentBuilder: 全局构建器

    Examples:
        >>> from fedcl.api.builder import get_builder
        >>> builder = get_builder()
        >>> dataset = builder.build_dataset("MNIST", root="./data")
    """
    global _global_builder
    if _global_builder is None:
        _global_builder = ComponentBuilder()
    return _global_builder


# 便捷函数
def build_dataset(name_or_class: Union[str, Type], **kwargs) -> Any:
    """便捷函数：构建数据集"""
    return get_builder().build_dataset(name_or_class, **kwargs)


def build_model(name_or_class: Union[str, Type], **kwargs) -> Any:
    """便捷函数：构建模型"""
    return get_builder().build_model(name_or_class, **kwargs)


def build_aggregator(name_or_class: Union[str, Type], **kwargs) -> Any:
    """便捷函数：构建聚合器"""
    return get_builder().build_aggregator(name_or_class, **kwargs)


def build_trainer(name_or_class: Union[str, Type], **kwargs) -> Any:
    """便捷函数：构建训练器"""
    return get_builder().build_trainer(name_or_class, **kwargs)


def build_learner(name_or_class: Union[str, Type], **kwargs) -> Any:
    """便捷函数：构建学习器"""
    return get_builder().build_learner(name_or_class, **kwargs)


def build_from_config(config: Union[Dict[str, Any], str, Path]) -> Dict[str, Any]:
    """便捷函数：从配置创建组件"""
    return get_builder().build_from_config(config)


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """便捷函数：加载配置文件"""
    return get_builder().load_config(config_path)


# 导出
__all__ = [
    'ComponentBuilder',
    'get_builder',
    'build_dataset',
    'build_model',
    'build_aggregator',
    'build_trainer',
    'build_learner',
    'build_from_config',
    'load_config',
]
