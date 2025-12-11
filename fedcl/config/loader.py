"""
配置加载器（简化版）
支持从 YAML 文件加载配置，处理继承和合并
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Union, Tuple

from .base import BaseConfig
from .communication import CommunicationConfig
from .training import TrainingConfig


class ConfigLoadError(Exception):
    """配置加载错误"""
    pass


class ConfigLoader:
    """
    配置加载器（简化版）

    功能：
    - 从 YAML 文件加载配置
    - 支持配置继承（extends 字段）
    - 支持深度合并配置
    - 将字典转换为配置对象
    """

    @staticmethod
    def load_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        加载 YAML 文件

        Args:
            file_path: YAML 文件路径

        Returns:
            配置字典

        Raises:
            ConfigLoadError: 文件不存在或格式错误
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise ConfigLoadError(f"Configuration file not found: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                if config is None:
                    config = {}
                return config
        except yaml.YAMLError as e:
            raise ConfigLoadError(f"YAML syntax error in {file_path}: {e}")
        except Exception as e:
            raise ConfigLoadError(f"Error loading {file_path}: {e}")

    @staticmethod
    def load_with_inheritance(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        加载配置并处理继承关系

        支持通过 extends 字段继承父配置：
        ```yaml
        extends: "base_config.yaml"

        # 覆盖或添加字段
        transport:
          port: 8001
        ```

        Args:
            file_path: 配置文件路径

        Returns:
            合并后的配置字典
        """
        file_path = Path(file_path)

        # 1. 加载当前配置
        config = ConfigLoader.load_yaml(file_path)

        # 2. 检查是否有 extends 字段
        if 'extends' not in config:
            return config

        # 3. 获取父配置路径（相对于当前文件）
        parent_path_str = config.pop('extends')

        # 如果是相对路径，相对于当前配置文件的目录
        if not os.path.isabs(parent_path_str):
            parent_path = file_path.parent / parent_path_str
        else:
            parent_path = Path(parent_path_str)

        # 4. 递归加载父配置
        parent_config = ConfigLoader.load_with_inheritance(parent_path)

        # 5. 深度合并配置（子配置覆盖父配置）
        merged_config = ConfigLoader.deep_merge(parent_config, config)

        return merged_config

    @staticmethod
    def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        深度合并两个字典

        规则：
        - 嵌套字典：递归合并
        - 列表：完全替换（不合并）
        - 其他类型：覆盖

        Args:
            base: 基础配置（父配置）
            override: 覆盖配置（子配置）

        Returns:
            合并后的配置
        """
        result = base.copy()

        for key, value in override.items():
            if key in result:
                # 如果两者都是字典，递归合并
                if isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = ConfigLoader.deep_merge(result[key], value)
                else:
                    # 其他情况：直接覆盖（包括列表）
                    result[key] = value
            else:
                # 新字段：直接添加
                result[key] = value

        return result

    @staticmethod
    def dict_to_config(data: Dict[str, Any], config_class: type) -> BaseConfig:
        """
        将字典转换为配置对象（简化版）

        Args:
            data: 配置字典
            config_class: 目标配置类（CommunicationConfig 或 TrainingConfig）

        Returns:
            配置对象实例
        """
        # 获取配置类的字段名
        if hasattr(config_class, '__dataclass_fields__'):
            known_fields = {
                f for f in config_class.__dataclass_fields__.keys()
                if f != 'extra_fields'
            }
        else:
            known_fields = set()

        # 分离标准字段和自定义字段
        kwargs = {}
        extra_fields = {}

        for key, value in data.items():
            if key in known_fields:
                kwargs[key] = value
            else:
                extra_fields[key] = value

        # 创建配置对象
        try:
            config_obj = config_class(**kwargs)
        except TypeError as e:
            raise ConfigLoadError(f"Error creating {config_class.__name__}: {e}")

        # 设置额外字段
        config_obj.extra_fields = extra_fields

        return config_obj

    # ========== 便捷加载方法 ==========

    @staticmethod
    def load_communication(file_path: Union[str, Path]) -> CommunicationConfig:
        """
        加载通信配置

        Args:
            file_path: 配置文件路径

        Returns:
            通信配置对象
        """
        config_dict = ConfigLoader.load_with_inheritance(file_path)
        return ConfigLoader.dict_to_config(config_dict, CommunicationConfig)

    @staticmethod
    def load_training(file_path: Union[str, Path]) -> TrainingConfig:
        """
        加载训练配置

        Args:
            file_path: 配置文件路径

        Returns:
            训练配置对象
        """
        config_dict = ConfigLoader.load_with_inheritance(file_path)
        return ConfigLoader.dict_to_config(config_dict, TrainingConfig)

    @staticmethod
    def load_combined(file_path: Union[str, Path]) -> Tuple[CommunicationConfig, TrainingConfig]:
        """
        从单个文件加载通信配置和训练配置

        配置文件格式：
        ```yaml
        communication:
          mode: "network"
          role: "server"
          ...

        training:
          trainer:
            name: "FedAvgTrainer"
          ...
        ```

        Args:
            file_path: 配置文件路径

        Returns:
            (通信配置, 训练配置) 元组
        """
        # 加载配置字典
        config_dict = ConfigLoader.load_with_inheritance(file_path)

        # 分离通信配置和训练配置
        # 注意：通信配置可能分布在两个地方：
        # 1. 顶层字段：mode, role, node_id, transport
        # 2. communication 部分：heartbeat, rpc等设置
        comm_dict = config_dict.get('communication', {}).copy()

        # 合并顶层的通信字段到 comm_dict
        top_level_comm_fields = {'mode', 'role', 'node_id', 'transport'}
        for field in top_level_comm_fields:
            if field in config_dict:
                comm_dict[field] = config_dict[field]

        train_dict = config_dict.get('training', {})

        # 将顶层的 logging 字段添加到 train_dict（如果存在）
        if 'logging' in config_dict:
            train_dict = train_dict.copy()  # 确保不修改原始字典
            train_dict['logging'] = config_dict['logging']

        # 创建配置对象
        comm_config = ConfigLoader.dict_to_config(comm_dict, CommunicationConfig)
        train_config = ConfigLoader.dict_to_config(train_dict, TrainingConfig)

        return comm_config, train_config

    @staticmethod
    def load(file_path: Union[str, Path]) -> Tuple[CommunicationConfig, TrainingConfig]:
        """
        智能加载配置（自动判断格式）

        支持两种格式：
        1. 分离格式：包含 communication 和 training 两个顶层键
        2. 扁平格式：直接在顶层包含所有字段

        Args:
            file_path: 配置文件路径

        Returns:
            (通信配置, 训练配置) 元组
            如果某个配置不存在，返回默认配置
        """
        config_dict = ConfigLoader.load_with_inheritance(file_path)

        # 判断配置格式
        has_communication_section = 'communication' in config_dict
        has_training_section = 'training' in config_dict

        # 先检查是否有扁平格式的通信/训练字段（在处理 section 格式之前）
        comm_fields = {'mode', 'role', 'node_id', 'transport'}  # 不包含 'communication'，避免误判
        train_fields = {'trainer', 'learner', 'aggregator', 'evaluator',
                       'global_model', 'local_model', 'dataset',
                       'max_rounds', 'min_clients', 'components', 'logging'}  # 添加 logging 字段
        has_comm_fields_toplevel = any(f in config_dict for f in comm_fields)
        has_train_fields_toplevel = any(f in config_dict for f in train_fields)

        # 方式1：分离格式（有独立的 communication 和 training 部分）
        if has_communication_section and has_training_section:
            return ConfigLoader.load_combined(file_path)
        elif has_communication_section:
            # 只有通信配置部分
            comm_config = ConfigLoader.dict_to_config(config_dict['communication'], CommunicationConfig)
            train_config = TrainingConfig()  # 默认训练配置
            return comm_config, train_config
        elif has_training_section and not has_comm_fields_toplevel:
            # 只有训练配置部分，且顶层没有通信字段
            comm_config = CommunicationConfig()  # 默认通信配置
            train_config = ConfigLoader.dict_to_config(config_dict['training'], TrainingConfig)
            return comm_config, train_config

        # 方式2：扁平格式（所有字段在顶层）或混合格式（training section + 顶层通信字段）
        has_comm_fields = has_comm_fields_toplevel or has_communication_section
        has_train_fields = has_train_fields_toplevel or has_training_section

        if has_comm_fields or has_train_fields:
            # 提取通信配置
            comm_dict = {}
            if has_comm_fields_toplevel:
                # 从顶层提取通信字段
                for key in comm_fields:
                    if key in config_dict:
                        comm_dict[key] = config_dict[key]

            # 提取训练配置
            train_dict = {}
            if has_training_section:
                # 如果有 training section，使用其内容
                train_dict = config_dict['training'].copy()
                # 如果顶层有 logging 字段，将其添加到 train_dict
                if 'logging' in config_dict:
                    train_dict['logging'] = config_dict['logging']
            elif has_train_fields_toplevel:
                # 从顶层提取训练字段
                for key in train_fields:
                    if key in config_dict:
                        train_dict[key] = config_dict[key]

            # 创建配置对象
            comm_config = ConfigLoader.dict_to_config(comm_dict, CommunicationConfig) if comm_dict else CommunicationConfig()
            train_config = ConfigLoader.dict_to_config(train_dict, TrainingConfig) if train_dict else TrainingConfig()
            return comm_config, train_config

        else:
            # 无法判断，返回默认配置
            comm_config = CommunicationConfig()
            train_config = TrainingConfig()
            return comm_config, train_config
