"""
配置加载器（支持继承和复用）
experiments/utils/config_loader.py

功能：
- 加载YAML配置文件
- 支持配置继承（includes字段）
- 深度合并配置
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List


def load_config_with_inheritance(config_path: str) -> Dict[str, Any]:
    """加载配置文件，支持 includes 和 extends 继承

    Args:
        config_path: 配置文件路径

    Returns:
        合并后的配置字典

    配置文件示例（方式1 - includes）:
        ```yaml
        # configs/experiments/fedavg_mnist.yaml
        experiment_name: "fedavg_mnist_iid"

        # 复用配置组件
        includes:
          - "../ingredients/datasets/mnist.yaml"
          - "../ingredients/trainers/fedavg.yaml"
          - "../ingredients/communication/memory.yaml"

        # 覆盖特定参数
        overrides:
          trainer:
            max_rounds: 20
          dataset:
            partition:
              num_clients: 5
        ```

    配置文件示例（方式2 - extends，兼容现有配置）:
        ```yaml
        # configs/distributed/experiments/iid/server.yaml
        extends: "../../base/server_base.yaml"

        node_id: "server_iid"
        ```
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # 处理 extends（单文件继承，兼容现有配置）
    if 'extends' in config:
        base_dir = config_path.parent
        extends_path = config['extends']
        extends_full_path = base_dir / extends_path

        if not extends_full_path.exists():
            raise FileNotFoundError(f"Extended file not found: {extends_full_path}")

        # 递归加载基础配置（支持链式继承）
        base_config = load_config_with_inheritance(str(extends_full_path))

        # 移除 extends 字段
        current_config = {k: v for k, v in config.items() if k != 'extends'}

        # 基础配置 + 当前配置覆盖
        merged_config = base_config.copy()
        deep_merge(merged_config, current_config)

        return merged_config

    # 处理 includes（多文件组合）
    if 'includes' in config:
        base_dir = config_path.parent
        merged_config = {}

        for include_path in config['includes']:
            include_full_path = base_dir / include_path

            if not include_full_path.exists():
                raise FileNotFoundError(f"Include file not found: {include_full_path}")

            with open(include_full_path) as f:
                include_config = yaml.safe_load(f)

            # 深度合并
            deep_merge(merged_config, include_config)

        # 应用当前配置的覆盖
        if 'overrides' in config:
            deep_merge(merged_config, config['overrides'])

        # 保留非继承相关的顶层字段
        for key in config:
            if key not in ['includes', 'overrides']:
                merged_config[key] = config[key]

        config = merged_config

    return config


def deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> None:
    """深度合并字典（修改base）

    Args:
        base: 基础字典（会被修改）
        update: 更新字典

    Examples:
        >>> base = {'a': {'b': 1, 'c': 2}}
        >>> update = {'a': {'c': 3, 'd': 4}}
        >>> deep_merge(base, update)
        >>> print(base)
        {'a': {'b': 1, 'c': 3, 'd': 4}}
    """
    for key, value in update.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            # 递归合并字典
            deep_merge(base[key], value)
        else:
            # 覆盖或新增
            base[key] = value


def load_multiple_configs(config_paths: List[str]) -> List[Dict[str, Any]]:
    """加载多个配置文件

    Args:
        config_paths: 配置文件路径列表

    Returns:
        配置字典列表
    """
    configs = []
    for path in config_paths:
        config = load_config_with_inheritance(path)
        configs.append(config)

    return configs


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """保存配置到YAML文件

    Args:
        config: 配置字典
        output_path: 输出文件路径
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
