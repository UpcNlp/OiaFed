"""
实验ID生成器
experiments/utils/experiment_id.py

功能：
- 生成可读性强的实验ID
- 格式：{algorithm}_{dataset}_{partition}_{timestamp}
"""

from datetime import datetime
from typing import Dict, Any, Optional


def generate_experiment_id(config: Dict[str, Any], timestamp: str = None) -> str:
    """
    生成可读性强的实验ID

    Args:
        config: 配置字典
        timestamp: 时间戳字符串（可选，默认使用当前时间）

    Returns:
        实验ID字符串

    Examples:
        >>> config = {
        ...     'trainer': {'name': 'FedAvgMNIST'},
        ...     'dataset': {'name': 'MNIST', 'partition': {'strategy': 'iid'}}
        ... }
        >>> exp_id = generate_experiment_id(config)
        >>> print(exp_id)
        fedavgmnist_mnist_iid_20250111_143020
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 提取关键信息
    trainer_name = _extract_trainer_name(config)
    dataset_name = _extract_dataset_name(config)
    partition_strategy = _extract_partition_strategy(config)

    # 构建实验ID
    exp_id = f"{trainer_name}_{dataset_name}_{partition_strategy}_{timestamp}"

    return exp_id.lower()


def generate_experiment_id_with_params(config: Dict[str, Any],
                                       include_params: bool = True,
                                       timestamp: str = None) -> str:
    """
    生成包含关键参数的实验ID

    Args:
        config: 配置字典
        include_params: 是否包含关键参数
        timestamp: 时间戳字符串

    Returns:
        实验ID字符串

    Examples:
        fedavg_mnist_iid_r10_lr0.01_20250111_143020
    """
    base_id = generate_experiment_id(config, timestamp)

    if not include_params:
        return base_id

    # 提取关键参数
    params_str = _extract_key_params(config)

    if params_str:
        # 插入参数到时间戳之前
        parts = base_id.rsplit('_', 2)  # 分离最后两个部分（日期_时间）
        if len(parts) == 3:
            return f"{parts[0]}_{params_str}_{parts[1]}_{parts[2]}"

    return base_id


def _extract_trainer_name(config: Dict[str, Any]) -> str:
    """提取训练器名称"""
    trainer_config = config.get('training', {}).get('trainer', {}) or config.get('trainer', {})

    if isinstance(trainer_config, dict):
        name = trainer_config.get('name', 'unknown')
    else:
        name = 'unknown'

    # 移除常见后缀
    name = name.replace('Trainer', '').replace('MNIST', '').replace('CIFAR', '')
    return name.strip() or 'unknown'


def _extract_dataset_name(config: Dict[str, Any]) -> str:
    """提取数据集名称"""
    dataset_config = config.get('training', {}).get('dataset', {}) or config.get('dataset', {})

    if isinstance(dataset_config, dict):
        name = dataset_config.get('name', 'unknown')
    else:
        name = 'unknown'

    return name.strip() or 'unknown'


def _extract_partition_strategy(config: Dict[str, Any]) -> str:
    """提取数据分区策略"""
    dataset_config = config.get('training', {}).get('dataset', {}) or config.get('dataset', {})

    if isinstance(dataset_config, dict):
        partition_config = dataset_config.get('partition', {})
        if isinstance(partition_config, dict):
            strategy = partition_config.get('strategy', 'unknown')
            return strategy.strip() or 'unknown'

    return 'unknown'


def _extract_key_params(config: Dict[str, Any]) -> str:
    """提取关键参数"""
    params_list = []

    # 提取训练轮数
    trainer_config = config.get('training', {}).get('trainer', {}) or config.get('trainer', {})
    if isinstance(trainer_config, dict):
        params = trainer_config.get('params', {})
        if isinstance(params, dict):
            # 最大轮数
            max_rounds = params.get('max_rounds')
            if max_rounds:
                params_list.append(f"r{max_rounds}")

            # 学习率
            lr = params.get('learning_rate')
            if lr:
                params_list.append(f"lr{lr}")

            # 批次大小
            batch_size = params.get('batch_size')
            if batch_size:
                params_list.append(f"b{batch_size}")

    return '_'.join(params_list) if params_list else ''


def parse_experiment_id(exp_id: str) -> Optional[Dict[str, str]]:
    """
    解析实验ID

    Args:
        exp_id: 实验ID字符串

    Returns:
        解析后的字典，包含algorithm, dataset, partition, timestamp等字段

    Examples:
        >>> info = parse_experiment_id("fedavg_mnist_iid_20250111_143020")
        >>> print(info)
        {'algorithm': 'fedavg', 'dataset': 'mnist', 'partition': 'iid',
         'date': '20250111', 'time': '143020'}
    """
    parts = exp_id.split('_')

    if len(parts) < 5:
        return None

    return {
        'algorithm': parts[0],
        'dataset': parts[1],
        'partition': parts[2],
        'date': parts[-2],
        'time': parts[-1]
    }
