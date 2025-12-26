"""
通用数据划分工厂

提供统一的数据集划分接口，支持任意 PyTorch Dataset
"""

from typing import Any, Dict, List, Optional, Union
from torch.utils.data import Dataset, Subset
import importlib

from .partitioner import Partitioner, IIDPartitioner, LabelPartitioner, DirichletPartitioner, QuantityPartitioner, DirichletQuantityPartitioner
from ..registry import registry
from ..infra import get_module_logger

logger = get_module_logger(__name__)


def create_partitioned_dataset(
    dataset: Dataset,
    partition_config: Dict[str, Any],
) -> Dataset:
    """
    对任意数据集应用划分策略

    Args:
        dataset: 原始完整数据集（任意 PyTorch Dataset）
        partition_config: 划分配置
            - strategy: 划分策略 (iid | label | dirichlet | quantity | custom)
            - partition_id: 当前分区ID
            - num_partitions: 总分区数
            - config: 划分器配置参数
            - custom_partitioner: 自定义 Partitioner 实例（可选）

    Returns:
        torch.utils.data.Subset (标准 Dataset 子类)

    Examples:
        >>> # 使用内置策略
        >>> partitioned = create_partitioned_dataset(
        ...     dataset=mnist_dataset,
        ...     partition_config={
        ...         'strategy': 'dirichlet',
        ...         'partition_id': 0,
        ...         'num_partitions': 5,
        ...         'config': {'alpha': 0.5}
        ...     }
        ... )

        >>> # 使用自定义 Partitioner
        >>> my_partitioner = MyCustomPartitioner(...)
        >>> partitioned = create_partitioned_dataset(
        ...     dataset=mnist_dataset,
        ...     partition_config={
        ...         'strategy': 'custom',
        ...         'partition_id': 0,
        ...         'num_partitions': 5,
        ...         'custom_partitioner': my_partitioner
        ...     }
        ... )
    """
    strategy = partition_config['strategy']
    partition_id = partition_config['partition_id']
    num_partitions = partition_config['num_partitions']
    config = partition_config.get('config', {})

    # 创建 Partitioner
    partitioner = _create_partitioner(
        strategy=strategy,
        config=config,
        custom_partitioner=partition_config.get('custom_partitioner', None)
    )

    # 提取标签（如果需要）
    labels = None
    if _needs_labels(partitioner):
        labels = _extract_labels(dataset)
        if labels is None:
            logger.warning(
                f"Partitioner '{strategy}' needs labels, but failed to extract from dataset. "
                f"Falling back to IID partition."
            )
            partitioner = IIDPartitioner(seed=config.get('seed'))

    # 执行划分
    logger.info(f"Partitioning dataset with strategy '{strategy}' (client {partition_id}/{num_partitions})")
    partition_dict = partitioner.partition(
        dataset_size=len(dataset),
        num_clients=num_partitions,
        labels=labels
    )

    # 获取当前客户端的索引
    if partition_id not in partition_dict:
        raise ValueError(f"Partition ID {partition_id} not found in partition result")

    indices = partition_dict[partition_id]

    if not indices:
        raise ValueError(f"Partition {partition_id} is empty")

    logger.info(f"Partition {partition_id}: {len(indices)} samples")

    # 返回标准 Subset（是 Dataset 子类）
    return Subset(dataset, indices)


def _create_partitioner(
    strategy: str,
    config: Dict[str, Any],
    custom_partitioner: Optional[Partitioner] = None
) -> Partitioner:
    """
    创建 Partitioner 实例

    支持三种方式：
    1. 内置策略 (iid, label, dirichlet, quantity)
    2. 自定义实例（直接传入）
    3. 自定义类（通过 class 路径加载）

    Args:
        strategy: 划分策略名称
        config: 配置参数
        custom_partitioner: 自定义 Partitioner 实例

    Returns:
        Partitioner 实例
    """
    seed = config.get('seed', None)

    # 方式1: 直接传入自定义 Partitioner 实例
    if custom_partitioner is not None:
        logger.info(f"Using custom partitioner: {type(custom_partitioner).__name__}")
        return custom_partitioner

    # 方式2: 从配置加载自定义类
    if strategy == 'custom':
        if 'class' not in config:
            raise ValueError("Custom partitioner requires 'class' in config")

        class_path = config['class']
        partitioner_class = _load_class(class_path)

        # 移除 'class' 和 'seed'，剩余参数传给构造函数
        init_kwargs = {k: v for k, v in config.items() if k not in ['class', 'seed']}
        init_kwargs['seed'] = seed

        logger.info(f"Loading custom partitioner from: {class_path}")
        return partitioner_class(**init_kwargs)

    # 方式3: 使用内置策略
    if strategy == 'iid':
        return IIDPartitioner(seed=seed)

    elif strategy in ['label', 'non_iid']:
        labels_per_client = config.get('labels_per_client', 2)
        return LabelPartitioner(labels_per_client=labels_per_client, seed=seed)

    elif strategy == 'dirichlet':
        alpha = config.get('alpha', 0.5)
        return DirichletPartitioner(alpha=alpha, seed=seed)

    elif strategy == 'dirichlet_quantity':
        alpha = config.get('alpha', 0.5)
        return DirichletQuantityPartitioner(alpha=alpha, seed=seed)

    elif strategy == 'quantity':
        quantities = config.get('quantities', None)
        ratios = config.get('ratios', None)
        return QuantityPartitioner(quantities=quantities, ratios=ratios, seed=seed)

    else:
        raise ValueError(
            f"Unknown partition strategy: {strategy}. "
            f"Supported: iid, label, dirichlet, dirichlet_quantity, quantity, custom"
        )


def _needs_labels(partitioner: Partitioner) -> bool:
    """
    检查 Partitioner 是否需要标签信息

    Args:
        partitioner: Partitioner 实例

    Returns:
        是否需要标签
    """
    # 检查是否有 needs_labels 属性
    if hasattr(partitioner, 'needs_labels'):
        return partitioner.needs_labels

    # 根据类型判断
    return isinstance(partitioner, (LabelPartitioner, DirichletPartitioner))


def _extract_labels(dataset: Dataset) -> Optional[List[Any]]:
    """
    从数据集中提取标签

    支持的数据集格式：
    1. dataset[i] 返回 (data, label) 元组
    2. dataset 有 .targets 属性
    3. dataset 有 .labels 属性

    Args:
        dataset: PyTorch Dataset

    Returns:
        标签列表，失败返回 None
    """
    try:
        # 方式1: 检查 targets 属性（torchvision 标准）
        if hasattr(dataset, 'targets'):
            targets = dataset.targets
            if hasattr(targets, 'tolist'):
                return targets.tolist()
            elif isinstance(targets, list):
                return targets
            else:
                return list(targets)

        # 方式2: 检查 labels 属性
        if hasattr(dataset, 'labels'):
            labels = dataset.labels
            if hasattr(labels, 'tolist'):
                return labels.tolist()
            elif isinstance(labels, list):
                return labels
            else:
                return list(labels)

        # 方式3: 遍历数据集提取标签
        logger.info("Extracting labels by iterating dataset...")
        labels = []
        for i in range(len(dataset)):
            item = dataset[i]
            # 假设返回 (data, label) 元组
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                label = item[1]
                # 转换为 Python 标量
                if hasattr(label, 'item'):
                    label = label.item()
                labels.append(label)
            else:
                logger.warning(f"Cannot extract label from dataset item at index {i}")
                return None

        return labels

    except Exception as e:
        logger.warning(f"Failed to extract labels: {e}")
        return None


def _load_class(class_path: str):
    """
    动态加载类

    Args:
        class_path: 类路径，例如 "my_module.MyClass"

    Returns:
        类对象
    """
    try:
        module_path, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except Exception as e:
        raise ImportError(f"Failed to load class '{class_path}': {e}")


def create_dataset_with_partition(
    dataset_config: Dict[str, Any],
    partition_config: Optional[Dict[str, Any]] = None
) -> Dataset:
    """
    创建数据集并应用划分（完整流程）

    Args:
        dataset_config: 数据集配置
            - type: 数据集类型 (例如 'mnist')
            - args: 数据集参数
        partition_config: 划分配置（可选）
            - 如果为 None，返回完整数据集

    Returns:
        Dataset 实例（可能是划分后的 Subset）

    Examples:
        >>> dataset = create_dataset_with_partition(
        ...     dataset_config={
        ...         'type': 'mnist',
        ...         'args': {'data_dir': './data'}
        ...     },
        ...     partition_config={
        ...         'strategy': 'dirichlet',
        ...         'partition_id': 0,
        ...         'num_partitions': 5,
        ...         'config': {'alpha': 0.5}
        ...     }
        ... )
    """
    # 1. 创建原始数据集
    dataset_type = dataset_config['type']
    dataset_args = dataset_config.get('args', {})

    logger.info(f"Creating dataset: {dataset_type}")
    dataset = registry.create(
        namespace=f"dataset.{dataset_type}",
        **dataset_args
    )

    # 2. 将数据集对象转换为 PyTorch Dataset（如果需要）
    pytorch_dataset = _to_pytorch_dataset(dataset)

    # 3. 如果没有划分配置，直接返回
    if not partition_config:
        logger.info("No partition config, returning full dataset")
        return pytorch_dataset

    # 4. 应用划分
    return create_partitioned_dataset(pytorch_dataset, partition_config)


def _to_pytorch_dataset(dataset: Any) -> Dataset:
    """
    将数据集对象转换为 PyTorch Dataset

    支持：
    1. 已经是 PyTorch Dataset → 直接返回
    2. 有 get_train_dataset() 方法 → 调用并返回
    3. 有 get_dataset() 方法 → 调用并返回

    Args:
        dataset: 数据集对象

    Returns:
        PyTorch Dataset
    """
    # 情况1: 已经是 PyTorch Dataset
    if isinstance(dataset, Dataset):
        return dataset

    # 情况2: 有 get_train_dataset() 方法
    if hasattr(dataset, 'get_train_dataset'):
        logger.debug("Converting dataset using get_train_dataset()")
        return dataset.get_train_dataset()

    # 情况3: 有 get_dataset() 方法
    if hasattr(dataset, 'get_dataset'):
        logger.debug("Converting dataset using get_dataset()")
        return dataset.get_dataset()

    # 情况4: 无法转换，抛出错误
    raise TypeError(
        f"Dataset of type {type(dataset)} cannot be converted to PyTorch Dataset. "
        f"It should either be a torch.utils.data.Dataset, or have get_train_dataset() / get_dataset() method."
    )
