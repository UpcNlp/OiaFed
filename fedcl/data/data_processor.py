# fedcl/data/data_processor.py
"""
数据处理器模块

提供完整的数据预处理、变换、增强和质量验证功能。
支持流式处理、并行处理和内存优化。
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
import copy
import hashlib
import warnings
import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoader, RandomSampler, SequentialSampler
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
from loguru import logger

from ..config.config_manager import DictConfig
from ..exceptions import FedCLError
from .dataset import Dataset, DatasetError, DatasetValidationError
from .split_strategy import SplitStrategy, IIDSplitStrategy, NonIIDSplitStrategy


class DataProcessorError(FedCLError):
    """数据处理器异常基类"""
    pass


class DataTransformError(DataProcessorError):
    """数据变换异常"""
    pass


class DataValidationError(DataProcessorError):
    """数据验证异常"""
    pass


class DataAugmentationError(DataProcessorError):
    """数据增强异常"""
    pass


class DataProcessor:
    """
    数据处理器类
    
    提供数据预处理、变换、增强、质量验证等功能。
    支持流式处理、并行处理和内存优化。
    
    Attributes:
        config: 数据处理配置
        transforms: 数据变换列表
        augmentations: 数据增强列表
        normalization_params: 标准化参数
        validation_config: 验证配置
    """
    
    def __init__(self, config: DictConfig):
        """
        初始化数据处理器
        
        Args:
            config: 数据处理配置
            
        Raises:
            DataProcessorError: 配置无效时抛出
        """
        self.config = config
        self._validate_config()
        
        # 初始化变换和增强
        self.transforms = self._initialize_transforms()
        self.augmentations = self._initialize_augmentations()
        
        # 初始化标准化参数
        self.normalization_params = self._initialize_normalization_params()
        
        # 初始化验证配置
        self.validation_config = config.get('validation', {})
        
        # 内存优化设置
        self.memory_efficient = config.get('memory_efficient', True)
        self.max_workers = config.get('max_workers', 4)
        
        # 缓存机制
        self._transform_cache: Dict[str, Any] = {}
        self._stats_cache: Dict[str, Dict[str, Any]] = {}
        
        logger.debug(f"DataProcessor initialized with {len(self.transforms)} transforms "
                   f"and {len(self.augmentations)} augmentations")
    
    def _validate_config(self) -> None:
        """验证配置有效性"""
        if not isinstance(self.config, DictConfig):
            raise DataProcessorError("Invalid config type")
        
        # 验证必要的配置项
        required_sections = ['transforms', 'augmentation', 'validation']
        for section in required_sections:
            if section not in self.config:
                logger.warning(f"Missing config section: {section}")
    
    def _initialize_transforms(self) -> List[Callable]:
        """初始化数据变换"""
        transform_list = []
        transform_configs = self.config.get('transforms', [])
        
        for transform_config in transform_configs:
            transform_name = transform_config.get('name')
            transform_params = transform_config.get('params', {})
            
            transform_func = self._create_transform(transform_name, transform_params)
            if transform_func:
                transform_list.append(transform_func)
        
        return transform_list
    
    def _create_transform(self, name: str, params: Dict[str, Any]) -> Optional[Callable]:
        """创建变换函数"""
        transform_map = {
            'resize': lambda p: transforms.Resize(p.get('size')),
            'center_crop': lambda p: transforms.CenterCrop(p.get('size')),
            'random_crop': lambda p: transforms.RandomCrop(p.get('size'), 
                                                          padding=p.get('padding', 0)),
            'normalize': lambda p: transforms.Normalize(
                mean=p.get('mean', [0.5]), std=p.get('std', [0.5])),
            'to_tensor': lambda p: transforms.ToTensor(),
            'random_horizontal_flip': lambda p: transforms.RandomHorizontalFlip(
                p=p.get('p', 0.5)),
            'random_vertical_flip': lambda p: transforms.RandomVerticalFlip(
                p=p.get('p', 0.5)),
            'random_rotation': lambda p: transforms.RandomRotation(
                degrees=p.get('degrees', 10)),
            'color_jitter': lambda p: transforms.ColorJitter(
                brightness=p.get('brightness', 0),
                contrast=p.get('contrast', 0),
                saturation=p.get('saturation', 0),
                hue=p.get('hue', 0))
        }
        
        if name in transform_map:
            try:
                return transform_map[name](params)
            except Exception as e:
                logger.error(f"Failed to create transform {name}: {e}")
                return None
        else:
            logger.warning(f"Unknown transform: {name}")
            return None
    
    def _initialize_augmentations(self) -> List[Callable]:
        """初始化数据增强"""
        aug_config = self.config.get('augmentation', {})
        if not aug_config.get('enable', False):
            return []
        
        augmentation_list = []
        strategies = aug_config.get('strategies', [])
        
        for strategy in strategies:
            aug_func = self._create_augmentation(strategy)
            if aug_func:
                augmentation_list.append(aug_func)
        
        return augmentation_list
    
    def _create_augmentation(self, strategy: str) -> Optional[Callable]:
        """创建增强函数"""
        augmentation_map = {
            'rotation': lambda x: TF.rotate(x, angle=np.random.uniform(-15, 15)),
            'color_jitter': lambda x: TF.adjust_brightness(x, brightness_factor=np.random.uniform(0.8, 1.2)),
            'gaussian_noise': lambda x: x + torch.randn_like(x) * 0.01,
            'random_erasing': lambda x: transforms.RandomErasing(p=0.5)(x)
        }
        
        if strategy in augmentation_map:
            return augmentation_map[strategy]
        else:
            logger.warning(f"Unknown augmentation strategy: {strategy}")
            return None
    
    def _initialize_normalization_params(self) -> Dict[str, Any]:
        """初始化标准化参数"""
        norm_config = self.config.get('normalization', {})
        return {
            'method': norm_config.get('method', 'standard'),
            'per_channel': norm_config.get('per_channel', True),
            'eps': norm_config.get('eps', 1e-8)
        }
    
    def preprocess_data(self, raw_data: Dataset) -> Dataset:
        """
        数据预处理
        
        Args:
            raw_data: 原始数据集
            
        Returns:
            预处理后的数据集
            
        Raises:
            DataProcessorError: 预处理失败时抛出
        """
        try:
            logger.debug(f"Preprocessing dataset: {getattr(raw_data, 'name', 'Unknown')}")
            
            # 验证输入数据
            if not isinstance(raw_data, Dataset):
                raise DataProcessorError("Input must be a Dataset instance")
            
            # 创建处理后的数据副本
            processed_data = copy.deepcopy(raw_data.data)
            processed_targets = copy.deepcopy(raw_data.targets)
            
            # 应用基础变换
            if self.transforms:
                processed_data = self._apply_transform_pipeline(processed_data)
            
            # 创建新的数据集
            processed_dataset = Dataset(
                name=f"{raw_data.name}_preprocessed",
                data=processed_data,
                targets=processed_targets,
                transform=raw_data.transform
            )
            
            logger.debug(f"Preprocessing completed for dataset: {raw_data.name}")
            return processed_dataset
            
        except Exception as e:
            dataset_name = getattr(raw_data, 'name', 'Unknown')
            logger.error(f"Preprocessing failed for dataset {dataset_name}: {e}")
            raise DataProcessorError(f"Preprocessing failed: {e}")
    
    def apply_transforms(self, data: Dataset, transform_config: DictConfig) -> Dataset:
        """
        应用数据变换
        
        Args:
            data: 输入数据集
            transform_config: 变换配置
            
        Returns:
            变换后的数据集
            
        Raises:
            DataTransformError: 变换失败时抛出
        """
        try:
            logger.debug(f"Applying transforms to dataset: {data.name}")
            
            # 创建临时变换列表
            temp_transforms = []
            for transform_cfg in transform_config.get('transforms', []):
                transform_func = self._create_transform(
                    transform_cfg.get('name'),
                    transform_cfg.get('params', {})
                )
                if transform_func:
                    temp_transforms.append(transform_func)
            
            # 应用变换
            transformed_data = self._apply_transform_pipeline(
                data.data, temp_transforms
            )
            
            # 创建新数据集
            transformed_dataset = Dataset(
                name=f"{data.name}_transformed",
                data=transformed_data,
                targets=data.targets,
                transform=data.transform
            )
            
            return transformed_dataset
            
        except Exception as e:
            logger.error(f"Transform application failed: {e}")
            raise DataTransformError(f"Transform failed: {e}")
    
    def _apply_transform_pipeline(self, data: torch.Tensor, 
                                 transform_list: Optional[List[Callable]] = None) -> torch.Tensor:
        """应用变换管道"""
        if transform_list is None:
            transform_list = self.transforms
        
        if not transform_list:
            return data
        
        try:
            # 组合变换
            composed_transform = transforms.Compose(transform_list)
            
            # 批量处理数据
            if self.memory_efficient and len(data) > 1000:
                return self._apply_transforms_batched(data, composed_transform)
            else:
                return torch.stack([composed_transform(sample) for sample in data])
                
        except Exception as e:
            logger.error(f"Transform pipeline failed: {e}")
            raise DataTransformError(f"Transform pipeline failed: {e}")
    
    def _apply_transforms_batched(self, data: torch.Tensor, 
                                 transform: Callable) -> torch.Tensor:
        """批量应用变换（内存优化）"""
        batch_size = min(100, len(data))
        transformed_batches = []
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            transformed_batch = torch.stack([transform(sample) for sample in batch])
            transformed_batches.append(transformed_batch)
        
        return torch.cat(transformed_batches, dim=0)
    
    def create_data_loaders(self, datasets: Dict[str, Dataset], 
                           batch_size: int, shuffle: bool = True) -> Dict[str, TorchDataLoader]:
        """
        创建数据加载器
        
        Args:
            datasets: 数据集字典
            batch_size: 批次大小
            shuffle: 是否打乱数据
            
        Returns:
            数据加载器字典
            
        Raises:
            DataProcessorError: 创建失败时抛出
        """
        try:
            logger.debug(f"Creating data loaders for {len(datasets)} datasets")
            
            data_loaders = {}
            for name, dataset in datasets.items():
                # 创建采样器
                sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
                
                # 创建数据加载器
                data_loader = TorchDataLoader(
                    dataset=dataset,
                    batch_size=batch_size,
                    sampler=sampler,
                    num_workers=self.max_workers,
                    pin_memory=torch.cuda.is_available(),
                    persistent_workers=self.max_workers > 0
                )
                
                data_loaders[name] = data_loader
                logger.debug(f"Created data loader for {name}: {len(data_loader)} batches")
            
            return data_loaders
            
        except Exception as e:
            logger.error(f"Data loader creation failed: {e}")
            raise DataProcessorError(f"Data loader creation failed: {e}")
    
    def split_data_federated(self, dataset: Dataset, num_clients: int, 
                            strategy: str = "iid") -> Dict[str, Dataset]:
        """
        联邦数据分割
        
        Args:
            dataset: 要分割的数据集
            num_clients: 客户端数量
            strategy: 分割策略 ("iid" 或 "non_iid")
            
        Returns:
            客户端数据集字典
            
        Raises:
            DataProcessorError: 分割失败时抛出
        """
        try:
            logger.debug(f"Splitting dataset {dataset.name} for {num_clients} clients using {strategy} strategy")
            
            if num_clients <= 0:
                raise DataProcessorError("Number of clients must be positive")
            
            if len(dataset) < num_clients:
                raise DataProcessorError("Dataset size must be >= number of 客户端")
            
            # 选择分割策略
            if strategy.lower() == "iid":
                strategy_config = DictConfig({
                    'stratified': True,
                    'shuffle': True,
                    'random_seed': 42
                })
                split_strategy = IIDSplitStrategy(strategy_config)
            elif strategy.lower() == "non_iid":
                alpha = self.config.get('non_iid_alpha', 0.5)
                strategy_config = DictConfig({
                    'alpha': alpha,
                    'min_samples_per_client': 1,
                    'random_seed': 42
                })
                split_strategy = NonIIDSplitStrategy(strategy_config)
            else:
                raise DataProcessorError(f"Unknown split strategy: {strategy}")
            
            # 执行分割
            client_datasets = split_strategy.split_data(dataset, num_clients)
            
            logger.debug(f"Successfully split dataset into {len(client_datasets)} client datasets")
            return client_datasets
            
        except Exception as e:
            logger.error(f"Federated data split failed: {e}")
            raise DataProcessorError(f"Federated data split failed: {e}")
    
    def balance_client_data(self, client_datasets: Dict[str, Dataset]) -> Dict[str, Dataset]:
        """
        平衡客户端数据
        
        Args:
            client_datasets: 客户端数据集字典
            
        Returns:
            平衡后的客户端数据集字典
            
        Raises:
            DataProcessorError: 平衡失败时抛出
        """
        try:
            logger.debug(f"Balancing data for {len(client_datasets)} 客户端")
            
            if not client_datasets:
                logger.warning("No client datasets provided for balancing")
                return {}
            
            # 计算目标样本数（最小客户端样本数）
            min_samples = min(len(dataset) for dataset in client_datasets.values())
            
            balanced_datasets = {}
            for client_id, dataset in client_datasets.items():
                if len(dataset) > min_samples:
                    # 随机选择样本以达到目标数量
                    indices = torch.randperm(len(dataset))[:min_samples]
                    balanced_data = dataset.data[indices]
                    balanced_targets = dataset.targets[indices]
                    
                    balanced_dataset = Dataset(
                        name=f"{dataset.name}_balanced",
                        data=balanced_data,
                        targets=balanced_targets,
                        transform=dataset.transform
                    )
                    balanced_datasets[client_id] = balanced_dataset
                else:
                    balanced_datasets[client_id] = dataset
                    
                logger.debug(f"Client {client_id}: {len(balanced_datasets[client_id])} samples")
            
            logger.debug("Data balancing 完成")
            return balanced_datasets
            
        except Exception as e:
            logger.error(f"Data balancing failed: {e}")
            raise DataProcessorError(f"Data balancing failed: {e}")
    
    def augment_data(self, dataset: Dataset, augmentation_config: DictConfig) -> Dataset:
        """
        数据增强
        
        Args:
            dataset: 输入数据集
            augmentation_config: 增强配置
            
        Returns:
            增强后的数据集
            
        Raises:
            DataAugmentationError: 增强失败时抛出
        """
        try:
            logger.debug(f"Augmenting dataset: {dataset.name}")
            
            if not augmentation_config.get('enable', False):
                logger.debug("Data augmentation disabled")
                return dataset
            
            # 创建增强函数
            augmentation_funcs = []
            for strategy in augmentation_config.get('strategies', []):
                aug_func = self._create_augmentation(strategy)
                if aug_func:
                    augmentation_funcs.append(aug_func)
            
            if not augmentation_funcs:
                logger.warning("No valid augmentation strategies found")
                return dataset
            
            # 应用增强
            augmented_data = []
            original_data = []
            
            for i, sample in enumerate(dataset.data):
                original_data.append(sample)
                
                # 应用随机选择的增强
                if augmentation_funcs and torch.rand(1).item() < 0.5:
                    aug_func = np.random.choice(augmentation_funcs)
                    try:
                        augmented_sample = aug_func(sample)
                        augmented_data.append(augmented_sample)
                    except Exception as aug_e:
                        logger.warning(f"Augmentation failed for sample {i}: {aug_e}")
                        augmented_data.append(sample)
                else:
                    augmented_data.append(sample)
            
            # 合并原始数据和增强数据
            factor = augmentation_config.get('factor', 1.0)
            if factor > 1.0:
                # 扩展数据集
                all_data = original_data + augmented_data[:int(len(original_data) * (factor - 1))]
                all_targets = torch.cat([dataset.targets, 
                                       dataset.targets[:int(len(dataset.targets) * (factor - 1))]])
            else:
                all_data = augmented_data
                all_targets = dataset.targets
            
            # 创建增强数据集
            augmented_dataset = Dataset(
                name=f"{dataset.name}_augmented",
                data=torch.stack(all_data),
                targets=all_targets,
                transform=dataset.transform
            )
            
            logger.debug(f"Data augmentation completed: {len(dataset)} -> {len(augmented_dataset)} samples")
            return augmented_dataset
            
        except Exception as e:
            logger.error(f"Data augmentation failed: {e}")
            raise DataAugmentationError(f"Data augmentation failed: {e}")
    
    def normalize_data(self, dataset: Dataset, normalization_config: DictConfig) -> Dataset:
        """
        数据标准化
        
        Args:
            dataset: 输入数据集
            normalization_config: 标准化配置
            
        Returns:
            标准化后的数据集
            
        Raises:
            DataProcessorError: 标准化失败时抛出
        """
        try:
            logger.debug(f"Normalizing dataset: {dataset.name}")
            
            method = normalization_config.get('method', 'standard')
            per_channel = normalization_config.get('per_channel', True)
            eps = normalization_config.get('eps', 1e-8)
            
            data = dataset.data.float()
            
            if method == 'standard':
                # 标准化 (z-score)
                if per_channel and len(data.shape) > 2:
                    # 按通道标准化
                    for c in range(data.shape[1]):
                        channel_data = data[:, c]
                        mean = channel_data.mean()
                        std = channel_data.std() + eps
                        data[:, c] = (channel_data - mean) / std
                else:
                    # 全局标准化
                    mean = data.mean()
                    std = data.std() + eps
                    data = (data - mean) / std
                    
            elif method == 'min_max':
                # 最小-最大标准化
                if per_channel and len(data.shape) > 2:
                    for c in range(data.shape[1]):
                        channel_data = data[:, c]
                        min_val = channel_data.min()
                        max_val = channel_data.max()
                        data[:, c] = (channel_data - min_val) / (max_val - min_val + eps)
                else:
                    min_val = data.min()
                    max_val = data.max()
                    data = (data - min_val) / (max_val - min_val + eps)
                    
            elif method == 'unit_norm':
                # 单位向量标准化
                norms = torch.norm(data.view(data.size(0), -1), dim=1, keepdim=True)
                data = data / (norms.view(-1, 1, 1, 1) + eps)
                
            else:
                raise DataProcessorError(f"Unknown normalization method: {method}")
            
            # 创建标准化数据集
            normalized_dataset = Dataset(
                name=f"{dataset.name}_normalized",
                data=data,
                targets=dataset.targets,
                transform=dataset.transform
            )
            
            logger.debug(f"Data normalization completed using {method} method")
            return normalized_dataset
            
        except Exception as e:
            logger.error(f"Data normalization failed: {e}")
            raise DataProcessorError(f"Data normalization failed: {e}")
    
    def validate_data_quality(self, dataset: Dataset) -> Dict[str, Any]:
        """
        数据质量验证
        
        Args:
            dataset: 要验证的数据集
            
        Returns:
            验证结果字典
            
        Raises:
            DataValidationError: 验证失败时抛出
        """
        try:
            logger.debug(f"Validating data quality for dataset: {dataset.name}")
            
            validation_results = {
                'dataset_name': dataset.name,
                'total_samples': len(dataset),
                'passed': True,
                'errors': [],
                'warnings': [],
                'checks': {}
            }
            
            data = dataset.data
            targets = dataset.targets
            
            # 检查NaN值
            if self.validation_config.get('check_nan', True):
                nan_count = torch.isnan(data).sum().item()
                validation_results['checks']['nan_check'] = {
                    'passed': nan_count == 0,
                    'nan_count': nan_count
                }
                if nan_count > 0:
                    validation_results['errors'].append(f"Found {nan_count} NaN values")
                    validation_results['passed'] = False
            
            # 检查数值范围
            if self.validation_config.get('check_range', True):
                min_val = data.min().item()
                max_val = data.max().item()
                inf_count = torch.isinf(data).sum().item()
                
                validation_results['checks']['range_check'] = {
                    'passed': inf_count == 0,
                    'min_value': min_val,
                    'max_value': max_val,
                    'inf_count': inf_count
                }
                
                if inf_count > 0:
                    validation_results['errors'].append(f"Found {inf_count} infinite values")
                    validation_results['passed'] = False
                
                if abs(min_val) > 1e6 or abs(max_val) > 1e6:
                    validation_results['warnings'].append("Extreme values detected")
            
            # 检查标签分布
            if self.validation_config.get('check_distribution', True):
                unique_targets, counts = torch.unique(targets, return_counts=True)
                min_count = counts.min().item()
                max_count = counts.max().item()
                
                validation_results['checks']['distribution_check'] = {
                    'passed': True,
                    'num_classes': len(unique_targets),
                    'min_samples_per_class': min_count,
                    'max_samples_per_class': max_count,
                    'class_distribution': dict(zip(unique_targets.tolist(), counts.tolist()))
                }
                
                # 检查类别不平衡
                if max_count > min_count * 5:  # 降低阈值使其更容易触发
                    validation_results['warnings'].append("Severe class imbalance detected")
                
                # 检查空类别
                if min_count == 0:
                    validation_results['errors'].append("Empty classes found")
                    validation_results['passed'] = False
            
            # 检查数据维度一致性
            # 检查样本数量是否匹配
            if data.shape[0] != targets.shape[0]:
                validation_results['errors'].append("Sample count mismatch between data and targets")
                validation_results['passed'] = False
            
            # 检查目标是否为1D
            if len(targets.shape) != 1:
                validation_results['errors'].append("Targets should be 1-dimensional")
                validation_results['passed'] = False
            
            logger.debug(f"Data quality validation completed for {dataset.name}: "
                       f"{'PASSED' if validation_results['passed'] else 'FAILED'}")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Data quality validation failed: {e}")
            raise DataValidationError(f"Data quality validation failed: {e}")
    
    def get_data_statistics(self, dataset: Dataset) -> Dict[str, Any]:
        """
        获取数据统计信息
        
        Args:
            dataset: 数据集
            
        Returns:
            统计信息字典
            
        Raises:
            DataProcessorError: 统计计算失败时抛出
        """
        try:
            # 检查缓存
            cache_key = f"{dataset.name}_{len(dataset)}"
            if cache_key in self._stats_cache:
                logger.debug(f"Returning cached statistics for {dataset.name}")
                return self._stats_cache[cache_key]
            
            logger.debug(f"Computing statistics for dataset: {dataset.name}")
            
            data = dataset.data.float()
            targets = dataset.targets
            
            # 基础统计
            stats = {
                'dataset_name': dataset.name,
                'total_samples': len(dataset),
                'data_shape': list(data.shape),
                'data_type': str(data.dtype),
                'targets_shape': list(targets.shape),
                'targets_type': str(targets.dtype)
            }
            
            # 数据统计
            stats['data_stats'] = {
                'mean': data.mean().item(),
                'std': data.std().item(),
                'min': data.min().item(),
                'max': data.max().item(),
                'median': data.median().item()
            }
            
            # 按通道统计（如果适用）
            if len(data.shape) == 4:  # 假设是图像数据 (N, C, H, W)
                channel_stats = {}
                for c in range(data.shape[1]):
                    channel_data = data[:, c]
                    channel_stats[f'channel_{c}'] = {
                        'mean': channel_data.mean().item(),
                        'std': channel_data.std().item(),
                        'min': channel_data.min().item(),
                        'max': channel_data.max().item()
                    }
                stats['channel_stats'] = channel_stats
            
            # 标签统计
            unique_targets, counts = torch.unique(targets, return_counts=True)
            stats['target_stats'] = {
                'num_classes': len(unique_targets),
                'unique_targets': unique_targets.tolist(),
                'class_counts': counts.tolist(),
                'class_distribution': dict(zip(unique_targets.tolist(), counts.tolist())),
                'most_frequent_class': unique_targets[counts.argmax()].item(),
                'least_frequent_class': unique_targets[counts.argmin()].item()
            }
            
            # 内存使用统计
            data_memory = data.element_size() * data.nelement()
            targets_memory = targets.element_size() * targets.nelement()
            stats['memory_stats'] = {
                'data_memory_mb': data_memory / (1024 * 1024),
                'targets_memory_mb': targets_memory / (1024 * 1024),
                'total_memory_mb': (data_memory + targets_memory) / (1024 * 1024)
            }
            
            # 缓存结果
            self._stats_cache[cache_key] = stats
            
            logger.debug(f"Statistics computed for {dataset.name}: "
                       f"{stats['total_samples']} samples, "
                       f"{stats['target_stats']['num_classes']} classes")
            
            return stats
            
        except Exception as e:
            logger.error(f"Statistics computation failed for {dataset.name}: {e}")
            raise DataProcessorError(f"Statistics computation failed: {e}")
    
    def clear_cache(self) -> None:
        """清空缓存"""
        self._transform_cache.clear()
        self._stats_cache.clear()
        logger.debug("DataProcessor cache cleared")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """获取缓存信息"""
        return {
            'transform_cache_size': len(self._transform_cache),
            'stats_cache_size': len(self._stats_cache),
            'total_cache_items': len(self._transform_cache) + len(self._stats_cache)
        }
