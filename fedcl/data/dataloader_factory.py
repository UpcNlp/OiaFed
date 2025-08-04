# fedcl/data/dataloader_factory.py
"""
DataLoader工厂系统

基于装饰器的数据集加载器注册和管理系统，支持：
- 装饰器注册自定义DataLoader类型
- 配置驱动的DataLoader创建
- 内存和性能优化
- 验证和错误处理
"""

from typing import Dict, Any, Optional, Type, Callable, List, Union
from abc import ABC, abstractmethod
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader as TorchDataLoader, TensorDataset, Subset
from pathlib import Path
import pickle
import numpy as np
from loguru import logger

from .dataloader import DataLoader
from .dataset import Dataset as FedCLDataset
from ..config.config_manager import DictConfig


class DataLoaderCreationError(Exception):
    """DataLoader创建错误"""
    pass


class DataLoaderRegistrationError(Exception):
    """DataLoader注册错误"""
    pass


class BaseDataLoaderCreator(ABC):
    """DataLoader创建器基类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化DataLoader创建器
        
        Args:
            config: DataLoader配置
        """
        self.config = config
        self.name = config.get('name', 'unknown')
        
    @abstractmethod
    def create_dataset(self) -> Dataset:
        """创建数据集"""
        pass
    
    @abstractmethod
    def create_transforms(self) -> Optional[transforms.Compose]:
        """创建数据变换"""
        pass
    
    def create_dataloader(self, **kwargs) -> DataLoader:
        """
        创建DataLoader
        
        Args:
            **kwargs: 额外的DataLoader参数
            
        Returns:
            DataLoader实例
        """
        try:
            # 创建数据集
            dataset = self.create_dataset()
            
            # 获取loader参数
            loader_params = self.config.get('loader_params', {})
            loader_params.update(kwargs)  # 允许运行时覆盖
            
            # 创建并返回DataLoader
            return DataLoader(
                dataset=dataset,
                **loader_params
            )
            
        except Exception as e:
            logger.error(f"Failed to create DataLoader for {self.name}: {e}")
            raise DataLoaderCreationError(f"Failed to create DataLoader: {e}")
    
    def validate_config(self) -> bool:
        """验证配置有效性"""
        required_fields = ['dataset', 'loader_params']
        for field in required_fields:
            if field not in self.config:
                logger.error(f"Missing required field '{field}' in DataLoader config")
                return False
        return True


class DataLoaderFactory:
    """DataLoader工厂类"""
    
    def __init__(self):
        """初始化工厂"""
        self._creators: Dict[str, Type[BaseDataLoaderCreator]] = {}
        self._instances: Dict[str, DataLoader] = {}
        self._stats = {
            'created_count': 0,
            'cache_hits': 0,
            'memory_usage_mb': 0.0
        }
        
    def register_creator(self, name: str, creator_class: Type[BaseDataLoaderCreator]) -> None:
        """
        注册DataLoader创建器
        
        Args:
            name: 创建器名称
            creator_class: 创建器类
        """
        if not issubclass(creator_class, BaseDataLoaderCreator):
            raise DataLoaderRegistrationError(
                f"Creator class must inherit from BaseDataLoaderCreator"
            )
        
        self._creators[name] = creator_class
        logger.debug(f"Registered DataLoader creator: {name}")
    
    def create_dataloader(self, name: str, config: Dict[str, Any], 
                         use_cache: bool = True, **kwargs) -> DataLoader:
        """
        创建DataLoader实例
        
        Args:
            name: DataLoader名称
            config: 配置字典
            use_cache: 是否使用缓存
            **kwargs: 额外参数
            
        Returns:
            DataLoader实例
        """
        # 检查缓存
        if use_cache and name in self._instances:
            self._stats['cache_hits'] += 1
            logger.debug(f"Using cached DataLoader: {name}")
            return self._instances[name]
        
        try:
            # 获取创建器类型
            creator_type = config.get('type', 'standard')
            
            if creator_type not in self._creators:
                raise DataLoaderCreationError(
                    f"Unknown DataLoader creator type: {creator_type}"
                )
            
            # 创建DataLoader创建器
            creator_class = self._creators[creator_type]
            creator = creator_class(config)
            
            # 验证配置
            if not creator.validate_config():
                raise DataLoaderCreationError(f"Invalid configuration for {name}")
            
            # 创建DataLoader
            dataloader = creator.create_dataloader(**kwargs)
            
            # 缓存实例
            if use_cache:
                self._instances[name] = dataloader
            
            # 更新统计
            self._stats['created_count'] += 1
            self._update_memory_stats(dataloader)
            
            logger.debug(f"Created DataLoader '{name}' of type '{creator_type}'")
            return dataloader
            
        except Exception as e:
            logger.error(f"Failed to create DataLoader '{name}': {e}")
            raise DataLoaderCreationError(f"Failed to create DataLoader '{name}': {e}")
    
    def create_multiple_dataloaders(self, configs: Dict[str, Dict[str, Any]], 
                                  use_cache: bool = True) -> Dict[str, DataLoader]:
        """
        批量创建多个DataLoader
        
        Args:
            configs: DataLoader配置字典
            use_cache: 是否使用缓存
            
        Returns:
            DataLoader字典
        """
        dataloaders = {}
        
        for name, config in configs.items():
            try:
                dataloaders[name] = self.create_dataloader(name, config, use_cache)
            except Exception as e:
                logger.error(f"Failed to create DataLoader '{name}': {e}")
                # 继续创建其他DataLoader
                continue
        
        logger.debug(f"Created {len(dataloaders)}/{len(configs)} DataLoaders")
        return dataloaders
    
    def get_dataloader(self, name: str) -> Optional[DataLoader]:
        """获取已创建的DataLoader"""
        return self._instances.get(name)
    
    def list_creators(self) -> List[str]:
        """列出所有已注册的创建器"""
        return list(self._creators.keys())
    
    def get_stats(self) -> Dict[str, Any]:
        """获取工厂统计信息"""
        return self._stats.copy()
    
    def clear_cache(self) -> None:
        """清空DataLoader缓存"""
        self._instances.clear()
        logger.debug("DataLoader cache cleared")
    
    def _update_memory_stats(self, dataloader: DataLoader) -> None:
        """更新内存统计"""
        try:
            memory_info = dataloader.get_memory_info()
            self._stats['memory_usage_mb'] += memory_info.get('estimated_total_memory_mb', 0)
        except Exception as e:
            logger.warning(f"Failed to update memory stats: {e}")


# 全局DataLoader工厂实例
dataloader_factory = DataLoaderFactory()


def dataloader_creator(name: str):
    """
    DataLoader创建器装饰器
    
    Args:
        name: 创建器名称
        
    Example:
        @dataloader_creator('my_custom_loader')
        class MyCustomDataLoaderCreator(BaseDataLoaderCreator):
            def create_dataset(self):
                # 实现自定义数据集创建逻辑
                pass
    """
    def decorator(creator_class: Type[BaseDataLoaderCreator]) -> Type[BaseDataLoaderCreator]:
        dataloader_factory.register_creator(name, creator_class)
        return creator_class
    
    return decorator


# ============= 内置DataLoader创建器 =============

@dataloader_creator('StandardDataLoader')
class StandardDataLoaderCreator(BaseDataLoaderCreator):
    """标准DataLoader创建器"""
    
    def create_dataset(self) -> Dataset:
        """创建标准数据集"""
        dataset_config = self.config['dataset']
        dataset_name = dataset_config['name']
        
        if dataset_name == 'CIFAR10':
            return self._create_cifar10_dataset(dataset_config)
        elif dataset_name == 'MNIST':
            return self._create_mnist_dataset(dataset_config)
        elif dataset_name == 'ImageNet':
            return self._create_imagenet_dataset(dataset_config)
        else:
            raise DataLoaderCreationError(f"Unsupported dataset: {dataset_name}")
    
    def create_transforms(self) -> Optional[transforms.Compose]:
        """创建数据变换"""
        transform_configs = self.config.get('transforms', [])
        if not transform_configs:
            return None
        
        transform_list = []
        for transform_config in transform_configs:
            transform_name = transform_config['name']
            transform_params = transform_config.get('params', {})
            
            # 创建变换
            if transform_name == 'Resize':
                transform_list.append(transforms.Resize(**transform_params))
            elif transform_name == 'ToTensor':
                transform_list.append(transforms.ToTensor())
            elif transform_name == 'Normalize':
                transform_list.append(transforms.Normalize(**transform_params))
            elif transform_name == 'RandomHorizontalFlip':
                transform_list.append(transforms.RandomHorizontalFlip(**transform_params))
            elif transform_name == 'RandomCrop':
                transform_list.append(transforms.RandomCrop(**transform_params))
            else:
                logger.warning(f"Unknown transform: {transform_name}")
        
        return transforms.Compose(transform_list) if transform_list else None
    
    def _create_cifar10_dataset(self, dataset_config: Dict[str, Any]) -> Dataset:
        """创建CIFAR10数据集"""
        data_path = Path(dataset_config['path'])
        split = dataset_config.get('split', 'train')
        
        transform = self.create_transforms()
        
        dataset = torchvision.datasets.CIFAR10(
            root=str(data_path),
            train=(split == 'train'),
            download=True,
            transform=transform
        )
        
        logger.debug(f"Created CIFAR10 dataset: {len(dataset)} samples")
        return dataset
    
    def _create_mnist_dataset(self, dataset_config: Dict[str, Any]) -> Dataset:
        """创建MNIST数据集"""
        data_path = Path(dataset_config['path'])
        split = dataset_config.get('split', 'train')
        
        transform = self.create_transforms()
        if transform is None:
            # 默认MNIST变换
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        
        dataset = torchvision.datasets.MNIST(
            root=str(data_path),
            train=(split == 'train'),
            download=True,
            transform=transform
        )
        
        logger.debug(f"Created MNIST dataset: {len(dataset)} samples")
        return dataset
    
    def _create_imagenet_dataset(self, dataset_config: Dict[str, Any]) -> Dataset:
        """创建ImageNet数据集"""
        data_path = Path(dataset_config['path'])
        split = dataset_config.get('split', 'train')
        
        transform = self.create_transforms()
        if transform is None:
            # 默认ImageNet变换
            if split == 'train':
                transform = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
            else:
                transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
        
        dataset = torchvision.datasets.ImageNet(
            root=str(data_path),
            split=split,
            transform=transform
        )
        
        logger.debug(f"Created ImageNet dataset: {len(dataset)} samples")
        return dataset


@dataloader_creator('DiffusionDataLoader')
class DiffusionDataLoaderCreator(BaseDataLoaderCreator):
    """扩散模型DataLoader创建器"""
    
    def create_dataset(self) -> Dataset:
        """创建扩散模型数据集"""
        dataset_config = self.config['dataset']
        
        # 创建基础数据集
        base_creator = StandardDataLoaderCreator(self.config)
        base_dataset = base_creator.create_dataset()
        
        # 包装为扩散数据集
        return DiffusionDatasetWrapper(base_dataset, self.config)
    
    def create_transforms(self) -> Optional[transforms.Compose]:
        """创建扩散模型专用变换"""
        transform_configs = self.config.get('transforms', [])
        transform_list = []
        
        for transform_config in transform_configs:
            transform_name = transform_config['name']
            transform_params = transform_config.get('params', {})
            
            if transform_name == 'AddGaussianNoise':
                transform_list.append(AddGaussianNoiseTransform(**transform_params))
            elif transform_name == 'TimestepSampling':
                transform_list.append(TimestepSamplingTransform(**transform_params))
            elif transform_name == 'Resize':
                transform_list.append(transforms.Resize(**transform_params))
            elif transform_name == 'ToTensor':
                transform_list.append(transforms.ToTensor())
            elif transform_name == 'Normalize':
                transform_list.append(transforms.Normalize(**transform_params))
        
        return transforms.Compose(transform_list) if transform_list else None


@dataloader_creator('FeatureDataLoader')
class FeatureDataLoaderCreator(BaseDataLoaderCreator):
    """特征DataLoader创建器"""
    
    def create_dataset(self) -> Dataset:
        """创建特征数据集"""
        dataset_config = self.config['dataset']
        
        # 创建基础数据集
        base_creator = StandardDataLoaderCreator(self.config)
        base_dataset = base_creator.create_dataset()
        
        # 包装为特征数据集
        return FeatureDatasetWrapper(base_dataset, self.config)
    
    def create_transforms(self) -> Optional[transforms.Compose]:
        """创建特征提取专用变换"""
        transform_configs = self.config.get('transforms', [])
        transform_list = []
        
        for transform_config in transform_configs:
            transform_name = transform_config['name']
            transform_params = transform_config.get('params', {})
            
            if transform_name == 'FeatureExtraction':
                transform_list.append(FeatureExtractionTransform(**transform_params))
            elif transform_name == 'AdaptivePooling':
                transform_list.append(AdaptivePoolingTransform(**transform_params))
            elif transform_name == 'Resize':
                transform_list.append(transforms.Resize(**transform_params))
            elif transform_name == 'ToTensor':
                transform_list.append(transforms.ToTensor())
        
        return transforms.Compose(transform_list) if transform_list else None


@dataloader_creator('CustomFileDataLoader')
class CustomFileDataLoaderCreator(BaseDataLoaderCreator):
    """自定义文件DataLoader创建器"""
    
    def create_dataset(self) -> Dataset:
        """从文件创建数据集"""
        dataset_config = self.config['dataset']
        file_path = Path(dataset_config['path'])
        
        if not file_path.exists():
            raise DataLoaderCreationError(f"Dataset file not found: {file_path}")
        
        file_type = dataset_config.get('file_type', 'pickle')
        
        if file_type == 'pickle':
            return self._load_pickle_dataset(file_path)
        elif file_type == 'npz':
            return self._load_npz_dataset(file_path)
        else:
            raise DataLoaderCreationError(f"Unsupported file type: {file_type}")
    
    def create_transforms(self) -> Optional[transforms.Compose]:
        """创建变换（文件数据集通常不需要变换）"""
        return None
    
    def _load_pickle_dataset(self, file_path: Path) -> Dataset:
        """加载pickle格式数据集"""
        try:
            with open(file_path, 'rb') as f:
                data_dict = pickle.load(f)
            
            data = torch.tensor(data_dict['data'], dtype=torch.float32)
            targets = torch.tensor(data_dict['targets'], dtype=torch.long)
            
            logger.debug(f"Loaded pickle dataset: {len(data)} samples")
            return TensorDataset(data, targets)
            
        except Exception as e:
            raise DataLoaderCreationError(f"Failed to load pickle dataset: {e}")
    
    def _load_npz_dataset(self, file_path: Path) -> Dataset:
        """加载npz格式数据集"""
        try:
            data_dict = np.load(file_path)
            
            data = torch.tensor(data_dict['data'], dtype=torch.float32)
            targets = torch.tensor(data_dict['targets'], dtype=torch.long)
            
            logger.debug(f"Loaded npz dataset: {len(data)} samples")
            return TensorDataset(data, targets)
            
        except Exception as e:
            raise DataLoaderCreationError(f"Failed to load npz dataset: {e}")


# ============= 数据集包装器 =============

class DiffusionDatasetWrapper(Dataset):
    """扩散模型数据集包装器"""
    
    def __init__(self, base_dataset: Dataset, config: Dict[str, Any]):
        self.base_dataset = base_dataset
        self.config = config
        self.max_timesteps = config.get('transforms', [{}])[-1].get('params', {}).get('max_timesteps', 1000)
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        data, target = self.base_dataset[idx]
        
        # 为扩散模型添加时间步
        timestep = torch.randint(0, self.max_timesteps, (1,)).item()
        
        return {
            'data': data,
            'target': target,
            'timestep': timestep
        }


class FeatureDatasetWrapper(Dataset):
    """特征数据集包装器"""
    
    def __init__(self, base_dataset: Dataset, config: Dict[str, Any]):
        self.base_dataset = base_dataset
        self.config = config
        self.feature_dim = config.get('transforms', [{}])[-1].get('params', {}).get('output_size', [2048])[0]
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        data, target = self.base_dataset[idx]
        
        # 模拟特征提取（实际应用中应该使用预训练模型）
        if isinstance(data, torch.Tensor):
            features = torch.randn(self.feature_dim)  # 模拟特征
        else:
            features = torch.randn(self.feature_dim)
        
        return features, target


# ============= 自定义变换 =============

class AddGaussianNoiseTransform:
    """添加高斯噪声变换"""
    
    def __init__(self, noise_schedule: str = 'linear', max_noise: float = 0.1):
        self.noise_schedule = noise_schedule
        self.max_noise = max_noise
    
    def __call__(self, x):
        noise_level = torch.rand(1).item() * self.max_noise
        noise = torch.randn_like(x) * noise_level
        return x + noise


class TimestepSamplingTransform:
    """时间步采样变换"""
    
    def __init__(self, max_timesteps: int = 1000):
        self.max_timesteps = max_timesteps
    
    def __call__(self, x):
        # 这个变换通常在数据集包装器中处理
        return x


class FeatureExtractionTransform:
    """特征提取变换"""
    
    def __init__(self, backbone: str = 'resnet50', layer: str = 'avgpool'):
        self.backbone = backbone
        self.layer = layer
    
    def __call__(self, x):
        # 这个变换通常在数据集包装器中处理
        return x


class AdaptivePoolingTransform:
    """自适应池化变换"""
    
    def __init__(self, output_size: List[int] = [2048]):
        self.output_size = output_size[0]
    
    def __call__(self, x):
        # 简单实现：如果输入是图像，进行全局平均池化
        if len(x.shape) == 3:  # CHW format
            x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
            x = x.view(-1)
            if len(x) != self.output_size:
                # 调整到目标大小
                x = torch.nn.functional.interpolate(
                    x.unsqueeze(0).unsqueeze(0), 
                    size=self.output_size, 
                    mode='linear', 
                    align_corners=False
                ).squeeze()
        return x


# ============= 工具函数 =============

def create_dataloader_from_config(name: str, config: Union[str, Dict[str, Any]], 
                                 factory: Optional[DataLoaderFactory] = None) -> DataLoader:
    """
    从配置创建DataLoader
    
    Args:
        name: DataLoader名称
        config: 配置字典或配置文件路径
        factory: DataLoader工厂实例（可选）
        
    Returns:
        DataLoader实例
    """
    if factory is None:
        factory = dataloader_factory
    
    if isinstance(config, str):
        # 从文件加载配置
        import yaml
        with open(config, 'r') as f:
            config = yaml.safe_load(f)
    
    return factory.create_dataloader(name, config)


def register_user_dataloader_creator(name: str):
    """
    用户注册DataLoader创建器的便利装饰器
    
    Args:
        name: 创建器名称
        
    Example:
        @register_user_dataloader_creator('my_dataset')
        class MyDatasetCreator(BaseDataLoaderCreator):
            def create_dataset(self):
                # 用户自定义逻辑
                pass
            
            def create_transforms(self):
                # 用户自定义变换
                pass
    """
    return dataloader_creator(name)


def get_available_creators() -> List[str]:
    """获取所有可用的DataLoader创建器"""
    return dataloader_factory.list_creators()


def validate_dataloader_config(config: Dict[str, Any]) -> bool:
    """
    验证DataLoader配置
    
    Args:
        config: DataLoader配置
        
    Returns:
        是否有效
    """
    required_fields = ['type', 'dataset', 'loader_params']
    
    for field in required_fields:
        if field not in config:
            logger.error(f"Missing required field: {field}")
            return False
    
    # 验证dataset配置
    dataset_config = config['dataset']
    if 'name' not in dataset_config:
        logger.error("Missing 'name' in dataset config")
        return False
    
    # 验证loader_params
    loader_params = config['loader_params']
    if 'batch_size' not in loader_params:
        logger.error("Missing 'batch_size' in loader_params")
        return False
    
    return True
