# fedcl/data/dataset_handlers.py
"""
数据集处理器注册系统

提供数据集处理器的注册和管理功能，支持不同类型数据集的加载和处理。
"""

from typing import Dict, Any, Optional, Callable
import torch
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
from loguru import logger

from ..registry.component_registry import ComponentRegistry


class DatasetHandlerRegistry:
    """数据集处理器注册表"""
    
    def __init__(self):
        self._handlers: Dict[str, Callable] = {}
        
    def register(self, name: str, handler: Callable):
        """注册数据集处理器"""
        self._handlers[name] = handler
        logger.debug(f"Registered dataset handler: {name}")
        
    def get_handler(self, name: str) -> Optional[Callable]:
        """获取数据集处理器"""
        return self._handlers.get(name)
        
    def list_handlers(self) -> list:
        """列出所有已注册的处理器"""
        return list(self._handlers.keys())


# 全局数据集处理器注册表
dataset_registry = DatasetHandlerRegistry()


def dataset_handler(name: str):
    """数据集处理器装饰器"""
    def decorator(handler_class):
        dataset_registry.register(name, handler_class)
        return handler_class
    return decorator


@dataset_handler('mnist_handler')
class MNISTDatasetHandler:
    """MNIST数据集处理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = config.get('name', 'mnist')
        self.download_dir = config.get('download_dir', './data')
        
    def load_dataset(self, client_id: str, split_config: Dict[str, Any]) -> DataLoader:
        """加载MNIST数据集"""
        try:
            # 定义数据变换
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            
            # 加载MNIST数据集
            train_dataset = torchvision.datasets.MNIST(
                root=self.download_dir,
                train=True,
                download=True,
                transform=transform
            )
            
            # 根据客户端ID和分割配置选择数据子集
            client_data = self._split_data_for_client(
                train_dataset, 
                client_id, 
                split_config
            )
            
            # 创建DataLoader
            dataloader = DataLoader(
                client_data,
                batch_size=self.config.get('batch_size', 32),
                shuffle=True
            )
            
            logger.debug(f"Loaded MNIST dataset for client {client_id}: {len(client_data)} samples")
            return dataloader
            
        except Exception as e:
            logger.error(f"Failed to load MNIST dataset: {e}")
            raise
    
    def _split_data_for_client(self, dataset, client_id: str, split_config: Dict[str, Any]):
        """为特定客户端分割数据"""
        num_clients = split_config.get('num_客户端', 1)
        client_idx = split_config.get('client_id', 0)
        method = split_config.get('method', 'iid')
        
        total_samples = len(dataset)
        samples_per_client = total_samples // num_clients
        
        if method == 'iid':
            # IID分割：随机分配
            start_idx = client_idx * samples_per_client
            end_idx = start_idx + samples_per_client
            indices = list(range(start_idx, min(end_idx, total_samples)))
        else:
            # 非IID分割可以在这里实现
            indices = list(range(0, min(samples_per_client, total_samples)))
        
        # 创建子集
        from torch.utils.data import Subset
        client_subset = Subset(dataset, indices)
        
        return client_subset


@dataset_handler('mock_handler')
class MockDatasetHandler:
    """模拟数据集处理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def load_dataset(self, client_id: str, split_config: Dict[str, Any]) -> DataLoader:
        """创建模拟数据集"""
        num_samples = self.config.get('num_samples', 100)
        input_dim = self.config.get('input_dim', 784)
        num_classes = self.config.get('num_classes', 10)
        
        # 创建模拟数据
        X = torch.randn(num_samples, input_dim)
        y = torch.randint(0, num_classes, (num_samples,))
        
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        logger.debug(f"Created mock dataset for client {client_id}: {num_samples} samples")
        return dataloader


def get_dataset_handler(handler_name: str, config: Dict[str, Any]):
    """获取数据集处理器实例"""
    handler_class = dataset_registry.get_handler(handler_name)
    if handler_class is None:
        logger.warning(f"Dataset handler '{handler_name}' not found, using mock handler")
        handler_class = dataset_registry.get_handler('mock_handler')
    
    return handler_class(config)
