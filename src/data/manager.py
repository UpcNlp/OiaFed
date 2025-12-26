"""
DataManager - 数据管理器

管理数据集的加载和划分
"""

from typing import Any, Dict, List, Optional, Tuple

from .provider import DataProvider, InMemoryDataProvider
from .partitioner import Partitioner, IIDPartitioner
from ..infra import get_module_logger

logger = get_module_logger(__name__)


class DataManager:
    """
    数据管理器
    
    职责：
    - 加载数据集
    - 划分数据（本地模式）
    - 管理数据分配
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        partitioner: Optional[Partitioner] = None,
    ):
        """
        初始化数据管理器
        
        Args:
            config: 数据配置
            partitioner: 划分器（可选）
        """
        self._config = config
        self._partitioner = partitioner or IIDPartitioner()
        
        # 数据集
        self._dataset: Optional[Tuple[Any, Any]] = None
        self._labels: Optional[List[Any]] = None
        
        # 划分结果
        self._partitions: Dict[str, List[int]] = {}
        
        # 数据集信息
        self._dataset_info: Dict[str, Any] = {}
    
    @property
    def dataset(self) -> Optional[Tuple[Any, Any]]:
        """获取数据集"""
        return self._dataset
    
    @property
    def partitions(self) -> Dict[str, List[int]]:
        """获取划分结果"""
        return self._partitions
    
    def load_dataset(self, path: Optional[str] = None) -> Tuple[Any, Any]:
        """
        加载数据集
        
        Args:
            path: 数据集路径（覆盖配置）
            
        Returns:
            (data, labels) 元组
        """
        path = path or self._config.get("path")
        dataset_name = self._config.get("dataset")
        
        if dataset_name:
            # 加载内置数据集
            self._dataset, self._labels = self._load_builtin_dataset(dataset_name)
        elif path:
            # 从文件加载
            self._dataset, self._labels = self._load_from_file(path)
        else:
            raise ValueError("No dataset specified")
        
        self._dataset_info = {
            "num_samples": len(self._dataset[0]),
            "path": path,
            "dataset": dataset_name,
        }
        
        logger.info(f"Loaded dataset: {self._dataset_info['num_samples']} samples")
        return self._dataset
    
    def partition_data(
        self,
        num_clients: int,
        partitioner: Optional[Partitioner] = None,
    ) -> Dict[str, List[int]]:
        """
        划分数据
        
        Args:
            num_clients: 客户端数量
            partitioner: 划分器（覆盖默认）
            
        Returns:
            划分结果 {client_id: [indices]}
        """
        if self._dataset is None:
            self.load_dataset()
        
        partitioner = partitioner or self._partitioner
        dataset_size = len(self._dataset[0])
        
        # 执行划分
        partition_by_idx = partitioner.partition(
            dataset_size=dataset_size,
            num_clients=num_clients,
            labels=self._labels,
        )
        
        # 转换为字符串 key
        self._partitions = {
            f"client_{idx}": indices
            for idx, indices in partition_by_idx.items()
        }
        
        # 记录统计
        for client_id, indices in self._partitions.items():
            logger.debug(f"Partition {client_id}: {len(indices)} samples")
        
        return self._partitions
    
    def get_partition(self, client_id: str) -> Optional[List[int]]:
        """
        获取客户端的数据分区
        
        Args:
            client_id: 客户端 ID
            
        Returns:
            样本索引列表
        """
        return self._partitions.get(client_id)
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """获取数据集信息"""
        return self._dataset_info
    
    def create_provider(
        self,
        client_id: str,
        batch_size: int = 32,
    ) -> Optional[DataProvider]:
        """
        为客户端创建数据提供者
        
        Args:
            client_id: 客户端 ID
            batch_size: 批大小
            
        Returns:
            DataProvider 实例
        """
        indices = self.get_partition(client_id)
        if indices is None or self._dataset is None:
            return None
        
        x, y = self._dataset
        
        # 提取子集
        subset_x = [x[i] for i in indices]
        subset_y = [y[i] for i in indices]
        
        return InMemoryDataProvider(
            train_x=subset_x,
            train_y=subset_y,
            batch_size=batch_size,
        )
    
    def _load_builtin_dataset(self, name: str) -> Tuple[Tuple[Any, Any], List[Any]]:
        """加载内置数据集"""
        name = name.lower()
        
        if name == "mnist":
            return self._load_mnist()
        elif name == "cifar10":
            return self._load_cifar10()
        else:
            raise ValueError(f"Unknown builtin dataset: {name}")
    
    def _load_mnist(self) -> Tuple[Tuple[Any, Any], List[Any]]:
        """加载 MNIST 数据集"""
        try:
            from torchvision import datasets, transforms
            import numpy as np
            
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            
            train_dataset = datasets.MNIST(
                root=self._config.get("path", "./data"),
                train=True,
                download=True,
                transform=transform,
            )
            
            # 转换为 numpy
            x = np.array([train_dataset[i][0].numpy() for i in range(len(train_dataset))])
            y = np.array([train_dataset[i][1] for i in range(len(train_dataset))])
            
            return (x, y), y.tolist()
            
        except ImportError:
            logger.warning("torchvision not available, generating dummy data")
            return self._generate_dummy_data()
    
    def _load_cifar10(self) -> Tuple[Tuple[Any, Any], List[Any]]:
        """加载 CIFAR-10 数据集"""
        try:
            from torchvision import datasets, transforms
            import numpy as np
            
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            
            train_dataset = datasets.CIFAR10(
                root=self._config.get("path", "./data"),
                train=True,
                download=True,
                transform=transform,
            )
            
            x = np.array([train_dataset[i][0].numpy() for i in range(len(train_dataset))])
            y = np.array([train_dataset[i][1] for i in range(len(train_dataset))])
            
            return (x, y), y.tolist()
            
        except ImportError:
            logger.warning("torchvision not available, generating dummy data")
            return self._generate_dummy_data()
    
    def _load_from_file(self, path: str) -> Tuple[Tuple[Any, Any], List[Any]]:
        """从文件加载数据集"""
        import os
        
        if path.endswith(".csv"):
            return self._load_csv(path)
        elif path.endswith(".npz"):
            return self._load_npz(path)
        elif os.path.isdir(path):
            return self._load_directory(path)
        else:
            raise ValueError(f"Unsupported file format: {path}")
    
    def _load_csv(self, path: str) -> Tuple[Tuple[Any, Any], List[Any]]:
        """加载 CSV 文件"""
        try:
            import pandas as pd
            import numpy as np
            
            df = pd.read_csv(path)
            
            # 假设最后一列是标签
            y = df.iloc[:, -1].values
            x = df.iloc[:, :-1].values
            
            return (x, y), y.tolist()
        except ImportError:
            raise ImportError("pandas required for CSV loading")
    
    def _load_npz(self, path: str) -> Tuple[Tuple[Any, Any], List[Any]]:
        """加载 NPZ 文件"""
        import numpy as np
        
        data = np.load(path)
        x = data.get("x", data.get("data"))
        y = data.get("y", data.get("labels"))
        
        if x is None or y is None:
            raise ValueError("NPZ file must contain 'x'/'data' and 'y'/'labels'")
        
        return (x, y), y.tolist()
    
    def _load_directory(self, path: str) -> Tuple[Tuple[Any, Any], List[Any]]:
        """从目录加载（图像分类格式）"""
        try:
            from torchvision import datasets, transforms
            import numpy as np
            
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ])
            
            dataset = datasets.ImageFolder(root=path, transform=transform)
            
            x = np.array([dataset[i][0].numpy() for i in range(len(dataset))])
            y = np.array([dataset[i][1] for i in range(len(dataset))])
            
            return (x, y), y.tolist()
        except ImportError:
            raise ImportError("torchvision required for directory loading")
    
    def _generate_dummy_data(
        self,
        num_samples: int = 1000,
        num_features: int = 784,
        num_classes: int = 10,
    ) -> Tuple[Tuple[Any, Any], List[Any]]:
        """生成虚拟数据（用于测试）"""
        import random
        
        x = [[random.random() for _ in range(num_features)] for _ in range(num_samples)]
        y = [random.randint(0, num_classes - 1) for _ in range(num_samples)]
        
        return (x, y), y
