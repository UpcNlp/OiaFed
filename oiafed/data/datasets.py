"""
数据集抽象

设计原则：
1. 直接使用 torch.utils.data.Dataset，不重复定义
2. 只提供联邦学习特定的工具（划分、采样等）
3. 支持 Hugging Face Datasets
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from torch.utils.data import Dataset as TorchDataset


# ========== 数据集工具函数 ==========

def create_dataloader(
    dataset: "TorchDataset",
    batch_size: int = 32,
    shuffle: bool = True,
    **kwargs
):
    """
    创建 PyTorch DataLoader

    Args:
        dataset: torch.utils.data.Dataset
        batch_size: 批次大小
        shuffle: 是否打乱
        **kwargs: 其他 DataLoader 参数

    Returns:
        torch.utils.data.DataLoader
    """
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        **kwargs
    )


def get_dataset_info(dataset: "TorchDataset") -> Dict[str, Any]:
    """
    获取数据集信息

    Args:
        dataset: torch.utils.data.Dataset

    Returns:
        数据集信息字典
    """
    info = {
        "num_samples": len(dataset),
        "type": type(dataset).__name__,
    }

    # 尝试获取更多信息
    if hasattr(dataset, 'classes'):
        info["num_classes"] = len(dataset.classes)
        info["classes"] = dataset.classes

    if hasattr(dataset, 'targets'):
        import numpy as np
        targets = np.array(dataset.targets)
        info["class_distribution"] = {
            int(c): int((targets == c).sum())
            for c in np.unique(targets)
        }

    return info


# ========== 联邦学习数据提供者 ==========

class DataProvider:
    """
    联邦学习数据提供者

    封装训练集和测试集，提供统一接口

    Example:
        # 使用 PyTorch 数据集
        from torchvision.datasets import MNIST
        from torchvision.transforms import ToTensor

        train_dataset = MNIST(root='./data', train=True, transform=ToTensor())
        test_dataset = MNIST(root='./data', train=False, transform=ToTensor())

        provider = DataProvider(train_dataset, test_dataset)
        train_loader = provider.get_train_loader(batch_size=32)
        test_loader = provider.get_test_loader(batch_size=64)
    """

    def __init__(
        self,
        train_dataset: Optional["TorchDataset"] = None,
        test_dataset: Optional["TorchDataset"] = None,
    ):
        """
        初始化数据提供者

        Args:
            train_dataset: 训练数据集（torch.utils.data.Dataset）
            test_dataset: 测试数据集（torch.utils.data.Dataset）
        """
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def get_train_loader(self, batch_size: int = 32, shuffle: bool = True, **kwargs):
        """获取训练 DataLoader"""
        if self.train_dataset is None:
            raise ValueError("No training dataset available")
        return create_dataloader(self.train_dataset, batch_size, shuffle, **kwargs)

    def get_test_loader(self, batch_size: int = 64, shuffle: bool = False, **kwargs):
        """获取测试 DataLoader"""
        if self.test_dataset is None:
            raise ValueError("No test dataset available")
        return create_dataloader(self.test_dataset, batch_size, shuffle, **kwargs)

    def get_info(self) -> Dict[str, Any]:
        """获取数据集信息"""
        info = {}

        if self.train_dataset is not None:
            info["train"] = get_dataset_info(self.train_dataset)

        if self.test_dataset is not None:
            info["test"] = get_dataset_info(self.test_dataset)

        return info

    def __repr__(self) -> str:
        info = self.get_info()
        train_samples = info.get("train", {}).get("num_samples", 0)
        test_samples = info.get("test", {}).get("num_samples", 0)
        return f"DataProvider(train={train_samples}, test={test_samples})"


# ========== Hugging Face Datasets 支持 ==========

def create_hf_dataloader(
    dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    collate_fn=None,
    **kwargs
):
    """
    为 Hugging Face Dataset 创建 DataLoader

    Args:
        dataset: datasets.Dataset
        batch_size: 批次大小
        shuffle: 是否打乱
        collate_fn: 数据整理函数
        **kwargs: 其他参数

    Returns:
        torch.utils.data.DataLoader

    Example:
        from datasets import load_dataset

        dataset = load_dataset("imdb", split="train")
        loader = create_hf_dataloader(dataset, batch_size=16)
    """
    from torch.utils.data import DataLoader

    # Hugging Face Dataset 可以直接用作 PyTorch Dataset
    dataset.set_format(type="torch")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        **kwargs
    )


class HuggingFaceDataProvider(DataProvider):
    """
    Hugging Face 数据提供者

    Example:
        from datasets import load_dataset

        dataset = load_dataset("imdb")
        provider = HuggingFaceDataProvider(
            train_dataset=dataset["train"],
            test_dataset=dataset["test"]
        )
    """

    def get_train_loader(self, batch_size: int = 32, shuffle: bool = True, **kwargs):
        """获取训练 DataLoader"""
        if self.train_dataset is None:
            raise ValueError("No training dataset available")
        return create_hf_dataloader(self.train_dataset, batch_size, shuffle, **kwargs)

    def get_test_loader(self, batch_size: int = 64, shuffle: bool = False, **kwargs):
        """获取测试 DataLoader"""
        if self.test_dataset is None:
            raise ValueError("No test dataset available")
        return create_hf_dataloader(self.test_dataset, batch_size, shuffle, **kwargs)
