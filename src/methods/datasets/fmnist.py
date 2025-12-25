"""
Fashion-MNIST 数据集

从 methods/datasets/fmnist.py 迁移到 src/
支持 split 参数 (train/test/valid)
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple
from pathlib import Path
from ...registry import dataset


@dataset(
    name='fmnist',
    description='Fashion-MNIST服装图像数据集',
    version='1.0',
    author='Federation Framework',
    dataset_type='image_classification',
    num_classes=10,
    input_shape=(28, 28, 1)
)
class FashionMNISTDataset(Dataset):
    """
    Fashion-MNIST 数据集 - 标准 PyTorch Dataset

    10个类别：
    0: T-shirt/top, 1: Trouser, 2: Pullover, 3: Dress, 4: Coat,
    5: Sandal, 6: Shirt, 7: Sneaker, 8: Bag, 9: Ankle boot
    """

    def __init__(
        self,
        data_dir: str = "./data",
        split: str = "train",
        download: bool = True,
    ):
        """
        Args:
            data_dir: 数据目录
            split: 数据集划分 ("train" / "test" / "valid")
            download: 是否下载数据
        """
        self.data_dir = Path(data_dir)
        self.split = split

        # 加载数据
        self.data, self.targets = self._load_data(download)

    def _load_data(self, download: bool) -> Tuple[np.ndarray, np.ndarray]:
        """根据 split 加载 Fashion-MNIST 数据"""
        import torchvision

        is_train = self.split in ("train", "valid")

        dataset = torchvision.datasets.FashionMNIST(
            root=str(self.data_dir),
            train=is_train,
            download=download,
            transform=None
        )

        data = dataset.data.numpy()
        targets = dataset.targets.numpy()

        # 归一化到 [0, 1]
        data = data.astype(np.float32) / 255.0

        # Fashion-MNIST的标准化参数
        mean = 0.2860
        std = 0.3530
        data = (data - mean) / std

        # 添加通道维度: (N, 28, 28) -> (N, 28, 28, 1)
        data = np.expand_dims(data, -1)

        return data, targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.targets[idx]

        # 转换为 torch tensor: (H, W, C) -> (C, H, W)
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        label = torch.tensor(label, dtype=torch.long)

        return image, label
