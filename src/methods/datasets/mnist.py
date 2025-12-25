"""
MNIST 数据集

标准 PyTorch Dataset 实现，支持联邦学习数据划分
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple
from pathlib import Path
from ...registry import dataset

"""
MNIST 数据集

标准 PyTorch Dataset 实现
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple
from pathlib import Path
from ...registry import dataset


@dataset(
    name='mnist',
    description='MNIST手写数字数据集',
    version='1.0',
    author='Federation Framework',
    dataset_type='image_classification',
    num_classes=10,
    input_shape=(28, 28, 1)
)
class MNISTDataset(Dataset):
    """
    MNIST 数据集 - 标准 PyTorch Dataset
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
            split: 数据集划分 ("train" / "test")
            download: 是否下载数据
        """
        self.data_dir = Path(data_dir)
        self.split = split

        # 加载数据
        self.data, self.targets = self._load_data(download)

    def _load_data(self, download: bool) -> Tuple[np.ndarray, np.ndarray]:
        """加载 MNIST 数据"""
        import torchvision

        is_train = self.split in ("train", "valid")
        
        dataset = torchvision.datasets.MNIST(
            root=str(self.data_dir),
            train=is_train,
            download=download,
            transform=None
        )

        data = dataset.data.numpy()
        targets = dataset.targets.numpy()

        # 归一化到 [0, 1]
        data = data.astype(np.float32) / 255.0

        # 添加通道维度: (N, 28, 28) -> (N, 28, 28, 1)
        data = np.expand_dims(data, -1)

        return data, targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.targets[idx]

        image = torch.from_numpy(image).permute(2, 0, 1).float()
        label = torch.tensor(label, dtype=torch.long)

        return image, label