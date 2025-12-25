"""
SVHN 数据集

从 methods/datasets/svhn.py 迁移到 src/
支持 split 参数 (train/test/valid)
"""

import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
from ...registry import dataset


@dataset(
    name='svhn',
    description='SVHN街道视图房屋号码数据集',
    version='1.0',
    author='Federation Framework',
    dataset_type='image_classification',
    num_classes=10,
    input_shape=(32, 32, 3)
)
class SVHNDataset(Dataset):
    """
    SVHN 数据集 - 标准 PyTorch Dataset

    Street View House Numbers (SVHN)
    10个类别：数字0-9
    图像大小：32x32x3 (RGB)

    训练集：73,257张图片
    测试集：26,032张图片
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

        # SVHN使用自己的split命名: 'train' or 'test'
        svhn_split = 'train' if self.split in ("train", "valid") else 'test'

        # 数据转换 (SVHN的标准化参数)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4377, 0.4438, 0.4728],
                std=[0.1980, 0.2010, 0.1970]
            )
        ])

        # 加载 SVHN 数据集
        self.dataset = torchvision.datasets.SVHN(
            root=str(self.data_dir),
            split=svhn_split,
            download=download,
            transform=transform
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
