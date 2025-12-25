"""
CIFAR-10 数据集

从 methods/datasets/cifar10.py 迁移到 src/
支持 split 参数 (train/test/valid)
"""

import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
from ...registry import dataset


@dataset(
    name='cifar10',
    description='CIFAR-10图像分类数据集',
    version='1.0',
    author='Federation Framework',
    dataset_type='image_classification',
    num_classes=10,
    input_shape=(32, 32, 3)
)
class CIFAR10Dataset(Dataset):
    """
    CIFAR-10 数据集 - 标准 PyTorch Dataset

    10个类别：
    0: airplane, 1: automobile, 2: bird, 3: cat, 4: deer,
    5: dog, 6: frog, 7: horse, 8: ship, 9: truck

    图像大小: 32x32x3 (RGB)
    训练集: 50,000张图像
    测试集: 10,000张图像
    """

    def __init__(
        self,
        data_dir: str = "./data",
        split: str = "train",
        download: bool = True,
        augmentation: bool = True,  # 是否使用数据增强 (仅训练时)
    ):
        """
        Args:
            data_dir: 数据目录
            split: 数据集划分 ("train" / "test" / "valid")
            download: 是否下载数据
            augmentation: 是否使用数据增强 (仅对 train split 生效)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.augmentation = augmentation

        # 根据 split 确定是否为训练集
        is_train = self.split in ("train", "valid")

        # 数据转换
        if is_train and self.augmentation:
            # 训练时使用数据增强
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2023, 0.1994, 0.2010]
                )
            ])
        else:
            # 测试时或不使用数据增强时
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2023, 0.1994, 0.2010]
                )
            ])

        # 加载 CIFAR-10 数据集
        self.dataset = torchvision.datasets.CIFAR10(
            root=str(self.data_dir),
            train=is_train,
            download=download,
            transform=transform
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
