"""
CIFAR-100 数据集

从 methods/datasets/cifar100.py 迁移到 src/
支持 split 参数 (train/test/valid)
"""

import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
from ...registry import dataset


@dataset(
    name='cifar100',
    description='CIFAR-100图像分类数据集',
    version='1.0',
    author='Federation Framework',
    dataset_type='image_classification',
    num_classes=100,
    input_shape=(32, 32, 3)
)
class CIFAR100Dataset(Dataset):
    """
    CIFAR-100 数据集 - 标准 PyTorch Dataset

    100个类别，分为20个超类
    图像大小: 32x32x3 (RGB)
    训练集: 50,000张图像
    测试集: 10,000张图像
    """

    def __init__(
        self,
        data_dir: str = "./data",
        split: str = "train",
        download: bool = True,
        augmentation: bool = True,
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
                    mean=[0.5071, 0.4867, 0.4408],
                    std=[0.2675, 0.2565, 0.2761]
                )
            ])
        else:
            # 测试时或不使用数据增强时
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5071, 0.4867, 0.4408],
                    std=[0.2675, 0.2565, 0.2761]
                )
            ])

        # 加载 CIFAR-100 数据集
        self.dataset = torchvision.datasets.CIFAR100(
            root=str(self.data_dir),
            train=is_train,
            download=download,
            transform=transform
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
