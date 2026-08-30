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
        max_samples: int | None = None,
        subset_seed: int = 42,
        transform_profile: str = "standard",
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
        self.transform_profile = transform_profile.lower()
        if self.transform_profile not in {"standard", "fedsra", "fafi", "oneshot_half"}:
            raise ValueError(
                "unsupported CIFAR-10 transform_profile"
            )

        # 根据 split 确定是否为训练集
        is_train = self.split in ("train", "valid")

        # 数据转换
        if self.transform_profile == "fafi":
            # The FAFI reference applies its two stochastic views inside the
            # learner and keeps CIFAR-10 evaluation tensors unnormalised.
            transform = transforms.ToTensor()
        elif self.transform_profile == "oneshot_half":
            operations = []
            if is_train and self.augmentation:
                operations.extend([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()])
            operations.extend([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            transform = transforms.Compose(operations)
        elif is_train and self.augmentation and self.transform_profile == "fedsra":
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2470, 0.2435, 0.2616],
                ),
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
            ])
        elif is_train and self.augmentation:
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
        elif self.transform_profile == "fedsra":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2470, 0.2435, 0.2616],
                ),
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
        self.indices = None
        if max_samples is not None and int(max_samples) < len(self.dataset):
            if int(max_samples) <= 0:
                raise ValueError("max_samples must be positive")
            generator = torch.Generator().manual_seed(int(subset_seed))
            self.indices = torch.randperm(
                len(self.dataset), generator=generator
            )[:int(max_samples)].tolist()

        # Expose labels so OiaFed partitioners and FedSRA can obtain class
        # counts without iterating through augmented images.
        if self.indices is None:
            self.targets = self.dataset.targets
        else:
            self.targets = [self.dataset.targets[index] for index in self.indices]

    def __len__(self):
        return len(self.indices) if self.indices is not None else len(self.dataset)

    def __getitem__(self, idx):
        source_index = self.indices[idx] if self.indices is not None else idx
        return self.dataset[source_index]
