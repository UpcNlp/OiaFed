"""
CINIC-10 数据集

从 methods/datasets/cinic10.py 迁移到 src/
支持 split 参数 (train/test/valid)

CINIC-10 (CIFAR-10 + ImageNet)数据集
注意:需要手动下载数据集到指定目录
下载地址:https://datashare.ed.ac.uk/handle/10283/3192
"""

import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from ...registry import dataset


@dataset(
    name='cinic10',
    description='CINIC-10数据集(CIFAR-10扩展版)',
    version='1.0',
    author='Federation Framework',
    dataset_type='image_classification',
    num_classes=10,
    input_shape=(32, 32, 3)
)
class CINIC10Dataset(Dataset):
    """
    CINIC-10 数据集 - 标准 PyTorch Dataset

    CINIC-10 (CIFAR-10 + ImageNet)
    10个类别:airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

    训练集:90,000张图片
    验证集:90,000张图片
    测试集:90,000张图片
    图像大小:32x32x3 (RGB)
    """

    # 类别定义
    CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

    def __init__(
        self,
        data_dir: str = "./data/cinic10",
        split: str = "train",
        download: bool = True,
        augmentation: bool = True,
    ):
        """
        Args:
            data_dir: 数据目录(应包含train/valid/test子目录)
            split: 数据集划分 ("train" / "test" / "valid")
            download: 是否下载数据(CINIC10需要手动下载)
            augmentation: 是否使用数据增强(仅对train split生效)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.augmentation = augmentation

        # CINIC10有3个split:train/valid/test
        # 使用框架的split名称映射
        if self.split == "train":
            cinic_split = "train"
        elif self.split == "valid":
            cinic_split = "valid"
        else:  # test
            cinic_split = "test"

        # 数据目录
        self.split_dir = self.data_dir / cinic_split

        # 检查目录是否存在
        if not self.split_dir.exists():
            error_msg = (
                f"CINIC10数据集目录不存在:{self.split_dir}\n"
                f"请从 https://datashare.ed.ac.uk/handle/10283/3192 下载CINIC10数据集\n"
                f"并解压到:{data_dir}\n"
                f"目录结构应为:{data_dir}/train/, {data_dir}/valid/, {data_dir}/test/"
            )
            if download:
                print("\n" + "=" * 80)
                print("CINIC10数据集需要手动下载:")
                print("1. 访问:https://datashare.ed.ac.uk/handle/10283/3192")
                print("2. 下载CINIC-10数据集")
                print(f"3. 解压到:{data_dir}")
                print(f"4. 确保目录结构为:{data_dir}/train/, {data_dir}/valid/, {data_dir}/test/")
                print("=" * 80 + "\n")
            raise RuntimeError(error_msg)

        # 类别到索引映射
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.CLASSES)}

        # 加载所有图像路径和标签
        self.samples = []
        for class_name in self.CLASSES:
            class_dir = self.split_dir / class_name
            if not class_dir.exists():
                continue

            class_idx = self.class_to_idx[class_name]
            for img_file in class_dir.glob('*.png'):
                if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    self.samples.append((str(img_file), class_idx))

        # 数据转换
        is_train = self.split == "train"
        if is_train and self.augmentation:
            # 训练集:数据增强
            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.47889522, 0.47227842, 0.43047404],
                    std=[0.24205776, 0.23828046, 0.25874835]
                )
            ])
        else:
            # 测试集或验证集:仅标准化
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.47889522, 0.47227842, 0.43047404],
                    std=[0.24205776, 0.23828046, 0.25874835]
                )
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # 加载图像
        img = Image.open(img_path).convert('RGB')

        # 应用转换
        if self.transform is not None:
            img = self.transform(img)

        return img, label
