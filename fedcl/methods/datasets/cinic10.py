"""
CINIC10数据集实现
fedcl/methods/datasets/cinic10.py

CINIC-10 (CIFAR-10 + ImageNet)数据集
注意：需要手动下载数据集到指定目录
下载地址：https://datashare.ed.ac.uk/handle/10283/3192
"""
from typing import Dict, Any
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from fedcl.api.decorators import dataset
from fedcl.methods.datasets.base import FederatedDataset


class CINIC10Dataset(Dataset):
    """CINIC10 PyTorch数据集包装器"""

    def __init__(self, root: str, train: bool = True, transform=None):
        """
        Args:
            root: CINIC10数据集根目录（应包含train/valid/test子目录）
            train: True表示训练集，False表示测试集
            transform: 数据转换
        """
        self.root = root
        self.train = train
        self.transform = transform

        # CINIC10类别
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                        'dog', 'frog', 'horse', 'ship', 'truck']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # 确定使用哪个split
        split = 'train' if train else 'test'
        self.split_dir = os.path.join(root, split)

        # 检查目录是否存在
        if not os.path.exists(self.split_dir):
            raise RuntimeError(
                f"CINIC10数据集目录不存在: {self.split_dir}\n"
                f"请从 https://datashare.ed.ac.uk/handle/10283/3192 下载CINIC10数据集\n"
                f"并解压到: {root}"
            )

        # 加载所有图像路径和标签
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.split_dir, class_name)
            if not os.path.exists(class_dir):
                continue

            class_idx = self.class_to_idx[class_name]
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))

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


@dataset(
    name='CINIC10',
    description='CINIC-10数据集（CIFAR-10扩展版）',
    dataset_type='image_classification',
    num_classes=10
)
class CINIC10FederatedDataset(FederatedDataset):
    """CINIC10联邦数据集实现

    10个类别（与CIFAR-10相同）：
    airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

    训练集：90,000张图片
    验证集：90,000张图片
    测试集：90,000张图片
    图像大小：32x32x3 (RGB)
    """

    def __init__(self, root: str = './data/cinic10', train: bool = True, download: bool = True):
        super().__init__(root, train, download)

        # 数据转换
        if train:
            # 训练集：数据增强
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.47889522, 0.47227842, 0.43047404),
                                     (0.24205776, 0.23828046, 0.25874835))
            ])
        else:
            # 测试集：仅标准化
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.47889522, 0.47227842, 0.43047404),
                                     (0.24205776, 0.23828046, 0.25874835))
            ])

        # 加载CINIC10数据集
        try:
            self.dataset = CINIC10Dataset(
                root=root,
                train=train,
                transform=transform
            )
        except RuntimeError as e:
            if download:
                print("\n" + "=" * 80)
                print("CINIC10数据集需要手动下载：")
                print("1. 访问: https://datashare.ed.ac.uk/handle/10283/3192")
                print("2. 下载CINIC-10数据集")
                print(f"3. 解压到: {root}")
                print("4. 确保目录结构为: {root}/train/, {root}/valid/, {root}/test/")
                print("=" * 80 + "\n")
            raise e

        # 设置属性
        self.num_classes = 10
        self.input_shape = (3, 32, 32)

    def get_statistics(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        return {
            'dataset_name': 'CINIC10',
            'num_samples': len(self.dataset),
            'num_classes': self.num_classes,
            'input_shape': self.input_shape,
            'train': self.train,
        }
