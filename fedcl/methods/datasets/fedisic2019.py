"""
FedISIC2019数据集实现
fedcl/methods/datasets/fedisic2019.py

ISIC 2019皮肤病变分类数据集（联邦学习版本）
注意：需要手动下载数据集
下载地址：https://challenge.isic-archive.com/data/ 或 Kaggle
"""
from typing import Dict, Any
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pandas as pd

from fedcl.api.decorators import dataset
from fedcl.methods.datasets.base import FederatedDataset


class ISIC2019Dataset(Dataset):
    """ISIC2019 PyTorch数据集包装器"""

    def __init__(self, root: str, train: bool = True, transform=None):
        """
        Args:
            root: ISIC2019数据集根目录
            train: True表示训练集，False表示测试集
            transform: 数据转换
        """
        self.root = root
        self.train = train
        self.transform = transform

        # 类别定义
        self.classes = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # 数据文件路径
        if train:
            csv_file = os.path.join(root, 'ISIC_2019_Training_GroundTruth.csv')
            img_dir = os.path.join(root, 'ISIC_2019_Training_Input')
        else:
            csv_file = os.path.join(root, 'ISIC_2019_Test_GroundTruth.csv')
            img_dir = os.path.join(root, 'ISIC_2019_Test_Input')

        # 检查文件是否存在
        if not os.path.exists(csv_file):
            raise RuntimeError(
                f"ISIC2019数据集文件不存在: {csv_file}\n"
                f"请从以下地址下载ISIC2019数据集：\n"
                f"  - https://challenge.isic-archive.com/data/\n"
                f"  - https://www.kaggle.com/c/isic-2019/data\n"
                f"并解压到: {root}"
            )

        # 读取CSV文件
        self.metadata = pd.read_csv(csv_file)
        self.img_dir = img_dir

        # 准备样本列表
        self.samples = []
        for idx, row in self.metadata.iterrows():
            img_name = row['image']
            img_path = os.path.join(img_dir, f"{img_name}.jpg")

            # 找到标签（one-hot编码）
            label = -1
            for cls in self.classes:
                if cls in row and row[cls] == 1.0:
                    label = self.class_to_idx[cls]
                    break

            if label >= 0 and os.path.exists(img_path):
                self.samples.append((img_path, label))

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
    name='FedISIC2019',
    description='ISIC 2019皮肤病变分类数据集',
    dataset_type='medical_image_classification',
    num_classes=9
)
class FedISIC2019Dataset(FederatedDataset):
    """FedISIC2019联邦数据集实现

    9个类别的皮肤病变：
    - MEL: Melanoma (黑色素瘤)
    - NV: Melanocytic nevus (黑色素痣)
    - BCC: Basal cell carcinoma (基底细胞癌)
    - AK: Actinic keratosis (光化性角化病)
    - BKL: Benign keratosis (良性角化病)
    - DF: Dermatofibroma (皮肤纤维瘤)
    - VASC: Vascular lesion (血管病变)
    - SCC: Squamous cell carcinoma (鳞状细胞癌)
    - UNK: Unknown (未知)

    训练集：25,331张图片
    测试集：8,238张图片
    图像大小：可变（会被resize到224x224）
    """

    def __init__(self, root: str = './data/isic2019', train: bool = True, download: bool = True):
        super().__init__(root, train, download)

        # 数据转换
        if train:
            # 训练集：数据增强
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            # 测试集：仅resize和标准化
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # 加载数据集
        try:
            self.dataset = ISIC2019Dataset(
                root=root,
                train=train,
                transform=transform
            )
        except RuntimeError as e:
            if download:
                print("\n" + "=" * 80)
                print("ISIC2019数据集需要手动下载：")
                print("1. 访问: https://challenge.isic-archive.com/data/")
                print("   或: https://www.kaggle.com/c/isic-2019/data")
                print("2. 下载以下文件：")
                print("   - ISIC_2019_Training_Input.zip (训练图像)")
                print("   - ISIC_2019_Training_GroundTruth.csv (训练标签)")
                print("   - ISIC_2019_Test_Input.zip (测试图像)")
                print("   - ISIC_2019_Test_GroundTruth.csv (测试标签)")
                print(f"3. 解压到: {root}")
                print("=" * 80 + "\n")
            raise e

        # 设置属性
        self.num_classes = 9
        self.input_shape = (3, 224, 224)

    def get_statistics(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        return {
            'dataset_name': 'FedISIC2019',
            'num_samples': len(self.dataset),
            'num_classes': self.num_classes,
            'input_shape': self.input_shape,
            'train': self.train,
        }
