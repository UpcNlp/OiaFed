"""
FedISIC2019数据集

从 methods/datasets/fedisic2019.py 迁移到 src/
支持 split 参数 (train/test/valid)

ISIC 2019皮肤病变分类数据集(联邦学习版本)
注意:需要手动下载数据集
下载地址:https://challenge.isic-archive.com/data/ 或 Kaggle
"""

import os
from pathlib import Path
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from ...registry import dataset


@dataset(
    name='fedisic2019',
    description='ISIC 2019皮肤病变分类数据集',
    version='1.0',
    author='Federation Framework',
    dataset_type='medical_image_classification',
    num_classes=9,
    input_shape=(224, 224, 3)
)
class FedISIC2019Dataset(Dataset):
    """
    FedISIC2019数据集 - 标准 PyTorch Dataset

    9个类别的皮肤病变:
    - MEL:Melanoma (黑色素瘤)
    - NV:Melanocytic nevus (黑色素痣)
    - BCC:Basal cell carcinoma (基底细胞癌)
    - AK:Actinic keratosis (光化性角化病)
    - BKL:Benign keratosis (良性角化病)
    - DF:Dermatofibroma (皮肤纤维瘤)
    - VASC:Vascular lesion (血管病变)
    - SCC:Squamous cell carcinoma (鳞状细胞癌)
    - UNK:Unknown (未知)

    训练集:25,331张图片
    测试集:8,238张图片
    图像大小:可变(会被resize到224x224)
    """

    # 类别定义
    CLASSES = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']

    def __init__(
        self,
        data_dir: str = "./data/isic2019",
        split: str = "train",
        download: bool = True,
        augmentation: bool = True,
    ):
        """
        Args:
            data_dir: 数据目录
            split: 数据集划分 ("train" / "test" / "valid")
            download: 是否下载数据(ISIC2019需要手动下载)
            augmentation: 是否使用数据增强(仅对train split生效)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.augmentation = augmentation

        # 类别到索引映射
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.CLASSES)}

        # 确定文件路径
        is_train = self.split in ("train", "valid")
        if is_train:
            csv_file = self.data_dir / 'ISIC_2019_Training_GroundTruth.csv'
            img_dir = self.data_dir / 'ISIC_2019_Training_Input'
        else:
            csv_file = self.data_dir / 'ISIC_2019_Test_GroundTruth.csv'
            img_dir = self.data_dir / 'ISIC_2019_Test_Input'

        # 检查文件是否存在
        if not csv_file.exists():
            error_msg = (
                f"ISIC2019数据集文件不存在:{csv_file}\n"
                f"请从以下地址下载ISIC2019数据集:\n"
                f"  - https://challenge.isic-archive.com/data/\n"
                f"  - https://www.kaggle.com/c/isic-2019/data\n"
                f"下载以下文件:\n"
                f"  - ISIC_2019_Training_Input.zip (训练图像)\n"
                f"  - ISIC_2019_Training_GroundTruth.csv (训练标签)\n"
                f"  - ISIC_2019_Test_Input.zip (测试图像)\n"
                f"  - ISIC_2019_Test_GroundTruth.csv (测试标签)\n"
                f"并解压到:{data_dir}"
            )
            if download:
                print("\n" + "=" * 80)
                print("ISIC2019数据集需要手动下载:")
                print("1. 访问:https://challenge.isic-archive.com/data/")
                print("   或:https://www.kaggle.com/c/isic-2019/data")
                print("2. 下载以下文件:")
                print("   - ISIC_2019_Training_Input.zip (训练图像)")
                print("   - ISIC_2019_Training_GroundTruth.csv (训练标签)")
                print("   - ISIC_2019_Test_Input.zip (测试图像)")
                print("   - ISIC_2019_Test_GroundTruth.csv (测试标签)")
                print(f"3. 解压到:{data_dir}")
                print("=" * 80 + "\n")
            raise RuntimeError(error_msg)

        # 读取CSV文件
        self.metadata = pd.read_csv(csv_file)
        self.img_dir = img_dir

        # 准备样本列表
        self.samples = []
        for idx, row in self.metadata.iterrows():
            img_name = row['image']
            img_path = img_dir / f"{img_name}.jpg"

            # 找到标签(one-hot编码)
            label = -1
            for cls in self.CLASSES:
                if cls in row and row[cls] == 1.0:
                    label = self.class_to_idx[cls]
                    break

            if label >= 0 and img_path.exists():
                self.samples.append((str(img_path), label))

        # 数据转换
        if is_train and self.augmentation:
            # 训练集:数据增强
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            # 测试集或验证集:仅resize和标准化
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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
