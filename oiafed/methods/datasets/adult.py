"""
Adult (Census Income)数据集

从 methods/datasets/adult.py 迁移到 src/
支持 split 参数 (train/test/valid)

Adult数据集用于预测收入是否超过50K
"""

import os
import urllib.request
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from ...registry import dataset


@dataset(
    name='adult',
    description='Adult(Census Income)数据集',
    version='1.0',
    author='Federation Framework',
    dataset_type='tabular_classification',
    num_classes=2,
    input_shape=(14,)  # 14个特征
)
class AdultDataset(Dataset):
    """
    Adult 数据集 - 标准 PyTorch Dataset

    2个类别:收入<=50K, 收入>50K
    特征:14个属性(年龄、工作类型、教育等)
    训练集:32,561个样本
    测试集:16,281个样本
    """

    # 数据集URL
    TRAIN_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    TEST_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

    # 列名
    COLUMNS = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
    ]

    # 类别特征
    CATEGORICAL_FEATURES = [
        'workclass', 'education', 'marital-status', 'occupation',
        'relationship', 'race', 'sex', 'native-country'
    ]

    def __init__(
        self,
        data_dir: str = "./data/adult",
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

        # 创建数据目录
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # 文件路径
        train_file = self.data_dir / 'adult.data'
        test_file = self.data_dir / 'adult.test'

        # 下载数据(如果需要)
        if download:
            if not train_file.exists():
                print(f"下载训练集:{self.TRAIN_URL}")
                urllib.request.urlretrieve(self.TRAIN_URL, train_file)
            if not test_file.exists():
                print(f"下载测试集:{self.TEST_URL}")
                urllib.request.urlretrieve(self.TEST_URL, test_file)

        # 根据split确定使用哪个文件
        is_train = self.split in ("train", "valid")
        file_path = train_file if is_train else test_file

        # 读取CSV
        df = pd.read_csv(
            file_path,
            names=self.COLUMNS,
            sep=r'\s*,\s*',
            engine='python',
            na_values='?',
            skiprows=0 if is_train else 1
        )

        # 删除缺失值
        df = df.dropna()

        # 处理标签
        df['income'] = df['income'].apply(lambda x: 1 if '>50K' in x else 0)

        # 分离特征和标签
        y = df['income'].values
        X = df.drop('income', axis=1)

        # 编码类别特征
        label_encoders = {}
        for col in self.CATEGORICAL_FEATURES:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le

        # 标准化数值特征
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.values)

        # 转换为torch tensor
        self.data = torch.FloatTensor(X_scaled)
        self.labels = torch.LongTensor(y)

        # 设置属性
        self.num_classes = 2
        self.num_features = X_scaled.shape[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
