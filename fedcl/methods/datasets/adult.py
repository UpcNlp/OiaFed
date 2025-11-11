"""
Adult (Census Income)数据集实现
fedcl/methods/datasets/adult.py

Adult数据集用于预测收入是否超过50K
"""
from typing import Dict, Any
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import urllib.request

from fedcl.api.decorators import dataset
from fedcl.methods.datasets.base import FederatedDataset


class AdultDataset(Dataset):
    """Adult数据集PyTorch包装器"""

    def __init__(self, data: np.ndarray, labels: np.ndarray):
        """
        Args:
            data: 特征数据 (N, num_features)
            labels: 标签 (N,)
        """
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


@dataset(
    name='Adult',
    description='Adult (Census Income)数据集',
    dataset_type='tabular_classification',
    num_classes=2
)
class AdultFederatedDataset(FederatedDataset):
    """Adult联邦数据集实现

    2个类别：收入<=50K, 收入>50K
    特征：14个属性（年龄、工作类型、教育等）
    训练集：32,561个样本
    测试集：16,281个样本
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

    def __init__(self, root: str = './data/adult', train: bool = True, download: bool = True):
        super().__init__(root, train, download)

        # 创建数据目录
        os.makedirs(root, exist_ok=True)

        # 文件路径
        train_file = os.path.join(root, 'adult.data')
        test_file = os.path.join(root, 'adult.test')

        # 下载数据（如果需要）
        if download:
            if not os.path.exists(train_file):
                print(f"下载训练集: {self.TRAIN_URL}")
                urllib.request.urlretrieve(self.TRAIN_URL, train_file)
            if not os.path.exists(test_file):
                print(f"下载测试集: {self.TEST_URL}")
                urllib.request.urlretrieve(self.TEST_URL, test_file)

        # 加载数据
        file_path = train_file if train else test_file

        # 读取CSV
        df = pd.read_csv(file_path, names=self.COLUMNS, sep=r'\s*,\s*',
                         engine='python', na_values='?', skiprows=0 if train else 1)

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

        # 创建PyTorch数据集
        self.dataset = AdultDataset(X_scaled, y)

        # 设置属性
        self.num_classes = 2
        self.num_features = X_scaled.shape[1]
        self.input_shape = (self.num_features,)

    def get_statistics(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        return {
            'dataset_name': 'Adult',
            'num_samples': len(self.dataset),
            'num_classes': self.num_classes,
            'num_features': self.num_features,
            'input_shape': self.input_shape,
            'train': self.train,
        }
