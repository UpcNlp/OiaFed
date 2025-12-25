"""
FCUBE合成数据集

从 methods/datasets/fcube.py 迁移到 src/
支持 split 参数 (train/test/valid)

参考论文:Measuring the Effects of Non-Identical Data Distribution (arXiv 2019)

FCUBE是一个3D合成数据集,用于测试特征分布倾斜(Feature Distribution Skew)。
数据点分布在一个立方体中,由平面 x1=0 分为两个类别。
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from ...registry import dataset


@dataset(
    name='fcube',
    description='FCUBE 3D合成分类数据集',
    version='1.0',
    author='Federation Framework',
    dataset_type='synthetic_classification',
    num_classes=2,
    input_shape=(3,)
)
class FCUBEDataset(Dataset):
    """
    FCUBE合成数据集 - 标准 PyTorch Dataset

    数据点均匀分布在3D立方体 [-1, 1]^3 中
    分类规则:x1 > 0 为类别0,x1 <= 0 为类别1

    用于测试特征分布倾斜(Feature Distribution Skew)
    """

    def __init__(
        self,
        data_dir: str = "./data",  # 不使用,保持接口一致性
        split: str = "train",
        num_samples: int = None,  # 如果为None,使用默认值
        seed: int = 42,
    ):
        """
        Args:
            data_dir: 数据目录(不使用,保持接口一致性)
            split: 数据集划分 ("train" / "test" / "valid")
            num_samples: 样本数量(如果为None,train=4000, test=1000)
            seed: 随机种子
        """
        self.split = split
        self.seed = seed

        # 确定样本数量
        if num_samples is None:
            if self.split in ("train", "valid"):
                self.num_samples = 4000
            else:  # test
                self.num_samples = 1000
        else:
            self.num_samples = num_samples

        # 设置随机种子(不同split使用不同种子)
        if self.split == "train":
            rng_seed = seed
        elif self.split == "valid":
            rng_seed = seed + 1
        else:  # test
            rng_seed = seed + 2

        rng = np.random.RandomState(rng_seed)

        # 生成3D点 (x1, x2, x3) 在 [-1, 1]^3 中
        self.data = rng.uniform(-1, 1, size=(self.num_samples, 3)).astype(np.float32)

        # 根据 x1 = 0 平面分类
        # x1 > 0 为类别0,x1 <= 0 为类别1
        self.targets = (self.data[:, 0] <= 0).astype(np.int64)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Returns:
            (features, label):特征张量和类别标签
        """
        features = torch.from_numpy(self.data[idx])
        label = self.targets[idx]

        return features, label

    @property
    def classes(self):
        """类别列表"""
        return [0, 1]
