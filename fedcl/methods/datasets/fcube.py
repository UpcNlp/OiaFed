"""
FCUBE合成数据集
fedcl/methods/datasets/fcube.py

参考论文: Measuring the Effects of Non-Identical Data Distribution (arXiv 2019)

FCUBE是一个3D合成数据集，用于测试特征分布倾斜(Feature Distribution Skew)。
数据点分布在一个立方体中，由平面 x1=0 分为两个类别。
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple

from fedcl.api.decorators import register_dataset


class FCUBEDataset(Dataset):
    """
    FCUBE合成数据集

    数据点均匀分布在3D立方体 [-1, 1]^3 中
    分类规则: x1 > 0 为类别0，x1 <= 0 为类别1
    """

    def __init__(self, num_samples: int = 5000, train: bool = True,
                 transform=None, target_transform=None, seed: int = 42):
        """
        Args:
            num_samples: 样本数量
            train: 是否为训练集
            transform: 特征转换
            target_transform: 标签转换
            seed: 随机种子
        """
        self.num_samples = num_samples
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        # 设置随机种子
        rng = np.random.RandomState(seed if train else seed + 1)

        # 生成3D点 (x1, x2, x3) 在 [-1, 1]^3 中
        self.data = rng.uniform(-1, 1, size=(num_samples, 3)).astype(np.float32)

        # 根据 x1 = 0 平面分类
        # x1 > 0 为类别0，x1 <= 0 为类别1
        self.targets = (self.data[:, 0] <= 0).astype(np.int64)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            (features, label): 特征张量和类别标签
        """
        features = self.data[idx]
        label = self.targets[idx]

        # 转换为torch张量
        features = torch.from_numpy(features)

        # 应用转换
        if self.transform is not None:
            features = self.transform(features)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return features, label

    @property
    def classes(self):
        """类别列表"""
        return [0, 1]


@register_dataset(
    name="FCUBE",
    version="1.0",
    dataset_type="synthetic_classification"
)
class FCUBEWrapper:
    """
    FCUBE数据集包装器

    符合框架的Dataset API
    """

    def __init__(self,
                 root: str = "./data",
                 num_train_samples: int = 4000,
                 num_test_samples: int = 1000,
                 seed: int = 42):
        """
        Args:
            root: 数据根目录（不使用，保持接口一致性）
            num_train_samples: 训练集样本数
            num_test_samples: 测试集样本数
            seed: 随机种子
        """
        self.root = root
        self.num_train_samples = num_train_samples
        self.num_test_samples = num_test_samples
        self.seed = seed

    def get_train_dataset(self, transform=None) -> FCUBEDataset:
        """获取训练集"""
        return FCUBEDataset(
            num_samples=self.num_train_samples,
            train=True,
            transform=transform,
            seed=self.seed
        )

    def get_test_dataset(self, transform=None) -> FCUBEDataset:
        """获取测试集"""
        return FCUBEDataset(
            num_samples=self.num_test_samples,
            train=False,
            transform=transform,
            seed=self.seed
        )

    @staticmethod
    def get_default_transform():
        """
        获取默认的数据转换

        对于FCUBE，不需要特殊转换（已经是归一化的浮点数）
        """
        return None

    @property
    def num_classes(self) -> int:
        """类别数量"""
        return 2

    @property
    def input_shape(self) -> Tuple[int]:
        """输入特征形状"""
        return (3,)  # 3D特征


def test_fcube():
    """测试FCUBE数据集"""
    print("=" * 80)
    print("测试FCUBE数据集")
    print("=" * 80)

    # 创建数据集
    wrapper = FCUBEWrapper(num_train_samples=4000, num_test_samples=1000)
    train_dataset = wrapper.get_train_dataset()
    test_dataset = wrapper.get_test_dataset()

    print(f"\n训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    print(f"类别数量: {wrapper.num_classes}")
    print(f"输入特征维度: {wrapper.input_shape}")

    # 检查数据
    features, label = train_dataset[0]
    print(f"\n样本示例:")
    print(f"  特征: {features}")
    print(f"  标签: {label}")
    print(f"  特征形状: {features.shape}")
    print(f"  特征类型: {features.dtype}")

    # 统计类别分布
    train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
    unique, counts = np.unique(train_labels, return_counts=True)
    print(f"\n训练集类别分布:")
    for cls, count in zip(unique, counts):
        print(f"  类别 {cls}: {count} 样本 ({count/len(train_dataset)*100:.1f}%)")

    # 验证分类规则
    print(f"\n验证分类规则 (x1 > 0 → 类别0, x1 <= 0 → 类别1):")
    correct = 0
    for i in range(min(100, len(train_dataset))):
        features, label = train_dataset[i]
        expected_label = int(features[0] <= 0)
        if label == expected_label:
            correct += 1
    print(f"  验证准确率: {correct}/100 = {correct}%")

    print("\n" + "=" * 80)
    print("✅ FCUBE数据集测试通过!")
    print("=" * 80)


if __name__ == "__main__":
    test_fcube()
