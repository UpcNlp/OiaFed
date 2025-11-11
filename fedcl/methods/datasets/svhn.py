"""
SVHN数据集实现
fedcl/methods/datasets/svhn.py

Street View House Numbers (SVHN)数据集
"""
from typing import Dict, Any
import torchvision
import torchvision.transforms as transforms

from fedcl.api.decorators import dataset
from fedcl.methods.datasets.base import FederatedDataset


@dataset(
    name='SVHN',
    description='SVHN街道视图房屋号码数据集',
    dataset_type='image_classification',
    num_classes=10
)
class SVHNFederatedDataset(FederatedDataset):
    """SVHN联邦数据集实现

    10个类别：数字0-9
    图像大小：32x32x3 (RGB)

    训练集：73,257张图片
    测试集：26,032张图片
    """

    def __init__(self, root: str = './data', train: bool = True, download: bool = True):
        super().__init__(root, train, download)

        # 数据转换
        if train:
            # 训练集：数据增强
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
            ])
        else:
            # 测试集：仅标准化
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
            ])

        # 加载SVHN数据集
        # split: 'train' or 'test'
        split = 'train' if train else 'test'
        self.dataset = torchvision.datasets.SVHN(
            root=root,
            split=split,
            download=download,
            transform=transform
        )

        # 设置属性
        self.num_classes = 10
        self.input_shape = (3, 32, 32)

    def get_statistics(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        return {
            'dataset_name': 'SVHN',
            'num_samples': len(self.dataset),
            'num_classes': self.num_classes,
            'input_shape': self.input_shape,
            'train': self.train,
        }
