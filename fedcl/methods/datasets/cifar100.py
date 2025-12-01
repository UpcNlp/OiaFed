from typing import Dict, Any
import torchvision
import torchvision.transforms as transforms

from ...api.decorators import dataset
from .base import FederatedDataset

@dataset(
    name='CIFAR100',
    description='CIFAR-100图像分类数据集',
    dataset_type='image_classification',
    num_classes=100
)
class CIFAR100FederatedDataset(FederatedDataset):
    """CIFAR-100联邦数据集实现

    100个类别，分为20个超类
    图像大小: 32x32x3 (RGB)
    训练集: 50,000张图像
    测试集: 10,000张图像
    """

    def __init__(self, root: str = './data', train: bool = True, download: bool = True):
        super().__init__(root, train, download)

        # 数据转换
        if train:
            # 训练时使用数据增强
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5071, 0.4867, 0.4408],
                    std=[0.2675, 0.2565, 0.2761]
                )
            ])
        else:
            # 测试时不使用数据增强
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5071, 0.4867, 0.4408],
                    std=[0.2675, 0.2565, 0.2761]
                )
            ])

        # 加载CIFAR-100数据集
        self.dataset = torchvision.datasets.CIFAR100(
            root=root,
            train=train,
            download=download,
            transform=transform
        )

        # 设置属性
        self.num_classes = 100
        self.input_shape = (3, 32, 32)

    def get_statistics(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        return {
            'dataset_name': 'CIFAR-100',
            'num_samples': len(self.dataset),
            'num_classes': self.num_classes,
            'input_shape': self.input_shape,
            'train': self.train,
        }
