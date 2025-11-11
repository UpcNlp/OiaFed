from typing import Dict, Any
import torchvision
import torchvision.transforms as transforms

from fedcl.api.decorators import dataset
from fedcl.methods.datasets.base import FederatedDataset

@dataset(
    name='CIFAR10',
    description='CIFAR-10图像分类数据集',
    dataset_type='image_classification',
    num_classes=10
)
class CIFAR10FederatedDataset(FederatedDataset):
    """CIFAR-10联邦数据集实现

    10个类别：
    0: airplane, 1: automobile, 2: bird, 3: cat, 4: deer,
    5: dog, 6: frog, 7: horse, 8: ship, 9: truck

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
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2023, 0.1994, 0.2010]
                )
            ])
        else:
            # 测试时不使用数据增强
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2023, 0.1994, 0.2010]
                )
            ])

        # 加载CIFAR-10数据集
        self.dataset = torchvision.datasets.CIFAR10(
            root=root,
            train=train,
            download=download,
            transform=transform
        )

        # 设置属性
        self.num_classes = 10
        self.input_shape = (3, 32, 32)

    def get_statistics(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        return {
            'dataset_name': 'CIFAR-10',
            'num_samples': len(self.dataset),
            'num_classes': self.num_classes,
            'input_shape': self.input_shape,
            'train': self.train,
        }
