from typing import Dict, Any
import torchvision
import torchvision.transforms as transforms

from fedcl.api.decorators import dataset
from fedcl.methods.datasets.base import FederatedDataset

@dataset(
    name='FMNIST',
    description='Fashion-MNIST服装图像数据集',
    dataset_type='image_classification',
    num_classes=10
)
class FMNISTFederatedDataset(FederatedDataset):
    """Fashion-MNIST联邦数据集实现

    10个类别：
    0: T-shirt/top, 1: Trouser, 2: Pullover, 3: Dress, 4: Coat,
    5: Sandal, 6: Shirt, 7: Sneaker, 8: Bag, 9: Ankle boot
    """

    def __init__(self, root: str = './data', train: bool = True, download: bool = True):
        super().__init__(root, train, download)

        # 数据转换
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))  # Fashion-MNIST的均值和标准差
        ])

        # 加载Fashion-MNIST数据集
        self.dataset = torchvision.datasets.FashionMNIST(
            root=root,
            train=train,
            download=download,
            transform=transform
        )

        # 设置属性
        self.num_classes = 10
        self.input_shape = (1, 28, 28)

    def get_statistics(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        return {
            'dataset_name': 'Fashion-MNIST',
            'num_samples': len(self.dataset),
            'num_classes': self.num_classes,
            'input_shape': self.input_shape,
            'train': self.train,
        }
