from typing import Dict, Any
import torchvision
import torchvision.transforms as transforms

# 导入新实现的数据集和模型管理
from fedcl.api.decorators import dataset, model
from fedcl.api.registry import registry
from fedcl.methods.datasets.base import FederatedDataset

@dataset(
    name='MNIST',
    description='MNIST手写数字数据集',
    dataset_type='image_classification',
    num_classes=10
)
class MNISTFederatedDataset(FederatedDataset):
    """MNIST联邦数据集实现"""

    def __init__(self, root: str = './data', train: bool = True, download: bool = True):
        super().__init__(root, train, download)

        # 数据转换
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # 加载MNIST数据集
        self.dataset = torchvision.datasets.MNIST(
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
            'dataset_name': 'MNIST',
            'num_samples': len(self.dataset),
            'num_classes': self.num_classes,
            'input_shape': self.input_shape,
            'train': self.train,
        }
