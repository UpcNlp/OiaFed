"""
内置数据集

提供常用数据集的加载和划分
"""

from .mnist import MNISTDataset
from .fmnist import FashionMNISTDataset
from .cifar10 import CIFAR10Dataset
from .cifar100 import CIFAR100Dataset
from .svhn import SVHNDataset
from .cinic10 import CINIC10Dataset
from .adult import AdultDataset
from .fcube import FCUBEDataset
from .fedisic2019 import FedISIC2019Dataset

__all__ = [
    "MNISTDataset",
    "FashionMNISTDataset",
    "CIFAR10Dataset",
    "CIFAR100Dataset",
    "SVHNDataset",
    "CINIC10Dataset",
    "AdultDataset",
    "FCUBEDataset",
    "FedISIC2019Dataset",
]
