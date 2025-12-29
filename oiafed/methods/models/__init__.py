"""
内置模型

注意:根据框架设计原则,我们不再定义自定义模型基类
直接使用 PyTorch nn.Module 和 Transformers 模型

同时提供一些常用的模型实现
"""

from .cnn import SimpleCNN
from .mnist_cnn import MNISTCNNModel
from .mnist_lenet import MNISTLeNetModel
from .paper_cnn import PaperCNN, CIFAR10_PaperCNN, MNIST_PaperCNN
from .paper_mlp import PaperMLP, Adult_PaperMLP, FCUBE_PaperMLP
from .cifar10_cnn import CNNModel, SimpleCNNModel
from .resnet import ResNet18_CIFAR, ResNet34_CIFAR

__all__ = [
    "SimpleCNN",
    "MNISTCNNModel",
    "MNISTLeNetModel",
    "PaperCNN",
    "CIFAR10_PaperCNN",
    "MNIST_PaperCNN",
    "PaperMLP",
    "Adult_PaperMLP",
    "FCUBE_PaperMLP",
    "CNNModel",
    "SimpleCNNModel",
    "ResNet18_CIFAR",
    "ResNet34_CIFAR",
]
