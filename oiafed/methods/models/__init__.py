"""
内置模型

按数据集组织的模型集合，直接使用 PyTorch nn.Module。

模型注册名:
-----------
CIFAR-10/100 (32x32 彩色图像):
- cifar10_cnn        : 主力 CNN (23篇论文使用)
- cifar10_simple_cnn : 轻量级 CNN
- cifar10_paper_cnn  : 论文标准 CNN (TPAMI 2025)
- resnet18           : ResNet18 (适配 CIFAR)
- resnet34           : ResNet34 (适配 CIFAR)

MNIST (28x28 灰度图像):
- mnist_cnn          : 主力 CNN
- mnist_lenet        : LeNet (TPAMI 2025)
- mnist_paper_cnn    : 论文标准 CNN (TPAMI 2025)

表格数据:
- paper_mlp          : 通用 MLP (TPAMI 2025)
- adult_mlp          : Adult 数据集专用
- fcube_mlp          : FCUBE 数据集专用

使用方式:
--------
在配置文件中指定模型类型:

    model:
      type: cifar10_cnn
      args:
        num_classes: 10

或在代码中直接使用:

    from oiafed.methods.models import CIFAR10CNN
    model = CIFAR10CNN(num_classes=10)
"""

# CIFAR 模型
from .cifar import (
    CIFAR10CNN,
    CIFAR10SimpleCNN,
    CIFAR10PaperCNN,
    ResNet18CIFAR,
    ResNet34CIFAR,
)

# MNIST 模型
from .mnist import (
    MNISTCNN,
    MNISTLeNet,
    MNISTPaperCNN,
)

# MLP 模型
from .mlp import (
    PaperMLP,
    AdultMLP,
    FCUBEMLP,
)

from .fot_alexnet import FOTAlexNet
from .fedsra import FedSRAResNet18Backbone, FedSRAEnsemble
from .oneshot import (
    OneShotEnsemble,
    FedCGSResNet18,
    FedCGSServerModel,
)


__all__ = [
    # CIFAR 模型
    "CIFAR10CNN",
    "CIFAR10SimpleCNN",
    "CIFAR10PaperCNN",
    "ResNet18CIFAR",
    "ResNet34CIFAR",
    # MNIST 模型
    "MNISTCNN",
    "MNISTLeNet",
    "MNISTPaperCNN",
    # MLP 模型
    "PaperMLP",
    "AdultMLP",
    "FCUBEMLP",
    "FOTAlexNet",
    "FedSRAResNet18Backbone",
    "FedSRAEnsemble",
    "OneShotEnsemble",
    "FedCGSResNet18",
    "FedCGSServerModel",
]
