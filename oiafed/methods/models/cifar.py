"""
CIFAR 数据集模型

包含适用于 CIFAR-10/100、SVHN、CINIC-10 等 32x32 彩色图像的模型。

注册名:
- cifar10_cnn: 主力 CNN 模型 (23篇论文使用)
- cifar10_simple_cnn: 轻量级 CNN
- cifar10_paper_cnn: 论文标准 CNN (TPAMI 2025)
- resnet18: ResNet18 (适配 CIFAR)
- resnet34: ResNet34 (适配 CIFAR)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...registry.decorators import model


# ==================== 主力 CNN 模型 ====================

@model(
    name='cifar10_cnn',
    description='CIFAR-10 主力 CNN 模型 - ResNet 风格轻量级网络',
    task='classification',
    version='1.0',
    input_shape=(3, 32, 32),
    output_shape=(10,)
)
class CIFAR10CNN(nn.Module):
    """
    CIFAR-10 主力 CNN 模型
    
    ResNet 风格的轻量级 CNN，适合联邦学习。
    这是大多数论文使用的默认模型。
    
    架构:
    - 4 个卷积块 (64 → 128 → 256 → 512)
    - 每块: Conv + BN + ReLU + MaxPool
    - 全局平均池化
    - 全连接分类层
    
    输入: (batch, 3, 32, 32)
    输出: (batch, num_classes)
    
    支持 VFL/SplitNN 分割: 通过 features/classifier 属性
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # 卷积部分（用于 VFL 分割）
        self.features = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32x32 -> 16x16
            
            # 第二个卷积块
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16x16 -> 8x8
            
            # 第三个卷积块
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 8x8 -> 4x4
            
            # 第四个卷积块
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # 全局平均池化
            nn.AdaptiveAvgPool2d((1, 1)),  # 4x4 -> 1x1
        )
        
        # 全连接部分（用于 VFL 分割）
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # (batch, 512)
        x = self.classifier(x)
        return x


# ==================== 轻量级 CNN ====================

@model(
    name='cifar10_simple_cnn',
    description='CIFAR-10 轻量级 CNN - 适合快速实验',
    task='classification',
    version='1.0',
    input_shape=(3, 32, 32),
    output_shape=(10,)
)
class CIFAR10SimpleCNN(nn.Module):
    """
    CIFAR-10 轻量级 CNN
    
    更轻量的版本，适合快速实验。
    
    架构:
    - 3 个卷积层 (32 → 64 → 64)
    - 2 个全连接层
    
    输入: (batch, 3, 32, 32)
    输出: (batch, num_classes)
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # 卷积部分
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32->16
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16->8
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 8->4
        )
        
        # 全连接部分
        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ==================== 论文标准 CNN (TPAMI 2025) ====================

@model(
    name='cifar10_paper_cnn',
    description='CIFAR-10 论文标准 CNN (TPAMI 2025)',
    task='classification',
    version='1.0',
    input_shape=(3, 32, 32),
    output_shape=(10,)
)
class CIFAR10PaperCNN(nn.Module):
    """
    CIFAR-10 论文标准 CNN
    
    符合 TPAMI 2025 论文标准架构。
    
    架构 (Section VI):
    - Conv1: 3 → 6, kernel=5
    - ReLU + MaxPool(2×2)
    - Conv2: 6 → 16, kernel=5
    - ReLU + MaxPool(2×2)
    - FC1: 400 → 120, ReLU
    - FC2: 120 → 84, ReLU
    - FC3: 84 → num_classes
    
    输入: (batch, 3, 32, 32)
    输出: (batch, num_classes)
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # 卷积层
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)   # 32 -> 28
        self.pool1 = nn.MaxPool2d(2, 2)               # 28 -> 14
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # 14 -> 10
        self.pool2 = nn.MaxPool2d(2, 2)               # 10 -> 5

        # 全连接层 (16 * 5 * 5 = 400)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ==================== ResNet ====================

class _BasicBlock(nn.Module):
    """ResNet BasicBlock"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class _ResNet(nn.Module):
    """ResNet 基类 (适配 CIFAR 32x32)"""

    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 64

        # 初始卷积 (CIFAR: kernel=3, stride=1, 无 maxpool)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # ResNet 层
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # 分类器
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward_features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        return torch.flatten(out, 1)

    def forward(self, x):
        return self.fc(self.forward_features(x))


@model(
    name='resnet18',
    description='ResNet18 (适配 CIFAR-10/100)',
    task='classification',
    version='1.0',
    input_shape=(3, 32, 32),
    output_shape=(10,)
)
class ResNet18CIFAR(_ResNet):
    """
    ResNet18 (适配 CIFAR)
    
    针对 32x32 图像优化:
    - conv1: kernel=3, stride=1 (无 stride=2)
    - 无初始 maxpool
    - 4 层: [2, 2, 2, 2] blocks
    
    参数量: ~11M
    """

    def __init__(self, num_classes=10):
        super().__init__(
            block=_BasicBlock,
            num_blocks=[2, 2, 2, 2],
            num_classes=num_classes
        )


@model(
    name='resnet34',
    description='ResNet34 (适配 CIFAR-10/100)',
    task='classification',
    version='1.0',
    input_shape=(3, 32, 32),
    output_shape=(10,)
)
class ResNet34CIFAR(_ResNet):
    """
    ResNet34 (适配 CIFAR)
    
    4 层: [3, 4, 6, 3] blocks
    参数量: ~21M
    """

    def __init__(self, num_classes=10):
        super().__init__(
            block=_BasicBlock,
            num_blocks=[3, 4, 6, 3],
            num_classes=num_classes
        )


# ==================== 导出 ====================

__all__ = [
    # 主力模型
    "CIFAR10CNN",
    "CIFAR10SimpleCNN",
    "CIFAR10PaperCNN",
    # ResNet
    "ResNet18CIFAR",
    "ResNet34CIFAR",
]
