import torch
import torch.nn as nn
import torch.nn.functional as F

from src.registry.decorators import model

@model(
    name='CNN',
    description='ResNet风格的CNN模型',
    task='classification',
    version='1.0'
)
class CNNModel(nn.Module):
    """ResNet风格的CNN模型

    基于ResNet思想的轻量级CNN，适合联邦学习
    输入: (batch, 3, 32, 32) - 适用于CIFAR-10, SVHN等
    输出: (batch, num_classes)
    """

    def __init__(self, num_classes: int = 10):
        super(CNNModel, self).__init__()

        # 第一个卷积块
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        # 第二个卷积块
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        # 第三个卷积块
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        # 第四个卷积块
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 全连接层
        self.fc = nn.Linear(512, num_classes)

        # Dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 第一块: conv1 -> bn1 -> relu -> pool
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  # 32x32 -> 16x16

        # 第二块: conv2 -> bn2 -> relu -> pool
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  # 16x16 -> 8x8

        # 第三块: conv3 -> bn3 -> relu -> pool
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  # 8x8 -> 4x4

        # 第四块: conv4 -> bn4 -> relu
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)

        # 全局平均池化
        x = self.avgpool(x)  # 4x4 -> 1x1
        x = torch.flatten(x, 1)  # (batch, 512)

        # Dropout + FC
        x = self.dropout(x)
        x = self.fc(x)

        return x


@model(
    name='SimpleCNN',
    description='简单的3层CNN模型',
    task='classification',
    version='1.0'
)
class SimpleCNNModel(nn.Module):
    """简单的3层CNN模型

    更轻量的版本，适合快速实验
    输入: (batch, 3, 32, 32) - 适用于CIFAR-10, SVHN等
    输出: (batch, num_classes)
    """

    def __init__(self, num_classes: int = 10):
        super(SimpleCNNModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # 卷积层
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # 32->16

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  # 16->8

        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)  # 8->4

        # 展平
        x = torch.flatten(x, 1)

        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
