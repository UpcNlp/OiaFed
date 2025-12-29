"""
CIFAR-10 CNN 分类模型

支持 VFL/SplitNN 的模型分割（通过 features/classifier 结构）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...registry.decorators import model


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
    
    支持 VFL/SplitNN 分割：通过 features/classifier 属性
    """

    def __init__(self, num_classes: int = 10):
        super(CNNModel, self).__init__()

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
    
    支持 VFL/SplitNN 分割：通过 features/classifier 属性
    """

    def __init__(self, num_classes: int = 10):
        super(SimpleCNNModel, self).__init__()

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