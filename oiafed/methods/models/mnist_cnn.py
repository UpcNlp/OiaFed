"""
MNIST CNN 分类模型

支持 VFL/SplitNN 的模型分割（通过 features/classifier 结构）
"""
import torch
import torch.nn as nn
from ...registry import model


@model(
    name='mnist_cnn',
    description='MNIST CNN分类模型',
    version='1.0',
    task='classification',
    input_shape=(1, 28, 28),
    output_shape=(10,)
)
class MNISTCNNModel(nn.Module):
    """MNIST CNN模型

    一个简单的卷积神经网络,用于MNIST手写数字分类。

    网络结构:
    - features: Conv2d(1, 32) + ReLU + Conv2d(32, 64) + ReLU + MaxPool + Dropout
    - classifier: Linear(64*14*14, 128) + ReLU + Dropout + Linear(128, 10)
    
    支持 VFL/SplitNN 分割：通过 features/classifier 属性
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # 卷积部分（用于 VFL 分割）
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )
        
        # 全连接部分（用于 VFL 分割）
        self.classifier = nn.Sequential(
            nn.Linear(64 * 14 * 14, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        """前向传播"""
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten: (batch, 64*14*14)
        x = self.classifier(x)
        return x