"""
基线模型模块。
提供用于对比实验的简单骨干网络 + 分类头模型。
"""

import torch
import torch.nn as nn

from .backbones import get_backbone


class BaselineModel(nn.Module):
    """
    基线分类模型。

    结构: 骨干网络 → 分类头
    用于 FedAvg、FedProx 和 FedSym 对比实验。
    """

    def __init__(
        self,
        num_classes: int,
        backbone: str = "cnn",
        input_channels: int = 3,
        input_size: int = 32,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        """
        初始化基线模型。

        参数:
            num_classes: 分类类别数
            backbone: 骨干网络类型 ('cnn' 或 'resnet18')
            input_channels: 输入通道数
            input_size: 输入图像尺寸
            hidden_dim: 分类头隐藏层维度
            dropout: Dropout 概率
        """
        super().__init__()

        self.num_classes = num_classes

        # 骨干网络
        self.backbone = get_backbone(backbone, input_channels, input_size)
        feature_dim = self.backbone.get_output_dim()

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数:
            x: 输入图像 [batch, channels, height, width]

        返回:
            logits [batch, num_classes]
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """获取特征表示。"""
        return self.backbone(x)


def create_baseline_model(
    num_classes: int,
    backbone: str = "cnn",
    input_channels: int = 3,
    input_size: int = 32,
    hidden_dim: int = 256,
    device: torch.device = None,
    **kwargs
) -> BaselineModel:
    """
    创建基线模型的工厂函数。

    参数:
        num_classes: 类别数
        backbone: 骨干网络类型
        input_channels: 输入通道数
        input_size: 输入图像尺寸
        hidden_dim: 分类头隐藏层维度
        device: 目标设备
        **kwargs: 其他参数（忽略）

    返回:
        BaselineModel 实例
    """
    model = BaselineModel(
        num_classes=num_classes,
        backbone=backbone,
        input_channels=input_channels,
        input_size=input_size,
        hidden_dim=hidden_dim
    )

    if device is not None:
        model = model.to(device)

    return model
