"""
通用 CNN 模型

适用于图像分类任务（MNIST, CIFAR-10, Fashion-MNIST 等）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple
from ...registry import model


@model(
    name='simple_cnn',
    description='简单的CNN模型 - 可配置的卷积神经网络，适用于图像分类',
    version='1.0',
    author='Federation Framework',
    model_type='cnn',
    framework='pytorch'
)
class SimpleCNN(nn.Module):
    """
    简单的 CNN 模型

    可配置的卷积神经网络，适用于多种图像分类任务
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (28, 28, 1),
        num_classes: int = 10,
        conv_channels: List[int] = None,
        dense_units: int = 128,
        dropout_rate: float = 0.5,
    ):
        """
        Args:
            input_shape: 输入形状 (height, width, channels)
            num_classes: 类别数量
            conv_channels: 卷积层通道数列表
            dense_units: 全连接层单元数
            dropout_rate: Dropout 比率
        """
        super(SimpleCNN, self).__init__()

        if conv_channels is None:
            conv_channels = [32, 64]

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.conv_channels = conv_channels
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate

        # 输入通道数（根据input_shape调整）
        # input_shape格式: (H, W, C)，需要转换为 (C, H, W) 供PyTorch使用
        in_channels = input_shape[2]

        # 构建卷积层
        self.conv_layers = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        prev_channels = in_channels
        for channels in conv_channels:
            self.conv_layers.append(
                nn.Conv2d(prev_channels, channels, kernel_size=3, padding=1)
            )
            prev_channels = channels

        # 计算卷积后的特征图大小
        # 每个MaxPool将尺寸减半
        feature_h = input_shape[0]
        feature_w = input_shape[1]
        for _ in conv_channels:
            feature_h = feature_h // 2
            feature_w = feature_w // 2

        self.flatten_size = conv_channels[-1] * feature_h * feature_w

        # 全连接层
        self.fc1 = nn.Linear(self.flatten_size, dense_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(dense_units, num_classes)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量，形状 (N, C, H, W)

        Returns:
            输出张量，形状 (N, num_classes)
        """
        # 卷积层 + 池化
        for conv in self.conv_layers:
            x = self.pool(F.relu(conv(x)))

        # 展平
        x = x.reshape(-1, self.flatten_size)

        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def get_weights(self) -> List[np.ndarray]:
        """
        获取模型权重（转换为numpy数组）

        Returns:
            权重列表
        """
        return [param.data.cpu().numpy() for param in self.parameters()]

    def set_weights(self, weights: List[np.ndarray]):
        """
        设置模型权重（从numpy数组）

        Args:
            weights: 权重列表
        """
        with torch.no_grad():
            for param, weight in zip(self.parameters(), weights):
                param.data = torch.from_numpy(weight).to(param.device)

    def train_step(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 1,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        device: str = 'cpu',
        verbose: bool = False
    ) -> dict:
        """
        训练模型

        Args:
            x_train: 训练数据，形状 (N, H, W, C)
            y_train: 训练标签，形状 (N,)
            epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            device: 设备 ('cpu' 或 'cuda')
            verbose: 是否打印训练信息

        Returns:
            训练历史字典
        """
        self.to(device)
        self.train()

        # 转换数据格式: (N, H, W, C) -> (N, C, H, W)
        x_train = torch.from_numpy(x_train).permute(0, 3, 1, 2).float().to(device)
        y_train = torch.from_numpy(y_train).long().to(device)

        # 创建数据加载器
        dataset = torch.utils.data.TensorDataset(x_train, y_train)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        # 优化器和损失函数
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        history = {'loss': [], 'accuracy': []}

        for epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in dataloader:
                # 前向传播
                outputs = self(inputs)
                loss = criterion(outputs, labels)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 统计
                epoch_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            # 记录
            avg_loss = epoch_loss / total
            accuracy = correct / total
            history['loss'].append(avg_loss)
            history['accuracy'].append(accuracy)

            if verbose:
                print(f'Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.4f}')

        return history

    def evaluate_model(
        self,
        x_test: np.ndarray,
        y_test: np.ndarray,
        batch_size: int = 32,
        device: str = 'cpu',
        verbose: bool = False
    ) -> Tuple[float, float]:
        """
        评估模型

        Args:
            x_test: 测试数据，形状 (N, H, W, C)
            y_test: 测试标签，形状 (N,)
            batch_size: 批次大小
            device: 设备
            verbose: 是否打印信息

        Returns:
            (loss, accuracy)
        """
        self.to(device)
        self.eval()

        # 转换数据格式
        x_test = torch.from_numpy(x_test).permute(0, 3, 1, 2).float().to(device)
        y_test = torch.from_numpy(y_test).long().to(device)

        # 创建数据加载器
        dataset = torch.utils.data.TensorDataset(x_test, y_test)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )

        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in dataloader:
                outputs = self(inputs)
                loss = criterion(outputs, labels)

                total_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / total
        accuracy = correct / total

        if verbose:
            print(f'Test Loss: {avg_loss:.4f} - Test Accuracy: {accuracy:.4f}')

        return avg_loss, accuracy

    def save(self, path: str):
        """
        保存模型

        Args:
            path: 保存路径
        """
        torch.save({
            'state_dict': self.state_dict(),
            'config': {
                'input_shape': self.input_shape,
                'num_classes': self.num_classes,
                'conv_channels': self.conv_channels,
                'dense_units': self.dense_units,
                'dropout_rate': self.dropout_rate,
            }
        }, path)

    def load(self, path: str, device: str = 'cpu'):
        """
        加载模型

        Args:
            path: 模型路径
            device: 设备
        """
        checkpoint = torch.load(path, map_location=device)
        self.load_state_dict(checkpoint['state_dict'])
        return self

    @classmethod
    def from_pretrained(cls, path: str, device: str = 'cpu') -> 'SimpleCNN':
        """
        从保存的模型加载

        Args:
            path: 模型路径
            device: 设备

        Returns:
            加载的模型实例
        """
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        model = cls(**config)
        model.load_state_dict(checkpoint['state_dict'])
        return model
