# fedcl/learners/generic_learner.py
"""
通用学习器实现

提供通用的学习器实现，支持通过配置文件指定注册的模型。
可以用于各种分类任务，通过注册系统动态加载模型。
"""

import time
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from loguru import logger

from ...core.base_learner import BaseLearner
from ...core.execution_context import ExecutionContext
from ...registry.component_registry import registry


# ===== 注册的模型组件 =====

@registry.auxiliary_model("MLP", model_type="classification")
class SimpleMLP(nn.Module):
    """
    简单的多层感知机模型
    
    适用于各种分类任务的基础MLP模型
    """
    
    def __init__(self, input_size: int = 784, hidden_sizes: List[int] = None, 
                 num_classes: int = 10, dropout_rate: float = 0.2, 
                 activation: str = "relu", use_batch_norm: bool = False):
        """
        初始化简单MLP
        
        Args:
            input_size: 输入特征维度
            hidden_sizes: 隐藏层维度列表
            num_classes: 输出类别数
            dropout_rate: Dropout概率
            activation: 激活函数类型
            use_batch_norm: 是否使用批归一化
        """
        super(SimpleMLP, self).__init__()
        
        if hidden_sizes is None:
            hidden_sizes = [512, 256]
        
        self.input_size = input_size
        self.num_classes = num_classes
        
        # 激活函数
        if activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation.lower() == "tanh":
            self.activation = nn.Tanh()
        elif activation.lower() == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.ReLU()
        
        # 构建网络层
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            # 线性层
            layers.append(nn.Linear(prev_size, hidden_size))
            
            # 批归一化
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            
            # 激活函数
            layers.append(self.activation)
            
            # Dropout
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_size = hidden_size
        
        # 输出层
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """前向传播"""
        # 自动展平输入
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        return self.network(x)


@registry.auxiliary_model("CNN", model_type="classification")
class SimpleCNN(nn.Module):
    """
    简单的卷积神经网络模型
    
    适用于图像分类任务的基础CNN模型
    """
    
    def __init__(self, input_channels: int = 1, num_classes: int = 10,
                 conv_channels: List[int] = None, kernel_size: int = 3,
                 hidden_size: int = 128, dropout_rate: float = 0.25,
                 use_batch_norm: bool = True):
        """
        初始化简单CNN
        
        Args:
            input_channels: 输入通道数
            num_classes: 输出类别数
            conv_channels: 卷积层通道数列表
            kernel_size: 卷积核大小
            hidden_size: 全连接层隐藏单元数
            dropout_rate: Dropout概率
            use_batch_norm: 是否使用批归一化
        """
        super(SimpleCNN, self).__init__()
        
        if conv_channels is None:
            conv_channels = [32, 64, 128]
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        
        # 卷积层
        conv_layers = []
        in_channels = input_channels
        
        for out_channels in conv_channels:
            conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=1))
            
            if use_batch_norm:
                conv_layers.append(nn.BatchNorm2d(out_channels))
            
            conv_layers.append(nn.ReLU(inplace=True))
            conv_layers.append(nn.MaxPool2d(2, 2))
            
            in_channels = out_channels
        
        self.conv_features = nn.Sequential(*conv_layers)
        
        # 计算卷积输出大小（假设输入为28x28）
        self.feature_size = self._get_conv_output_size()
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _get_conv_output_size(self, input_size: int = 28):
        """计算卷积层输出尺寸"""
        # 每个卷积+池化层都会将尺寸减半
        size = input_size
        num_pools = len([layer for layer in self.conv_features if isinstance(layer, nn.MaxPool2d)])
        for _ in range(num_pools):
            size = size // 2
        
        # 最后一个卷积层的通道数
        last_conv_channels = None
        for layer in reversed(self.conv_features):
            if isinstance(layer, nn.Conv2d):
                last_conv_channels = layer.out_channels
                break
        
        return last_conv_channels * size * size
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """前向传播"""
        x = self.conv_features(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.classifier(x)
        return x


@registry.auxiliary_model("resnet", model_type="classification")
class ResNetLike(nn.Module):
    """
    类ResNet结构的简单模型
    
    包含残差连接的简化版ResNet
    """
    
    def __init__(self, input_channels: int = 1, num_classes: int = 10,
                 base_channels: int = 32, num_blocks: int = 2):
        """
        初始化类ResNet模型
        
        Args:
            input_channels: 输入通道数
            num_classes: 输出类别数
            base_channels: 基础通道数
            num_blocks: 残差块数量
        """
        super(ResNetLike, self).__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        
        # 初始卷积
        self.conv1 = nn.Conv2d(input_channels, base_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(base_channels)
        
        # 残差块
        self.blocks = nn.ModuleList()
        channels = base_channels
        
        for i in range(num_blocks):
            self.blocks.append(self._make_block(channels, channels * 2, stride=2 if i > 0 else 1))
            channels *= 2
        
        # 全局平均池化和分类器
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels, num_classes)
        
        self._initialize_weights()
    
    def _make_block(self, in_channels, out_channels, stride=1):
        """创建残差块"""
        layers = []
        
        # 第一个卷积
        layers.append(nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        # 第二个卷积
        layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        
        block = nn.Sequential(*layers)
        
        # 残差连接的shortcut
        if stride != 1 or in_channels != out_channels:
            shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            shortcut = nn.Identity()
        
        class ResidualBlock(nn.Module):
            def __init__(self, block, shortcut):
                super().__init__()
                self.block = block
                self.shortcut = shortcut
                
            def forward(self, x):
                residual = self.shortcut(x)
                out = self.block(x)
                out += residual
                return F.relu(out)
        
        return ResidualBlock(block, shortcut)
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """前向传播"""
        x = F.relu(self.bn1(self.conv1(x)))
        
        for block in self.blocks:
            x = block(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x

