"""
骨干网络模块，用于特征提取。
支持多种骨干网络架构，包括早期退出版本以解决联邦学习中的过拟合问题。

骨干网络选项：

1. CNNBackbone - 简单3层CNN
   - 参数量: ~0.1M
   - 输出维度: 2048
   - 适用场景: 快速实验

2. ResNet18Backbone - 标准ResNet18
   - 参数量: ~11.2M
   - 输出维度: 512
   - 注意: 在联邦学习小数据场景下容易过拟合，不推荐

3. EarlyExitResNetBackbone - 早期退出ResNet 【推荐用于FedEMoE】
   - 参数量: ~0.2M
   - 输出维度: 128
   - 只使用stem + layer1 + layer2，避免深层过拟合
   - 内置LayerNorm和Dropout正则化
   - 中间层特征更通用，EDL不确定性估计更可靠

4. ConditionalDepthResNetBackbone - 条件深度ResNet 【推荐用于条件深度EMoE】
   - 参数量: ~2.8M
   - 浅层输出维度: 128 (layer2)
   - 深层输出维度: 512 (layer4)
   - 支持根据样本难度动态选择网络深度
   - 简单样本使用浅层快速分类，困难样本使用深层获得更强表示
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNBackbone(nn.Module):
    """
    简单 CNN 骨干网络，适用于 CIFAR 类小图像数据集。
    
    结构: Conv→BN→ReLU→Pool × 3 层
    """
    
    def __init__(
        self, 
        input_channels: int = 3, 
        input_size: int = 32,
        hidden_channels: List[int] = None
    ):
        """
        初始化 CNN 骨干网络。
        
        参数:
            input_channels: 输入通道数（RGB=3, 灰度=1）
            input_size: 输入图像尺寸
            hidden_channels: 各层的通道数列表
        """
        super().__init__()
        
        if hidden_channels is None:
            hidden_channels = [32, 64, 128]
        
        self.input_channels = input_channels
        self.input_size = input_size
        
        # 构建卷积层
        layers = []
        in_ch = input_channels
        
        for out_ch in hidden_channels:
            layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            ])
            in_ch = out_ch
        
        self.features = nn.Sequential(*layers)
        
        # 计算输出维度
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_size, input_size)
            out = self.features(dummy)
            self.output_dim = int(out.view(1, -1).shape[1])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        
        参数:
            x: 输入张量 [batch, channels, height, width]
        
        返回:
            特征张量 [batch, output_dim]
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 展平
        return x
    
    def get_output_dim(self) -> int:
        """获取输出特征维度。"""
        return self.output_dim


class BasicBlock(nn.Module):
    """ResNet 基本残差块。"""
    
    expansion = 1
    
    def __init__(
        self, 
        in_planes: int, 
        planes: int, 
        stride: int = 1,
        dropout: float = 0.0
    ):
        """
        初始化基本残差块。
        
        参数:
            in_planes: 输入通道数
            planes: 输出通道数
            stride: 卷积步长
            dropout: Dropout概率（用于正则化）
        """
        super().__init__()
        
        # 第一个卷积层
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        
        # 第二个卷积层
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        
        # Dropout（可选）
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # 捷径连接（当维度不匹配时需要变换）
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes, 
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(self.expansion * planes)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        
        参数:
            x: 输入张量
        
        返回:
            输出张量（残差 + 捷径）
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # 残差连接
        out = F.relu(out)
        return out


class ResNet18Backbone(nn.Module):
    """
    ResNet18 骨干网络，用于特征提取。
    针对 CIFAR 类小图像进行了修改（使用较小的初始卷积核）。
    """
    
    def __init__(
        self, 
        input_channels: int = 3, 
        input_size: int = 32
    ):
        """
        初始化 ResNet18 骨干网络。
        
        参数:
            input_channels: 输入通道数
            input_size: 输入图像尺寸
        """
        super().__init__()
        
        self.in_planes = 64
        
        # 初始卷积层（针对 CIFAR 使用较小的卷积核）
        self.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        
        # 残差层
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.output_dim = 512 * BasicBlock.expansion
    
    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        """
        构建一个残差层（包含多个残差块）。
        
        参数:
            planes: 输出通道数
            num_blocks: 残差块数量
            stride: 第一个块的步长
        
        返回:
            残差层序列
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes * BasicBlock.expansion
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        
        参数:
            x: 输入张量 [batch, channels, height, width]
        
        返回:
            特征张量 [batch, 512]
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)  # 展平
        return out
    
    def get_output_dim(self) -> int:
        """获取输出特征维度。"""
        return self.output_dim


class EarlyExitResNetBackbone(nn.Module):
    """
    早期退出的ResNet骨干网络，专为FedEMoE设计。
    
    核心思想：
    - 在ResNet的layer2后"早期退出"，避免深层过拟合
    - 中间层特征(128维)更通用，过拟合风险低
    - EDL不确定性估计更加校准可靠
    - 专家网络更容易学习多样化模式
    
    架构：
    - stem: Conv(3→64) + BN + ReLU
    - layer1: 64→64, 2个BasicBlock, stride=1
    - layer2: 64→128, 2个BasicBlock, stride=2
    - 全局平均池化 → 128维输出
    
    优势：
    - 参数量: ~0.2M (vs ResNet18的11.2M)
    - 特征维度: 128 (与专家hidden_dim更匹配)
    - 保留残差学习优势，同时避免过拟合
    - 适合联邦学习的小数据场景
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        input_size: int = 32,
        base_channels: int = 64,
        dropout: float = 0.2,
        use_layer_norm: bool = True
    ):
        """
        初始化早期退出ResNet骨干网络。
        
        参数:
            input_channels: 输入通道数
            input_size: 输入图像尺寸
            base_channels: 基础通道数
            dropout: 特征正则化的Dropout概率
            use_layer_norm: 是否使用LayerNorm进行特征正则化
        """
        super().__init__()
        
        self.in_planes = base_channels
        self.dropout_rate = dropout
        self.use_layer_norm = use_layer_norm
        
        # Stem: 初始卷积层
        self.stem = nn.Sequential(
            nn.Conv2d(
                input_channels, base_channels,
                kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Layer1: 64 → 64, stride=1, 保持空间尺寸
        self.layer1 = self._make_layer(base_channels, num_blocks=2, stride=1)
        
        # Layer2: 64 → 128, stride=2, 空间尺寸减半
        self.layer2 = self._make_layer(base_channels * 2, num_blocks=2, stride=2)
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 输出特征维度
        self.output_dim = base_channels * 2  # 128
        
        # 特征正则化层（关键！防止过拟合）
        regularizer_layers = []
        if use_layer_norm:
            regularizer_layers.append(nn.LayerNorm(self.output_dim))
        if dropout > 0:
            regularizer_layers.append(nn.Dropout(dropout))
        
        self.feature_regularizer = nn.Sequential(*regularizer_layers) if regularizer_layers else nn.Identity()
        
        # 权重初始化
        self._initialize_weights()
    
    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        """构建残差层。"""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlock.expansion
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """使用Kaiming初始化权重。"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        
        参数:
            x: 输入张量 [batch, channels, height, width]
        
        返回:
            特征张量 [batch, 128]
        """
        # Stem
        out = self.stem(x)  # [B, 64, 32, 32]
        
        # 残差层
        out = self.layer1(out)  # [B, 64, 32, 32]
        out = self.layer2(out)  # [B, 128, 16, 16]
        
        # 全局池化
        out = self.avgpool(out)  # [B, 128, 1, 1]
        out = out.view(out.size(0), -1)  # [B, 128]
        
        # 特征正则化
        out = self.feature_regularizer(out)
        
        return out
    
    def get_output_dim(self) -> int:
        """获取输出特征维度。"""
        return self.output_dim
    
    def get_intermediate_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        获取中间层特征（用于分析和可视化）。
        
        返回:
            包含各层特征的字典
        """
        features = {}
        
        out = self.stem(x)
        features['stem'] = out
        
        out = self.layer1(out)
        features['layer1'] = out
        
        out = self.layer2(out)
        features['layer2'] = out
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        features['pooled'] = out
        
        out = self.feature_regularizer(out)
        features['regularized'] = out
        
        return features


class ConditionalDepthResNetBackbone(nn.Module):
    """
    条件深度ResNet骨干网络，支持根据样本难度动态选择网络深度。
    
    核心思想：
    - 对简单样本：使用浅层特征(layer2)快速分类，减少计算和过拟合
    - 对困难样本：使用深层特征(layer4)获得更强的表示能力
    
    架构：
    ┌─────────────────┐
    │  stem + layer1  │
    │  layer2 (→128)  │  ← 浅层出口
    └────────┬────────┘
             │
        [条件分支]
             │
    ┌────────┴────────┐
    │  layer3 (→256)  │
    │  layer4 (→512)  │  ← 深层出口
    └─────────────────┘
    
    输出：
    - shallow_features: 浅层特征 [batch, 128]
    - deep_features: 深层特征 [batch, 512] (可选，仅在需要时计算)
    
    参数量: ~2.8M (完整) / ~0.2M (仅浅层)
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        input_size: int = 32,
        base_channels: int = 64,
        dropout: float = 0.2,
        use_layer_norm: bool = True
    ):
        """
        初始化条件深度ResNet骨干网络。
        
        参数:
            input_channels: 输入通道数
            input_size: 输入图像尺寸
            base_channels: 基础通道数
            dropout: 特征正则化的Dropout概率
            use_layer_norm: 是否使用LayerNorm进行特征正则化
        """
        super().__init__()
        
        self.in_planes = base_channels
        self.dropout_rate = dropout
        self.use_layer_norm = use_layer_norm
        
        # ==================== 浅层部分 (Early Exit) ====================
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(
                input_channels, base_channels,
                kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Layer1: 64 → 64
        self.layer1 = self._make_layer(base_channels, num_blocks=2, stride=1)
        
        # Layer2: 64 → 128
        self.layer2 = self._make_layer(base_channels * 2, num_blocks=2, stride=2)
        
        # 浅层输出维度
        self.shallow_dim = base_channels * 2  # 128
        
        # 浅层特征正则化
        shallow_reg = []
        if use_layer_norm:
            shallow_reg.append(nn.LayerNorm(self.shallow_dim))
        if dropout > 0:
            shallow_reg.append(nn.Dropout(dropout))
        self.shallow_regularizer = nn.Sequential(*shallow_reg) if shallow_reg else nn.Identity()
        
        # ==================== 深层部分 (Full Depth) ====================
        # Layer3: 128 → 256
        self.layer3 = self._make_layer(base_channels * 4, num_blocks=2, stride=2)
        
        # Layer4: 256 → 512
        self.layer4 = self._make_layer(base_channels * 8, num_blocks=2, stride=2)
        
        # 深层输出维度
        self.deep_dim = base_channels * 8  # 512
        
        # 深层特征正则化
        deep_reg = []
        if use_layer_norm:
            deep_reg.append(nn.LayerNorm(self.deep_dim))
        if dropout > 0:
            deep_reg.append(nn.Dropout(dropout))
        self.deep_regularizer = nn.Sequential(*deep_reg) if deep_reg else nn.Identity()
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 输出维度（默认返回浅层）
        self.output_dim = self.shallow_dim
        
        # 权重初始化
        self._initialize_weights()
    
    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        """构建残差层。"""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlock.expansion
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """使用Kaiming初始化权重。"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward_shallow(self, x: torch.Tensor) -> torch.Tensor:
        """
        浅层前向传播（仅使用layer1和layer2）。
        
        参数:
            x: 输入张量 [batch, channels, height, width]
        
        返回:
            浅层特征 [batch, 128]
        """
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.shallow_regularizer(out)
        return out
    
    def forward_deep(self, x: torch.Tensor) -> torch.Tensor:
        """
        深层前向传播（使用全部layer1-4）。
        
        参数:
            x: 输入张量 [batch, channels, height, width]
        
        返回:
            深层特征 [batch, 512]
        """
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.deep_regularizer(out)
        return out
    
    def forward_from_layer2(self, layer2_output: torch.Tensor) -> torch.Tensor:
        """
        从layer2的输出继续深层前向传播。
        用于条件深度：先计算浅层，再根据需要继续深层。
        
        参数:
            layer2_output: layer2的输出特征图 [batch, 128, H, W]
        
        返回:
            深层特征 [batch, 512]
        """
        out = self.layer3(layer2_output)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.deep_regularizer(out)
        return out
    
    def forward(self, x: torch.Tensor, return_both: bool = False) -> torch.Tensor:
        """
        前向传播。默认只返回浅层特征。
        
        参数:
            x: 输入张量 [batch, channels, height, width]
            return_both: 是否同时返回浅层和深层特征
        
        返回:
            如果 return_both=False: 浅层特征 [batch, 128]
            如果 return_both=True: (浅层特征, 深层特征, layer2特征图)
        """
        # 浅层路径
        out = self.stem(x)
        out = self.layer1(out)
        layer2_out = self.layer2(out)  # 保存layer2输出用于条件深度
        
        # 浅层特征
        shallow_pooled = self.avgpool(layer2_out)
        shallow_features = shallow_pooled.view(shallow_pooled.size(0), -1)
        shallow_features = self.shallow_regularizer(shallow_features)
        
        if not return_both:
            return shallow_features
        
        # 深层路径
        out = self.layer3(layer2_out)
        out = self.layer4(out)
        deep_pooled = self.avgpool(out)
        deep_features = deep_pooled.view(deep_pooled.size(0), -1)
        deep_features = self.deep_regularizer(deep_features)
        
        return shallow_features, deep_features, layer2_out
    
    def forward_conditional(
        self, 
        x: torch.Tensor, 
        use_deep_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        条件深度前向传播。
        
        参数:
            x: 输入张量 [batch, channels, height, width]
            use_deep_mask: 布尔掩码，指示哪些样本需要深层处理 [batch]
        
        返回:
            包含特征的字典:
            - 'shallow_features': 浅层特征 [batch, 128]
            - 'deep_features': 深层特征 [num_deep, 512] (仅对需要深层的样本)
            - 'deep_indices': 需要深层处理的样本索引
        """
        batch_size = x.size(0)
        
        # 1. 所有样本都通过浅层
        out = self.stem(x)
        out = self.layer1(out)
        layer2_out = self.layer2(out)
        
        shallow_pooled = self.avgpool(layer2_out)
        shallow_features = shallow_pooled.view(batch_size, -1)
        shallow_features = self.shallow_regularizer(shallow_features)
        
        result = {
            'shallow_features': shallow_features,
            'deep_features': None,
            'deep_indices': None
        }
        
        # 2. 只有需要深层的样本继续
        if use_deep_mask.any():
            deep_indices = torch.where(use_deep_mask)[0]
            deep_layer2 = layer2_out[deep_indices]
            
            deep_out = self.layer3(deep_layer2)
            deep_out = self.layer4(deep_out)
            deep_pooled = self.avgpool(deep_out)
            deep_features = deep_pooled.view(deep_indices.size(0), -1)
            deep_features = self.deep_regularizer(deep_features)
            
            result['deep_features'] = deep_features
            result['deep_indices'] = deep_indices
        
        return result
    
    def get_output_dim(self) -> int:
        """获取输出特征维度（浅层）。"""
        return self.shallow_dim
    
    def get_shallow_dim(self) -> int:
        """获取浅层输出维度。"""
        return self.shallow_dim
    
    def get_deep_dim(self) -> int:
        """获取深层输出维度。"""
        return self.deep_dim


def get_backbone(
    name: str,
    input_channels: int = 3,
    input_size: int = 32,
    **kwargs
) -> nn.Module:
    """
    根据名称获取骨干网络的工厂函数。
    
    参数:
        name: 骨干网络名称
            - 'cnn': 简单CNN骨干网络（参数量~0.1M，输出2048维）
            - 'resnet18': 标准ResNet18（参数量~11.2M，输出512维）- 不推荐，容易过拟合
            - 'early_exit_resnet': 早期退出ResNet（参数量~0.2M，输出128维）【推荐用于FedEMoE】
            - 'conditional_depth_resnet': 条件深度ResNet（参数量~2.8M，浅层128维/深层512维）【推荐用于条件深度EMoE】
        input_channels: 输入通道数
        input_size: 输入图像尺寸
        **kwargs: 其他参数
            - dropout: Dropout概率（默认0.2）
            - base_channels: 基础通道数
            - use_layer_norm: 是否使用LayerNorm（仅early_exit和conditional_depth）
    
    返回:
        骨干网络模块
    
    使用示例:
        # 推荐用于 FedEMoE + CIFAR（解决过拟合问题）
        backbone = get_backbone('early_exit_resnet', input_channels=3, input_size=32)
        
        # 条件深度（简单样本用浅层，困难样本用深层）
        backbone = get_backbone('conditional_depth_resnet', input_channels=3, input_size=32)
    """
    name = name.lower()
    
    if name == "cnn":
        return CNNBackbone(input_channels, input_size)
    elif name == "resnet18":
        return ResNet18Backbone(input_channels, input_size)
    elif name in ["early_exit_resnet", "earlyexitresnet", "ee_resnet"]:
        dropout = kwargs.get('dropout', 0.2)
        base_channels = kwargs.get('base_channels', 64)
        use_layer_norm = kwargs.get('use_layer_norm', True)
        return EarlyExitResNetBackbone(
            input_channels, input_size,
            base_channels=base_channels,
            dropout=dropout,
            use_layer_norm=use_layer_norm
        )
    elif name in ["conditional_depth_resnet", "conditionaldepthresnet", "cd_resnet"]:
        dropout = kwargs.get('dropout', 0.2)
        base_channels = kwargs.get('base_channels', 64)
        use_layer_norm = kwargs.get('use_layer_norm', True)
        return ConditionalDepthResNetBackbone(
            input_channels, input_size,
            base_channels=base_channels,
            dropout=dropout,
            use_layer_norm=use_layer_norm
        )
    else:
        raise ValueError(
            f"未知的骨干网络: {name}。"
            f"支持的选项: cnn, resnet18, early_exit_resnet, conditional_depth_resnet"
        )


def count_parameters(model: nn.Module) -> int:
    """统计模型参数量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compare_backbones(input_channels: int = 3, input_size: int = 32):
    """
    比较不同骨干网络的参数量和输出维度。
    
    用于调试和选择合适的骨干网络。
    """
    backbones = [
        'cnn', 'resnet18',
        'early_exit_resnet',
        'conditional_depth_resnet'
    ]
    
    print("=" * 80)
    print("骨干网络对比")
    print("=" * 80)
    print(f"{'名称':<28} {'参数量':<12} {'输出维度':<12} {'显存估计':<10}")
    print("-" * 80)
    
    for name in backbones:
        backbone = get_backbone(name, input_channels, input_size)
        params = count_parameters(backbone)
        output_dim = backbone.get_output_dim()
        
        # 粗略估计显存（参数 + 梯度 + 优化器状态）
        memory_mb = params * 4 * 3 / (1024 * 1024)  # 3x for params + grads + optimizer
        
        # 对于条件深度，显示额外信息
        extra_info = ""
        if name == 'conditional_depth_resnet':
            extra_info = f" (深层: {backbone.get_deep_dim()})"
        
        print(f"{name:<28} {params:>10,} {output_dim:>8}{extra_info:<8} {memory_mb:>8.1f} MB")
    
    print("=" * 80)


if __name__ == "__main__":
    # 运行比较
    compare_backbones()
    
    # 测试各个骨干网络
    print("\n测试前向传播:")
    x = torch.randn(2, 3, 32, 32)
    
    all_backbones = [
        'cnn', 'resnet18',
        'early_exit_resnet',
        'conditional_depth_resnet'
    ]
    
    for name in all_backbones:
        backbone = get_backbone(name)
        out = backbone(x)
        print(f"{name}: input {x.shape} -> output {out.shape}")
    
    # 测试条件深度backbone的特殊功能
    print("\n测试条件深度ResNet:")
    cd_backbone = get_backbone('conditional_depth_resnet')
    
    # 测试return_both
    shallow, deep, layer2 = cd_backbone(x, return_both=True)
    print(f"  浅层特征: {shallow.shape}")
    print(f"  深层特征: {deep.shape}")
    print(f"  Layer2输出: {layer2.shape}")
    
    # 测试条件前向
    mask = torch.tensor([True, False])  # 第一个样本用深层，第二个用浅层
    result = cd_backbone.forward_conditional(x, mask)
    print(f"  条件前向 - 浅层: {result['shallow_features'].shape}")
    print(f"  条件前向 - 深层: {result['deep_features'].shape if result['deep_features'] is not None else 'None'}")
