"""
MLP 模型 (表格数据)

包含适用于 Adult、FCUBE 等表格数据集的 MLP 模型。

注册名:
- paper_mlp: 通用 MLP (TPAMI 2025)
- adult_mlp: Adult 数据集专用
- fcube_mlp: FCUBE 数据集专用
"""

import torch
import torch.nn as nn

from ...registry.decorators import model


# ==================== 通用 MLP (TPAMI 2025) ====================

@model(
    name='paper_mlp',
    description='论文标准 MLP (TPAMI 2025) - 适配表格数据',
    task='classification',
    version='1.0'
)
class PaperMLP(nn.Module):
    """
    论文标准 MLP 模型
    
    符合 TPAMI 2025 论文 Section VI 标准架构。
    
    架构:
    "For the tabular datasets, we employ a standard MLP with
    three hidden layers (32, 16, and 8 units)."
    
    - FC1: input_dim → 32, ReLU
    - FC2: 32 → 16, ReLU
    - FC3: 16 → 8, ReLU
    - FC4: 8 → num_classes
    
    支持的数据集:
    - Adult: 99 特征, 2 分类
    - FCUBE: 3 特征, 2 分类
    """

    def __init__(self, input_dim: int, num_classes: int = 2):
        """
        Args:
            input_dim: 输入特征维度
            num_classes: 分类数量
        """
        super().__init__()

        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, num_classes)

    def forward(self, x):
        # 确保输入是 2D
        if x.dim() == 1:
            x = x.unsqueeze(0)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# ==================== Adult 数据集专用 ====================

@model(
    name='adult_mlp',
    description='Adult 数据集 MLP (99 特征, 2 分类)',
    task='classification',
    version='1.0',
    input_shape=(99,),
    output_shape=(2,)
)
class AdultMLP(PaperMLP):
    """
    Adult 数据集专用 MLP
    
    输入: 99 维特征向量
    输出: 2 分类 (收入 >50K / <=50K)
    """

    def __init__(self, num_classes: int = 2):
        super().__init__(input_dim=99, num_classes=num_classes)


# ==================== FCUBE 数据集专用 ====================

@model(
    name='fcube_mlp',
    description='FCUBE 数据集 MLP (3 特征, 2 分类)',
    task='classification',
    version='1.0',
    input_shape=(3,),
    output_shape=(2,)
)
class FCUBEMLP(PaperMLP):
    """
    FCUBE 数据集专用 MLP
    
    输入: 3 维特征向量
    输出: 2 分类
    """

    def __init__(self, num_classes: int = 2):
        super().__init__(input_dim=3, num_classes=num_classes)


# ==================== 导出 ====================

__all__ = [
    "PaperMLP",
    "AdultMLP",
    "FCUBEMLP",
]
