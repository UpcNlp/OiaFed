"""
论文标准MLP模型 - TPAMI 2025

从 methods/models/paper_mlp.py 迁移到 src/

符合论文 "Fundamentals and Experimental Analysis of Federated Learning Algorithms:
A Comparative Study on Non-IID Data Silos" (TPAMI 2025) Table III实验设置
"""
import torch
import torch.nn as nn
from ...registry import model


@model(
    name='paper_mlp',
    description='论文标准MLP模型(TPAMI 2025) - 适配所有表格数据集',
    version='1.0',
    task='classification'
)
class PaperMLP(nn.Module):
    """
    论文标准MLP模型 - 适配所有表格数据集

    Architecture (TPAMI 2025 Paper, Section VI):
    -------------------------------------------
    "For the tabular datasets, we employ a standard MLP with
    three hidden layers (32, 16, and 8 units)."

    - FC1: input_dim → 32, ReLU
    - FC2: 32 → 16, ReLU
    - FC3: 16 → 8, ReLU
    - FC4: 8 → num_classes

    支持的数据集:
    - Adult: 99特征, 2分类
    - FCUBE: 3特征, 2分类
    """

    def __init__(self,
                 input_dim: int,
                 num_classes: int = 2):
        """
        初始化论文标准MLP模型

        Args:
            input_dim: 输入特征维度
            num_classes: 分类数量
        """
        super().__init__()

        # 全连接层(论文标准: 32 → 16 → 8)
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, num_classes)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量, shape=(batch_size, input_dim)

        Returns:
            logits: 分类logits, shape=(batch_size, num_classes)
        """
        # 确保输入是2D张量
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # 第一层
        x = self.fc1(x)
        x = torch.relu(x)

        # 第二层
        x = self.fc2(x)
        x = torch.relu(x)

        # 第三层
        x = self.fc3(x)
        x = torch.relu(x)

        # 输出层
        x = self.fc4(x)

        return x


# 为不同数据集创建便捷的工厂类

@model(
    name='adult_paper_mlp',
    description='Adult数据集论文标准MLP(99特征,2分类)',
    version='1.0',
    task='classification',
    input_shape=(99,),
    output_shape=(2,)
)
class Adult_PaperMLP(PaperMLP):
    """Adult数据集的论文标准MLP"""
    def __init__(self, num_classes: int = 2):
        super().__init__(
            input_dim=99,
            num_classes=num_classes
        )


@model(
    name='fcube_paper_mlp',
    description='FCUBE数据集论文标准MLP(3特征,2分类)',
    version='1.0',
    task='classification',
    input_shape=(3,),
    output_shape=(2,)
)
class FCUBE_PaperMLP(PaperMLP):
    """FCUBE数据集的论文标准MLP"""
    def __init__(self, num_classes: int = 2):
        super().__init__(
            input_dim=3,
            num_classes=num_classes
        )
