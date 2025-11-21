"""
论文标准MLP模型 - TPAMI 2025
fedcl/methods/models/paper_mlp.py

符合论文 "Fundamentals and Experimental Analysis of Federated Learning Algorithms:
A Comparative Study on Non-IID Data Silos" (TPAMI 2025) Table III实验设置

论文第11页描述（Section VI）：
"For the tabular datasets, we employ a standard MLP with three hidden layers (32, 16, and 8 units)."
"""
import torch
import torch.nn as nn
from typing import Dict

from fedcl.api.decorators import model
from fedcl.methods.models.base import FederatedModel


@model(
    name='PaperMLP',
    description='论文标准MLP模型（TPAMI 2025）- 适配所有表格数据集',
    task='classification',
    version='1.0'
)
class PaperMLP(FederatedModel):
    """
    论文标准MLP模型 - 适配所有表格数据集

    Architecture (TPAMI 2025 Paper, Section VI):
    -------------------------------------------
    - FC1: input_dim → 32
    - ReLU
    - FC2: 32 → 16
    - ReLU
    - FC3: 16 → 8
    - ReLU
    - FC4: 8 → num_classes

    支持的数据集：
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

        # 设置元数据
        self.set_metadata(
            task_type='classification',
            input_shape=(input_dim,),
            output_shape=(num_classes,)
        )

        # 全连接层（论文标准: 32 → 16 → 8）
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, num_classes)

        # 损失函数
        self.criterion = nn.CrossEntropyLoss()

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

    def get_weights_as_dict(self) -> Dict[str, torch.Tensor]:
        """获取模型权重为字典格式"""
        return {k: v.cpu().clone() for k, v in self.state_dict().items()}

    def set_weights_from_dict(self, weights: Dict[str, torch.Tensor], strict: bool = True):
        """从字典设置模型权重"""
        self.load_state_dict(weights, strict=strict)

    def get_param_count(self) -> int:
        """获取参数数量"""
        return sum(p.numel() for p in self.parameters())


# 为不同数据集创建便捷的工厂函数

@model(
    name='Adult_PaperMLP',
    description='Adult数据集论文标准MLP（99特征，2分类）',
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
    name='FCUBE_PaperMLP',
    description='FCUBE数据集论文标准MLP（3特征，2分类）',
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
