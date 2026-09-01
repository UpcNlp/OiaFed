"""
MoE (混合专家) 模型，不带 EDL 路由器。

这是 EMoE 的简化版本，使用普通的 Softmax 路由器而非 EDL 路由器。
用于消融实验，验证 EDL 路由器的贡献。
"""

from typing import Dict, NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import get_backbone
from .experts import ExpertGroup


class SoftmaxRouterOutput(NamedTuple):
    """Softmax 路由器输出结构。"""
    expert_probs: torch.Tensor  # 专家概率 [batch, num_experts]
    top_k_indices: torch.Tensor  # Top-K 专家索引 [batch, k]
    top_k_weights: torch.Tensor  # Top-K 专家权重 [batch, k]
    router_logits: torch.Tensor = None  # 路由器原始logits [batch, num_experts]（用于校准分析）


class SoftmaxRouter(nn.Module):
    """
    传统 Softmax 路由器。

    与 EDL 路由器不同，这个路由器:
    1. 使用固定的 K 值（不是动态的）
    2. 输出 Softmax 概率而非证据
    3. 没有不确定性量化

    网络结构与 EDLRouter 保持一致（单层 Linear），
    仅将路由机制从 EDL (Softplus → Dirichlet) 替换为 Softmax。
    """

    def __init__(
            self,
            input_dim: int,
            num_experts: int,
            hidden_dim: int = 256,
            top_k: int = 2,
            temperature: float = 1.0
    ):
        """
        初始化 Softmax 路由器。

        参数:
            input_dim: 输入特征维度
            num_experts: 专家数量
            hidden_dim: 隐藏层维度（保留接口兼容，未使用）
            top_k: 激活的专家数量（固定）
            temperature: Softmax 温度参数
        """
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k
        self.temperature = temperature

        # 路由网络: 单层 Linear，与 EDLRouter.evidence_network 结构一致
        self.router_network = nn.Linear(input_dim, num_experts)

    def forward(self, x: torch.Tensor) -> SoftmaxRouterOutput:
        """
        前向传播。

        参数:
            x: 输入特征 [batch, input_dim]

        返回:
            SoftmaxRouterOutput 包含路由信息
        """
        # 计算路由 logits
        logits = self.router_network(x)  # [batch, num_experts]

        # Softmax 概率
        expert_probs = F.softmax(logits / self.temperature, dim=-1)

        # Top-K 专家选择
        top_values, top_indices = torch.topk(expert_probs, self.top_k, dim=-1)

        # 归一化权重
        top_weights = top_values / (top_values.sum(dim=-1, keepdim=True) + 1e-8)

        return SoftmaxRouterOutput(
            expert_probs=expert_probs,
            top_k_indices=top_indices,
            top_k_weights=top_weights,
            router_logits=logits
        )


class MoEOutput(NamedTuple):
    """MoE 模型输出结构。"""
    logits: torch.Tensor  # 分类 logits [batch, num_classes]
    router_output: SoftmaxRouterOutput  # 路由器输出
    expert_outputs: torch.Tensor  # 各专家输出 [batch, k, num_classes]


class MoE(nn.Module):
    """
    混合专家模型 (Mixture-of-Experts)，不使用 EDL。

    整体架构:
        输入 → 骨干网络 → Softmax路由器 → 固定Top-K选择 → 专家加权聚合 → 输出

    与 EMoE 的区别:
        1. 使用 Softmax 路由器而非 EDL 路由器
        2. 使用固定的 K 值而非动态 K
        3. 没有不确定性量化
    """

    def __init__(
            self,
            num_classes: int,
            num_experts: int = 8,
            backbone: str = "cnn",
            input_channels: int = 3,
            input_size: int = 32,
            expert_hidden_dim: int = 256,
            router_hidden_dim: int = 256,
            top_k: int = 2,
            dropout: float = 0.1
    ):
        """
        初始化 MoE 模型。

        参数:
            num_classes: 分类类别数
            num_experts: 专家数量
            backbone: 骨干网络类型 ('cnn' 或 'resnet18')
            input_channels: 输入通道数
            input_size: 输入图像尺寸
            expert_hidden_dim: 专家隐藏层维度
            router_hidden_dim: 路由器隐藏层维度
            top_k: 激活的专家数量（固定）
            dropout: Dropout 概率
        """
        super().__init__()

        self.num_classes = num_classes
        self.num_experts = num_experts
        self.top_k = top_k

        # 1. 骨干网络：提取特征
        self.backbone = get_backbone(backbone, input_channels, input_size)
        feature_dim = self.backbone.get_output_dim()

        # 2. Softmax 路由器
        self.router = SoftmaxRouter(
            input_dim=feature_dim,
            num_experts=num_experts,
            hidden_dim=router_hidden_dim,
            top_k=top_k
        )

        # 3. 专家组：多个独立的专家网络
        self.experts = ExpertGroup(
            num_experts=num_experts,
            input_dim=feature_dim,
            hidden_dim=expert_hidden_dim,
            output_dim=num_classes,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> MoEOutput:
        """
        前向传播。

        参数:
            x: 输入图像 [batch, channels, height, width]

        返回:
            MoEOutput 包含 logits、路由信息和专家输出
        """
        # 1. 特征提取
        features = self.backbone(x)  # [batch, feature_dim]

        # 2. 路由
        router_output = self.router(features)

        # 3. 专家计算（只计算被选中的专家）
        expert_outputs = self.experts(
            features,
            router_output.top_k_indices
        )  # [batch, top_k, num_classes]

        # 4. 加权聚合
        weights = router_output.top_k_weights.unsqueeze(-1)
        logits = (expert_outputs * weights).sum(dim=1)

        return MoEOutput(
            logits=logits,
            router_output=router_output,
            expert_outputs=expert_outputs
        )

    def get_expert_utilization(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        获取专家使用情况统计。

        参数:
            x: 输入图像

        返回:
            包含专家使用统计的字典
        """
        output = self.forward(x)

        selected_experts = output.router_output.top_k_indices
        weights = output.router_output.top_k_weights

        utilization = torch.zeros(self.num_experts, device=x.device)
        for i in range(self.num_experts):
            mask = (selected_experts == i)
            utilization[i] = (weights * mask.float()).sum()

        utilization = utilization / (weights.sum() + 1e-8)

        return {
            "utilization": utilization,
            "avg_k": torch.tensor(self.top_k, device=x.device).float()
        }


def create_moe_model(
        num_classes: int,
        num_experts: int = 8,
        backbone: str = "cnn",
        input_channels: int = 3,
        input_size: int = 32,
        expert_hidden_dim: int = 256,
        top_k: int = 2,
        device: torch.device = None,
        **kwargs
) -> MoE:
    """
    创建 MoE 模型的工厂函数。

    参数:
        num_classes: 类别数
        num_experts: 专家数量
        backbone: 骨干网络类型
        input_channels: 输入通道数
        input_size: 输入图像尺寸
        expert_hidden_dim: 专家隐藏层维度
        top_k: 激活的专家数量
        device: 目标设备
        **kwargs: 其他参数（忽略）

    返回:
        MoE 模型实例
    """
    model = MoE(
        num_classes=num_classes,
        num_experts=num_experts,
        backbone=backbone,
        input_channels=input_channels,
        input_size=input_size,
        expert_hidden_dim=expert_hidden_dim,
        top_k=top_k
    )

    if device is not None:
        model = model.to(device)

    return model