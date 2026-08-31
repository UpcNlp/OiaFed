"""
EMoE (证据混合专家) 完整模型。
集成骨干网络、EDL 路由器和专家组。
"""

from typing import Dict, NamedTuple

import torch
import torch.nn as nn

from .backbones import get_backbone
from .edl_router import EDLRouter, RouterOutput
from .experts import ExpertGroup


class EMoEOutput(NamedTuple):
    """EMoE 模型输出结构。"""
    logits: torch.Tensor            # 分类 logits [batch, num_classes]
    router_output: RouterOutput     # 路由器输出
    expert_outputs: torch.Tensor    # 各专家输出 [batch, k, num_classes]


class EMoE(nn.Module):
    """
    证据混合专家模型 (Evidential Mixture-of-Experts)。
    
    整体架构:
        输入 → 骨干网络 → EDL路由器 → 动态Top-K选择 → 专家加权聚合 → 输出
    
    核心创新:
        1. EDL 路由器输出证据，量化认知不确定性
        2. 根据不确定性动态调整激活专家数量
        3. 不确定时多参考几个专家，确定时只用最相关的专家
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
        k_min: int = 1,
        k_max: int = 4,
        dropout: float = 0.1
    ):
        """
        初始化 EMoE 模型。
        
        参数:
            num_classes: 分类类别数
            num_experts: 专家数量
            backbone: 骨干网络类型 ('cnn' 或 'resnet18')
            input_channels: 输入通道数
            input_size: 输入图像尺寸
            expert_hidden_dim: 专家隐藏层维度
            router_hidden_dim: 路由器隐藏层维度
            k_min: 最少激活专家数
            k_max: 最多激活专家数
            dropout: Dropout 概率
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.num_experts = num_experts
        self.k_min = k_min
        self.k_max = k_max
        
        # 1. 骨干网络：提取特征
        self.backbone = get_backbone(backbone, input_channels, input_size)
        feature_dim = self.backbone.get_output_dim()
        
        # 2. EDL 路由器：计算证据和不确定性，选择专家
        self.router = EDLRouter(
            input_dim=feature_dim,
            num_experts=num_experts,
            hidden_dim=router_hidden_dim,
            k_min=k_min,
            k_max=k_max
        )
        
        # 3. 专家组：多个独立的专家网络
        self.experts = ExpertGroup(
            num_experts=num_experts,
            input_dim=feature_dim,
            hidden_dim=expert_hidden_dim,
            output_dim=num_classes,
            dropout=dropout
        )
    
    def forward(self, x: torch.Tensor) -> EMoEOutput:
        """
        前向传播。
        
        参数:
            x: 输入图像 [batch, channels, height, width]
        
        返回:
            EMoEOutput 包含 logits、路由信息和专家输出
        """
        # 1. 特征提取
        features = self.backbone(x)  # [batch, feature_dim]
        
        # 2. EDL 路由
        router_output = self.router(features)
        
        # 3. 专家计算（只计算被选中的专家）
        expert_outputs = self.experts(
            features, 
            router_output.top_k_indices
        )  # [batch, k_max, num_classes]
        
        # 4. 加权聚合
        # weights: [batch, k_max] → [batch, k_max, 1]
        weights = router_output.top_k_weights.unsqueeze(-1)
        
        # 加权求和: [batch, k_max, num_classes] * [batch, k_max, 1] → [batch, num_classes]
        logits = (expert_outputs * weights).sum(dim=1)
        
        return EMoEOutput(
            logits=logits,
            router_output=router_output,
            expert_outputs=expert_outputs
        )
    
    def forward_with_fixed_k(self, x: torch.Tensor, k: int) -> EMoEOutput:
        """
        使用固定 K 值的前向传播（用于消融实验）。
        
        参数:
            x: 输入图像
            k: 固定的专家数量
        
        返回:
            EMoEOutput
        """
        # 特征提取
        features = self.backbone(x)
        
        # 获取路由概率（不使用动态 K）
        evidence = self.router.softplus(self.router.evidence_network(features))
        alpha, uncertainty, expert_probs = self.router.evidence_to_dirichlet(evidence)
        
        # 固定 K 的 Top-K 选择
        top_values, top_indices = torch.topk(expert_probs, k, dim=-1)
        top_weights = top_values / (top_values.sum(dim=-1, keepdim=True) + 1e-8)
        
        # 填充到 k_max 大小（便于统一处理）
        batch_size = x.shape[0]
        device = x.device
        
        if k < self.k_max:
            # 填充索引和权重
            padded_indices = torch.zeros(batch_size, self.k_max, dtype=torch.long, device=device)
            padded_weights = torch.zeros(batch_size, self.k_max, device=device)
            padded_indices[:, :k] = top_indices
            padded_weights[:, :k] = top_weights
        else:
            padded_indices = top_indices
            padded_weights = top_weights
        
        # 创建 RouterOutput
        router_output = RouterOutput(
            expert_probs=expert_probs,
            evidence=evidence,
            uncertainty=uncertainty,
            alpha=alpha,
            top_k_indices=padded_indices,
            top_k_weights=padded_weights,
            dynamic_k=torch.full((batch_size,), k, device=device)
        )
        
        # 专家计算
        expert_outputs = self.experts(features, padded_indices)
        
        # 加权聚合
        weights = padded_weights.unsqueeze(-1)
        logits = (expert_outputs * weights).sum(dim=1)
        
        return EMoEOutput(
            logits=logits,
            router_output=router_output,
            expert_outputs=expert_outputs
        )
    
    def get_uncertainty(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取输入的不确定性。
        
        参数:
            x: 输入图像
        
        返回:
            不确定性值 [batch, 1]
        """
        features = self.backbone(x)
        router_output = self.router(features)
        return router_output.uncertainty
    
    def get_expert_utilization(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        获取专家使用情况统计。
        
        参数:
            x: 输入图像
        
        返回:
            包含专家使用统计的字典
        """
        output = self.forward(x)
        
        # 统计每个专家被选中的次数
        selected_experts = output.router_output.top_k_indices  # [batch, k_max]
        weights = output.router_output.top_k_weights  # [batch, k_max]
        
        # 计算每个专家的加权使用率
        utilization = torch.zeros(self.num_experts, device=x.device)
        for i in range(self.num_experts):
            mask = (selected_experts == i)
            utilization[i] = (weights * mask.float()).sum()
        
        # 归一化
        utilization = utilization / (weights.sum() + 1e-8)
        
        return {
            "utilization": utilization,
            "avg_k": output.router_output.dynamic_k.float().mean(),
            "avg_uncertainty": output.router_output.uncertainty.mean()
        }


def create_emoe_model(
    num_classes: int,
    num_experts: int = 8,
    backbone: str = "cnn",
    input_channels: int = 3,
    input_size: int = 32,
    expert_hidden_dim: int = 256,
    k_min: int = 1,
    k_max: int = 4,
    device: torch.device = None,
    **kwargs
) -> EMoE:
    """
    创建 EMoE 模型的工厂函数。
    
    参数:
        num_classes: 类别数
        num_experts: 专家数量
        backbone: 骨干网络类型
        input_channels: 输入通道数
        input_size: 输入图像尺寸
        expert_hidden_dim: 专家隐藏层维度
        k_min: 最少激活专家数
        k_max: 最多激活专家数
        device: 目标设备
        **kwargs: 其他参数（忽略）
    
    返回:
        EMoE 模型实例
    """
    model = EMoE(
        num_classes=num_classes,
        num_experts=num_experts,
        backbone=backbone,
        input_channels=input_channels,
        input_size=input_size,
        expert_hidden_dim=expert_hidden_dim,
        k_min=k_min,
        k_max=k_max
    )
    
    if device is not None:
        model = model.to(device)
    
    return model
