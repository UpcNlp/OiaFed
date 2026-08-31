"""
证据深度学习 (EDL) 路由器模块。
实现基于不确定性的动态专家选择。

核心思想:
- 传统路由器输出 softmax 概率
- EDL 路由器输出证据 (evidence)，可以量化认知不确定性
- 不确定性高时激活更多专家，不确定性低时激活少量专家
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, NamedTuple, Optional


class RouterOutput(NamedTuple):
    """路由器输出结构。"""
    expert_probs: torch.Tensor      # 专家概率 [batch, num_experts]
    evidence: torch.Tensor          # 证据向量 [batch, num_experts]
    uncertainty: torch.Tensor       # 不确定性 [batch, 1]
    alpha: torch.Tensor             # Dirichlet 参数 [batch, num_experts]
    top_k_indices: torch.Tensor     # Top-K 专家索引 [batch, k_max]
    top_k_weights: torch.Tensor     # Top-K 专家权重 [batch, k_max]
    dynamic_k: torch.Tensor         # 动态 K 值 [batch]
    router_logits: torch.Tensor     # 路由器原始logits（softplus之前）[batch, num_experts]


class EDLRouter(nn.Module):
    """
    证据深度学习路由器。
    
    输出证据而非传统的 softmax 概率，用于:
    1. 量化认知不确定性
    2. 动态选择激活的专家数量
    
    数学原理:
    - 证据: e = Softplus(W·h + b)
    - Dirichlet 参数: α = e + 1
    - 总强度: S = Σα
    - 不确定性: u = K/S (K 为专家数)
    - 专家概率: p = α/S
    """
    
    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        hidden_dim: int = 256,
        k_min: int = 1,
        k_max: int = 4
    ):
        """
        初始化 EDL 路由器。
        
        参数:
            input_dim: 输入特征维度
            num_experts: 专家数量
            hidden_dim: 隐藏层维度
            k_min: 最少激活专家数
            k_max: 最多激活专家数
        """
        super().__init__()
        
        self.num_experts = num_experts
        self.k_min = k_min
        self.k_max = k_max
        
        # 证据网络: 输出非负证据值
        # self.evidence_network = nn.Sequential(
        #     nn.Linear(input_dim, hidden_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(hidden_dim, num_experts)
        # )

        self.evidence_network = nn.Linear(input_dim, num_experts)
        
        # Softplus 激活确保证据非负
        self.softplus = nn.Softplus()
    
    def forward(self, x: torch.Tensor) -> RouterOutput:
        """
        前向传播。
        
        参数:
            x: 输入特征 [batch, input_dim]
        
        返回:
            RouterOutput 包含所有路由信息
        """
        # 计算原始logits（softplus之前）
        router_logits = self.evidence_network(x)  # [batch, num_experts]
        
        # 计算证据（非负）
        evidence = self.softplus(router_logits)  # [batch, num_experts]
        
        # 转换为 Dirichlet 参数和不确定性
        alpha, uncertainty, expert_probs = self.evidence_to_dirichlet(evidence)
        
        # 计算动态 K 值
        dynamic_k = self.compute_dynamic_k(uncertainty)  # [batch]
        
        # Top-K 专家选择
        top_k_indices, top_k_weights = self.select_top_k_experts(
            expert_probs, dynamic_k
        )
        
        return RouterOutput(
            expert_probs=expert_probs,
            evidence=evidence,
            uncertainty=uncertainty,
            alpha=alpha,
            top_k_indices=top_k_indices,
            top_k_weights=top_k_weights,
            dynamic_k=dynamic_k,
            router_logits=router_logits
        )
    
    def evidence_to_dirichlet(
        self, 
        evidence: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        将证据转换为 Dirichlet 分布参数。
        
        参数:
            evidence: 证据向量 [batch, num_experts]
        
        返回:
            alpha: Dirichlet 参数 [batch, num_experts]
            uncertainty: 认知不确定性 [batch, 1]
            probs: 专家概率 [batch, num_experts]
        """
        # Dirichlet 参数 = 证据 + 1（先验）
        alpha = evidence + 1.0
        
        # 总强度 S = Σα
        S = alpha.sum(dim=-1, keepdim=True)  # [batch, 1]
        
        # 不确定性 u = K/S
        # S 越大（证据越充分），不确定性越低
        uncertainty = self.num_experts / S  # [batch, 1]
        
        # 专家概率（期望）
        probs = alpha / S  # [batch, num_experts]
        
        return alpha, uncertainty, probs
    
    def compute_dynamic_k(self, uncertainty: torch.Tensor) -> torch.Tensor:
        """
        根据不确定性计算动态 K 值。
        
        公式: k = k_min + (k_max - k_min) × u
        
        直觉:
        - 不确定性高 (u ≈ 1): 使用更多专家，综合多个意见
        - 不确定性低 (u ≈ 0): 使用少量专家，相信主要专家
        
        参数:
            uncertainty: 不确定性值 [batch, 1]
        
        返回:
            dynamic_k: 每个样本的 K 值 [batch]
        """
        # 将不确定性裁剪到 [0, 1] 范围
        u = uncertainty.squeeze(-1).clamp(0, 1)  # [batch]
        
        # 线性插值: k = k_min + (k_max - k_min) * u
        k_float = self.k_min + (self.k_max - self.k_min) * u
        
        # 取整（向上取整更保守）
        k = torch.ceil(k_float).long()
        
        # 确保在有效范围内
        k = k.clamp(self.k_min, self.k_max)
        
        return k
    
    def select_top_k_experts(
        self,
        expert_probs: torch.Tensor,
        dynamic_k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        动态 Top-K 专家选择。
        
        由于每个样本的 K 值可能不同，使用掩码处理变长选择。
        
        参数:
            expert_probs: 专家概率 [batch, num_experts]
            dynamic_k: 每个样本的 K 值 [batch]
        
        返回:
            indices: Top-K 专家索引 [batch, k_max]（无效位置填充 0）
            weights: Top-K 权重 [batch, k_max]（无效位置权重为 0）
        """
        device = expert_probs.device
        
        # 先选择 k_max 个专家
        top_values, top_indices = torch.topk(expert_probs, self.k_max, dim=-1)
        
        # 创建掩码：只保留前 dynamic_k 个
        # mask[i, j] = 1 if j < dynamic_k[i] else 0
        position_indices = torch.arange(self.k_max, device=device).unsqueeze(0)  # [1, k_max]
        mask = (position_indices < dynamic_k.unsqueeze(1)).float()  # [batch, k_max]
        
        # 应用掩码
        masked_values = top_values * mask
        
        # 重新归一化权重（只在有效专家上）
        weight_sum = masked_values.sum(dim=-1, keepdim=True) + 1e-8
        normalized_weights = masked_values / weight_sum
        
        return top_indices, normalized_weights


class LoadBalancingLoss(nn.Module):
    """
    负载均衡损失。
    
    鼓励各专家被均匀使用，避免某些专家过载或闲置。
    使用 KL 散度衡量实际分布与均匀分布的差异。
    """
    
    def __init__(self, num_experts: int):
        """
        初始化负载均衡损失。
        
        参数:
            num_experts: 专家数量
        """
        super().__init__()
        self.num_experts = num_experts
    
    def forward(
        self, 
        expert_probs: torch.Tensor,
        top_k_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算负载均衡损失。
        
        参数:
            expert_probs: 专家概率 [batch, num_experts]
            top_k_indices: 选中的专家索引（可选）
        
        返回:
            负载均衡损失标量
        """
        # 计算每个专家的平均负载
        if top_k_indices is not None:
            # 基于选择计算负载
            one_hot = F.one_hot(top_k_indices, self.num_experts).float()
            load = one_hot.sum(dim=1).mean(dim=0)  # [num_experts]
        else:
            # 基于概率计算负载
            load = expert_probs.mean(dim=0)  # [num_experts]
        
        # 目标：均匀分布
        uniform = torch.ones_like(load) / self.num_experts
        
        # KL 散度
        load = load + 1e-8  # 避免 log(0)
        kl_loss = (load * (load.log() - uniform.log())).sum()
        
        return kl_loss
