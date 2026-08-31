"""
MoE 专家网络模块。
实现单个专家网络和专家组。
"""

from typing import Optional

import torch
import torch.nn as nn


class Expert(nn.Module):
    """
    单个专家网络。
    
    结构: Linear→ReLU→Linear→ReLU→Linear（3层 MLP）
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1
    ):
        """
        初始化专家网络。
        
        参数:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度（类别数）
            dropout: Dropout 概率
        """
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        
        参数:
            x: 输入特征 [batch, input_dim]
        
        返回:
            输出 logits [batch, output_dim]
        """
        return self.network(x)


class ExpertGroup(nn.Module):
    """
    专家组：管理多个专家网络。
    """
    
    def __init__(
        self,
        num_experts: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1
    ):
        """
        初始化专家组。
        
        参数:
            num_experts: 专家数量
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
            dropout: Dropout 概率
        """
        super().__init__()
        
        self.num_experts = num_experts
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 创建专家列表
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim, output_dim, dropout)
            for _ in range(num_experts)
        ])
    
    def forward(
        self, 
        x: torch.Tensor, 
        expert_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播。
        
        参数:
            x: 输入特征 [batch, input_dim]
            expert_indices: 要使用的专家索引 [batch, k] 或 None（使用全部专家）
        
        返回:
            专家输出 [batch, num_experts/k, output_dim]
        """
        batch_size = x.shape[0]
        
        if expert_indices is None:
            # 使用所有专家
            outputs = torch.stack([
                expert(x) for expert in self.experts
            ], dim=1)  # [batch, num_experts, output_dim]
        else:
            # 只使用选中的专家
            k = expert_indices.shape[1]
            outputs = torch.zeros(
                batch_size, k, self.output_dim, 
                device=x.device, dtype=x.dtype
            )
            
            for i in range(k):
                # 获取第 i 个位置的专家索引
                idx = expert_indices[:, i]  # [batch]
                
                # 按专家分组计算（提高效率）
                for expert_id in range(self.num_experts):
                    mask = (idx == expert_id)
                    if mask.any():
                        outputs[mask, i] = self.experts[expert_id](x[mask])
        
        return outputs
    
    def forward_single_expert(self, x: torch.Tensor, expert_id: int) -> torch.Tensor:
        """
        只使用单个专家进行前向传播。
        
        参数:
            x: 输入特征 [batch, input_dim]
            expert_id: 专家索引
        
        返回:
            专家输出 [batch, output_dim]
        """
        return self.experts[expert_id](x)


class SharedBottomExperts(nn.Module):
    """
    共享底层的专家组（参数高效版本）。
    
    所有专家共享一个底层网络，只有顶层是独立的。
    可以减少参数量，同时保持专家的多样性。
    """
    
    def __init__(
        self,
        num_experts: int,
        input_dim: int,
        shared_dim: int,
        expert_dim: int,
        output_dim: int,
        dropout: float = 0.1
    ):
        """
        初始化共享底层专家组。
        
        参数:
            num_experts: 专家数量
            input_dim: 输入特征维度
            shared_dim: 共享层维度
            expert_dim: 专家特有层维度
            output_dim: 输出维度
            dropout: Dropout 概率
        """
        super().__init__()
        
        self.num_experts = num_experts
        self.output_dim = output_dim
        
        # 共享底层网络
        self.shared_bottom = nn.Sequential(
            nn.Linear(input_dim, shared_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # 各专家独立的顶层网络
        self.expert_tops = nn.ModuleList([
            nn.Sequential(
                nn.Linear(shared_dim, expert_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(expert_dim, output_dim)
            )
            for _ in range(num_experts)
        ])
    
    def forward(
        self, 
        x: torch.Tensor, 
        expert_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播。
        
        参数:
            x: 输入特征 [batch, input_dim]
            expert_indices: 要使用的专家索引 [batch, k] 或 None
        
        返回:
            专家输出 [batch, num_experts/k, output_dim]
        """
        # 先通过共享底层
        shared_features = self.shared_bottom(x)  # [batch, shared_dim]
        
        batch_size = x.shape[0]
        
        if expert_indices is None:
            # 使用所有专家
            outputs = torch.stack([
                top(shared_features) for top in self.expert_tops
            ], dim=1)
        else:
            # 只使用选中的专家
            k = expert_indices.shape[1]
            outputs = torch.zeros(
                batch_size, k, self.output_dim,
                device=x.device, dtype=x.dtype
            )
            
            for i in range(k):
                idx = expert_indices[:, i]
                for expert_id in range(self.num_experts):
                    mask = (idx == expert_id)
                    if mask.any():
                        outputs[mask, i] = self.expert_tops[expert_id](shared_features[mask])
        
        return outputs
