"""
证据深度学习 (EDL) 损失函数模块。
实现基于 Dirichlet 分布的不确定性感知损失。

参考文献:
    Sensoy et al. "Evidential Deep Learning to Quantify Classification Uncertainty"
    NeurIPS 2018
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


def compute_kl_divergence(
    alpha: torch.Tensor,
    num_classes: int,
    target: torch.Tensor
) -> torch.Tensor:
    """
    计算 Dirichlet 分布与均匀先验之间的 KL 散度。
    
    KL[Dir(α̃) || Dir(1)]，其中 α̃ 移除了真实类别的证据。
    
    参数:
        alpha: Dirichlet 参数 [batch, num_classes]
        num_classes: 类别数
        target: 目标类别索引 [batch]
    
    返回:
        KL 散度 [batch]
    """
    # 创建 one-hot 编码
    one_hot = F.one_hot(target, num_classes).float()
    
    # 移除真实类别的证据: α̃ = 1 + (1 - y) ⊙ (α - 1)
    # 这样做是为了只惩罚错误类别的证据
    alpha_tilde = 1.0 + (1.0 - one_hot) * (alpha - 1.0)
    
    # 计算 KL 散度
    S_tilde = alpha_tilde.sum(dim=-1, keepdim=True)
    
    # KL[Dir(α̃) || Dir(1)] 的解析形式
    kl = (
        torch.lgamma(S_tilde.squeeze(-1)) 
        - torch.lgamma(torch.tensor(num_classes, dtype=torch.float, device=alpha.device))
        - torch.lgamma(alpha_tilde).sum(dim=-1)
        + ((alpha_tilde - 1.0) * (
            torch.digamma(alpha_tilde) - torch.digamma(S_tilde)
        )).sum(dim=-1)
    )
    
    return kl


def compute_edl_mse_loss(
    alpha: torch.Tensor,
    target: torch.Tensor,
    num_classes: int
) -> torch.Tensor:
    """
    计算基于 MSE 的 EDL 损失（贝叶斯风险）。
    
    L = Σ_k [(y_k - p̂_k)² + p̂_k(1 - p̂_k)/(S + 1)]
    
    其中:
    - p̂_k = α_k / S 是预测概率
    - 第一项是预测误差
    - 第二项是方差（不确定性惩罚）
    
    参数:
        alpha: Dirichlet 参数 [batch, num_classes]
        target: 目标类别索引 [batch]
        num_classes: 类别数
    
    返回:
        MSE EDL 损失 [batch]
    """
    # one-hot 编码
    y = F.one_hot(target, num_classes).float()
    
    # 总强度
    S = alpha.sum(dim=-1, keepdim=True)
    
    # 预测概率
    p = alpha / S
    
    # 误差项: (y - p)²
    err = (y - p) ** 2
    
    # 方差项: p(1-p)/(S+1)
    var = p * (1 - p) / (S + 1)
    
    # 总损失
    loss = (err + var).sum(dim=-1)
    
    return loss


def compute_edl_digamma_loss(
    alpha: torch.Tensor,
    target: torch.Tensor,
    num_classes: int
) -> torch.Tensor:
    """
    计算基于 Digamma 函数的 EDL 损失。
    
    使用 digamma 函数处理对数期望。
    
    参数:
        alpha: Dirichlet 参数 [batch, num_classes]
        target: 目标类别索引 [batch]
        num_classes: 类别数
    
    返回:
        Digamma EDL 损失 [batch]
    """
    y = F.one_hot(target, num_classes).float()
    S = alpha.sum(dim=-1, keepdim=True)
    
    # 使用 digamma 函数
    loss = (y * (torch.digamma(S) - torch.digamma(alpha))).sum(dim=-1)
    
    return loss


def compute_edl_log_loss(
    alpha: torch.Tensor,
    target: torch.Tensor,
    num_classes: int
) -> torch.Tensor:
    """
    计算基于对数的 EDL 损失。
    
    参数:
        alpha: Dirichlet 参数 [batch, num_classes]
        target: 目标类别索引 [batch]
        num_classes: 类别数
    
    返回:
        Log EDL 损失 [batch]
    """
    y = F.one_hot(target, num_classes).float()
    S = alpha.sum(dim=-1, keepdim=True)
    
    # 对数损失
    loss = (y * (torch.log(S) - torch.log(alpha))).sum(dim=-1)
    
    return loss


class EDLLoss(nn.Module):
    """
    EDL 综合损失函数。
    
    总损失: L = L_task + λ₁·L_EDL + λ_t·λ₂·L_KL
    
    其中:
    - L_task: 任务损失（交叉熵）
    - L_EDL: EDL 损失（MSE/Digamma/Log 三选一）
    - L_KL: KL 散度正则化
    - λ_t: 退火系数，随训练逐渐增加
    """
    
    def __init__(
        self,
        num_classes: int,
        lambda1: float = 1.0,
        lambda2: float = 0.1,
        annealing_epochs: int = 10,
        edl_loss_type: str = "mse"
    ):
        """
        初始化 EDL 损失函数。
        
        参数:
            num_classes: 类别数
            lambda1: EDL 损失权重
            lambda2: KL 散度权重
            annealing_epochs: KL 退火轮数
            edl_loss_type: EDL 损失类型 ('mse', 'digamma', 'log')
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.annealing_epochs = annealing_epochs
        self.edl_loss_type = edl_loss_type
        
        # 任务损失（交叉熵）
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(
        self,
        logits: torch.Tensor,
        evidence: torch.Tensor,
        target: torch.Tensor,
        epoch: int = 0
    ) -> Dict[str, torch.Tensor]:
        """
        计算综合损失。
        
        参数:
            logits: 模型输出 logits [batch, num_classes]
            evidence: 路由器证据向量 [batch, num_experts]
                      注意：此参数仅用于专家选择，不直接参与EDL损失计算
            target: 目标标签 [batch]
            epoch: 当前轮数（用于退火）
        
        返回:
            包含各项损失的字典
            
        设计说明:
            - 路由器的evidence维度是num_experts，用于动态K选择
            - 路由器通过最终分类损失的反向传播间接学习，没有直接的损失约束
            - EDL损失基于分类logits计算，对"类别预测"进行不确定性建模
        """
        batch_size = logits.shape[0]
        
        # 1. 任务损失（交叉熵）
        loss_task = self.ce_loss(logits, target)
        
        # 2. EDL 损失 - 始终基于分类logits计算
        # 路由器的evidence（维度=num_experts）仅用于专家选择，不用于此处
        # 从logits生成分类证据（维度=num_classes）
        evidence_for_edl = F.softplus(logits)  # [batch, num_classes]
        
        alpha = evidence_for_edl + 1.0
        
        if self.edl_loss_type == "mse":
            loss_edl = compute_edl_mse_loss(alpha, target, self.num_classes)
        elif self.edl_loss_type == "digamma":
            loss_edl = compute_edl_digamma_loss(alpha, target, self.num_classes)
        else:  # log
            loss_edl = compute_edl_log_loss(alpha, target, self.num_classes)
        
        # 3. KL 散度损失（带退火）
        loss_kl = compute_kl_divergence(alpha, self.num_classes, target)
        
        # 退火系数: 从 0 逐渐增加到 1
        annealing_coef = min(1.0, epoch / self.annealing_epochs)
        
        # 4. 总损失
        loss = (
            loss_task 
            + self.lambda1 * loss_edl 
            + annealing_coef * self.lambda2 * loss_kl
        )
        
        return {
            "loss": loss.mean(),
            "loss_task": loss_task.mean(),
            "loss_edl": loss_edl.mean(),
            "loss_kl": loss_kl.mean(),
            "annealing_coef": torch.tensor(annealing_coef)
        }


class RouterEDLLoss(nn.Module):
    """
    路由器专用的 EDL 损失。
    
    鼓励路由器:
    - 对正确预测降低不确定性
    - 对错误预测提高不确定性
    """
    
    def __init__(self, num_experts: int):
        """
        初始化路由器 EDL 损失。
        
        参数:
            num_experts: 专家数量
        """
        super().__init__()
        self.num_experts = num_experts
    
    def forward(
        self,
        evidence: torch.Tensor,
        uncertainty: torch.Tensor,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        计算路由器损失。
        
        参数:
            evidence: 路由器证据 [batch, num_experts]
            uncertainty: 不确定性 [batch, 1]
            predictions: 模型预测 [batch]
            targets: 真实标签 [batch]
        
        返回:
            损失标量
        """
        # 判断预测是否正确
        correct = (predictions == targets).float()
        
        # 正确预测应该有低不确定性，错误预测应该有高不确定性
        # 损失 = 正确时的不确定性 + (1 - 错误时的不确定性)
        uncertainty = uncertainty.squeeze(-1)
        
        loss = correct * uncertainty + (1 - correct) * (1 - uncertainty)
        
        return loss.mean()


def compute_edl_loss(
    logits: torch.Tensor,
    evidence: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    lambda1: float = 1.0,
    lambda2: float = 0.1,
    epoch: int = 0,
    annealing_epochs: int = 10
) -> Dict[str, torch.Tensor]:
    """
    计算 EDL 损失的便捷函数。
    
    参数:
        logits: 模型输出
        evidence: 证据向量
        target: 目标标签
        num_classes: 类别数
        lambda1: EDL 损失权重
        lambda2: KL 损失权重
        epoch: 当前轮数
        annealing_epochs: 退火轮数
    
    返回:
        损失字典
    """
    loss_fn = EDLLoss(
        num_classes=num_classes,
        lambda1=lambda1,
        lambda2=lambda2,
        annealing_epochs=annealing_epochs
    )
    return loss_fn(logits, evidence, target, epoch)
