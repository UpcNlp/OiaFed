"""
FedEMoE 评估指标模块。
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class Metrics:
    """指标跟踪和计算容器。"""
    
    # 跟踪列表
    predictions: List[torch.Tensor] = field(default_factory=list)
    targets: List[torch.Tensor] = field(default_factory=list)
    uncertainties: List[torch.Tensor] = field(default_factory=list)
    confidences: List[torch.Tensor] = field(default_factory=list)
    dynamic_ks: List[torch.Tensor] = field(default_factory=list)
    
    def reset(self):
        """重置所有跟踪列表。"""
        self.predictions = []
        self.targets = []
        self.uncertainties = []
        self.confidences = []
        self.dynamic_ks = []
    
    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        uncertainties: Optional[torch.Tensor] = None,
        confidences: Optional[torch.Tensor] = None,
        dynamic_ks: Optional[torch.Tensor] = None
    ):
        """
        使用新批次数据更新指标。
        
        参数:
            preds: 预测的类别索引 [batch]
            targets: 目标类别索引 [batch]
            uncertainties: 不确定性值 [batch]
            confidences: 置信度值 [batch]
            dynamic_ks: 动态 K 值 [batch]
        """
        self.predictions.append(preds.detach().cpu())
        self.targets.append(targets.detach().cpu())
        
        if uncertainties is not None:
            self.uncertainties.append(uncertainties.detach().cpu())
        if confidences is not None:
            self.confidences.append(confidences.detach().cpu())
        if dynamic_ks is not None:
            self.dynamic_ks.append(dynamic_ks.detach().cpu())
    
    def compute(self) -> Dict[str, float]:
        """
        计算所有指标。
        
        返回:
            指标名称和值的字典
        """
        results = {}
        
        # 拼接所有批次
        preds = torch.cat(self.predictions)
        targets = torch.cat(self.targets)
        
        # 准确率
        results["accuracy"] = compute_accuracy(preds, targets)
        
        # 不确定性指标（如果可用）
        if self.uncertainties:
            uncertainties = torch.cat(self.uncertainties)
            results["avg_uncertainty"] = uncertainties.mean().item()
            results["uncertainty_std"] = uncertainties.std().item()
            
            # 正确/错误预测的不确定性
            correct_mask = (preds == targets)
            if correct_mask.any():
                results["uncertainty_correct"] = uncertainties[correct_mask].mean().item()
            if (~correct_mask).any():
                results["uncertainty_incorrect"] = uncertainties[~correct_mask].mean().item()
        
        # 置信度指标（如果可用）
        if self.confidences:
            confidences = torch.cat(self.confidences)
            results["avg_confidence"] = confidences.mean().item()
            
            # 期望校准误差
            results["ece"] = compute_ece(confidences, preds, targets)
        
        # 动态 K 统计（如果可用）
        if self.dynamic_ks:
            dynamic_ks = torch.cat(self.dynamic_ks).float()
            results["avg_k"] = dynamic_ks.mean().item()
            results["k_std"] = dynamic_ks.std().item()
        
        return results


def compute_accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """
    计算分类准确率。
    
    参数:
        preds: 预测的类别索引 [N]
        targets: 目标类别索引 [N]
    
    返回:
        准确率（浮点数）
    """
    correct = (preds == targets).sum().item()
    total = len(targets)
    return correct / total if total > 0 else 0.0


def compute_ece(
    confidences: torch.Tensor,
    preds: torch.Tensor,
    targets: torch.Tensor,
    n_bins: int = 15
) -> float:
    """
    计算期望校准误差 (Expected Calibration Error, ECE)。
    
    ECE 衡量预测概率与实际准确率的匹配程度。
    
    参数:
        confidences: 预测置信度 [N]
        preds: 预测的类别索引 [N]
        targets: 目标类别索引 [N]
        n_bins: 校准分箱数量
    
    返回:
        ECE 值
    """
    confidences = confidences.numpy() if isinstance(confidences, torch.Tensor) else confidences
    preds = preds.numpy() if isinstance(preds, torch.Tensor) else preds
    targets = targets.numpy() if isinstance(targets, torch.Tensor) else targets
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # 找出该分箱中的样本
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            # 该分箱的准确率
            accuracy_in_bin = (preds[in_bin] == targets[in_bin]).mean()
            # 该分箱的平均置信度
            avg_confidence_in_bin = confidences[in_bin].mean()
            # 对 ECE 的贡献
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return float(ece)


def compute_uncertainty_auroc(
    uncertainties: torch.Tensor,
    preds: torch.Tensor,
    targets: torch.Tensor
) -> float:
    """
    计算基于不确定性的错误检测 AUROC。
    
    较高的不确定性应该与错误预测相关。
    
    参数:
        uncertainties: 不确定性值 [N]
        preds: 预测的类别索引 [N]
        targets: 目标类别索引 [N]
    
    返回:
        AUROC 值
    """
    try:
        from sklearn.metrics import roc_auc_score
        
        # 二分类标签: 1 表示错误, 0 表示正确
        # 高不确定性应该预测错误
        is_incorrect = (preds != targets).numpy().astype(int)
        uncertainties_np = uncertainties.numpy()
        
        # 计算 AUROC
        auroc = roc_auc_score(is_incorrect, uncertainties_np)
        return float(auroc)
    except ImportError:
        return 0.0
    except ValueError:
        # 如果所有预测都正确或都错误
        return 0.5


def compute_client_fairness(
    client_accuracies: List[float]
) -> Dict[str, float]:
    """
    计算客户端间的公平性指标。
    
    参数:
        client_accuracies: 每个客户端的准确率列表
    
    返回:
        包含公平性指标的字典
    """
    accs = np.array(client_accuracies)
    
    return {
        "mean_accuracy": float(accs.mean()),
        "std_accuracy": float(accs.std()),
        "min_accuracy": float(accs.min()),
        "max_accuracy": float(accs.max()),
        "fairness_gap": float(accs.max() - accs.min())
    }


class AverageMeter:
    """计算并存储平均值和当前值。"""
    
    def __init__(self, name: str = ""):
        """
        初始化。
        
        参数:
            name: 指标名称
        """
        self.name = name
        self.reset()
    
    def reset(self):
        """重置所有计数器。"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        """
        更新指标。
        
        参数:
            val: 当前值
            n: 样本数
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0
    
    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"


class ExpertUtilizationTracker:
    """跟踪训练过程中的专家使用情况。"""
    
    def __init__(self, num_experts: int):
        """
        初始化。
        
        参数:
            num_experts: 专家数量
        """
        self.num_experts = num_experts
        self.utilization_counts = np.zeros(num_experts)
        self.total_samples = 0
    
    def update(self, expert_indices: torch.Tensor, weights: torch.Tensor):
        """
        更新使用计数。
        
        参数:
            expert_indices: 选中的专家索引 [batch, k]
            weights: 专家权重 [batch, k]
        """
        batch_size = expert_indices.shape[0]
        self.total_samples += batch_size
        
        # 转换为 numpy
        indices = expert_indices.detach().cpu().numpy()
        w = weights.detach().cpu().numpy()
        
        # 按选择权重更新计数
        for b in range(batch_size):
            for i, (idx, weight) in enumerate(zip(indices[b], w[b])):
                if weight > 0:
                    self.utilization_counts[idx] += weight
    
    def get_utilization(self) -> Dict[str, float]:
        """获取使用统计。"""
        if self.total_samples == 0:
            return {}
        
        rates = self.utilization_counts / self.total_samples
        
        return {
            "utilization_rates": rates.tolist(),
            "utilization_std": np.std(rates),
            "utilization_max": np.max(rates),
            "utilization_min": np.min(rates),
            "utilization_balance": np.min(rates) / (np.max(rates) + 1e-8)
        }
    
    def reset(self):
        """重置跟踪器。"""
        self.utilization_counts = np.zeros(self.num_experts)
        self.total_samples = 0
