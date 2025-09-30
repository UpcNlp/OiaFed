"""
对比学习器

实现联邦对比学习，通过对比损失提高表征学习质量。
适用于无监督或半监督联邦学习场景。
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from loguru import logger

from ...api.decorators import learner


@learner("contrastive", description="联邦对比学习器")
class ContrastiveLearner:
    """对比学习器实现"""
    
    def __init__(self, config: Dict[str, Any] = None, context: Optional[Any] = None):
        self.config = config or {}
        self.context = context
        
        # 对比学习参数
        self.temperature = self.config.get("temperature", 0.5)
        self.projection_dim = self.config.get("projection_dim", 128)
        
        # 设备配置
        self.device = self.config.get("device", "auto")
        if self.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"✅ 对比学习器初始化完成 - 温度: {self.temperature}")
    
    def train_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行对比学习训练"""
        # 验证必需数据
        if "data" not in task_data:
            raise ValueError("缺少必需的训练数据 'data'")
        
        data = task_data["data"]
        if not isinstance(data, torch.Tensor):
            raise ValueError("data 必须是 torch.Tensor 类型")
        
        batch_size = data.shape[0]
        
        # 创建数据增强（必须使用真实的增强策略）
        augmented_data1 = self._augment_data(data)
        augmented_data2 = self._augment_data(data)
        
        # 计算特征表示（必须使用真实的模型）
        features1 = self._extract_features(augmented_data1)
        features2 = self._extract_features(augmented_data2)
        
        # 计算对比损失（真实计算）
        contrastive_loss = self._compute_contrastive_loss(features1, features2)
        
        # 计算准确率（基于最近邻）
        accuracy = self._compute_contrastive_accuracy(features1, features2)
        
        # 计算额外指标
        feature_std = torch.std(features1, dim=0).mean().item()
        feature_norm = torch.norm(features1, dim=1).mean().item()
        
        return {
            "loss": contrastive_loss.item(),
            "contrastive_accuracy": accuracy,
            "num_samples": batch_size,
            "feature_dimension": features1.shape[1],
            "feature_std": feature_std,
            "feature_norm": feature_norm,
            "representation_quality": feature_std * feature_norm
        }
    
    def evaluate_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """评估对比学习效果"""
        if "data" not in task_data:
            raise ValueError("缺少必需的评估数据 'data'")
        
        data = task_data["data"]
        if not isinstance(data, torch.Tensor):
            raise ValueError("data 必须是 torch.Tensor 类型")
        
        # 提取特征
        features = self._extract_features(data)
        
        # 计算特征质量指标
        feature_std = torch.std(features, dim=0).mean().item()
        feature_norm = torch.norm(features, dim=1).mean().item()
        
        # 计算特征分布指标
        feature_min = torch.min(features).item()
        feature_max = torch.max(features).item()
        feature_mean = torch.mean(features).item()
        
        # 计算维度间相关性
        if features.shape[1] > 1:
            correlation_matrix = torch.corrcoef(features.T)
            avg_correlation = torch.mean(torch.abs(correlation_matrix - torch.eye(features.shape[1]))).item()
        else:
            avg_correlation = 0.0
        
        return {
            "feature_std": feature_std,
            "feature_norm": feature_norm,
            "feature_min": feature_min,
            "feature_max": feature_max,
            "feature_mean": feature_mean,
            "avg_correlation": avg_correlation,
            "representation_quality": feature_std * feature_norm,
            "feature_diversity": feature_std / max(abs(feature_mean), 1e-8)
        }
    
    def _augment_data(self, data: torch.Tensor) -> torch.Tensor:
        """数据增强（必须使用真实的增强策略）"""
        if not hasattr(self, '_augmentation_fn') or self._augmentation_fn is None:
            raise NotImplementedError(
                "对比学习器必须提供真实的数据增强策略。"
                "请设置self._augmentation_fn或重写此方法。"
                "不允许使用torch.randn_like()等模拟增强，这是生产环境。"
            )
        
        return self._augmentation_fn(data)
    
    def _extract_features(self, data: torch.Tensor) -> torch.Tensor:
        """提取特征表示（必须使用真实模型）"""
        # 检查是否有可用的模型
        if not hasattr(self, '_feature_extractor') or self._feature_extractor is None:
            raise NotImplementedError(
                "对比学习器必须提供真实的特征提取器。"
                "请在初始化时设置 self._feature_extractor 或重写此方法。"
            )
        
        # 使用真实的特征提取器
        with torch.no_grad():
            features = self._feature_extractor(data)
            # 归一化特征
            return nn.functional.normalize(features, dim=1)
    
    def _compute_contrastive_loss(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        """计算对比损失"""
        batch_size = features1.shape[0]
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(features1, features2.T) / self.temperature
        
        # 正样本：对角线元素
        positive_mask = torch.eye(batch_size, dtype=torch.bool)
        
        # 计算对比损失
        exp_sim = torch.exp(similarity_matrix)
        positive_sum = exp_sim[positive_mask].sum()
        total_sum = exp_sim.sum()
        
        loss = -torch.log(positive_sum / total_sum)
        return loss
    
    def _compute_contrastive_accuracy(self, features1: torch.Tensor, features2: torch.Tensor) -> float:
        """计算对比学习准确率"""
        # 简化：计算最近邻准确率
        similarity_matrix = torch.matmul(features1, features2.T)
        predicted = torch.argmax(similarity_matrix, dim=1)
        targets = torch.arange(features1.shape[0])
        
        accuracy = (predicted == targets).float().mean().item()
        return accuracy