"""
原型评估器

基于原型（prototype）的评估方法，适用于联邦学习中的表征学习评估。
通过计算类原型来评估模型的表征质量。
"""

import torch
from typing import Dict, Any, Optional, List
from loguru import logger

from ...api.decorators import evaluator


@evaluator("prototype", description="基于原型的联邦评估器")
class PrototypeEvaluator:
    """原型评估器实现"""
    
    def __init__(self, config: Dict[str, Any] = None, context: Optional[Any] = None):
        self.config = config or {}
        self.context = context
        
        # 评估参数
        self.num_classes = self.config.get("num_classes", 10)
        self.prototype_momentum = self.config.get("prototype_momentum", 0.9)
        
        # 原型状态
        self.class_prototypes: Optional[torch.Tensor] = None
        self.prototype_counts: Optional[torch.Tensor] = None
        
        logger.info(f"✅ 原型评估器初始化完成 - 类别数: {self.num_classes}")
    
    def evaluate(self, model: Any, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行基于原型的评估"""
        # 验证必需参数
        if model is None:
            raise ValueError("必须提供有效的模型对象")
        
        if "data" not in test_data:
            raise ValueError("缺少必需的测试数据 'data'")
        
        if "labels" not in test_data:
            raise ValueError("缺少必需的标签数据 'labels'")
        
        # 获取数据
        data = test_data["data"]
        labels = test_data["labels"]
        
        # 验证数据类型
        if not isinstance(data, torch.Tensor):
            raise ValueError("data 必须是 torch.Tensor 类型")
        
        if not isinstance(labels, torch.Tensor):
            raise ValueError("labels 必须是 torch.Tensor 类型")
        
        if data.shape[0] != labels.shape[0]:
            raise ValueError(f"数据和标签数量不匹配: {data.shape[0]} vs {labels.shape[0]}")
        
        # 提取特征
        features = self._extract_features(model, data)
        
        # 计算或更新原型
        if self.class_prototypes is None:
            self._initialize_prototypes(features, labels)
        else:
            self._update_prototypes(features, labels)
        
        # 基于原型的分类
        prototype_predictions = self._classify_with_prototypes(features)
        
        # 计算评估指标
        metrics = self._compute_prototype_metrics(prototype_predictions, labels, features)
        
        return metrics
    
    def _extract_features(self, model: Any, data: torch.Tensor) -> torch.Tensor:
        """提取模型特征（必须使用真实模型）"""
        if not hasattr(model, 'forward') and not callable(model):
            raise ValueError("模型必须是可调用的或具有 forward 方法")
        
        # 尝试使用模型的特征提取方法
        if hasattr(model, 'extract_features'):
            with torch.no_grad():
                return model.extract_features(data)
        elif hasattr(model, 'forward_features'):
            with torch.no_grad():
                return model.forward_features(data)
        elif hasattr(model, 'feature_extractor'):
            with torch.no_grad():
                return model.feature_extractor(data)
        else:
            # 如果模型没有特征提取方法，抛出错误
            raise NotImplementedError(
                f"模型 {type(model).__name__} 必须实现以下方法之一："
                f"extract_features(), forward_features(), 或提供 feature_extractor 属性。"
                f"原型评估器需要真实的特征表示，不允许使用模拟数据。"
            )
    
    def _initialize_prototypes(self, features: torch.Tensor, labels: torch.Tensor):
        """初始化类原型"""
        feature_dim = features.shape[1]
        self.class_prototypes = torch.zeros(self.num_classes, feature_dim)
        self.prototype_counts = torch.zeros(self.num_classes)
        
        # 计算每个类的原型
        for class_id in range(self.num_classes):
            class_mask = (labels == class_id)
            if class_mask.sum() > 0:
                class_features = features[class_mask]
                self.class_prototypes[class_id] = class_features.mean(dim=0)
                self.prototype_counts[class_id] = class_mask.sum().float()
        
        logger.debug("类原型初始化完成")
    
    def _update_prototypes(self, features: torch.Tensor, labels: torch.Tensor):
        """更新类原型"""
        for class_id in range(self.num_classes):
            class_mask = (labels == class_id)
            if class_mask.sum() > 0:
                class_features = features[class_mask]
                new_prototype = class_features.mean(dim=0)
                
                # 动量更新
                self.class_prototypes[class_id] = (
                    self.prototype_momentum * self.class_prototypes[class_id] +
                    (1 - self.prototype_momentum) * new_prototype
                )
                
                # 更新计数
                self.prototype_counts[class_id] += class_mask.sum().float()
    
    def _classify_with_prototypes(self, features: torch.Tensor) -> torch.Tensor:
        """基于原型进行分类"""
        # 计算特征与原型的相似度
        similarities = torch.matmul(features, self.class_prototypes.T)
        
        # 返回最相似的类别
        predictions = torch.argmax(similarities, dim=1)
        return predictions
    
    def _compute_prototype_metrics(self, predictions: torch.Tensor, 
                                 labels: torch.Tensor, features: torch.Tensor) -> Dict[str, Any]:
        """计算原型相关的评估指标"""
        # 基础准确率
        accuracy = (predictions == labels).float().mean().item()
        
        # 原型质量指标
        prototype_metrics = self._compute_prototype_quality()
        
        # 特征聚类质量
        clustering_metrics = self._compute_clustering_quality(features, labels)
        
        # 类间分离度
        separation_metrics = self._compute_class_separation()
        
        return {
            "prototype_accuracy": accuracy,
            "overall_accuracy": accuracy,  # 兼容标准接口
            **prototype_metrics,
            **clustering_metrics,
            **separation_metrics
        }
    
    def _compute_prototype_quality(self) -> Dict[str, float]:
        """计算原型质量指标"""
        if self.class_prototypes is None:
            return {}
        
        # 原型范数
        prototype_norms = torch.norm(self.class_prototypes, dim=1)
        avg_prototype_norm = prototype_norms.mean().item()
        std_prototype_norm = prototype_norms.std().item()
        
        # 原型稳定性（基于计数）
        min_count = self.prototype_counts.min().item()
        max_count = self.prototype_counts.max().item()
        count_balance = min_count / max(max_count, 1)
        
        return {
            "avg_prototype_norm": avg_prototype_norm,
            "std_prototype_norm": std_prototype_norm,
            "prototype_balance": count_balance
        }
    
    def _compute_clustering_quality(self, features: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """计算聚类质量指标"""
        # 类内距离
        intra_class_distances = []
        for class_id in range(self.num_classes):
            class_mask = (labels == class_id)
            if class_mask.sum() > 1:
                class_features = features[class_mask]
                class_center = class_features.mean(dim=0)
                distances = torch.norm(class_features - class_center, dim=1)
                intra_class_distances.append(distances.mean().item())
        
        avg_intra_distance = sum(intra_class_distances) / max(len(intra_class_distances), 1)
        
        # 类间距离
        inter_class_distances = []
        for i in range(self.num_classes):
            for j in range(i + 1, self.num_classes):
                if self.prototype_counts[i] > 0 and self.prototype_counts[j] > 0:
                    distance = torch.norm(self.class_prototypes[i] - self.class_prototypes[j]).item()
                    inter_class_distances.append(distance)
        
        avg_inter_distance = sum(inter_class_distances) / max(len(inter_class_distances), 1)
        
        # 聚类质量比率
        clustering_ratio = avg_inter_distance / max(avg_intra_distance, 1e-8)
        
        return {
            "avg_intra_class_distance": avg_intra_distance,
            "avg_inter_class_distance": avg_inter_distance,
            "clustering_quality_ratio": clustering_ratio
        }
    
    def _compute_class_separation(self) -> Dict[str, float]:
        """计算类间分离度"""
        if self.class_prototypes is None:
            return {}
        
        # 最小类间距离
        min_separation = float('inf')
        max_separation = 0.0
        
        for i in range(self.num_classes):
            for j in range(i + 1, self.num_classes):
                if self.prototype_counts[i] > 0 and self.prototype_counts[j] > 0:
                    distance = torch.norm(self.class_prototypes[i] - self.class_prototypes[j]).item()
                    min_separation = min(min_separation, distance)
                    max_separation = max(max_separation, distance)
        
        return {
            "min_class_separation": min_separation if min_separation != float('inf') else 0.0,
            "max_class_separation": max_separation,
            "separation_ratio": max_separation / max(min_separation, 1e-8) if min_separation != float('inf') else 1.0
        }
    
    def get_prototypes(self) -> Optional[torch.Tensor]:
        """获取当前的类原型"""
        return self.class_prototypes
    
    def reset_prototypes(self):
        """重置原型状态"""
        self.class_prototypes = None
        self.prototype_counts = None
        logger.info("🔄 原型评估器状态已重置")