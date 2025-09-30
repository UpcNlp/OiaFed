"""
公平性评估器

评估联邦学习中的公平性问题，包括性能公平性、表示公平性等。
适用于需要关注公平性的联邦学习应用。
"""

import torch
from typing import Dict, Any, Optional, List
from loguru import logger

from ...api.decorators import evaluator


@evaluator("fairness", description="联邦学习公平性评估器")
class FairnessEvaluator:
    """公平性评估器实现"""
    
    def __init__(self, config: Dict[str, Any] = None, context: Optional[Any] = None):
        self.config = config or {}
        self.context = context
        
        # 公平性评估参数
        self.sensitive_attributes = self.config.get("sensitive_attributes", ["age", "gender"])
        self.fairness_metrics = self.config.get("fairness_metrics", ["demographic_parity", "equalized_odds"])
        
        # 客户端公平性历史
        self.client_performance_history = {}
        self.group_performance_history = {}
        
        logger.info(f"✅ 公平性评估器初始化完成 - 敏感属性: {self.sensitive_attributes}")
    
    def evaluate(self, model: Any, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行公平性评估"""
        # 验证必需参数
        if model is None:
            raise ValueError("必须提供有效的模型对象")
        
        required_fields = ["data", "labels"]
        for field in required_fields:
            if field not in test_data:
                raise ValueError(f"缺少必需的测试数据 '{field}'")
        
        # 获取数据
        data = test_data["data"]
        labels = test_data["labels"]
        
        # 验证数据类型
        if not isinstance(data, torch.Tensor) or not isinstance(labels, torch.Tensor):
            raise ValueError("data 和 labels 必须是 torch.Tensor 类型")
        
        if data.shape[0] != labels.shape[0]:
            raise ValueError(f"数据和标签数量不匹配: {data.shape[0]} vs {labels.shape[0]}")
        
        # 获取或检查敏感属性
        sensitive_attrs = test_data.get("sensitive_attributes")
        if sensitive_attrs is None:
            raise ValueError(
                "公平性评估器必须提供真实的敏感属性数据。"
                "请在test_data中提供'sensitive_attributes'字段。"
                "不允许使用模拟或随机生成的敏感属性，这是生产环境。"
            )
        
        # 获取客户端ID
        client_ids = test_data.get("client_ids")
        if client_ids is None:
            # 如果没有提供客户端ID，生成默认值
            client_ids = [f"client_{i % 5}" for i in range(len(data))]
        
        # 模型预测
        predictions = self._get_model_predictions(model, data)
        
        # 计算公平性指标
        fairness_metrics = {}
        
        # 1. 客户端间公平性
        client_fairness = self._evaluate_client_fairness(predictions, labels, client_ids)
        fairness_metrics.update(client_fairness)
        
        # 2. 群体公平性
        group_fairness = self._evaluate_group_fairness(predictions, labels, sensitive_attrs)
        fairness_metrics.update(group_fairness)
        
        # 3. 整体性能指标
        overall_metrics = self._compute_overall_metrics(predictions, labels)
        fairness_metrics.update(overall_metrics)
        
        # 4. 公平性趋势分析
        trend_metrics = self._analyze_fairness_trends()
        fairness_metrics.update(trend_metrics)
        
        return fairness_metrics
    
    def _get_model_predictions(self, model: Any, data: torch.Tensor) -> torch.Tensor:
        """获取模型预测（必须使用真实模型）"""
        if not hasattr(model, 'forward') and not callable(model):
            raise ValueError("模型必须是可调用的或具有 forward 方法")
        
        # 尝试使用模型的预测方法
        with torch.no_grad():
            if hasattr(model, 'predict'):
                return model.predict(data)
            elif hasattr(model, 'forward'):
                output = model.forward(data)
                # 如果输出是 logits，转换为预测
                if output.dim() > 1 and output.shape[1] > 1:
                    return torch.argmax(output, dim=1)
                else:
                    return (output > 0.5).long().squeeze()
            elif callable(model):
                output = model(data)
                if output.dim() > 1 and output.shape[1] > 1:
                    return torch.argmax(output, dim=1)
                else:
                    return (output > 0.5).long().squeeze()
            else:
                raise NotImplementedError(
                    f"模型 {type(model).__name__} 必须实现 predict() 方法或是可调用的。"
                    f"公平性评估器需要真实的模型预测，不允许使用模拟数据。"
                )
    

    
    def _evaluate_client_fairness(self, predictions: torch.Tensor, 
                                labels: torch.Tensor, client_ids: List[str]) -> Dict[str, Any]:
        """评估客户端间公平性"""
        client_metrics = {}
        client_accuracies = {}
        
        # 计算每个客户端的性能
        unique_clients = list(set(client_ids))
        for client_id in unique_clients:
            client_mask = torch.tensor([cid == client_id for cid in client_ids])
            if client_mask.sum() > 0:
                client_pred = predictions[client_mask]
                client_labels = labels[client_mask]
                
                accuracy = (client_pred == client_labels).float().mean().item()
                client_accuracies[client_id] = accuracy
                
                # 更新历史记录
                if client_id not in self.client_performance_history:
                    self.client_performance_history[client_id] = []
                self.client_performance_history[client_id].append(accuracy)
        
        # 计算客户端公平性指标
        if len(client_accuracies) > 1:
            accuracies = list(client_accuracies.values())
            client_metrics = {
                "client_accuracy_mean": sum(accuracies) / len(accuracies),
                "client_accuracy_std": torch.tensor(accuracies).std().item(),
                "client_accuracy_min": min(accuracies),
                "client_accuracy_max": max(accuracies),
                "client_fairness_gap": max(accuracies) - min(accuracies),
                "client_gini_coefficient": self._compute_gini_coefficient(accuracies)
            }
        
        return {"client_fairness": client_metrics}
    
    def _evaluate_group_fairness(self, predictions: torch.Tensor, 
                               labels: torch.Tensor, 
                               sensitive_attrs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """评估群体公平性"""
        group_metrics = {}
        for attr_name, attr_values in sensitive_attrs.items():
            attr_metrics = {}
            
            # 获取属性的唯一值
            unique_values = torch.unique(attr_values)            
            group_accuracies = []
            group_tpr = []  # True Positive Rate
            group_fpr = []  # False Positive Rate
            
            for value in unique_values:
                group_mask = (attr_values == value)
                if group_mask.sum() > 0:
                    group_pred = predictions[group_mask]
                    group_labels = labels[group_mask]
                    
                    # 准确率
                    accuracy = (group_pred == group_labels).float().mean().item()
                    group_accuracies.append(accuracy)
                    
                    # TPR和FPR（假设二分类）
                    if len(torch.unique(group_labels)) > 1:
                        tp = ((group_pred == 1) & (group_labels == 1)).sum().float()
                        fp = ((group_pred == 1) & (group_labels == 0)).sum().float()
                        tn = ((group_pred == 0) & (group_labels == 0)).sum().float()
                        fn = ((group_pred == 0) & (group_labels == 1)).sum().float()
                        
                        tpr = tp / (tp + fn + 1e-8)
                        fpr = fp / (fp + tn + 1e-8)
                        
                        group_tpr.append(tpr.item())
                        group_fpr.append(fpr.item())
            
            # 计算群体间差异
            if len(group_accuracies) > 1:
                attr_metrics[f"{attr_name}_accuracy_gap"] = max(group_accuracies) - min(group_accuracies)
                attr_metrics[f"{attr_name}_demographic_parity"] = self._compute_demographic_parity(group_accuracies)
                
                if len(group_tpr) > 1:
                    attr_metrics[f"{attr_name}_equalized_odds_tpr"] = max(group_tpr) - min(group_tpr)
                
                if len(group_fpr) > 1:
                    attr_metrics[f"{attr_name}_equalized_odds_fpr"] = max(group_fpr) - min(group_fpr)
            
            group_metrics[attr_name] = attr_metrics
        
        return {"group_fairness": group_metrics}
    
    def _compute_overall_metrics(self, predictions: torch.Tensor, labels: torch.Tensor) -> Dict[str, Any]:
        """计算整体性能指标（支持任意分类任务）"""
        # 基础准确率
        accuracy = (predictions == labels).float().mean().item()
        
        # 获取唯一类别
        unique_labels = torch.unique(labels)
        unique_preds = torch.unique(predictions)
        
        metrics = {
            "overall_accuracy": accuracy,
            "num_classes_true": len(unique_labels),
            "num_classes_pred": len(unique_preds)
        }
        
        # 如果是二分类任务，计算额外指标
        if len(unique_labels) == 2 and len(unique_preds) <= 2:
            # 计算混淆矩阵相关指标
            tp = ((predictions == 1) & (labels == 1)).sum().float()
            fp = ((predictions == 1) & (labels == 0)).sum().float()
            tn = ((predictions == 0) & (labels == 0)).sum().float()
            fn = ((predictions == 0) & (labels == 1)).sum().float()
            
            # 防止除零错误
            epsilon = 1e-8
            
            precision = tp / (tp + fp + epsilon)
            recall = tp / (tp + fn + epsilon)
            f1_score = 2 * precision * recall / (precision + recall + epsilon)
            specificity = tn / (tn + fp + epsilon)
            
            metrics.update({
                "precision": precision.item(),
                "recall": recall.item(),
                "f1_score": f1_score.item(),
                "specificity": specificity.item(),
                "true_positive_rate": recall.item(),
                "false_positive_rate": (fp / (fp + tn + epsilon)).item()
            })
        
        # 多分类任务的指标
        elif len(unique_labels) > 2:
            # 计算每个类别的准确率
            class_accuracies = []
            for class_id in unique_labels:
                class_mask = (labels == class_id)
                if class_mask.sum() > 0:
                    class_acc = (predictions[class_mask] == class_id).float().mean().item()
                    class_accuracies.append(class_acc)
            
            if class_accuracies:
                metrics.update({
                    "mean_class_accuracy": sum(class_accuracies) / len(class_accuracies),
                    "min_class_accuracy": min(class_accuracies),
                    "max_class_accuracy": max(class_accuracies),
                    "class_accuracy_std": torch.tensor(class_accuracies).std().item()
                })
        
        return metrics
    
    def _analyze_fairness_trends(self) -> Dict[str, Any]:
        """分析公平性趋势"""
        trends = {}
        
        # 分析客户端公平性趋势
        if len(self.client_performance_history) > 1:
            recent_fairness_gaps = []
            
            for round_idx in range(min(len(hist) for hist in self.client_performance_history.values())):
                round_accuracies = []
                for client_hist in self.client_performance_history.values():
                    if round_idx < len(client_hist):
                        round_accuracies.append(client_hist[round_idx])
                
                if len(round_accuracies) > 1:
                    gap = max(round_accuracies) - min(round_accuracies)
                    recent_fairness_gaps.append(gap)
            
            if len(recent_fairness_gaps) > 1:
                trends["fairness_gap_trend"] = "improving" if recent_fairness_gaps[-1] < recent_fairness_gaps[0] else "degrading"
                trends["avg_fairness_gap"] = sum(recent_fairness_gaps) / len(recent_fairness_gaps)
        
        return {"fairness_trends": trends}
    
    def _compute_demographic_parity(self, group_accuracies: List[float]) -> float:
        """计算人口统计学平等"""
        if len(group_accuracies) < 2:
            return 0.0
        
        # 简化的人口统计学平等：最大差异
        return max(group_accuracies) - min(group_accuracies)
    
    def _compute_gini_coefficient(self, values: List[float]) -> float:
        """计算基尼系数"""
        if len(values) < 2:
            return 0.0
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        cumsum = torch.cumsum(torch.tensor(sorted_values), dim=0)
        
        # 基尼系数计算
        gini = (n + 1 - 2 * torch.sum(cumsum) / cumsum[-1]) / n
        return gini.item()
    
    def get_fairness_report(self) -> Dict[str, Any]:
        """获取详细的公平性报告"""
        report = {
            "client_count": len(self.client_performance_history),
            "evaluation_rounds": max(len(hist) for hist in self.client_performance_history.values()) if self.client_performance_history else 0,
            "sensitive_attributes": self.sensitive_attributes,
            "fairness_metrics": self.fairness_metrics
        }
        
        # 添加历史统计
        if self.client_performance_history:
            all_accuracies = []
            for hist in self.client_performance_history.values():
                all_accuracies.extend(hist)
            
            if all_accuracies:
                report["historical_stats"] = {
                    "mean_accuracy": sum(all_accuracies) / len(all_accuracies),
                    "std_accuracy": torch.tensor(all_accuracies).std().item(),
                    "min_accuracy": min(all_accuracies),
                    "max_accuracy": max(all_accuracies)
                }
        
        return report