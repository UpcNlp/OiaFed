# fedcl/implementations/evaluators/accuracy_evaluator.py
"""
准确率评估器实现
"""

from typing import Dict, Any
import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from ...core.base_evaluator import BaseEvaluator
from ...core.execution_context import ExecutionContext
from ...registry import registry


@registry.evaluator("accuracy", metadata={
    "description": "Accuracy Evaluator for Classification Tasks"
})
class AccuracyEvaluator(BaseEvaluator):
    """准确率评估器"""
    
    def __init__(self, context: ExecutionContext, config: DictConfig):
        super().__init__(context, config)
        
    def evaluate(self, model: torch.nn.Module, data_loader: DataLoader, 
                task_id: int = None) -> Dict[str, Any]:
        """评估模型准确率"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "task_id": task_id
        }
    
    def compute_task_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        计算任务级别的评估指标
        
        Args:
            predictions: 模型预测结果 (batch_size, num_classes)
            targets: 真实标签 (batch_size,)
            
        Returns:
            Dict[str, float]: 评估指标字典
        """
        # 获取预测类别
        _, predicted = predictions.max(1)
        
        # 计算准确率
        correct = predicted.eq(targets).sum().item()
        total = targets.size(0)
        accuracy = correct / total if total > 0 else 0.0
        
        # 计算top-k准确率 (如果类别数量 > 1)
        metrics = {"accuracy": accuracy}
        
        if predictions.size(1) > 1:  # 多分类任务
            # Top-5准确率（如果类别数量足够）
            k = min(5, predictions.size(1))
            _, top_k_pred = predictions.topk(k, 1, True, True)
            top_k_correct = 0
            for i in range(targets.size(0)):
                if targets[i] in top_k_pred[i]:
                    top_k_correct += 1
            metrics[f"top_{k}_accuracy"] = top_k_correct / total if total > 0 else 0.0
        
        return metrics
