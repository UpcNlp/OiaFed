# fedcl/implementations/evaluators/forgetting_evaluator.py
"""
遗忘程度评估器实现
"""

from typing import Dict, Any, List
import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from ...core.base_evaluator import BaseEvaluator
from ...core.execution_context import ExecutionContext
from ...registry import registry


@registry.evaluator("forgetting", metadata={
    "description": "Forgetting Evaluator for Continual Learning"
})
class ForgettingEvaluator(BaseEvaluator):
    """遗忘程度评估器"""
    
    def __init__(self, context: ExecutionContext, config: DictConfig):
        super().__init__(context, config)
        self.task_accuracies = {}  # 存储每个任务的历史准确率
        
    def evaluate(self, model: torch.nn.Module, data_loader: DataLoader, 
                task_id: int = None) -> Dict[str, Any]:
        """评估遗忘程度"""
        # 计算当前准确率
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
                
        current_accuracy = correct / total if total > 0 else 0.0
        
        # 计算遗忘程度
        forgetting = 0.0
        if task_id in self.task_accuracies:
            max_accuracy = max(self.task_accuracies[task_id])
            forgetting = max(0, max_accuracy - current_accuracy)
            
        # 更新历史记录
        if task_id not in self.task_accuracies:
            self.task_accuracies[task_id] = []
        self.task_accuracies[task_id].append(current_accuracy)
        
        return {
            "current_accuracy": current_accuracy,
            "forgetting": forgetting,
            "task_id": task_id
        }
