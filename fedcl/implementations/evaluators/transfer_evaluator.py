# fedcl/implementations/evaluators/transfer_evaluator.py
"""
迁移能力评估器实现
"""

from typing import Dict, Any
import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from ...core.base_evaluator import BaseEvaluator
from ...core.execution_context import ExecutionContext
from ...registry import registry


@registry.evaluator("transfer", metadata={
    "description": "Transfer Ability Evaluator for Continual Learning"
})
class TransferEvaluator(BaseEvaluator):
    """迁移能力评估器"""
    
    def __init__(self, context: ExecutionContext, config: DictConfig):
        super().__init__(context, config)
        self.baseline_accuracies = {}
        
    def evaluate(self, model: torch.nn.Module, data_loader: DataLoader, 
                task_id: int = None) -> Dict[str, Any]:
        """评估迁移能力"""
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
        
        # 计算相对于基线的提升
        transfer_gain = 0.0
        if task_id in self.baseline_accuracies:
            transfer_gain = accuracy - self.baseline_accuracies[task_id]
            
        return {
            "accuracy": accuracy,
            "transfer_gain": transfer_gain,
            "task_id": task_id
        }
