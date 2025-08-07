# fedcl/implementations/evaluators/loss_evaluator.py
"""
损失评估器实现
"""

from typing import Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from ...core.base_evaluator import BaseEvaluator
from ...core.execution_context import ExecutionContext
from ...registry import registry


@registry.evaluator("loss", metadata={
    "description": "Loss Evaluator for Model Training"
})
class LossEvaluator(BaseEvaluator):
    """损失评估器"""
    
    def __init__(self, context: ExecutionContext, config: DictConfig):
        super().__init__(context, config)
        
        # 获取损失函数类型
        self.loss_function = config.get("loss_function", "cross_entropy")
        
        # 创建损失函数
        if self.loss_function == "cross_entropy":
            self.criterion = nn.CrossEntropyLoss()
        elif self.loss_function == "mse":
            self.criterion = nn.MSELoss()
        elif self.loss_function == "nll":
            self.criterion = nn.NLLLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()  # 默认使用交叉熵
    
    def evaluate(self, model: torch.nn.Module, data_loader: DataLoader, 
                task_id: int = None) -> Dict[str, Any]:
        """评估模型损失"""
        model.eval()
        total_loss = 0.0
        total_samples = 0
        correct = 0
        num_batches = 0
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                total_samples += inputs.size(0)
                num_batches += 1
                
                # 同时计算准确率
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        accuracy = correct / total_samples if total_samples > 0 else 0.0
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "total_loss": total_loss,
            "total_samples": total_samples,
            "num_batches": num_batches,
            "task_id": task_id
        }
    
    def compute_task_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """计算任务指标"""
        # 计算损失
        loss = self.criterion(predictions, targets)
        
        # 计算准确率
        _, predicted = predictions.max(1)
        total = targets.size(0)
        correct = predicted.eq(targets).sum().item()
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            "loss": loss.item(),
            "accuracy": accuracy
        }
