# fedcl/implementations/learners/lwf_learner.py
"""
Learning without Forgetting (LwF) 实现

基于论文: https://arxiv.org/abs/1606.09282
LwF通过知识蒸馏机制保持之前任务的知识，在学习新任务时约束模型输出。

主要特性:
- 知识蒸馏损失
- 温度缩放的软标签
- 渐进式任务学习
- 无需存储之前任务数据
"""

from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from loguru import logger
import copy

from ...core.base_learner import BaseLearner
from ...core.execution_context import ExecutionContext
from ...data.results import TaskResults
from ...registry import registry


@registry.learner("lwf", metadata={
    "description": "Learning without Forgetting",
    "paper": "https://arxiv.org/abs/1606.09282",
    "supported_tasks": ["class_incremental"],
    "requires_pretrained": False
})
class LwFLearner(BaseLearner):
    """
    LwF (Learning without Forgetting) 持续学习算法实现
    
    主要特点：
    1. 使用知识蒸馏保持旧知识
    2. 不需要存储旧任务数据
    3. 通过软标签传递知识
    4. 支持类增量学习
    """
    
    def __init__(self, context: ExecutionContext, config: DictConfig):
        super().__init__(context, config)
        
        # LwF特定配置
        self.distillation_weight = config.get("distillation_weight", 1.0)
        self.temperature = config.get("temperature", 4.0)
        
        # 存储旧模型用于蒸馏
        self.old_model = None
        self.old_classes = []
        
        self._build_model()
        self._build_optimizer()
        
        logger.debug(f"LwFLearner initialized with distillation_weight={self.distillation_weight}")
        
    def _build_model(self) -> None:
        """构建神经网络模型"""
        input_channels = self.config.get("input_channels", 3)
        num_classes = self.config.get("num_classes", 10)
        
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        ).to(self.device)
        
    def _build_optimizer(self) -> None:
        """构建优化器"""
        lr = self.config.get("learning_rate", 0.001)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
    def train_task(self, task_data: DataLoader) -> TaskResults:
        """训练单个任务"""
        self.model.train()
        
        # 保存旧模型用于蒸馏
        if self.old_model is not None:
            self.old_model.eval()
            
        total_loss = 0.0
        total_ce_loss = 0.0
        total_distill_loss = 0.0
        total_samples = 0
        correct_predictions = 0
        
        epochs = self.config.get("epochs", 10)
        
        for epoch in range(epochs):
            for inputs, targets in task_data:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                self.optimizer.zero_grad()
                
                outputs = self.model(inputs)
                ce_loss = F.cross_entropy(outputs, targets)
                
                # 计算蒸馏损失
                distill_loss = torch.tensor(0.0, device=self.device)
                if self.old_model is not None:
                    with torch.no_grad():
                        old_outputs = self.old_model(inputs)
                    distill_loss = self._distillation_loss(outputs[:, :len(self.old_classes)], old_outputs)
                
                total_loss_batch = ce_loss + self.distillation_weight * distill_loss
                total_loss_batch.backward()
                self.optimizer.step()
                
                total_loss += total_loss_batch.item()
                total_ce_loss += ce_loss.item()
                total_distill_loss += distill_loss.item()
                total_samples += inputs.size(0)
                
                _, predicted = outputs.max(1)
                correct_predictions += predicted.eq(targets).sum().item()
                
        # 更新旧模型
        self._update_old_model()
        
        avg_loss = total_loss / (len(task_data) * epochs)
        accuracy = correct_predictions / total_samples
        
        return TaskResults(
            task_id=self.current_task_id,
            metrics={
                "total_loss": avg_loss,
                "ce_loss": total_ce_loss / (len(task_data) * epochs),
                "distill_loss": total_distill_loss / (len(task_data) * epochs),
                "accuracy": accuracy
            },
            model_state=self.get_model_state(),
            metadata={"algorithm": "LwF"}
        )
        
    def _distillation_loss(self, new_outputs: torch.Tensor, old_outputs: torch.Tensor) -> torch.Tensor:
        """计算知识蒸馏损失"""
        new_soft = F.softmax(new_outputs / self.temperature, dim=1)
        old_soft = F.softmax(old_outputs / self.temperature, dim=1)
        return F.kl_div(new_soft.log(), old_soft, reduction='batchmean') * (self.temperature ** 2)
        
    def _update_old_model(self) -> None:
        """更新旧模型"""
        self.old_model = copy.deepcopy(self.model)
        # 更新已学习的类别
        if self.current_task_id not in self.old_classes:
            self.old_classes.append(self.current_task_id)
            
    def evaluate_task(self, task_data: DataLoader) -> Dict[str, float]:
        """评估任务性能"""
        self.model.eval()
        total_samples = 0
        correct_predictions = 0
        
        with torch.no_grad():
            for inputs, targets in task_data:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                correct_predictions += predicted.eq(targets).sum().item()
                total_samples += inputs.size(0)
                
        return {"accuracy": correct_predictions / total_samples}
