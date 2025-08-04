# fedcl/implementations/learners/packnet_learner.py
"""
PackNet 持续学习实现

基于论文: https://arxiv.org/abs/1711.05769
PackNet通过为每个任务分配专用的网络容量来避免干扰，使用二进制掩码管理参数。

主要特性:
- 任务特定的参数掩码
- 网络容量分配
- 参数共享和隔离
- 支持任务增量学习
"""

from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from loguru import logger

from ...core.base_learner import BaseLearner
from ...core.execution_context import ExecutionContext
from ...data.results import TaskResults
from ...registry import registry


@registry.learner("packnet", metadata={
    "description": "PackNet: Adding Multiple Tasks to a Single Network",
    "paper": "https://arxiv.org/abs/1711.05769",
    "supported_tasks": ["task_incremental"],
    "requires_pretrained": False
})
class PackNetLearner(BaseLearner):
    """
    PackNet 持续学习算法实现
    
    主要特点：
    1. 为每个任务分配专用参数
    2. 使用二进制掩码控制参数使用
    3. 防止任务间干扰
    4. 支持任务增量学习
    """
    
    def __init__(self, context: ExecutionContext, config: DictConfig):
        super().__init__(context, config)
        
        # PackNet特定配置
        self.sparsity_ratio = config.get("sparsity_ratio", 0.5)
        self.task_masks = {}
        
        self._build_model()
        self._build_optimizer()
        
        logger.debug(f"PackNetLearner initialized with sparsity_ratio={self.sparsity_ratio}")
        
    def _build_model(self) -> None:
        """构建神经网络模型"""
        input_channels = self.config.get("input_channels", 3)
        num_classes = self.config.get("num_classes", 10)
        
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 16, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        ).to(self.device)
        
    def _build_optimizer(self) -> None:
        """构建优化器"""
        lr = self.config.get("learning_rate", 0.001)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
    def train_task(self, task_data: DataLoader) -> TaskResults:
        """训练单个任务"""
        self.model.train()
        
        # 为当前任务创建掩码
        self._create_task_mask(self.current_task_id)
        
        total_loss = 0.0
        total_samples = 0
        correct_predictions = 0
        
        epochs = self.config.get("epochs", 10)
        
        for epoch in range(epochs):
            for inputs, targets in task_data:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                self.optimizer.zero_grad()
                
                # 应用当前任务掩码
                self._apply_mask(self.current_task_id)
                
                outputs = self.model(inputs)
                loss = F.cross_entropy(outputs, targets)
                loss.backward()
                
                # 只更新当前任务可用的参数
                self._mask_gradients(self.current_task_id)
                
                self.optimizer.step()
                
                total_loss += loss.item()
                total_samples += inputs.size(0)
                
                _, predicted = outputs.max(1)
                correct_predictions += predicted.eq(targets).sum().item()
                
        avg_loss = total_loss / (len(task_data) * epochs)
        accuracy = correct_predictions / total_samples
        
        return TaskResults(
            task_id=self.current_task_id,
            metrics={"loss": avg_loss, "accuracy": accuracy},
            model_state=self.get_model_state(),
            metadata={"algorithm": "PackNet"}
        )
        
    def _create_task_mask(self, task_id: int) -> None:
        """为任务创建参数掩码"""
        mask = {}
        for name, param in self.model.named_parameters():
            # 简化的掩码创建：随机选择参数
            mask[name] = torch.rand_like(param) > self.sparsity_ratio
        self.task_masks[task_id] = mask
        
    def _apply_mask(self, task_id: int) -> None:
        """应用任务掩码"""
        if task_id in self.task_masks:
            for name, param in self.model.named_parameters():
                if name in self.task_masks[task_id]:
                    param.data *= self.task_masks[task_id][name]
                    
    def _mask_gradients(self, task_id: int) -> None:
        """掩码梯度"""
        if task_id in self.task_masks:
            for name, param in self.model.named_parameters():
                if param.grad is not None and name in self.task_masks[task_id]:
                    param.grad *= self.task_masks[task_id][name]
                    
    def evaluate_task(self, task_data: DataLoader) -> Dict[str, float]:
        """评估任务性能"""
        self.model.eval()
        
        # 应用对应任务的掩码
        self._apply_mask(self.current_task_id)
        
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
