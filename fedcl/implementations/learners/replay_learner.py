# fedcl/implementations/learners/replay_learner.py
"""
Experience Replay 持续学习实现

基于经验重放的持续学习方法，通过存储和重放之前任务的样本来防止灾难性遗忘。

主要特性:
- 样本存储缓冲区
- 多种采样策略（随机、按类平衡等）
- 内存管理和样本更新
- 支持多种重放策略
"""

from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from omegaconf import DictConfig
from loguru import logger
import random
import numpy as np
from collections import defaultdict

from ...core.base_learner import BaseLearner
from ...core.execution_context import ExecutionContext
from ...data.results import TaskResults
from ...registry import registry


class ReplayBuffer:
    """
    经验重放缓冲区
    
    存储之前任务的样本，支持多种采样和更新策略。
    """
    
    def __init__(
        self,
        capacity: int,
        input_shape: Tuple[int, ...],
        num_classes: int,
        device: torch.device,
        sampling_strategy: str = "random"
    ):
        """
        初始化重放缓冲区
        
        Args:
            capacity: 缓冲区容量
            input_shape: 输入数据形状
            num_classes: 类别数量
            device: 计算设备
            sampling_strategy: 采样策略 ("random", "balanced", "oldest")
        """
        self.capacity = capacity
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.device = device
        self.sampling_strategy = sampling_strategy
        
        # 存储缓冲区
        self.inputs = torch.zeros((capacity, *input_shape), device=device)
        self.targets = torch.zeros(capacity, dtype=torch.long, device=device)
        self.task_ids = torch.zeros(capacity, dtype=torch.long, device=device)
        self.timestamps = torch.zeros(capacity, dtype=torch.long, device=device)
        
        # 管理信息
        self.size = 0
        self.current_idx = 0
        self.global_timestamp = 0
        
        # 按类别索引
        self.class_indices = defaultdict(list)
        
    def add_samples(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        task_id: int
    ) -> None:
        """
        添加样本到缓冲区
        
        Args:
            inputs: 输入数据
            targets: 目标标签
            task_id: 任务ID
        """
        batch_size = inputs.size(0)
        
        for i in range(batch_size):
            # 存储样本
            idx = self.current_idx % self.capacity
            
            # 如果位置已被占用，需要从class_indices中移除旧样本
            if self.size >= self.capacity:
                old_target = self.targets[idx].item()
                if idx in self.class_indices[old_target]:
                    self.class_indices[old_target].remove(idx)
                    
            # 存储新样本
            self.inputs[idx] = inputs[i]
            self.targets[idx] = targets[i]
            self.task_ids[idx] = task_id
            self.timestamps[idx] = self.global_timestamp
            
            # 更新类别索引
            target_class = targets[i].item()
            self.class_indices[target_class].append(idx)
            
            # 更新指针
            self.current_idx += 1
            self.global_timestamp += 1
            self.size = min(self.size + 1, self.capacity)
            
    def sample_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        从缓冲区采样批次
        
        Args:
            batch_size: 批次大小
            
        Returns:
            (inputs, targets, task_ids): 采样的数据
        """
        if self.size == 0:
            return torch.empty(0), torch.empty(0), torch.empty(0)
            
        batch_size = min(batch_size, self.size)
        
        if self.sampling_strategy == "random":
            indices = torch.randperm(self.size)[:batch_size]
        elif self.sampling_strategy == "balanced":
            indices = self._balanced_sampling(batch_size)
        elif self.sampling_strategy == "oldest":
            # 按时间戳排序，选择最旧的样本
            _, sorted_indices = torch.sort(self.timestamps[:self.size])
            indices = sorted_indices[:batch_size]
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")
            
        return (
            self.inputs[indices],
            self.targets[indices],
            self.task_ids[indices]
        )
        
    def _balanced_sampling(self, batch_size: int) -> torch.Tensor:
        """
        按类别平衡采样
        
        Args:
            batch_size: 批次大小
            
        Returns:
            采样索引
        """
        available_classes = [cls for cls, indices in self.class_indices.items() if indices]
        
        if not available_classes:
            return torch.randperm(self.size)[:batch_size]
            
        samples_per_class = batch_size // len(available_classes)
        remainder = batch_size % len(available_classes)
        
        selected_indices = []
        
        for i, cls in enumerate(available_classes):
            class_indices = self.class_indices[cls]
            num_samples = samples_per_class + (1 if i < remainder else 0)
            num_samples = min(num_samples, len(class_indices))
            
            if num_samples > 0:
                sampled = random.sample(class_indices, num_samples)
                selected_indices.extend(sampled)
                
        # 如果采样不足，随机补充
        if len(selected_indices) < batch_size:
            remaining = batch_size - len(selected_indices)
            all_indices = set(range(self.size))
            available_indices = list(all_indices - set(selected_indices))
            if available_indices:
                additional = random.sample(
                    available_indices,
                    min(remaining, len(available_indices))
                )
                selected_indices.extend(additional)
                
        return torch.tensor(selected_indices[:batch_size], dtype=torch.long)
        
    def get_statistics(self) -> Dict[str, Any]:
        """获取缓冲区统计信息"""
        class_counts = {}
        for cls, indices in self.class_indices.items():
            class_counts[cls] = len(indices)
            
        return {
            "size": self.size,
            "capacity": self.capacity,
            "utilization": self.size / self.capacity,
            "class_counts": class_counts,
            "num_classes_stored": len([cls for cls, indices in self.class_indices.items() if indices])
        }


@registry.learner("replay", metadata={
    "description": "Experience Replay for Continual Learning",
    "paper": "Classic experience replay mechanism",
    "supported_tasks": ["class_incremental", "domain_incremental", "task_incremental"],
    "requires_pretrained": False
})
class ReplayLearner(BaseLearner):
    """
    经验重放持续学习算法实现
    
    主要特点：
    1. 存储之前任务的样本
    2. 训练时混合当前任务和重放样本
    3. 支持多种采样策略
    4. 防止灾难性遗忘
    """
    
    def __init__(self, context: ExecutionContext, config: DictConfig):
        """
        初始化Replay学习器
        
        Args:
            context: 执行上下文
            config: 配置参数，应包含：
                - replay_capacity: 重放缓冲区容量
                - replay_batch_size: 重放批次大小
                - sampling_strategy: 采样策略
                - replay_frequency: 重放频率
        """
        super().__init__(context, config)
        
        # Replay特定配置
        self.replay_capacity = config.get("replay_capacity", 1000)
        self.replay_batch_size = config.get("replay_batch_size", 32)
        self.sampling_strategy = config.get("sampling_strategy", "balanced")
        self.replay_frequency = config.get("replay_frequency", 1)  # 每N个batch重放一次
        
        # 获取数据形状信息
        input_shape = config.get("input_shape", (3, 32, 32))
        num_classes = config.get("num_classes", 10)
        
        # 初始化重放缓冲区
        self.replay_buffer = ReplayBuffer(
            capacity=self.replay_capacity,
            input_shape=input_shape,
            num_classes=num_classes,
            device=self.device,
            sampling_strategy=self.sampling_strategy
        )
        
        # 构建模型
        self._build_model()
        self._build_optimizer()
        
        logger.info(f"ReplayLearner initialized with capacity={self.replay_capacity}, "
                   f"strategy={self.sampling_strategy}")
        
    def _build_model(self) -> None:
        """构建神经网络模型"""
        input_channels = self.config.get("input_channels", 3)
        num_classes = self.config.get("num_classes", 10)
        
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(256 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        ).to(self.device)
        
    def _build_optimizer(self) -> None:
        """构建优化器"""
        lr = self.config.get("learning_rate", 0.001)
        weight_decay = self.config.get("weight_decay", 1e-4)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
    def train_task(self, task_data: DataLoader) -> TaskResults:
        """
        训练单个任务
        
        Args:
            task_data: 任务数据加载器
            
        Returns:
            TaskResults: 训练结果
        """
        self.model.train()
        
        # 首先将当前任务的样本添加到重放缓冲区
        self._populate_buffer(task_data)
        
        total_loss = 0.0
        total_current_loss = 0.0
        total_replay_loss = 0.0
        total_samples = 0
        correct_predictions = 0
        
        epochs = self.config.get("epochs", 10)
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_samples = 0
            
            for batch_idx, (inputs, targets) in enumerate(task_data):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                self.optimizer.zero_grad()
                
                # 当前任务训练
                outputs = self.model(inputs)
                current_loss = F.cross_entropy(outputs, targets)
                
                total_loss_batch = current_loss
                total_current_loss += current_loss.item()
                
                # 重放训练
                if batch_idx % self.replay_frequency == 0 and self.replay_buffer.size > 0:
                    replay_inputs, replay_targets, _ = self.replay_buffer.sample_batch(
                        self.replay_batch_size
                    )
                    
                    if replay_inputs.size(0) > 0:
                        replay_outputs = self.model(replay_inputs)
                        replay_loss = F.cross_entropy(replay_outputs, replay_targets)
                        total_loss_batch += replay_loss
                        total_replay_loss += replay_loss.item()
                
                # 反向传播
                total_loss_batch.backward()
                self.optimizer.step()
                
                # 统计
                epoch_loss += total_loss_batch.item()
                epoch_samples += inputs.size(0)
                
                _, predicted = outputs.max(1)
                correct_predictions += predicted.eq(targets).sum().item()
                
            total_loss += epoch_loss
            total_samples += epoch_samples
            
            if epoch % 2 == 0:
                logger.debug(f"Epoch {epoch}, Loss: {epoch_loss/len(task_data):.4f}")
                
        # 计算平均指标
        num_batches_total = len(task_data) * epochs
        avg_total_loss = total_loss / num_batches_total
        avg_current_loss = total_current_loss / num_batches_total
        avg_replay_loss = total_replay_loss / max(1, total_replay_loss) if total_replay_loss > 0 else 0.0
        accuracy = correct_predictions / total_samples
        
        return TaskResults(
            task_id=self.current_task_id,
            metrics={
                "total_loss": avg_total_loss,
                "current_loss": avg_current_loss,
                "replay_loss": avg_replay_loss,
                "accuracy": accuracy,
                "total_samples": total_samples,
                "buffer_size": self.replay_buffer.size
            },
            model_state=self.get_model_state(),
            metadata={
                "algorithm": "Replay",
                "replay_capacity": self.replay_capacity,
                "sampling_strategy": self.sampling_strategy,
                "buffer_utilization": self.replay_buffer.size / self.replay_capacity
            }
        )
        
    def _populate_buffer(self, task_data: DataLoader) -> None:
        """
        将当前任务的样本添加到重放缓冲区
        
        Args:
            task_data: 当前任务数据
        """
        logger.info(f"Adding samples from task {self.current_task_id} to replay buffer...")
        
        samples_added = 0
        for inputs, targets in task_data:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # 可以选择性地添加样本（例如只添加部分样本以节省内存）
            add_ratio = self.config.get("buffer_add_ratio", 1.0)
            if add_ratio < 1.0:
                indices = torch.randperm(inputs.size(0))[:int(inputs.size(0) * add_ratio)]
                inputs = inputs[indices]
                targets = targets[indices]
                
            self.replay_buffer.add_samples(inputs, targets, self.current_task_id)
            samples_added += inputs.size(0)
            
        logger.info(f"Added {samples_added} samples to replay buffer. "
                   f"Buffer size: {self.replay_buffer.size}/{self.replay_capacity}")
        
    def evaluate_task(self, task_data: DataLoader) -> Dict[str, float]:
        """
        评估任务性能
        
        Args:
            task_data: 任务数据加载器
            
        Returns:
            评估指标字典
        """
        self.model.eval()
        
        total_samples = 0
        correct_predictions = 0
        total_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in task_data:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = F.cross_entropy(outputs, targets)
                
                total_loss += loss.item()
                total_samples += inputs.size(0)
                _, predicted = outputs.max(1)
                correct_predictions += predicted.eq(targets).sum().item()
                
        return {
            "accuracy": correct_predictions / total_samples,
            "loss": total_loss / len(task_data),
            "total_samples": total_samples,
            "buffer_size": self.replay_buffer.size
        }
        
    def get_buffer_statistics(self) -> Dict[str, Any]:
        """
        获取重放缓冲区统计信息
        
        Returns:
            缓冲区统计信息
        """
        return self.replay_buffer.get_statistics()
        
    def clear_buffer(self) -> None:
        """清空重放缓冲区"""
        self.replay_buffer.size = 0
        self.replay_buffer.current_idx = 0
        self.replay_buffer.class_indices.clear()
        logger.info("Replay buffer cleared")
        
    def get_replay_samples(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取重放样本用于可视化或分析
        
        Args:
            num_samples: 样本数量
            
        Returns:
            (inputs, targets): 重放样本
        """
        if self.replay_buffer.size == 0:
            return torch.empty(0), torch.empty(0)
            
        inputs, targets, _ = self.replay_buffer.sample_batch(num_samples)
        return inputs.cpu(), targets.cpu()
