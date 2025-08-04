# fedcl/implementations/learners/ewc_learner.py
"""
Elastic Weight Consolidation (EWC) 实现

基于论文: https://arxiv.org/abs/1612.00796
EWC通过计算Fisher信息矩阵来估计参数重要性，在训练新任务时对重要参数施加弹性约束。

主要特性:
- Fisher信息矩阵计算
- 弹性权重约束
- 多任务渐进学习
- 防止灾难性遗忘
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


@registry.learner("ewc", metadata={
    "description": "Elastic Weight Consolidation for Continual Learning",
    "paper": "https://arxiv.org/abs/1612.00796",
    "supported_tasks": ["class_incremental", "domain_incremental", "task_incremental"],
    "requires_pretrained": False
})
class EWCLearner(BaseLearner):
    """
    EWC (Elastic Weight Consolidation) 持续学习算法实现
    
    主要特点：
    1. 计算Fisher信息矩阵估计参数重要性
    2. 对重要参数施加弹性约束
    3. 保持之前任务的知识
    4. 支持多任务增量学习
    """
    
    def __init__(self, context: ExecutionContext, config: DictConfig):
        """
        初始化EWC学习器
        
        Args:
            context: 执行上下文
            config: 配置参数，应包含：
                - ewc_lambda: EWC正则化强度
                - fisher_sample_size: Fisher矩阵采样大小
                - online_ewc: 是否使用在线EWC
                - gamma: 在线EWC的折扣因子
        """
        super().__init__(context, config)
        
        # EWC特定配置
        self.ewc_lambda = config.get("ewc_lambda", 1000.0)
        self.fisher_sample_size = config.get("fisher_sample_size", 1000)
        self.online_ewc = config.get("online_ewc", False)
        self.gamma = config.get("gamma", 1.0)
        
        # 存储之前任务的参数和Fisher信息
        self.previous_params: Dict[str, torch.Tensor] = {}
        self.fisher_information: Dict[str, torch.Tensor] = {}
        self.task_count = 0
        
        # 构建模型
        self._build_model()
        self._build_optimizer()
        
        logger.info(f"EWCLearner initialized with lambda={self.ewc_lambda}, "
                   f"online_ewc={self.online_ewc}")
        
    def _build_model(self) -> None:
        """构建神经网络模型"""
        # 简化的CNN模型示例
        input_channels = self.config.get("input_channels", 3)
        num_classes = self.config.get("num_classes", 10)
        
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
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
        self.task_count += 1
        
        total_loss = 0.0
        total_ce_loss = 0.0
        total_ewc_loss = 0.0
        total_samples = 0
        correct_predictions = 0
        
        epochs = self.config.get("epochs", 10)
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_samples = 0
            
            for batch_idx, (inputs, targets) in enumerate(task_data):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                self.optimizer.zero_grad()
                
                # 前向传播
                outputs = self.model(inputs)
                
                # 计算交叉熵损失
                ce_loss = F.cross_entropy(outputs, targets)
                
                # 计算EWC损失
                ewc_loss = self._compute_ewc_loss()
                
                # 总损失
                total_loss_batch = ce_loss + self.ewc_lambda * ewc_loss
                
                # 反向传播
                total_loss_batch.backward()
                self.optimizer.step()
                
                # 统计
                epoch_loss += total_loss_batch.item()
                epoch_samples += inputs.size(0)
                total_ce_loss += ce_loss.item()
                total_ewc_loss += ewc_loss.item()
                
                _, predicted = outputs.max(1)
                correct_predictions += predicted.eq(targets).sum().item()
                
            total_loss += epoch_loss
            total_samples += epoch_samples
            
            if epoch % 2 == 0:
                if len(task_data) > 0:
                    logger.debug(f"Epoch {epoch}, Loss: {epoch_loss/len(task_data):.4f}")
                else:
                    logger.debug(f"Epoch {epoch}, Loss: 0.0000 (empty dataset)")
                
        # 训练完成后更新Fisher信息矩阵
        if len(task_data) > 0:
            self._update_fisher_information(task_data)
        
        # 计算平均指标
        total_batches = len(task_data) * epochs
        if total_batches > 0:
            avg_total_loss = total_loss / total_batches
            avg_ce_loss = total_ce_loss / total_batches  
            avg_ewc_loss = total_ewc_loss / total_batches
        else:
            avg_total_loss = 0.0
            avg_ce_loss = 0.0
            avg_ewc_loss = 0.0
            
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
        
        return TaskResults(
            task_id=self.current_task_id,
            metrics={
                "total_loss": avg_total_loss,
                "ce_loss": avg_ce_loss,
                "ewc_loss": avg_ewc_loss,
                "accuracy": accuracy,
                "total_samples": total_samples
            },
            metadata={
                "algorithm": "EWC",
                "ewc_lambda": self.ewc_lambda,
                "task_count": self.task_count,
                "online_ewc": self.online_ewc,
                "model_state": self.get_model_state()
            }
        )
        
    def _compute_ewc_loss(self) -> torch.Tensor:
        """
        计算EWC正则化损失
        
        Returns:
            EWC损失值
        """
        if not self.previous_params or not self.fisher_information:
            return torch.tensor(0.0, device=self.device)
            
        ewc_loss = 0.0
        
        for name, param in self.model.named_parameters():
            if name in self.previous_params:
                # 计算参数变化的平方差
                param_diff = param - self.previous_params[name]
                
                # 使用Fisher信息加权
                if name in self.fisher_information:
                    fisher_weight = self.fisher_information[name]
                    ewc_loss += (fisher_weight * param_diff ** 2).sum()
                    
        return ewc_loss / 2.0
        
    def _update_fisher_information(self, task_data: DataLoader) -> None:
        """
        更新Fisher信息矩阵
        
        Args:
            task_data: 当前任务数据
        """
        logger.info("Computing Fisher Information Matrix...")
        
        self.model.eval()
        
        # 初始化Fisher信息字典
        fisher_dict = {}
        for name, param in self.model.named_parameters():
            fisher_dict[name] = torch.zeros_like(param.data)
            
        # 采样计算Fisher信息
        sample_count = 0
        max_samples = min(self.fisher_sample_size, len(task_data.dataset))
        
        for inputs, targets in task_data:
            if sample_count >= max_samples:
                break
                
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            batch_size = inputs.size(0)
            
            # 前向传播
            outputs = self.model(inputs)
            
            # 对每个样本计算梯度
            for i in range(batch_size):
                if sample_count >= max_samples:
                    break
                    
                self.model.zero_grad()
                
                # 计算单个样本的负对数似然
                sample_output = outputs[i:i+1]
                sample_target = targets[i:i+1]
                loss = F.cross_entropy(sample_output, sample_target)
                
                # 反向传播获取梯度
                loss.backward(retain_graph=True)
                
                # 累积梯度的平方
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        fisher_dict[name] += param.grad.data ** 2
                        
                sample_count += 1
                
        # 归一化Fisher信息
        for name in fisher_dict:
            fisher_dict[name] /= sample_count
            
        # 更新Fisher信息矩阵
        if self.online_ewc and self.fisher_information:
            # 在线EWC：使用移动平均
            for name in fisher_dict:
                if name in self.fisher_information:
                    self.fisher_information[name] = (
                        self.gamma * self.fisher_information[name] +
                        (1 - self.gamma) * fisher_dict[name]
                    )
                else:
                    self.fisher_information[name] = fisher_dict[name]
        else:
            # 标准EWC：直接替换
            self.fisher_information = fisher_dict
            
        # 保存当前参数
        self.previous_params = {}
        for name, param in self.model.named_parameters():
            self.previous_params[name] = param.data.clone()
            
        logger.info(f"Fisher Information Matrix updated for task {self.task_count}")
        
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
        total_ce_loss = 0.0
        total_ewc_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in task_data:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                ce_loss = F.cross_entropy(outputs, targets)
                ewc_loss = self._compute_ewc_loss()
                total_loss_batch = ce_loss + self.ewc_lambda * ewc_loss
                
                total_loss += total_loss_batch.item()
                total_ce_loss += ce_loss.item()
                total_ewc_loss += ewc_loss.item()
                total_samples += inputs.size(0)
                
                _, predicted = outputs.max(1)
                correct_predictions += predicted.eq(targets).sum().item()
                
        return {
            "accuracy": correct_predictions / total_samples,
            "total_loss": total_loss / len(task_data),
            "ce_loss": total_ce_loss / len(task_data),
            "ewc_loss": total_ewc_loss / len(task_data),
            "total_samples": total_samples
        }
        
    def get_fisher_statistics(self) -> Dict[str, Any]:
        """
        获取Fisher信息统计
        
        Returns:
            Fisher信息统计
        """
        if not self.fisher_information:
            return {"fisher_computed": False}
            
        total_fisher_sum = 0.0
        layer_stats = {}
        
        for name, fisher in self.fisher_information.items():
            fisher_sum = fisher.sum().item()
            fisher_mean = fisher.mean().item()
            fisher_std = fisher.std().item()
            
            total_fisher_sum += fisher_sum
            layer_stats[name] = {
                "sum": fisher_sum,
                "mean": fisher_mean,
                "std": fisher_std
            }
            
        return {
            "fisher_computed": True,
            "total_fisher_sum": total_fisher_sum,
            "task_count": self.task_count,
            "layer_stats": layer_stats,
            "ewc_lambda": self.ewc_lambda
        }
        
    def reset_fisher_information(self) -> None:
        """重置Fisher信息矩阵"""
        self.fisher_information.clear()
        self.previous_params.clear()
        self.task_count = 0
        logger.info("Fisher information matrix reset")
