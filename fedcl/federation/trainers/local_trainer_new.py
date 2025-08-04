# fedcl/federation/trainers/local_trainer.py
"""
本地训练器实现

专门用于联邦学习环境中的本地训练，继承自BaseLearner。
实现纯粹的训练逻辑，不包含通信或协调逻辑。
"""

from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from loguru import logger
import copy
import time

from ...core.base_learner import BaseLearner
from ...core.execution_context import ExecutionContext
from ...exceptions import ConfigurationError, LearnerError
from ...data.results import TaskResults


class LocalTrainer(BaseLearner):
    """
    本地训练器
    
    专门用于联邦学习环境的BaseLearner实现，提供纯粹的本地训练功能。
    不包含通信、协调等联邦逻辑，专注于模型训练和评估。
    
    重构后的职责：
    - 实现BaseLearner接口
    - 提供本地训练算法
    - 管理模型、优化器、损失函数
    - 支持模型更新和参数同步
    
    Attributes:
        model: 神经网络模型
        optimizer: 优化器
        criterion: 损失函数
        device: 计算设备
        training_stats: 训练统计信息
    """
    
    def __init__(self, context: ExecutionContext, config: DictConfig) -> None:
        """
        初始化本地训练器
        
        Args:
            context: 执行上下文
            config: 训练配置参数
            
        Raises:
            ConfigurationError: 配置参数无效时抛出
        """
        super().__init__(context, config)
        
        # 训练相关配置
        self.learning_rate = config.get("learning_rate", 0.001)
        self.batch_size = config.get("batch_size", 32)
        self.local_epochs = config.get("local_epochs", 1)
        self.weight_decay = config.get("weight_decay", 0.0)
        
        # 初始化组件
        self.criterion: Optional[nn.Module] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        
        # 训练统计信息
        self.training_stats = {
            "total_batches": 0,
            "total_samples": 0,
            "total_loss": 0.0,
            "epoch_losses": [],
            "gradient_norms": [],
            "learning_rates": []
        }
        
        # 建模和优化器初始化（延迟到需要时）
        self._initialize_components()
        
        logger.debug(f"Initialized LocalTrainer with device: {self.device}")
    
    def _initialize_components(self) -> None:
        """初始化训练组件"""
        try:
            # 初始化模型
            if self.model is None:
                self.model = self._build_model()
                self.model.to(self.device)
            
            # 初始化优化器
            if self.optimizer is None:
                self.optimizer = self._build_optimizer()
            
            # 初始化损失函数
            if self.criterion is None:
                self.criterion = self._build_criterion()
            
            # 初始化学习率调度器
            if self.scheduler is None and self.config.get("use_scheduler", False):
                self.scheduler = self._build_scheduler()
                
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise ConfigurationError(f"Component initialization failed: {e}") from e
    
    def train_task(self, task_data: DataLoader) -> TaskResults:
        """
        训练单个任务
        
        实现BaseLearner接口，执行本地训练任务。
        
        Args:
            task_data: 任务训练数据加载器
            
        Returns:
            TaskResults: 训练结果
            
        Raises:
            LearnerError: 训练过程中出现错误时抛出
        """
        try:
            logger.info("Starting local training task")
            start_time = time.time()
            
            if self.model is None or self.optimizer is None:
                self._initialize_components()
            
            self.model.train()
            task_metrics = {}
            epoch_losses = []
            
            # 多轮本地训练
            for epoch in range(self.local_epochs):
                epoch_loss = self._train_epoch(task_data, epoch)
                epoch_losses.append(epoch_loss)
                
                logger.debug(f"Epoch {epoch + 1}/{self.local_epochs}, Loss: {epoch_loss:.4f}")
            
            # 计算任务指标
            final_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
            task_metrics = {
                "final_loss": final_loss,
                "epochs_trained": self.local_epochs,
                "total_batches": self.training_stats["total_batches"],
                "total_samples": self.training_stats["total_samples"]
            }
            
            # 评估训练后的性能
            if hasattr(task_data, 'dataset') and len(task_data.dataset) > 0:
                eval_metrics = self.evaluate_task(task_data)
                task_metrics.update(eval_metrics)
            
            training_time = time.time() - start_time
            
            # 创建任务结果
            results = TaskResults(
                task_id=getattr(task_data, 'task_id', 0),
                metrics=task_metrics,
                training_time=training_time,
                memory_usage=self._get_memory_usage(),
                model_size=self._get_model_size(),
                convergence_step=len(epoch_losses),
                metadata={
                    "local_epochs": self.local_epochs,
                    "learning_rate": self.learning_rate,
                    "device": str(self.device)
                }
            )
            
            logger.info(f"Completed local training in {training_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"Local training failed: {e}")
            raise LearnerError(f"Training task failed: {e}") from e
    
    def evaluate_task(self, task_data: DataLoader) -> Dict[str, float]:
        """
        评估单个任务
        
        实现BaseLearner接口，评估模型在指定任务上的性能。
        
        Args:
            task_data: 任务评估数据加载器
            
        Returns:
            Dict[str, float]: 评估指标字典
            
        Raises:
            LearnerError: 评估过程中出现错误时抛出
        """
        try:
            if self.model is None:
                raise LearnerError("Model not initialized for evaluation")
            
            self.model.eval()
            total_loss = 0.0
            total_samples = 0
            correct_predictions = 0
            
            with torch.no_grad():
                for batch_idx, (data, targets) in enumerate(task_data):
                    data, targets = data.to(self.device), targets.to(self.device)
                    
                    outputs = self.model(data)
                    loss = self.criterion(outputs, targets)
                    
                    total_loss += loss.item() * data.size(0)
                    total_samples += data.size(0)
                    
                    # 计算准确率（假设是分类任务）
                    if outputs.dim() > 1 and outputs.size(1) > 1:
                        _, predicted = torch.max(outputs.data, 1)
                        correct_predictions += (predicted == targets).sum().item()
            
            avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
            accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
            
            metrics = {
                "eval_loss": avg_loss,
                "eval_accuracy": accuracy,
                "eval_samples": float(total_samples)
            }
            
            logger.debug(f"Evaluation metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Task evaluation failed: {e}")
            raise LearnerError(f"Evaluation failed: {e}") from e
    
    def _train_epoch(self, task_data: DataLoader, epoch: int) -> float:
        """
        训练单个轮次
        
        Args:
            task_data: 训练数据加载器
            epoch: 轮次编号
            
        Returns:
            float: 轮次平均损失
        """
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, targets) in enumerate(task_data):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 前向传播
            outputs = self.model(data)
            
            # 计算损失
            loss = self.criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            
            # 计算梯度范数（用于监控）
            grad_norm = self._compute_gradient_norm()
            self.training_stats["gradient_norms"].append(grad_norm)
            
            # 优化器更新
            self.optimizer.step()
            
            # 学习率调度器更新
            if self.scheduler is not None:
                self.scheduler.step()
            
            # 统计信息
            epoch_loss += loss.item()
            num_batches += 1
            self.training_stats["total_batches"] += 1
            self.training_stats["total_samples"] += data.size(0)
        
        # 计算平均损失
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        self.training_stats["epoch_losses"].append(avg_loss)
        self.training_stats["learning_rates"].append(self._get_current_lr())
        
        return avg_loss
    
    def _build_model(self) -> nn.Module:
        """构建模型"""
        # 这里应该根据配置构建具体的模型
        # 作为示例，我们创建一个简单的全连接网络
        model_config = self.config.get("model", {})
        model_type = model_config.get("type", "simple_fc")
        
        if model_type == "simple_fc":
            input_size = model_config.get("input_size", 784)
            hidden_size = model_config.get("hidden_size", 256)
            num_classes = model_config.get("num_classes", 10)
            
            model = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_size, num_classes)
            )
        else:
            raise ConfigurationError(f"Unsupported model type: {model_type}")
        
        return model
    
    def _build_optimizer(self) -> torch.optim.Optimizer:
        """构建优化器"""
        if self.model is None:
            raise ConfigurationError("Model must be initialized before optimizer")
        
        optimizer_type = self.config.get("optimizer", "adam").lower()
        
        if optimizer_type == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif optimizer_type == "sgd":
            momentum = self.config.get("momentum", 0.9)
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=momentum,
                weight_decay=self.weight_decay
            )
        else:
            raise ConfigurationError(f"Unsupported optimizer: {optimizer_type}")
    
    def _build_criterion(self) -> nn.Module:
        """构建损失函数"""
        criterion_type = self.config.get("criterion", "cross_entropy").lower()
        
        if criterion_type == "cross_entropy":
            return nn.CrossEntropyLoss()
        elif criterion_type == "mse":
            return nn.MSELoss()
        else:
            raise ConfigurationError(f"Unsupported criterion: {criterion_type}")
    
    def _build_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """构建学习率调度器"""
        if self.optimizer is None:
            return None
        
        scheduler_config = self.config.get("scheduler", {})
        scheduler_type = scheduler_config.get("type", "step").lower()
        
        if scheduler_type == "step":
            step_size = scheduler_config.get("step_size", 30)
            gamma = scheduler_config.get("gamma", 0.1)
            return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_type == "cosine":
            T_max = scheduler_config.get("T_max", 100)
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max)
        else:
            logger.warning(f"Unsupported scheduler type: {scheduler_type}")
            return None
    
    def _compute_gradient_norm(self) -> float:
        """计算梯度范数"""
        if self.model is None:
            return 0.0
        
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm
    
    def _get_current_lr(self) -> float:
        """获取当前学习率"""
        if self.optimizer is None:
            return 0.0
        return self.optimizer.param_groups[0]['lr']
    
    def _get_memory_usage(self) -> float:
        """获取内存使用情况"""
        if torch.cuda.is_available() and self.device.type == 'cuda':
            return torch.cuda.memory_allocated(self.device) / (1024 ** 2)  # MB
        return 0.0
    
    def _get_model_size(self) -> int:
        """获取模型参数数量"""
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters())
    
    def get_model_update(self) -> Dict[str, torch.Tensor]:
        """获取模型更新（状态字典）"""
        if self.model is None:
            raise LearnerError("Model not initialized")
        return copy.deepcopy(self.model.state_dict())
    
    def apply_model_update(self, model_state_dict: Dict[str, torch.Tensor]) -> None:
        """应用模型更新"""
        if self.model is None:
            raise LearnerError("Model not initialized")
        
        try:
            self.model.load_state_dict(model_state_dict)
            logger.debug("Applied model update successfully")
        except Exception as e:
            logger.error(f"Failed to apply model update: {e}")
            raise LearnerError(f"Model update failed: {e}") from e
    
    def reset_training_stats(self) -> None:
        """重置训练统计信息"""
        self.training_stats = {
            "total_batches": 0,
            "total_samples": 0,
            "total_loss": 0.0,
            "epoch_losses": [],
            "gradient_norms": [],
            "learning_rates": []
        }
    
    def get_training_stats(self) -> Dict[str, Any]:
        """获取训练统计信息"""
        return copy.deepcopy(self.training_stats)
