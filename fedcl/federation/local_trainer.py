# fedcl/federation/local_trainer.py
"""
本地训练器实现

负责在客户端或伪联邦模式下执行本地训练，包括模型训练、评估、
参数更新、梯度计算等功能。与BaseLearner配合完成持续学习任务。
"""

from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from loguru import logger
import copy

from ..core.base_learner import BaseLearner
from ..core.exceptions import ConfigurationError, TrainingError


class LocalTrainer:
    """
    本地训练器
    
    负责在联邦学习环境中执行本地训练任务，作为 BaseLearner 的协调器。
    LocalTrainer 专注于联邦学习的训练执行细节，而将模型和优化器管理
    委托给 BaseLearner，避免职责重复。
    
    Attributes:
        learner: 持续学习算法实现（管理模型、优化器等核心组件）
        config: 训练配置参数
        device: 计算设备（委托给 BaseLearner）
        criterion: 损失函数（LocalTrainer 特有）
        training_stats: 训练统计信息（LocalTrainer 特有）
    
    设计原则:
        - BaseLearner: 管理模型、优化器、设备等核心组件
        - LocalTrainer: 管理训练执行、损失函数、统计信息等
    """
    
    def __init__(self, learner: BaseLearner, config: DictConfig) -> None:
        """
        初始化本地训练器
        
        Args:
            learner: 基础学习器实例
            config: 训练配置参数
            
        Raises:
            ConfigurationError: 配置参数无效时抛出
        """
        if not isinstance(learner, BaseLearner):
            raise ConfigurationError("Invalid learner provided")
            
        if not isinstance(config, DictConfig):
            raise ConfigurationError("Invalid configuration provided")
            
        self.learner = learner
        self.config = config
        
        # 委托给 BaseLearner 管理设备
        self.device = self.learner.get_device()
        
        # 训练相关配置
        self.learning_rate = config.get("learning_rate", 0.001)
        self.batch_size = config.get("batch_size", 32)
        self.local_epochs = config.get("local_epochs", 1)
        
        # LocalTrainer 特有的组件（不与 BaseLearner 重复）
        self.criterion: Optional[nn.Module] = None
        self.training_stats = {
            "total_batches": 0,
            "total_samples": 0,
            "total_loss": 0.0,
            "epoch_losses": [],
            "gradient_norms": [],
            "learning_rates": []
        }
        
        logger.info(f"Initialized LocalTrainer with device: {self.device}")
    
    def train_epoch(self, model: torch.nn.Module, task_data: DataLoader) -> Dict[str, float]:
        """
        训练一个轮次
        
        Args:
            model: 要训练的模型
            task_data: 训练数据加载器
            
        Returns:
            训练指标字典
        """
        try:
            # 设置模型到 BaseLearner（统一管理）
            self.learner.set_model(model)
            current_model = self.learner.get_model()
            current_model.train()
            
            # 初始化优化器（委托给 BaseLearner）
            optimizer = self._get_or_create_optimizer(current_model)
                
            # 初始化损失函数（LocalTrainer 特有）
            if self.criterion is None:
                self._initialize_criterion()
            
            epoch_loss = 0.0
            num_batches = 0
            num_samples = 0
            
            for batch_idx, (data, target) in enumerate(task_data):
                data, target = data.to(self.device), target.to(self.device)
                
                # 清零梯度
                optimizer.zero_grad()
                
                # 前向传播
                output = current_model(data)
                
                # 计算损失
                loss = self.criterion(output, target)
                
                # 反向传播
                loss.backward()
                
                # 优化器更新
                optimizer.step()
                
                # 统计信息
                epoch_loss += loss.item()
                num_batches += 1
                num_samples += data.size(0)
                
                # 记录梯度范数
                grad_norm = self._compute_gradient_norm(current_model)
                self.training_stats["gradient_norms"].append(grad_norm)
                
                self.training_stats["total_batches"] += 1
                
            # 计算平均损失
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            
            # 更新统计信息
            self.training_stats["total_samples"] += num_samples
            self.training_stats["total_loss"] += epoch_loss
            self.training_stats["epoch_losses"].append(avg_loss)
            self.training_stats["learning_rates"].append(self._get_current_lr())
            
            metrics = {
                "train_loss": avg_loss,
                "num_batches": num_batches,
                "num_samples": num_samples,
                "gradient_norm": grad_norm
            }
            
            logger.debug(f"Epoch training completed: loss={avg_loss:.6f}, "
                        f"batches={num_batches}, samples={num_samples}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Training epoch failed: {e}")
            raise TrainingError(f"Training epoch failed: {e}")
    
    def evaluate_model(self, model: torch.nn.Module, test_data: DataLoader) -> Dict[str, float]:
        """
        评估模型
        
        Args:
            model: 要评估的模型
            test_data: 测试数据加载器
            
        Returns:
            评估指标字典
        """
        try:
            model.eval()
            model = model.to(self.device)
            
            if self.criterion is None:
                self._initialize_criterion()
            
            total_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, target in test_data:
                    data, target = data.to(self.device), target.to(self.device)
                    
                    output = model(data)
                    loss = self.criterion(output, target)
                    
                    total_loss += loss.item()
                    
                    # 计算准确率
                    if len(output.shape) > 1 and output.shape[1] > 1:
                        # 分类任务
                        _, predicted = torch.max(output.data, 1)
                        total += target.size(0)
                        correct += (predicted == target).sum().item()
                    else:
                        # 回归任务或二分类
                        total += target.size(0)
                        if output.shape[1] == 1:  # 回归
                            # 使用MAE作为准确率替代
                            mae = torch.abs(output.squeeze() - target).mean().item()
                            correct += len(target) * (1.0 - min(mae, 1.0))  # 简化的准确率计算
            
            avg_loss = total_loss / len(test_data) if len(test_data) > 0 else 0.0
            accuracy = correct / total if total > 0 else 0.0
            
            metrics = {
                "test_loss": avg_loss,
                "test_accuracy": accuracy,
                "num_samples": total
            }
            
            logger.debug(f"Model evaluation completed: loss={avg_loss:.6f}, "
                        f"accuracy={accuracy:.4f}, samples={total}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            raise TrainingError(f"Model evaluation failed: {e}")
    
    def compute_model_update(self, old_model: torch.nn.Module, 
                           new_model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        """
        计算模型更新（新模型参数 - 旧模型参数）
        
        Args:
            old_model: 旧模型
            new_model: 新模型
            
        Returns:
            模型更新字典
        """
        try:
            old_params = dict(old_model.named_parameters())
            new_params = dict(new_model.named_parameters())
            
            model_update = {}
            for name in old_params:
                if name in new_params:
                    model_update[name] = new_params[name].data - old_params[name].data
                    
            logger.debug(f"Computed model update with {len(model_update)} parameters")
            return model_update
            
        except Exception as e:
            logger.error(f"Failed to compute model update: {e}")
            raise TrainingError(f"Failed to compute model update: {e}")
    
    def apply_model_update(self, model: torch.nn.Module, 
                          update: Dict[str, torch.Tensor]) -> torch.nn.Module:
        """
        应用模型更新
        
        Args:
            model: 目标模型
            update: 模型更新字典
            
        Returns:
            更新后的模型
        """
        try:
            # 创建模型副本以避免原地修改
            updated_model = copy.deepcopy(model)
            
            with torch.no_grad():
                for name, param in updated_model.named_parameters():
                    if name in update:
                        param.data += update[name].to(param.device)
                        
            logger.debug(f"Applied model update with {len(update)} parameters")
            return updated_model
            
        except Exception as e:
            logger.error(f"Failed to apply model update: {e}")
            raise TrainingError(f"Failed to apply model update: {e}")
    
    def get_model_parameters(self, model: Optional[torch.nn.Module] = None) -> Dict[str, torch.Tensor]:
        """
        获取模型参数
        
        Args:
            model: 可选的模型，如果不提供则使用 BaseLearner 中的模型
            
        Returns:
            模型参数字典
        """
        try:
            if model is None:
                # 从 BaseLearner 获取模型
                model = self.learner.get_model()
                
            parameters = {}
            for name, param in model.named_parameters():
                parameters[name] = param.data.clone()
                
            logger.debug(f"Retrieved {len(parameters)} model parameters")
            return parameters
            
        except Exception as e:
            logger.error(f"Failed to get model parameters: {e}")
            raise TrainingError(f"Failed to get model parameters: {e}")
    
    def set_model_parameters(self, parameters: Dict[str, torch.Tensor], 
                           model: Optional[torch.nn.Module] = None) -> None:
        """
        设置模型参数
        
        Args:
            parameters: 参数字典
            model: 可选的模型，如果不提供则使用 BaseLearner 中的模型
        """
        try:
            if model is None:
                # 从 BaseLearner 获取模型
                model = self.learner.get_model()
                
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in parameters:
                        param.data.copy_(parameters[name].to(param.device))
                        
            logger.debug(f"Set {len(parameters)} model parameters")
            
        except Exception as e:
            logger.error(f"Failed to set model parameters: {e}")
            raise TrainingError(f"Failed to set model parameters: {e}")
    
    def compute_gradient_norms(self, model: torch.nn.Module) -> Dict[str, float]:
        """
        计算梯度范数
        
        Args:
            model: 模型
            
        Returns:
            梯度范数字典
        """
        try:
            grad_norms = {}
            total_norm = 0.0
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2).item()
                    grad_norms[name] = param_norm
                    total_norm += param_norm ** 2
                    
            grad_norms["total_norm"] = total_norm ** 0.5
            
            return grad_norms
            
        except Exception as e:
            logger.error(f"Failed to compute gradient norms: {e}")
            return {"total_norm": 0.0}
    
    def get_training_stats(self) -> Dict[str, Any]:
        """
        获取训练统计信息
        
        Returns:
            训练统计信息
        """
        stats = self.training_stats.copy()
        
        # 计算额外统计信息
        if stats["epoch_losses"]:
            stats["avg_epoch_loss"] = sum(stats["epoch_losses"]) / len(stats["epoch_losses"])
            stats["best_epoch_loss"] = min(stats["epoch_losses"])
            stats["latest_epoch_loss"] = stats["epoch_losses"][-1]
        
        if stats["gradient_norms"]:
            stats["avg_gradient_norm"] = sum(stats["gradient_norms"]) / len(stats["gradient_norms"])
            stats["max_gradient_norm"] = max(stats["gradient_norms"])
            
        stats["current_lr"] = self._get_current_lr()
        
        return stats
    
    def reset_optimizer(self) -> None:
        """重置优化器"""
        # 委托给 BaseLearner 重置
        try:
            self.learner.set_optimizer(None)
        except:
            pass  # BaseLearner 可能不允许设置 None
        logger.info("Optimizer reset")
    
    def _get_or_create_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        """获取或创建优化器，优先使用 BaseLearner 中的优化器"""
        try:
            # 尝试从 BaseLearner 获取优化器
            optimizer = self.learner.get_optimizer()
            return optimizer
        except:
            # 如果 BaseLearner 中没有优化器，创建并设置
            optimizer = self._create_optimizer(model)
            self.learner.set_optimizer(optimizer)
            return optimizer
    
    def save_training_state(self) -> Dict[str, Any]:
        """
        保存训练状态
        
        Returns:
            训练状态字典
        """
        state = {
            "training_stats": self.training_stats.copy(),
            "config": dict(self.config),
            "learning_rate": self.learning_rate,
            "learner_state": None  # 将包含 BaseLearner 的完整状态
        }
        
        # 保存 BaseLearner 状态（包括模型和优化器）
        try:
            learner_state = self.learner.save_learner_state()
            state["learner_state"] = learner_state
        except Exception as e:
            logger.warning(f"Failed to save learner state: {e}")
            
        return state
    
    def load_training_state(self, state: Dict[str, Any]) -> None:
        """
        加载训练状态
        
        Args:
            state: 训练状态字典
        """
        try:
            if "training_stats" in state:
                self.training_stats = state["training_stats"]
                
            if "learning_rate" in state:
                self.learning_rate = state["learning_rate"]
                
            # 加载 BaseLearner 状态（包括模型和优化器）
            if "learner_state" in state and state["learner_state"] is not None:
                try:
                    self.learner.load_learner_state(state["learner_state"])
                except Exception as e:
                    logger.warning(f"Failed to load learner state: {e}")
                    
            logger.info("Training state loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load training state: {e}")
            raise TrainingError(f"Failed to load training state: {e}")
    
    def _create_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        """创建优化器（不直接存储，而是返回给 BaseLearner 管理）"""
        optimizer_type = self.config.get("optimizer", "adam").lower()
        
        if optimizer_type == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.config.get("weight_decay", 0.0)
            )
        elif optimizer_type == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=self.learning_rate,
                momentum=self.config.get("momentum", 0.9),
                weight_decay=self.config.get("weight_decay", 0.0)
            )
        elif optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.config.get("weight_decay", 0.01)
            )
        else:
            raise ConfigurationError(f"Unsupported optimizer: {optimizer_type}")
            
        logger.debug(f"Created {optimizer_type} optimizer with lr={self.learning_rate}")
        return optimizer
    
    def _initialize_criterion(self) -> None:
        """初始化损失函数"""
        criterion_type = self.config.get("criterion", "cross_entropy").lower()
        
        if criterion_type == "cross_entropy":
            self.criterion = nn.CrossEntropyLoss()
        elif criterion_type == "mse":
            self.criterion = nn.MSELoss()
        elif criterion_type == "bce":
            self.criterion = nn.BCEWithLogitsLoss()
        elif criterion_type == "l1":
            self.criterion = nn.L1Loss()
        else:
            raise ConfigurationError(f"Unsupported criterion: {criterion_type}")
            
        logger.debug(f"Initialized {criterion_type} criterion")
    
    def _compute_gradient_norm(self, model: torch.nn.Module) -> float:
        """计算总梯度范数"""
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5
    
    def _get_current_lr(self) -> float:
        """获取当前学习率"""
        try:
            optimizer = self.learner.get_optimizer()
            return optimizer.param_groups[0]["lr"]
        except:
            return self.learning_rate
