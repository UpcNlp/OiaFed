# fedcl/learners/default_learner.py
"""
默认通用学习器

提供一个完全通用的学习器实现，不依赖任何特定模型。
模型完全通过外部配置传递（auxiliary_models或model_factory）。
"""

import time
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from loguru import logger

from ...core.base_learner import BaseLearner
from ...core.execution_context import ExecutionContext
from ...data.results import TaskResults
from ...exceptions import LearnerError
from ...registry.component_registry import registry


@registry.learner("default", 
                  version="1.0.0",
                  author="FedCL Team", 
                  description="Default generic learner that works with any externally provided model",
                  supported_features=["classification", "federated_learning", "continual_learning", "model_agnostic"])
class DefaultLearner(BaseLearner):
    """
    默认通用学习器
    
    完全通用的学习器实现，不依赖任何特定模型。
    模型完全通过外部配置传递：
    1. 通过auxiliary_models参数传入预创建的模型
    2. 通过model_factory配置传入模型创建函数
    3. 如果都没有，使用简单的默认模型
    """
    
    def __init__(self, context: ExecutionContext, config: DictConfig, **kwargs):
        """
        初始化默认学习器
        
        Args:
            context: 执行上下文
            config: 学习器配置
            **kwargs: 额外参数，支持auxiliary_models传入预创建的模型
        """
        super().__init__(context, config, **kwargs)
        
        # 基础学习参数
        self.learning_rate = config.get('learning_rate', 0.001)
        self.weight_decay = config.get('weight_decay', 1e-4)
        
        # 训练参数
        self.epochs_per_task = config.get('epochs_per_task', 5)
        self.early_stopping_patience = config.get('early_stopping_patience', 10)
        self.min_improvement = config.get('min_improvement', 0.001)
        self.loss_function = config.get('loss_function', 'cross_entropy')
        
        # 初始化优化器
        if self.model is not None:
            self._initialize_optimizer()
        
        # 训练状态
        self.current_epoch = 0
        self.best_metric = 0.0
        self.training_history = []
        
        # 记录模型来源
        self.model_source = self._determine_model_source()
        
        logger.debug(f"DefaultLearner initialized (model source: {self.model_source})")
    
    def _determine_model_source(self) -> str:
        """确定模型来源"""
        if hasattr(self, '_model_from_kwargs') and self._model_from_kwargs:
            return "direct_model"
        elif hasattr(self, '_model_from_auxiliary') and self._model_from_auxiliary:
            return "auxiliary_models"
        else:
            return "default_fallback"
    
    def _create_default_model(self) -> nn.Module:
        """
        创建默认回退模型
        
        当没有外部提供模型时，创建一个简单的通用模型作为回退。
        这个模型会尝试从配置中推断合适的架构。
        
        Returns:
            默认模型实例
        """
        try:
            logger.debug("Creating default fallback model")
            
            # 从配置中获取模型参数提示
            default_config = self.config.get('default_model_config', {})
            
            # 尝试推断模型类型
            input_size = default_config.get('input_size', 784)
            num_classes = default_config.get('num_classes', 10)
            hidden_sizes = default_config.get('hidden_sizes', [256, 128])
            dropout_rate = default_config.get('dropout_rate', 0.2)
            
            # 创建简单的MLP作为默认模型
            layers = []
            prev_size = input_size
            
            for hidden_size in hidden_sizes:
                layers.append(nn.Linear(prev_size, hidden_size))
                layers.append(nn.ReLU())
                if dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))
                prev_size = hidden_size
            
            layers.append(nn.Linear(prev_size, num_classes))
            
            model = nn.Sequential(*layers)
            
            logger.debug(f"Created default MLP model: input={input_size}, hidden={hidden_sizes}, output={num_classes}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to create default model: {e}")
            
            # 最简单的回退模型
            return nn.Sequential(
                nn.Flatten(),
                nn.Linear(784, 128),
                nn.ReLU(),
                nn.Linear(128, 10)
            )
    
    def _initialize_optimizer(self):
        """初始化优化器"""
        try:
            optimizer_config = self.config.get('optimizer', {})
            optimizer_type = optimizer_config.get('type', 'Adam').lower()
            
            if optimizer_type == 'adam':
                self.optimizer = optim.Adam(
                    self.model.parameters(),
                    lr=self.learning_rate,
                    weight_decay=self.weight_decay,
                    betas=optimizer_config.get('betas', (0.9, 0.999))
                )
            elif optimizer_type == 'sgd':
                self.optimizer = optim.SGD(
                    self.model.parameters(),
                    lr=self.learning_rate,
                    momentum=optimizer_config.get('momentum', 0.9),
                    weight_decay=self.weight_decay
                )
            elif optimizer_type == 'adamw':
                self.optimizer = optim.AdamW(
                    self.model.parameters(),
                    lr=self.learning_rate,
                    weight_decay=self.weight_decay,
                    betas=optimizer_config.get('betas', (0.9, 0.999))
                )
            else:
                logger.warning(f"Unknown optimizer {optimizer_type}, using Adam")
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            
            logger.debug(f"Initialized {optimizer_type} optimizer")
            
        except Exception as e:
            logger.error(f"Failed to initialize optimizer: {e}")
            raise LearnerError(f"Optimizer initialization failed: {e}")
    
    def _get_loss_function(self):
        """获取损失函数"""
        loss_functions = {
            'cross_entropy': F.cross_entropy,
            'nll_loss': F.nll_loss,
            'mse': F.mse_loss,
            'l1_loss': F.l1_loss,
            'binary_cross_entropy': F.binary_cross_entropy,
            'binary_cross_entropy_with_logits': F.binary_cross_entropy_with_logits
        }
        
        loss_fn = loss_functions.get(self.loss_function)
        if loss_fn is None:
            logger.warning(f"Unknown loss function {self.loss_function}, using cross_entropy")
            return F.cross_entropy
        
        return loss_fn
    
    def train_task(self, task_data: DataLoader) -> TaskResults:
        """
        训练任务
        
        Args:
            task_data: 任务训练数据加载器
            
        Returns:
            TaskResults: 训练结果
        """
        try:
            logger.info(f"Starting training for task {self.current_task_id} (model source: {self.model_source})")
            start_time = time.time()
            
            if self.model is None:
                raise LearnerError("Model not initialized")
            
            if self.optimizer is None:
                self._initialize_optimizer()
            
            self.model.train()
            loss_fn = self._get_loss_function()
            
            # 训练指标
            epoch_losses = []
            epoch_metrics = []
            best_metric = 0.0
            patience_counter = 0
            
            # 训练循环
            for epoch in range(self.epochs_per_task):
                self.current_epoch = epoch
                
                # 执行前钩子
                self.before_epoch_hook(epoch)
                
                epoch_loss, epoch_acc = self._train_epoch(task_data, loss_fn, epoch)
                
                epoch_losses.append(epoch_loss)
                epoch_metrics.append(epoch_acc)
                
                # 早停检查
                if epoch_acc > best_metric + self.min_improvement:
                    best_metric = epoch_acc
                    patience_counter = 0
                    self.best_metric = best_metric
                else:
                    patience_counter += 1
                
                # 执行后钩子
                metrics = {
                    'loss': epoch_loss,
                    'accuracy': epoch_acc,
                    'epoch': epoch
                }
                self.after_epoch_hook(epoch, metrics)
                
                # 早停
                if patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
                
                logger.info(f"Epoch {epoch}: Loss={epoch_loss:.4f}, Accuracy={epoch_acc:.4f}")
            
            training_time = time.time() - start_time
            
            # 构建训练结果
            final_metrics = {
                'final_loss': epoch_losses[-1] if epoch_losses else 0.0,
                'final_accuracy': epoch_metrics[-1] if epoch_metrics else 0.0,
                'best_accuracy': self.best_metric,
                'training_time': training_time,
                'epochs_trained': len(epoch_losses)
            }
            
            # 更新训练历史
            self.training_history.append({
                'task_id': self.current_task_id,
                'metrics': final_metrics,
                'epoch_losses': epoch_losses,
                'epoch_accuracies': epoch_metrics
            })
            
            # 创建任务结果
            task_results = TaskResults(
                task_id=self.current_task_id,
                metrics=final_metrics,
                model_state=self.get_model_state(),
                training_time=training_time,
                metadata={
                    'learner_type': 'default',
                    'model_source': self.model_source,
                    'epochs_trained': len(epoch_losses),
                    'early_已停止': patience_counter >= self.early_stopping_patience
                }
            )
            
            logger.info(f"Training completed for task {self.current_task_id}")
            logger.info(f"Final metrics: {final_metrics}")
            
            return task_results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise LearnerError(f"Training failed: {e}")
    
    def _train_epoch(self, dataloader: DataLoader, loss_fn, epoch: int) -> tuple:
        """
        训练单个epoch
        
        Args:
            dataloader: 数据加载器
            loss_fn: 损失函数
            epoch: 当前epoch
            
        Returns:
            tuple: (平均损失, 平均准确率)
        """
        self.model.train()
        
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            # 移动数据到设备
            data = data.to(self.device)
            target = target.to(self.device)
            
            # 自动处理数据形状
            if len(data.shape) > 2 and self.model_source == "default_fallback":
                # 如果是默认模型且输入是多维的，自动展平
                data = data.view(data.size(0), -1)
            
            # 前向传播
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = loss_fn(output, target)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct_predictions += pred.eq(target.view_as(pred)).sum().item()
            total_samples += data.size(0)
            
            # 定期日志
            if batch_idx % 100 == 0:
                logger.debug(f"Epoch {epoch}, Batch {batch_idx}: Loss={loss.item():.6f}")
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_samples
        
        return avg_loss, accuracy
    
    def evaluate_task(self, task_data: DataLoader) -> Dict[str, float]:
        """
        评估任务
        
        Args:
            task_data: 任务评估数据加载器
            
        Returns:
            Dict[str, float]: 评估指标字典
        """
        try:
            logger.info(f"Starting evaluation (model source: {self.model_source})")
            
            if self.model is None:
                raise LearnerError("Model not initialized")
            
            self.model.eval()
            loss_fn = self._get_loss_function()
            
            total_loss = 0.0
            correct_predictions = 0
            total_samples = 0
            
            with torch.no_grad():
                for data, target in task_data:
                    # 移动数据到设备
                    data = data.to(self.device)
                    target = target.to(self.device)
                    
                    # 自动处理数据形状
                    if len(data.shape) > 2 and self.model_source == "default_fallback":
                        data = data.view(data.size(0), -1)
                    
                    # 前向传播
                    output = self.model(data)
                    loss = loss_fn(output, target, reduction='sum')
                    
                    # 统计
                    total_loss += loss.item()
                    pred = output.argmax(dim=1, keepdim=True)
                    correct_predictions += pred.eq(target.view_as(pred)).sum().item()
                    total_samples += data.size(0)
            
            # 计算指标
            avg_loss = total_loss / total_samples
            accuracy = correct_predictions / total_samples
            
            evaluation_metrics = {
                'loss': avg_loss,
                'accuracy': accuracy,
                'correct_predictions': correct_predictions,
                'total_samples': total_samples
            }
            
            logger.info(f"Evaluation completed: Accuracy={accuracy:.4f}, Loss={avg_loss:.4f}")
            
            return evaluation_metrics
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise LearnerError(f"Evaluation failed: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            Dict[str, Any]: 模型相关信息
        """
        base_info = super().get_model_info()
        
        # 添加默认学习器特定信息
        default_info = {
            'model_source': self.model_source,
            'loss_function': self.loss_function,
            'current_epoch': self.current_epoch,
            'best_metric': self.best_metric,
            'training_history_length': len(self.training_history),
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'early_stopping_patience': self.early_stopping_patience
        }
        
        # 合并信息
        base_info.update(default_info)
        return base_info
    
    def update_model_from_server(self, global_parameters: Dict[str, torch.Tensor]):
        """
        从服务端更新模型参数
        
        Args:
            global_parameters: 全局模型参数
        """
        try:
            if self.model is None:
                raise LearnerError("Model not initialized")
            
            # 加载参数
            self.model.load_state_dict(global_parameters, strict=False)
            
            logger.info(f"Model updated from server parameters (source: {self.model_source})")
            
        except Exception as e:
            logger.error(f"Failed to update model from server: {e}")
            raise LearnerError(f"Model update failed: {e}")
    
    def reset_for_new_task(self, task_id: int) -> None:
        """
        为新任务重置学习器
        
        Args:
            task_id: 新任务的ID
        """
        super().reset_for_new_task(task_id)
        
        # 重置训练状态
        self.current_epoch = 0
        
        # 根据配置决定是否重置最佳指标
        reset_best_metric = self.config.get('reset_best_metric_per_task', False)
        if reset_best_metric:
            self.best_metric = 0.0
        
        logger.info(f"Default learner reset for new task: {task_id} (model source: {self.model_source})")
    
    def save_checkpoint(self, checkpoint_path: str):
        """
        保存检查点
        
        Args:
            checkpoint_path: 检查点路径
        """
        try:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
                'current_epoch': self.current_epoch,
                'best_metric': self.best_metric,
                'training_history': self.training_history,
                'config': self.config,
                'current_task_id': self.current_task_id,
                'model_source': self.model_source
            }
            
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Checkpoint saved to {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise LearnerError(f"Checkpoint save failed: {e}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        加载检查点
        
        Args:
            checkpoint_path: 检查点路径
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # 加载模型状态
            if self.model:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # 加载优化器状态
            if self.optimizer and 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # 加载训练状态
            self.current_epoch = checkpoint.get('current_epoch', 0)
            self.best_metric = checkpoint.get('best_metric', 0.0)
            self.training_history = checkpoint.get('training_history', [])
            self.current_task_id = checkpoint.get('current_task_id')
            self.model_source = checkpoint.get('model_source', 'unknown')
            
            logger.info(f"Checkpoint loaded from {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise LearnerError(f"Checkpoint load failed: {e}")
    
    def get_custom_parameter_selection(self) -> Dict[str, Any]:
        """
        自定义参数选择策略（重写父类方法）
        
        根据模型来源提供不同的参数选择策略
        
        Returns:
            Dict[str, Any]: 自定义选择的参数
        """
        if self.model_source == "auxiliary_models":
            # 如果模型来自auxiliary_models，可能需要特殊处理
            logger.debug("Using auxiliary model parameter selection")
            return self.model.state_dict()
        elif self.model_source == "direct_model":
            # 如果模型直接传入，使用全部参数
            logger.debug("Using direct model parameter selection")
            return self.model.state_dict()
        else:
            # 默认回退模型，使用全部参数
            logger.debug("Using default parameter selection")
            return self.model.state_dict()


# ===== 便利函数 =====

def create_default_learner(context: ExecutionContext, config: DictConfig, 
                          model: nn.Module = None, **kwargs) -> DefaultLearner:
    """
    创建默认学习器实例
    
    Args:
        context: 执行上下文
        config: 配置
        model: 预创建的模型（可选）
        **kwargs: 额外参数
        
    Returns:
        默认学习器实例
    """
    if model is not None:
        kwargs['model'] = model
    
    return DefaultLearner(context, config, **kwargs)


def create_learner_with_auxiliary_model(context: ExecutionContext, config: DictConfig,
                                       model_name: str, model_instance: nn.Module) -> DefaultLearner:
    """
    使用辅助模型创建学习器
    
    Args:
        context: 执行上下文
        config: 配置
        model_name: 模型名称
        model_instance: 模型实例
        
    Returns:
        学习器实例
    """
    auxiliary_models = {model_name: model_instance}
    config['model_name'] = model_name
    
    return DefaultLearner(context, config, auxiliary_models=auxiliary_models)


# ===== 示例使用 =====

if __name__ == "__main__":
    # 示例：验证learner可以正常创建和使用
    from omegaconf import OmegaConf
    from ...core.execution_context import ExecutionContext
    
    # 创建测试配置
    config = OmegaConf.create({
        'learning_rate': 0.001,
        'epochs_per_task': 2,
        'loss_function': 'cross_entropy',
        'optimizer': {'type': 'Adam'},
        'default_model_config': {
            'input_size': 784,
            'num_classes': 10,
            'hidden_sizes': [128, 64]
        }
    })
    
    # 创建执行上下文
    context = ExecutionContext(
        config=OmegaConf.create({}),
        experiment_id="default_learner_test"
    )
    
    # 创建学习器
    learner = DefaultLearner(context, config)
    
    print(f"Created learner: {learner}")
    print(f"Model info: {learner.get_model_info()}")
    print("Default learner test completed successfully!")