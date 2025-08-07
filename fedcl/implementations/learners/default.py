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
from tqdm import tqdm

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
        # 创建context-aware logger
        super().__init__(context, config, **kwargs)
        
        # 基础学习参数
        self.learning_rate = config.get('learning_rate', 0.001)
        self.weight_decay = config.get('weight_decay', 1e-4)
        
        # 训练参数 - 支持多种配置路径
        # 优先读取 training.local_epochs，然后是 epochs_per_task
        training_config = config.get('training', {})
        self.epochs_per_task = training_config.get('local_epochs') or config.get('epochs_per_task', 5)
        
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
        
        # 进度条配置
        self._progress_position = 0  # 进度条显示位置，用于多进度条场景
        self._enable_progress_bar = config.get('enable_progress_bar', True)  # 是否启用进度条
        
        # 记录模型来源
        self.model_source = self._determine_model_source()
        
        # 添加调试信息
        self.logger.debug(f"DefaultLearner initialized (model source: {self.model_source})")
        self.logger.debug(f"Training config: epochs_per_task={self.epochs_per_task}, learning_rate={self.learning_rate}")
        self.logger.debug(f"Raw training config: {training_config}")
        self.logger.debug(f"Raw config: {dict(config) if hasattr(config, 'items') else config}")
    
    
    
    def _determine_model_source(self) -> str:
        """确定模型来源"""
        if hasattr(self, '_model_from_kwargs') and self._model_from_kwargs:
            return "direct_model"
        elif hasattr(self, '_model_from_auxiliary') and self._model_from_auxiliary:
            return "auxiliary_models"
        else:
            return "default_fallback"
    
    def set_progress_bar_position(self, position: int):
        """
        设置进度条显示位置
        
        在多客户端或多任务并行训练场景中，可以设置不同的位置来避免进度条重叠
        
        Args:
            position: 进度条位置（从0开始）
        """
        self._progress_position = position
        self.logger.debug(f"Progress bar position set to {position}")
    
    def enable_progress_bar(self, enable: bool = True):
        """
        启用或禁用进度条显示
        
        Args:
            enable: 是否启用进度条
        """
        self._enable_progress_bar = enable
        self.logger.debug(f"Progress bar {'enabled' if enable else 'disabled'}")
    
    def _create_default_model(self) -> nn.Module:
        """
        创建默认回退模型
        
        当没有外部提供模型时，创建一个简单的通用模型作为回退。
        这个模型会尝试从配置中推断合适的架构。
        
        Returns:
            默认模型实例
        """
        try:
            self.logger.debug("Creating default fallback model")
            
            # 优先使用配置中的模型类型
            model_config = self.config.get('model', {})
            if model_config and 'type' in model_config:
                model_type = model_config.get('type')
                self.logger.debug(f"Using configured model type: {model_type}")
                
                # 尝试使用ModelFactory（支持注册的模型名称）
                try:
                    from ..factory import ModelFactory
                    if model_type == "mnist_cnn":
                        # 使用ModelFactory创建注册的CNN模型
                        model = ModelFactory.create_model(model_config)
                        self.logger.debug(f"Created {model_type} model via ModelFactory")
                        return model
                except Exception as e:
                    self.logger.warning(f"Failed to create model via ModelFactory: {e}, trying direct import")
                
                # 尝试导入并创建指定的模型类型（向后兼容）
                try:
                    from ..models.mnist import SimpleMLP, SimpleCNN
                    
                    if model_type in ["SimpleMLP", "mnist_mlp"]:
                        input_size = model_config.get('input_size', 784)
                        hidden_sizes = model_config.get('hidden_sizes', [256, 128])
                        num_classes = model_config.get('num_classes', 10)
                        dropout_rate = model_config.get('dropout_rate', 0.2)
                        activation = model_config.get('activation', 'relu')
                        use_batch_norm = model_config.get('use_batch_norm', False)
                        
                        model = SimpleMLP(
                            input_size=input_size,
                            hidden_sizes=hidden_sizes,
                            num_classes=num_classes,
                            dropout_rate=dropout_rate,
                            activation=activation,
                            use_batch_norm=use_batch_norm
                        )
                        self.logger.debug(f"Created {model_type} model with config: {model_config}")
                        return model
                        
                    elif model_type in ["SimpleCNN", "mnist_cnn"]:
                        # CNN 模型配置
                        model = SimpleCNN(**{k: v for k, v in model_config.items() if k != 'type'})
                        self.logger.debug(f"Created {model_type} model with config: {model_config}")
                        return model
                        
                except Exception as e:
                    self.logger.warning(f"Failed to create configured model {model_type}: {e}, falling back to Sequential")
            
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
            
            self.logger.debug(f"Created default MLP model: input={input_size}, hidden={hidden_sizes}, output={num_classes}")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to create default model: {e}")
            
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
                self.logger.warning(f"Unknown optimizer {optimizer_type}, using Adam")
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            
            self.logger.debug(f"Initialized {optimizer_type} optimizer")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize optimizer: {e}")
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
            self.logger.warning(f"Unknown loss function {self.loss_function}, using cross_entropy")
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
            self.logger.info(f"Starting training for task {self.current_task_id} (model source: {self.model_source})")
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
            
            # 创建epoch级别的进度条
            if self._enable_progress_bar:
                epoch_progress = tqdm(
                    range(self.epochs_per_task),
                    desc=f"Task {self.current_task_id} Training",
                    unit="epoch",
                    ncols=100,
                    position=max(0, self._progress_position - 1) if self._progress_position > 0 else 0,
                    leave=True,
                    colour='blue'
                )
                epoch_iterator = epoch_progress
            else:
                epoch_iterator = range(self.epochs_per_task)
            print("epoch_iterator",epoch_iterator)
            try:
                # 训练循环
                for epoch in epoch_iterator:
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
                    
                    # 更新epoch进度条信息
                    if self._enable_progress_bar and hasattr(epoch_iterator, 'set_postfix'):
                        epoch_iterator.set_postfix({
                            'Loss': f'{epoch_loss:.4f}',
                            'Acc': f'{epoch_acc:.4f}',
                            'Best': f'{best_metric:.4f}',
                            'Patience': f'{patience_counter}/{self.early_stopping_patience}'
                        })
                    
                    # 早停
                    if patience_counter >= self.early_stopping_patience:
                        self.logger.info(f"Early stopping at epoch {epoch}")
                        if self._enable_progress_bar and hasattr(epoch_iterator, 'set_description'):
                            epoch_iterator.set_description(f"Task {self.current_task_id} Early Stopped")
                        break
                    
                    self.logger.info(f"Epoch {epoch}: Loss={epoch_loss:.4f}, Accuracy={epoch_acc:.4f}")
            
            finally:
                # 关闭epoch进度条
                if self._enable_progress_bar and hasattr(epoch_iterator, 'close'):
                    epoch_iterator.close()
            
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
                training_time=training_time,
                metadata={
                    'learner_type': 'default',
                    'model_source': self.model_source,
                    'epochs_trained': len(epoch_losses),
                    'early_stopped': patience_counter >= self.early_stopping_patience,
                    'model_state': self.get_model_state()  # 将model_state放到metadata中
                }
            )
            
            self.logger.info(f"Training completed for task {self.current_task_id}")
            self.logger.info(f"Final metrics: {final_metrics}")
            
            return task_results
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
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
        print(f"\n=== 训练 Epoch {epoch} ===")
        print(f"DataLoader batch_size: {dataloader.batch_size}")
        print(f"DataLoader dataset size (总样本数): {len(dataloader.dataset)}")
        print(f"DataLoader total batches (总批次数): {len(dataloader)}")
        print(f"验证: {len(dataloader.dataset)} 样本 ÷ {dataloader.batch_size} batch_size = {len(dataloader.dataset) / dataloader.batch_size:.1f} 批次")
        
        self.model.train()
        
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        # 根据配置决定是否使用进度条
        if self._enable_progress_bar:
            # 创建进度条，支持多进度条显示
            progress_bar = tqdm(
                enumerate(dataloader), 
                total=len(dataloader),
                desc=f"Epoch {epoch:3d} [Task {self.current_task_id}]",
                unit="batch",
                ncols=140,  # 增加进度条宽度以显示更多信息
                position=self._progress_position,  # 支持多进度条位置
                leave=True,  # 保持进度条在完成后显示
                ascii=False,  # 使用Unicode字符
                colour='green'  # 设置进度条颜色
            )
            data_iterator = progress_bar
        else:
            # 不使用进度条时的普通迭代器
            data_iterator = enumerate(dataloader)
        try:
            for batch_idx, (data, target) in data_iterator:
                # 移动数据到设备
                data = data.to(self.device)
                target = target.to(self.device)
                
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
                
                # 计算当前准确率和平均损失
                current_acc = correct_predictions / total_samples
                current_avg_loss = total_loss / (batch_idx + 1)
                
                # 更新进度条描述（仅在使用进度条时）
                if self._enable_progress_bar and hasattr(data_iterator, 'set_postfix'):
                    data_iterator.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Avg Loss': f'{current_avg_loss:.4f}',
                        'Acc': f'{current_acc:.4f}'
                    })
                
                # 定期日志
                log_interval = 500 if self._enable_progress_bar else 100
                if batch_idx % log_interval == 0 and batch_idx > 0:
                    self.logger.debug(f"Epoch {epoch}, Batch {batch_idx}: Loss={loss.item():.6f}, Acc={current_acc:.4f}")
        
        finally:
            # 确保进度条正确关闭（仅在使用时）
            if self._enable_progress_bar and hasattr(data_iterator, 'close'):
                data_iterator.close()
        
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
                    
                    # 自动处理数据形状 - SimpleMLP等模型自己会处理展平，跳过手动展平
                    # if len(data.shape) > 2 and self.model_source == "default_fallback":
                    #     data = data.view(data.size(0), -1)
                    
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

    def train_epoch(self, dataloader, epoch: int) -> Dict[str, float]:
        """
        训练一个epoch
        
        Args:
            dataloader: 数据加载器
            epoch: 当前epoch编号
            
        Returns:
            Dict[str, float]: 训练指标（loss, accuracy等）
        """
        self.logger.info(f"🔥 [DefaultLearner训练] 开始train_epoch - epoch {epoch}")
        
        if self.model is None:
            self.logger.warning("🔥 [DefaultLearner训练] No model available for training")
            return {"loss": 0.0, "accuracy": 0.0}
        
        if self.optimizer is None:
            self.logger.warning("🔥 [DefaultLearner训练] No optimizer available for training")
            return {"loss": 0.0, "accuracy": 0.0}
        
        try:
            # 设置模型为训练模式
            self.model.train()
            # 获取损失函数
            if hasattr(self, 'criterion') and self.criterion is not None:
                loss_fn = self.criterion
            else:
                # 创建默认损失函数
                loss_fn = nn.CrossEntropyLoss()
            
            self.logger.info(f"🔥 [DefaultLearner训练] 开始训练epoch {epoch}，数据集大小: {len(dataloader) if hasattr(dataloader, '__len__') else 'unknown'}")
            # 调用内部的_train_epoch方法
            epoch_loss, epoch_acc = self._train_epoch(dataloader, loss_fn, epoch)
            # 更新当前epoch
            self.current_epoch = epoch
            
            # 记录训练历史
            epoch_metrics = {
                "loss": float(epoch_loss),
                "accuracy": float(epoch_acc),
                "epoch": epoch
            }
            self.training_history.append(epoch_metrics)
            self.logger.info(f"✅ [DefaultLearner训练] Epoch {epoch} 完成 - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
            return epoch_metrics
            
        except Exception as e:
            self.logger.error(f"❌ [DefaultLearner训练] 训练epoch {epoch} 失败: {e}")
            return {"loss": float('inf'), "accuracy": 0.0, "epoch": epoch}


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


def create_learner_with_progress_config(context: ExecutionContext, config: DictConfig,
                                       progress_position: int = 0, 
                                       enable_progress: bool = True,
                                       **kwargs) -> DefaultLearner:
    """
    创建带有进度条配置的学习器
    
    在多客户端或多任务并行训练场景中特别有用
    
    Args:
        context: 执行上下文
        config: 配置
        progress_position: 进度条显示位置（用于多进度条）
        enable_progress: 是否启用进度条
        **kwargs: 额外参数
        
    Returns:
        配置好进度条的学习器实例
    """
    # 在配置中设置进度条选项
    config = config.copy() if hasattr(config, 'copy') else dict(config)
    config['enable_progress_bar'] = enable_progress
    
    learner = DefaultLearner(context, config, **kwargs)
    learner.set_progress_bar_position(progress_position)
    
    return learner


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