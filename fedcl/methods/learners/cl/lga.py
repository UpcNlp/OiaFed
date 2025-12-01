"""
LGA: Layerwise Gradient Accumulation for Federated Continual Learning
TPAMI 2023

Paper: Layerwise Gradient Accumulation for Continual Learning in Federated Settings

核心思想：
1. 逐层梯度累积(Layerwise Gradient Accumulation)
2. 选择性参数更新(Selective Parameter Update)
3. 基于重要性的参数保护(Importance-based Parameter Protection)
4. 高效的内存使用和通信开销
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import copy
import numpy as np

from .._decorators import learner
from fedcl.learner.base_learner import BaseLearner
from fedcl.types import TrainingResponse, EvaluationResult, ModelData
from fedcl.methods.models.base import get_weights_as_dict, set_weights_from_dict


@learner('cl', 'LGA', description='LGA: Layerwise Gradient Accumulation for Federated Continual Learning (TPAMI 2023)')
class LGALearner(BaseLearner):
    """
    LGA (Layerwise Gradient Accumulation) 持续学习方法的Learner实现

    特点：
    - 逐层梯度累积策略
    - 基于重要性的参数保护
    - 选择性参数更新
    - 高效的通信和存储
    """

    def __init__(self, client_id: str, config: Optional[Dict[str, Any]] = None, lazy_init: bool = True):
        # 提取配置
        learner_params = (config or {}).get('learner', {}).get('params', {})

        # 保存LGA特定配置
        self._model_cfg = learner_params.get('model', {})
        self._optimizer_cfg = learner_params.get('optimizer', {
            'type': 'SGD',
            'lr': learner_params.get('learning_rate', 0.01),
            'momentum': 0.9
        })
        self._loss_cfg = learner_params.get('loss', 'CrossEntropyLoss')

        # 训练参数
        self._lr = learner_params.get('learning_rate', 0.01)
        self._bs = learner_params.get('batch_size', 32)
        self._epochs = learner_params.get('local_epochs', 5)

        # 持续学习参数
        self.num_tasks = learner_params.get('num_tasks', 5)
        self.classes_per_task = learner_params.get('classes_per_task', 2)
        self.scenario = learner_params.get('scenario', 'class_incremental')

        # LGA特定参数
        self.accumulation_steps = learner_params.get('accumulation_steps', 4)
        self.use_layerwise_accumulation = learner_params.get('use_layerwise_accumulation', True)
        self.importance_method = learner_params.get('importance_method', 'fisher')
        self.protection_threshold = learner_params.get('protection_threshold', 0.1)
        self.selective_update = learner_params.get('selective_update', True)
        self.update_ratio = learner_params.get('update_ratio', 0.8)

        # 清理配置
        clean_config = config.copy() if config else {}
        if 'learner' in clean_config and 'params' in clean_config['learner']:
            clean_params = clean_config['learner']['params'].copy()
            clean_params.pop('model', None)
            clean_params.pop('optimizer', None)
            clean_params.pop('loss', None)
            clean_config['learner'] = clean_config['learner'].copy()
            clean_config['learner']['params'] = clean_params

        super().__init__(client_id, clean_config, lazy_init)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 持续学习状态
        self.current_task_id = 0
        self.seen_classes = []

        # 存储每层参数的重要性分数
        self.parameter_importance = {}  # {layer_name: importance_tensor}

        # 存储每层的梯度累积
        self.accumulated_gradients = {}  # {layer_name: accumulated_gradient}

        # 旧任务的参数快照(用于约束)
        self.old_task_params = {}  # {task_id: state_dict}

        # 组件占位符
        self._model = None
        self._optimizer = None
        self._criterion = None
        self._train_loader = None

        self.logger.info(
            f"LGALearner {client_id} initialized: "
            f"num_tasks={self.num_tasks}, classes_per_task={self.classes_per_task}, "
            f"accumulation_steps={self.accumulation_steps}, importance_method={self.importance_method}"
        )

    @property
    def model(self):
        """延迟加载模型"""
        if self._model is None:
            from ....api.registry import registry

            # 优先使用learner.params.model
            if self._model_cfg and self._model_cfg.get('name'):
                model_name = self._model_cfg['name']
                model_params = self._model_cfg.get('params', {})
            # 否则使用顶层model配置
            elif 'model' in self.config:
                model_config = self.config['model']
                model_name = model_config.get('name')
                model_params = model_config.get('params', {})
            else:
                raise ValueError("未找到模型配置")

            model_class = registry.get_model(model_name)
            self._model = model_class(**model_params).to(self.device)
            self.logger.info(f"Client {self.client_id}: 模型 {model_name} 创建完成")
        return self._model

    @property
    def optimizer(self):
        """延迟加载优化器"""
        if self._optimizer is None:
            opt_type = self._optimizer_cfg.get('type', 'SGD').upper()
            lr = self._optimizer_cfg.get('lr', self._lr)

            if opt_type == 'SGD':
                self._optimizer = optim.SGD(
                    self.model.parameters(),
                    lr=lr,
                    momentum=self._optimizer_cfg.get('momentum', 0.9),
                    weight_decay=self._optimizer_cfg.get('weight_decay', 5e-4)
                )
            elif opt_type == 'ADAM':
                self._optimizer = optim.Adam(
                    self.model.parameters(),
                    lr=lr
                )
            else:
                raise ValueError(f"不支持的优化器类型: {opt_type}")

            self.logger.debug(f"Client {self.client_id}: 优化器 {opt_type} 创建完成")
        return self._optimizer

    @property
    def criterion(self):
        """延迟加载损失函数"""
        if self._criterion is None:
            if isinstance(self._loss_cfg, str):
                if self._loss_cfg == 'CrossEntropyLoss':
                    self._criterion = nn.CrossEntropyLoss()
                elif self._loss_cfg == 'MSELoss':
                    self._criterion = nn.MSELoss()
                else:
                    raise ValueError(f"不支持的损失函数: {self._loss_cfg}")
            self.logger.debug(f"Client {self.client_id}: 损失函数创建完成")
        return self._criterion

    @property
    def train_loader(self):
        """延迟加载数据加载器"""
        if self._train_loader is None:
            dataset = self.dataset
            self._train_loader = DataLoader(
                dataset,
                batch_size=self._bs,
                shuffle=True,
                drop_last=False
            )
            self.logger.info(
                f"Client {self.client_id}: 数据加载器创建完成 "
                f"(samples={len(dataset)}, batch_size={self._bs})"
            )
        return self._train_loader

    def _get_task_data_loader(self, task_id: int) -> DataLoader:
        """
        获取特定任务的数据加载器

        根据当前任务ID筛选对应的类别数据
        """
        dataset = self.dataset

        # 计算当前任务的类别范围
        start_class = task_id * self.classes_per_task
        end_class = min(start_class + self.classes_per_task,
                       self.num_tasks * self.classes_per_task)
        task_classes = list(range(start_class, end_class))

        # 筛选当前任务的样本
        indices = []
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            if isinstance(label, torch.Tensor):
                label = label.item()
            if label in task_classes:
                indices.append(idx)

        if not indices:
            self.logger.warning(
                f"Task {task_id} has no samples for classes {task_classes}"
            )
            return self.train_loader

        task_dataset = Subset(dataset, indices)

        self.logger.info(
            f"Task {task_id} data loader: {len(indices)} samples, "
            f"classes {task_classes}"
        )

        return DataLoader(
            task_dataset,
            batch_size=self._bs,
            shuffle=True,
            drop_last=False
        )

    async def train(self, params: Dict[str, Any]) -> TrainingResponse:
        """
        执行LGA方法的本地训练

        LGA的训练流程：
        1. 逐层累积梯度
        2. 计算参数重要性
        3. 选择性更新参数(保护重要参数)
        4. 更新重要性分数
        """
        start_time = datetime.now()

        # 提取参数
        num_epochs = params.get("num_epochs", self._epochs)
        round_number = params.get("round_number", 1)
        task_id = params.get("task_id", self.current_task_id)

        # 更新任务状态
        if task_id != self.current_task_id:
            self.logger.info(
                f"[{self.client_id}] Switching task: {self.current_task_id} -> {task_id}"
            )

            # 保存当前任务的参数快照
            if self._model is not None:
                self.old_task_params[self.current_task_id] = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }

            self.current_task_id = task_id

            # 更新已见类别
            start_class = task_id * self.classes_per_task
            end_class = min(start_class + self.classes_per_task,
                           self.num_tasks * self.classes_per_task)
            new_classes = list(range(start_class, end_class))
            self.seen_classes.extend(new_classes)

        # 获取当前任务的数据
        task_loader = self._get_task_data_loader(task_id)

        self.logger.info(
            f"[{self.client_id}] Task {task_id} Round {round_number}, "
            f"Training {num_epochs} epochs..."
        )

        # 训练
        self.model.train()

        total_loss = 0.0
        total_accuracy = 0.0
        total_samples = 0

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            # 重置梯度累积
            if self.use_layerwise_accumulation:
                self._reset_accumulated_gradients()

            for batch_idx, (data, target) in enumerate(task_loader):
                data, target = data.to(self.device), target.to(self.device)

                # 前向传播
                output = self.model(data)

                # 1. 当前任务的分类损失
                ce_loss = self.criterion(output, target)

                # 2. 参数保护损失(防止遗忘旧任务)
                protection_loss = 0.0
                if task_id > 0 and self.old_task_params:
                    protection_loss = self._compute_protection_loss()

                # 总损失
                if task_id > 0:
                    loss = ce_loss + 0.1 * protection_loss
                else:
                    loss = ce_loss

                # 反向传播(不立即更新参数)
                loss.backward()

                # 累积梯度
                if self.use_layerwise_accumulation:
                    self._accumulate_gradients()

                # 每accumulation_steps步或最后一个batch更新一次
                if (batch_idx + 1) % self.accumulation_steps == 0 or \
                   (batch_idx + 1) == len(task_loader):

                    # 选择性参数更新
                    if self.selective_update and task_id > 0 and self.parameter_importance:
                        self._selective_parameter_update()
                    else:
                        self.optimizer.step()

                    self.optimizer.zero_grad()

                    # 重置累积梯度
                    if self.use_layerwise_accumulation:
                        self._reset_accumulated_gradients()

                # 统计
                epoch_loss += loss.item()
                _, predicted = output.max(1)
                epoch_total += target.size(0)
                epoch_correct += predicted.eq(target).sum().item()

            # Epoch统计
            epoch_accuracy = epoch_correct / epoch_total if epoch_total > 0 else 0
            avg_epoch_loss = epoch_loss / len(task_loader) if len(task_loader) > 0 else 0

            self.logger.info(
                f"  [{self.client_id}] Task {task_id} Epoch {epoch+1}/{num_epochs}: "
                f"Loss={avg_epoch_loss:.4f}, Acc={epoch_accuracy:.4f}"
            )

            total_loss += epoch_loss
            total_accuracy += epoch_accuracy
            total_samples = epoch_total

        # 训练完成后，更新参数重要性
        self._update_parameter_importance(task_loader)

        # 计算平均值
        avg_loss = total_loss / (num_epochs * len(task_loader)) if num_epochs > 0 else 0
        avg_accuracy = total_accuracy / num_epochs if num_epochs > 0 else 0

        # 更新统计
        self.training_count += 1
        self.last_training_time = datetime.now()

        # 提取模型权重
        model_weights = get_weights_as_dict(self.model)

        # 返回结果（使用TrainingResponse）
        response = TrainingResponse(
            request_id="",
            client_id=self.client_id,
            success=True,
            result={
                'loss': avg_loss,
                'accuracy': avg_accuracy,
                'task_id': task_id,
                'seen_classes': len(self.seen_classes),
                'num_protected_params': self._count_protected_parameters(),
                'model_weights': model_weights,
                'epochs_completed': num_epochs,
                'samples_used': total_samples
            },
            execution_time=(datetime.now() - start_time).total_seconds()
        )

        # 触发回调
        for callback in self._callbacks.get('after_train', []):
            callback(response)

        return response

    def _reset_accumulated_gradients(self):
        """重置梯度累积"""
        self.accumulated_gradients = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.accumulated_gradients[name] = torch.zeros_like(param.data)

    def _accumulate_gradients(self):
        """累积当前batch的梯度"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if name not in self.accumulated_gradients:
                    self.accumulated_gradients[name] = torch.zeros_like(param.data)
                self.accumulated_gradients[name] += param.grad.data

    def _compute_protection_loss(self) -> torch.Tensor:
        """
        计算参数保护损失

        防止重要参数偏离旧任务的值太远

        Returns:
            保护损失
        """
        if not self.old_task_params or not self.parameter_importance:
            return torch.tensor(0.0, device=self.device)

        total_loss = 0.0
        current_params = self.model.state_dict()

        # 取最近一个旧任务的参数
        latest_old_task = max(self.old_task_params.keys())
        old_params = self.old_task_params[latest_old_task]

        # 计算重要性加权的L2损失
        for name, param in current_params.items():
            if name in old_params and name in self.parameter_importance:
                old_param = old_params[name].to(self.device)
                importance = self.parameter_importance[name].to(self.device)

                # 重要性加权的L2距离
                param_loss = importance * ((param - old_param) ** 2)
                total_loss += param_loss.sum()

        return total_loss

    def _selective_parameter_update(self):
        """
        选择性参数更新

        只更新重要性低的参数，保护重要性高的参数
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if name in self.parameter_importance:
                    importance = self.parameter_importance[name].to(self.device)

                    # 创建更新掩码：重要性低于阈值的参数允许更新
                    update_mask = (importance < self.protection_threshold).float()

                    # 应用掩码到梯度
                    param.grad.data *= update_mask

        # 执行参数更新
        self.optimizer.step()

    def _update_parameter_importance(self, train_loader: DataLoader):
        """
        更新参数重要性

        使用Fisher信息矩阵或梯度幅度估计参数重要性

        Args:
            train_loader: 训练数据加载器
        """
        self.model.eval()

        if self.importance_method == 'fisher':
            self._compute_fisher_importance(train_loader)
        elif self.importance_method == 'gradient_magnitude':
            self._compute_gradient_importance(train_loader)
        else:
            self.logger.warning(f"Unknown importance method: {self.importance_method}")

        self.model.train()

    def _compute_fisher_importance(self, train_loader: DataLoader):
        """
        使用Fisher信息矩阵计算参数重要性

        Args:
            train_loader: 训练数据加载器
        """
        fisher = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param.data)

        self.model.zero_grad()

        # 采样一部分数据计算Fisher信息
        num_samples = min(100, len(train_loader.dataset))
        sample_count = 0

        for data, target in train_loader:
            if sample_count >= num_samples:
                break

            data, target = data.to(self.device), target.to(self.device)

            output = self.model(data)
            loss = self.criterion(output, target)

            loss.backward()

            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.data ** 2

            self.model.zero_grad()
            sample_count += data.size(0)

        # 归一化Fisher信息
        for name in fisher:
            fisher[name] /= num_samples

        # 更新或合并重要性分数
        for name, importance in fisher.items():
            if name in self.parameter_importance:
                # 与之前的重要性合并(累积)
                self.parameter_importance[name] = 0.9 * self.parameter_importance[name] + 0.1 * importance
            else:
                self.parameter_importance[name] = importance

        self.logger.info(f"Updated Fisher importance for {len(fisher)} parameters")

    def _compute_gradient_importance(self, train_loader: DataLoader):
        """
        使用梯度幅度计算参数重要性

        Args:
            train_loader: 训练数据加载器
        """
        grad_magnitude = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                grad_magnitude[name] = torch.zeros_like(param.data)

        self.model.zero_grad()

        # 采样一部分数据计算梯度幅度
        num_samples = min(100, len(train_loader.dataset))
        sample_count = 0

        for data, target in train_loader:
            if sample_count >= num_samples:
                break

            data, target = data.to(self.device), target.to(self.device)

            output = self.model(data)
            loss = self.criterion(output, target)

            loss.backward()

            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad_magnitude[name] += param.grad.data.abs()

            self.model.zero_grad()
            sample_count += data.size(0)

        # 归一化梯度幅度
        for name in grad_magnitude:
            grad_magnitude[name] /= num_samples

        # 更新重要性分数
        self.parameter_importance = grad_magnitude

        self.logger.info(f"Updated gradient importance for {len(grad_magnitude)} parameters")

    def _count_protected_parameters(self) -> int:
        """
        统计被保护的参数数量

        Returns:
            被保护的参数总数
        """
        if not self.parameter_importance:
            return 0

        protected_count = 0
        for name, importance in self.parameter_importance.items():
            protected_mask = (importance >= self.protection_threshold)
            protected_count += protected_mask.sum().item()

        return protected_count

    async def evaluate(self, params: Dict[str, Any]) -> EvaluationResult:
        """评估模型性能"""
        start_time = datetime.now()

        task_id = params.get("task_id", self.current_task_id)

        # 获取评估数据
        if task_id is not None:
            eval_loader = self._get_task_data_loader(task_id)
        else:
            eval_loader = self.train_loader

        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in eval_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        accuracy = correct / total if total > 0 else 0
        avg_loss = total_loss / len(eval_loader) if len(eval_loader) > 0 else 0

        self.evaluation_count += 1
        self.last_evaluation_time = datetime.now()

        result = EvaluationResult(
            success=True,
            result={
                'accuracy': accuracy,
                'loss': avg_loss,
                'task_id': task_id
            },
            metadata={
                'duration_seconds': (datetime.now() - start_time).total_seconds(),
                'samples_evaluated': total
            }
        )

        # 触发回调
        for callback in self._callbacks.get('after_evaluate', []):
            callback(result)

        return result

    async def get_local_model(self) -> ModelData:
        """获取本地模型参数"""
        return ModelData(
            model_data=get_weights_as_dict(self.model),
            metadata={
                'client_id': self.client_id,
                'task_id': self.current_task_id,
                'seen_classes': self.seen_classes,
                'num_protected_params': self._count_protected_parameters()
            }
        )

    async def set_local_model(self, model_data: ModelData) -> bool:
        """设置本地模型参数"""
        try:
            # ModelData是dict[str, Any]的类型别名，直接检查dict
            if isinstance(model_data, dict):
                # 支持多种格式:
                # 1. {'model_data': {...weights...}} - 新格式
                # 2. {'parameters': {'weights': {...}}} - 老格式（from fl learners）
                # 3. 直接权重字典
                if 'model_data' in model_data:
                    weights = model_data['model_data']
                elif 'parameters' in model_data and isinstance(model_data['parameters'], dict):
                    weights = model_data['parameters'].get('weights', model_data['parameters'])
                else:
                    weights = model_data
            else:
                weights = model_data

            set_weights_from_dict(self.model, weights)
            return True
        except Exception as e:
            self.logger.error(f"Failed to set model: {e}")
            return False

    def get_importance_summary(self) -> Dict[str, Any]:
        """
        获取参数重要性摘要

        Returns:
            重要性统计信息
        """
        if not self.parameter_importance:
            return {
                'num_parameters': 0,
                'num_protected': 0,
                'protection_ratio': 0.0
            }

        total_params = sum(imp.numel() for imp in self.parameter_importance.values())
        protected_params = self._count_protected_parameters()

        return {
            'num_parameters': total_params,
            'num_protected': protected_params,
            'protection_ratio': protected_params / total_params if total_params > 0 else 0.0,
            'importance_method': self.importance_method,
            'protection_threshold': self.protection_threshold,
            'current_task': self.current_task_id,
            'seen_classes': self.seen_classes
        }
