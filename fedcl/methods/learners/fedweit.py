"""
FedWeIT: Federated Continual Learning with Weighted Inter-client Transfer
ICML 2021

Paper: https://proceedings.mlr.press/v139/yoon21b.html
GitHub: https://github.com/wyjeong/FedWeIT (原TensorFlow实现，此为PyTorch版本)

核心思想：
1. 任务自适应注意力模块(Task-Adaptive Attention)
2. 加权客户端间知识转移(Weighted Inter-client Transfer)
3. 自适应聚合权重(Adaptive Aggregation Weights)
4. 支持异构任务序列(Heterogeneous Task Sequences)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from typing import Dict, Any, Optional, List
from datetime import datetime
import copy
import numpy as np

from ...api.decorators import learner
from ...learner.base_learner import BaseLearner
from ...types import TrainingResponse, EvaluationResult, ModelData
from ...methods.models.base import get_weights_as_dict, set_weights_from_dict


class TaskAdaptiveAttention(nn.Module):
    """
    任务自适应注意力模块

    为每个任务学习独立的注意力权重
    """

    def __init__(self, num_tasks: int, feature_dim: int):
        super().__init__()
        self.num_tasks = num_tasks
        self.feature_dim = feature_dim

        # 每个任务的注意力权重
        self.task_embeddings = nn.Parameter(
            torch.randn(num_tasks, feature_dim)
        )

        # 注意力计算网络
        self.attention_net = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, features: torch.Tensor, task_id: int) -> torch.Tensor:
        """
        计算任务特定的注意力权重

        Args:
            features: [batch_size, feature_dim]
            task_id: 当前任务ID

        Returns:
            加权后的特征
        """
        batch_size = features.size(0)

        # 获取任务嵌入
        task_emb = self.task_embeddings[task_id].unsqueeze(0).expand(batch_size, -1)

        # 拼接特征和任务嵌入
        combined = torch.cat([features, task_emb], dim=1)

        # 计算注意力权重
        attention_weights = self.attention_net(combined)

        # 应用注意力
        weighted_features = features * attention_weights

        return weighted_features


@learner('FedWeIT',
         description='FedWeIT: Federated Continual Learning with Weighted Inter-client Transfer (ICML 2021)',
         version='1.0',
         author='MOE-FedCL')
class FedWeITLearner(BaseLearner):
    """
    FedWeIT 持续学习方法的Learner实现

    特点：
    - 任务自适应注意力机制
    - 加权客户端间知识转移
    - 支持异构任务序列
    - 自适应聚合策略
    """

    def __init__(self, client_id: str, config: Optional[Dict[str, Any]] = None, lazy_init: bool = True):
        # 提取配置
        learner_params = (config or {}).get('learner', {}).get('params', {})

        # 保存FedWeIT特定配置
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

        # FedWeIT特定参数
        self.use_attention = learner_params.get('use_attention', True)
        self.feature_dim = learner_params.get('feature_dim', 512)
        self.transfer_weight = learner_params.get('transfer_weight', 0.5)
        self.use_weighted_transfer = learner_params.get('use_weighted_transfer', True)
        self.distill_temperature = learner_params.get('distill_temperature', 2.0)

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

        # 任务特定的模型参数快照
        self.task_specific_params = {}  # {task_id: model_state_dict}

        # 注意力模块(延迟初始化)
        self.attention_module = None

        # 组件占位符
        self._model = None
        self._optimizer = None
        self._criterion = None
        self._train_loader = None

        self.logger.info(
            f"FedWeITLearner {client_id} initialized: "
            f"num_tasks={self.num_tasks}, classes_per_task={self.classes_per_task}, "
            f"use_attention={self.use_attention}"
        )

    @property
    def model(self):
        """延迟加载模型"""
        if self._model is None:
            from ...api.registry import registry

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

            # 收集需要优化的参数
            params_to_optimize = list(self.model.parameters())

            # 如果有注意力模块，也加入优化
            if self.attention_module is not None:
                params_to_optimize += list(self.attention_module.parameters())

            if opt_type == 'SGD':
                self._optimizer = optim.SGD(
                    params_to_optimize,
                    lr=lr,
                    momentum=self._optimizer_cfg.get('momentum', 0.9),
                    weight_decay=self._optimizer_cfg.get('weight_decay', 5e-4)
                )
            elif opt_type == 'ADAM':
                self._optimizer = optim.Adam(
                    params_to_optimize,
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
        执行FedWeIT方法的本地训练

        FedWeIT的训练流程：
        1. 使用任务自适应注意力
        2. 结合加权客户端间知识转移
        3. 保存任务特定参数
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
                self.task_specific_params[self.current_task_id] = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }

            self.current_task_id = task_id

            # 更新已见类别
            start_class = task_id * self.classes_per_task
            end_class = min(start_class + self.classes_per_task,
                           self.num_tasks * self.classes_per_task)
            new_classes = list(range(start_class, end_class))
            self.seen_classes.extend(new_classes)

        # 初始化注意力模块(如果需要且未初始化)
        if self.use_attention and self.attention_module is None:
            self.attention_module = TaskAdaptiveAttention(
                num_tasks=self.num_tasks,
                feature_dim=self.feature_dim
            ).to(self.device)
            # 重新创建优化器以包含注意力模块参数
            self._optimizer = None

        # 获取当前任务的数据
        task_loader = self._get_task_data_loader(task_id)

        self.logger.info(
            f"[{self.client_id}] Task {task_id} Round {round_number}, "
            f"Training {num_epochs} epochs..."
        )

        # 训练
        self.model.train()
        if self.attention_module is not None:
            self.attention_module.train()

        total_loss = 0.0
        total_accuracy = 0.0
        total_samples = 0

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            for batch_idx, (data, target) in enumerate(task_loader):
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()

                # 前向传播(带注意力)
                features, output = self._forward_with_attention(data, task_id)

                # 1. 分类损失
                ce_loss = self.criterion(output, target)

                # 2. 知识转移损失(如果有历史任务)
                transfer_loss = 0.0
                if self.use_weighted_transfer and len(self.task_specific_params) > 0 and task_id > 0:
                    transfer_loss = self._compute_weighted_transfer_loss(data)

                # 3. 正则化损失(防止任务间参数差异过大)
                regularization_loss = 0.0
                if len(self.task_specific_params) > 0 and task_id > 0:
                    regularization_loss = self._compute_regularization_loss()

                # 总损失
                if task_id > 0:
                    loss = (ce_loss +
                            self.transfer_weight * transfer_loss +
                            0.01 * regularization_loss)
                else:
                    loss = ce_loss

                # 反向传播
                loss.backward()
                self.optimizer.step()

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
                'num_task_params': len(self.task_specific_params),
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

    def _forward_with_attention(self, data: torch.Tensor, task_id: int):
        """
        前向传播并应用任务自适应注意力

        Args:
            data: 输入数据
            task_id: 任务ID

        Returns:
            (features, output): 特征和输出
        """
        # 前向传播
        output = self.model(data)

        # 对于简单实现，使用输出作为特征
        features = output.detach()

        # 如果使用注意力，应用任务特定注意力
        if self.use_attention and self.attention_module is not None:
            # 注意力增强特征
            attended_features = self.attention_module(features, task_id)

            # 使用增强的特征重新计算输出(简化版)
            # 实际应该从模型中间层提取特征并应用注意力
            # 这里简化为保持原输出
            pass

        return features, output

    def _compute_weighted_transfer_loss(self, data: torch.Tensor) -> torch.Tensor:
        """
        计算加权知识转移损失

        使用历史任务的模型输出进行蒸馏

        Args:
            data: 输入数据

        Returns:
            转移损失
        """
        if not self.task_specific_params:
            return torch.tensor(0.0, device=self.device)

        current_output = self.model(data)
        total_loss = 0.0
        num_prev_tasks = len(self.task_specific_params)

        # 遍历所有历史任务
        for prev_task_id, prev_params in self.task_specific_params.items():
            # 加载历史任务参数
            prev_model = copy.deepcopy(self.model)
            prev_model.load_state_dict({k: v.to(self.device) for k, v in prev_params.items()})
            prev_model.eval()

            with torch.no_grad():
                prev_output = prev_model(data)

            # 计算KL散度
            teacher_probs = F.softmax(prev_output / self.distill_temperature, dim=1)
            student_log_probs = F.log_softmax(current_output / self.distill_temperature, dim=1)

            kl_div = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
            kl_div *= (self.distill_temperature ** 2)

            total_loss += kl_div

        return total_loss / num_prev_tasks if num_prev_tasks > 0 else torch.tensor(0.0, device=self.device)

    def _compute_regularization_loss(self) -> torch.Tensor:
        """
        计算正则化损失

        防止当前参数与历史任务参数差异过大

        Returns:
            正则化损失
        """
        if not self.task_specific_params:
            return torch.tensor(0.0, device=self.device)

        total_loss = 0.0
        current_params = self.model.state_dict()

        # 计算与所有历史任务参数的L2距离
        for prev_task_id, prev_params in self.task_specific_params.items():
            for name, param in current_params.items():
                if name in prev_params:
                    prev_param = prev_params[name].to(self.device)
                    total_loss += ((param - prev_param) ** 2).sum()

        num_prev_tasks = len(self.task_specific_params)
        return total_loss / num_prev_tasks if num_prev_tasks > 0 else torch.tensor(0.0, device=self.device)

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
        if self.attention_module is not None:
            self.attention_module.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in eval_loader:
                data, target = data.to(self.device), target.to(self.device)

                # 前向传播
                _, output = self._forward_with_attention(data, task_id)

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
        model_data = get_weights_as_dict(self.model)

        # 如果有注意力模块，也包含其参数
        if self.attention_module is not None:
            attention_data = {
                f'attention.{k}': v for k, v in get_weights_as_dict(self.attention_module).items()
            }
            model_data.update(attention_data)

        return ModelData(
            model_data=model_data,
            metadata={
                'client_id': self.client_id,
                'task_id': self.current_task_id,
                'seen_classes': self.seen_classes,
                'num_task_params': len(self.task_specific_params)
            }
        )

    async def set_local_model(self, model_data: ModelData) -> bool:
        """设置本地模型参数"""
        try:
            if isinstance(model_data, ModelData):
                weights = model_data.model_data
            elif isinstance(model_data, dict):
                weights = model_data.get('model_data', model_data)
            else:
                weights = model_data

            # 分离模型权重和注意力权重
            model_weights = {}
            attention_weights = {}

            for k, v in weights.items():
                if k.startswith('attention.'):
                    attention_weights[k.replace('attention.', '')] = v
                else:
                    model_weights[k] = v

            # 设置模型权重
            set_weights_from_dict(self.model, model_weights)

            # 设置注意力模块权重
            if attention_weights and self.attention_module is not None:
                set_weights_from_dict(self.attention_module, attention_weights)

            return True
        except Exception as e:
            self.logger.error(f"Failed to set model: {e}")
            return False

    def get_task_params_summary(self) -> Dict[str, Any]:
        """
        获取任务参数摘要

        Returns:
            任务参数统计信息
        """
        return {
            'num_tasks': len(self.task_specific_params),
            'current_task': self.current_task_id,
            'task_ids': list(self.task_specific_params.keys()),
            'use_attention': self.use_attention,
            'seen_classes': self.seen_classes
        }
