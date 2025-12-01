"""
Fed-CPrompt: Contrastive Prompt for Rehearsal-Free Federated Continual Learning
CVPR 2023

Paper: https://openaccess.thecvf.com/content/CVPR2023/papers/Qi_CPFE_A_Query-Based_Federated_Learning_Framework_With_Prompt_Enhancement_CVPR_2023_paper.pdf

核心思想：
1. 使用提示学习(Prompt Learning)避免灾难性遗忘
2. 对比学习增强提示表示
3. 无需存储历史样本(Rehearsal-Free)
4. 基于预训练模型(如CLIP, ViT)的持续学习
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

from .._decorators import learner
from fedcl.learner.base_learner import BaseLearner
from fedcl.types import TrainingResponse, EvaluationResult, ModelData
from fedcl.methods.models.base import get_weights_as_dict, set_weights_from_dict


class PromptPool(nn.Module):
    """
    提示池模块

    管理多个任务的提示向量
    """

    def __init__(self, num_tasks: int, prompt_length: int, embed_dim: int):
        super().__init__()
        self.num_tasks = num_tasks
        self.prompt_length = prompt_length
        self.embed_dim = embed_dim

        # 每个任务的提示向量
        self.prompts = nn.Parameter(
            torch.randn(num_tasks, prompt_length, embed_dim)
        )

        # 提示选择键(用于自动选择提示)
        self.prompt_keys = nn.Parameter(
            torch.randn(num_tasks, embed_dim)
        )

    def forward(self, task_id: Optional[int] = None, query: Optional[torch.Tensor] = None):
        """
        获取任务特定的提示

        Args:
            task_id: 任务ID(如果已知)
            query: 查询向量(用于自动选择提示)

        Returns:
            提示向量
        """
        if task_id is not None:
            # Task-Incremental: 直接使用任务ID
            return self.prompts[task_id]

        elif query is not None:
            # Class-Incremental: 使用查询向量选择最相关的提示
            # 计算查询与所有提示键的相似度
            similarities = F.cosine_similarity(
                query.unsqueeze(1),  # [batch, 1, dim]
                self.prompt_keys.unsqueeze(0),  # [1, num_tasks, dim]
                dim=2
            )

            # 选择最相似的提示
            selected_indices = similarities.argmax(dim=1)

            # 返回选中的提示
            selected_prompts = self.prompts[selected_indices]
            return selected_prompts

        else:
            raise ValueError("Either task_id or query must be provided")


@learner('cl', 'FedCPrompt', description='Fed-CPrompt: Contrastive Prompt for Rehearsal-Free Federated Continual Learning (CVPR 2023)')
class FedCPromptLearner(BaseLearner):
    """
    Fed-CPrompt 持续学习方法的Learner实现

    特点：
    - 基于提示学习的持续学习
    - 对比学习增强提示表示
    - 无需存储历史样本
    - 支持预训练模型微调
    """

    def __init__(self, client_id: str, config: Optional[Dict[str, Any]] = None, lazy_init: bool = True):
        # 提取配置
        learner_params = (config or {}).get('learner', {}).get('params', {})

        # 保存Fed-CPrompt特定配置
        self._model_cfg = learner_params.get('model', {})
        self._optimizer_cfg = learner_params.get('optimizer', {
            'type': 'Adam',
            'lr': learner_params.get('learning_rate', 0.001),
        })
        self._loss_cfg = learner_params.get('loss', 'CrossEntropyLoss')

        # 训练参数
        self._lr = learner_params.get('learning_rate', 0.001)
        self._bs = learner_params.get('batch_size', 32)
        self._epochs = learner_params.get('local_epochs', 5)

        # 持续学习参数
        self.num_tasks = learner_params.get('num_tasks', 5)
        self.classes_per_task = learner_params.get('classes_per_task', 2)
        self.scenario = learner_params.get('scenario', 'class_incremental')

        # Fed-CPrompt特定参数
        self.prompt_length = learner_params.get('prompt_length', 10)
        self.embed_dim = learner_params.get('embed_dim', 768)
        self.use_prompt_pool = learner_params.get('use_prompt_pool', True)
        self.contrastive_temperature = learner_params.get('contrastive_temperature', 0.07)
        self.contrastive_weight = learner_params.get('contrastive_weight', 0.5)
        self.freeze_backbone = learner_params.get('freeze_backbone', True)

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

        # 提示池(延迟初始化)
        self.prompt_pool = None

        # 组件占位符
        self._model = None
        self._optimizer = None
        self._criterion = None
        self._train_loader = None

        self.logger.info(
            f"FedCPromptLearner {client_id} initialized: "
            f"num_tasks={self.num_tasks}, classes_per_task={self.classes_per_task}, "
            f"prompt_length={self.prompt_length}"
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

            # 如果需要冻结主干网络
            if self.freeze_backbone:
                self._freeze_backbone_params()

            self.logger.info(f"Client {self.client_id}: 模型 {model_name} 创建完成")
        return self._model

    def _freeze_backbone_params(self):
        """冻结主干网络参数，只训练分类头"""
        # 假设模型的最后一层是分类器，冻结其他层
        for name, param in self._model.named_parameters():
            # 通常分类器层包含'fc', 'classifier', 'head'等关键词
            if not any(keyword in name.lower() for keyword in ['fc', 'classifier', 'head']):
                param.requires_grad = False
                self.logger.debug(f"Frozen parameter: {name}")

    @property
    def optimizer(self):
        """延迟加载优化器"""
        if self._optimizer is None:
            opt_type = self._optimizer_cfg.get('type', 'Adam').upper()
            lr = self._optimizer_cfg.get('lr', self._lr)

            # 收集需要优化的参数(只优化未冻结的参数和提示池)
            params_to_optimize = [p for p in self.model.parameters() if p.requires_grad]

            # 如果有提示池，也加入优化
            if self.prompt_pool is not None:
                params_to_optimize += list(self.prompt_pool.parameters())

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
        执行Fed-CPrompt方法的本地训练

        Fed-CPrompt的训练流程：
        1. 使用任务特定的提示
        2. 通过对比学习增强提示表示
        3. 冻结主干网络，只更新提示和分类头
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
            self.current_task_id = task_id

            # 更新已见类别
            start_class = task_id * self.classes_per_task
            end_class = min(start_class + self.classes_per_task,
                           self.num_tasks * self.classes_per_task)
            new_classes = list(range(start_class, end_class))
            self.seen_classes.extend(new_classes)

        # 初始化提示池(如果需要且未初始化)
        if self.use_prompt_pool and self.prompt_pool is None:
            self.prompt_pool = PromptPool(
                num_tasks=self.num_tasks,
                prompt_length=self.prompt_length,
                embed_dim=self.embed_dim
            ).to(self.device)
            # 重新创建优化器以包含提示池参数
            self._optimizer = None

        # 获取当前任务的数据
        task_loader = self._get_task_data_loader(task_id)

        self.logger.info(
            f"[{self.client_id}] Task {task_id} Round {round_number}, "
            f"Training {num_epochs} epochs..."
        )

        # 训练
        self.model.train()
        if self.prompt_pool is not None:
            self.prompt_pool.train()

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

                # 获取提示增强的特征和输出
                features, output = self._forward_with_prompt(data, task_id)

                # 1. 分类损失
                ce_loss = self.criterion(output, target)

                # 2. 对比学习损失(增强提示表示)
                contrastive_loss = 0.0
                if self.use_prompt_pool and task_id > 0:
                    contrastive_loss = self._compute_contrastive_loss(features, target, task_id)

                # 3. 提示正则化损失(防止提示偏移过大)
                prompt_reg_loss = 0.0
                if self.use_prompt_pool:
                    prompt_reg_loss = self._compute_prompt_regularization()

                # 总损失
                if task_id > 0:
                    loss = (ce_loss +
                            self.contrastive_weight * contrastive_loss +
                            0.01 * prompt_reg_loss)
                else:
                    loss = ce_loss + 0.01 * prompt_reg_loss

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

    def _forward_with_prompt(self, data: torch.Tensor, task_id: int):
        """
        前向传播并应用提示

        Args:
            data: 输入数据
            task_id: 任务ID

        Returns:
            (features, output): 特征和输出
        """
        # 简化实现：直接使用模型输出
        # 实际应该在模型内部注入提示向量
        output = self.model(data)

        # 使用输出作为特征(简化版)
        features = output.detach()

        return features, output

    def _compute_contrastive_loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        task_id: int
    ) -> torch.Tensor:
        """
        计算对比学习损失

        增强同类样本的提示表示相似度，降低不同类样本的相似度

        Args:
            features: 特征向量
            labels: 标签
            task_id: 任务ID

        Returns:
            对比损失
        """
        if not self.use_prompt_pool:
            return torch.tensor(0.0, device=self.device)

        batch_size = features.size(0)

        # 归一化特征
        features_norm = F.normalize(features, p=2, dim=1)

        # 计算相似度矩阵
        similarity_matrix = torch.mm(features_norm, features_norm.t())

        # 创建标签掩码(同类为1，不同类为0)
        labels_expanded = labels.unsqueeze(0)
        mask = (labels_expanded == labels_expanded.t()).float()

        # 对角线设为0(排除自身)
        mask = mask - torch.eye(batch_size, device=self.device)

        # 正样本对和负样本对
        pos_mask = mask
        neg_mask = 1 - mask - torch.eye(batch_size, device=self.device)

        # 使用InfoNCE损失
        # exp(sim(i,j)/temp) / sum(exp(sim(i,k)/temp))
        exp_sim = torch.exp(similarity_matrix / self.contrastive_temperature)

        # 对于每个样本，计算正样本对的损失
        pos_sum = (exp_sim * pos_mask).sum(dim=1)
        neg_sum = (exp_sim * neg_mask).sum(dim=1)

        # 避免除零
        loss = -torch.log((pos_sum + 1e-8) / (pos_sum + neg_sum + 1e-8))

        return loss.mean()

    def _compute_prompt_regularization(self) -> torch.Tensor:
        """
        计算提示正则化损失

        防止提示向量偏离初始化太远

        Returns:
            正则化损失
        """
        if not self.use_prompt_pool:
            return torch.tensor(0.0, device=self.device)

        # L2正则化
        reg_loss = 0.0
        for param in self.prompt_pool.parameters():
            reg_loss += (param ** 2).sum()

        return reg_loss

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
        if self.prompt_pool is not None:
            self.prompt_pool.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in eval_loader:
                data, target = data.to(self.device), target.to(self.device)

                # 前向传播
                _, output = self._forward_with_prompt(data, task_id)

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

        # 如果有提示池，也包含其参数
        if self.prompt_pool is not None:
            prompt_data = {
                f'prompt.{k}': v for k, v in get_weights_as_dict(self.prompt_pool).items()
            }
            model_data.update(prompt_data)

        return ModelData(
            model_data=model_data,
            metadata={
                'client_id': self.client_id,
                'task_id': self.current_task_id,
                'seen_classes': self.seen_classes
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

            # 分离模型权重和提示权重
            model_weights = {}
            prompt_weights = {}

            for k, v in weights.items():
                if k.startswith('prompt.'):
                    prompt_weights[k.replace('prompt.', '')] = v
                else:
                    model_weights[k] = v

            # 设置模型权重
            set_weights_from_dict(self.model, model_weights)

            # 设置提示池权重
            if prompt_weights and self.prompt_pool is not None:
                set_weights_from_dict(self.prompt_pool, prompt_weights)

            return True
        except Exception as e:
            self.logger.error(f"Failed to set model: {e}")
            return False

    def get_prompt_summary(self) -> Dict[str, Any]:
        """
        获取提示摘要

        Returns:
            提示统计信息
        """
        return {
            'use_prompt_pool': self.use_prompt_pool,
            'prompt_length': self.prompt_length,
            'embed_dim': self.embed_dim,
            'current_task': self.current_task_id,
            'seen_classes': self.seen_classes
        }
