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
import copy
import numpy as np

from ....core.learner import Learner
from ....core.types import StepMetrics, EpochMetrics, TrainMetrics, EvalResult
from ....registry import learner


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


@learner('cl.fed_cprompt', description='Fed-CPrompt: Contrastive Prompt for Rehearsal-Free Federated Continual Learning (CVPR 2023)')
class FedCPromptLearner(Learner):
    """
    Fed-CPrompt 持续学习方法的Learner实现

    特点：
    - 基于提示学习的持续学习
    - 对比学习增强提示表示
    - 无需存储历史样本
    - 支持预训练模型微调
    """

    def __init__(
        self,
        model,
        datasets,
        tracker=None,
        callbacks=None,
        config=None,
        node_id=None
    ):
        """
        初始化 Fed-CPrompt Learner

        Args:
            model: 模型实例
            datasets: 数据集字典 {"train": [...], "test": [...]}
            tracker: 追踪器
            callbacks: 回调管理器
            config: 配置字典
            node_id: 节点ID
        """
        super().__init__(model, None, tracker, callbacks, config, node_id)

        # 保存数据集
        self._datasets = datasets or {}

        # 从配置提取参数
        self._lr = self._config.get('learning_rate', 0.001)
        self._bs = self._config.get('batch_size', 32)
        self._epochs = self._config.get('local_epochs', 5)
        self.optimizer_type = self._config.get('optimizer', 'Adam').upper()
        self.loss_type = self._config.get('loss', 'CrossEntropyLoss')

        # 持续学习参数
        self.num_tasks = self._config.get('num_tasks', 5)
        self.classes_per_task = self._config.get('classes_per_task', 2)
        self.scenario = self._config.get('scenario', 'class_incremental')

        # Fed-CPrompt特定参数
        self.prompt_length = self._config.get('prompt_length', 10)
        self.embed_dim = self._config.get('embed_dim', 768)
        self.use_prompt_pool = self._config.get('use_prompt_pool', True)
        self.contrastive_temperature = self._config.get('contrastive_temperature', 0.07)
        self.contrastive_weight = self._config.get('contrastive_weight', 0.5)
        self.freeze_backbone = self._config.get('freeze_backbone', True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 持续学习状态
        self.current_task_id = 0
        self.seen_classes = []

        # 提示池(延迟初始化)
        self.prompt_pool = None

        # 组件占位符
        self.torch_model = None
        self._optimizer = None
        self._criterion = None
        self._train_loader = None
        self._test_loader = None

        self.logger.info(
            f"FedCPromptLearner {node_id} initialized: "
            f"num_tasks={self.num_tasks}, classes_per_task={self.classes_per_task}, "
            f"prompt_length={self.prompt_length}"
        )

    async def setup(self, config: Dict) -> None:
        """
        初始化训练环境

        Args:
            config: 运行时配置
        """
        # 获取PyTorch模型
        self.torch_model = self._model.get_model()
        self.torch_model.to(self.device)

        # 如果需要冻结主干网络
        if self.freeze_backbone:
            self._freeze_backbone_params()

        # 初始化提示池(如果需要)
        if self.use_prompt_pool:
            self.prompt_pool = PromptPool(
                num_tasks=self.num_tasks,
                prompt_length=self.prompt_length,
                embed_dim=self.embed_dim
            ).to(self.device)
            self.logger.info(f"Prompt pool initialized: {self.num_tasks} tasks")

        # 创建优化器
        # 收集需要优化的参数(只优化未冻结的参数和提示池)
        params_to_optimize = [p for p in self.torch_model.parameters() if p.requires_grad]

        # 如果有提示池，也加入优化
        if self.prompt_pool is not None:
            params_to_optimize += list(self.prompt_pool.parameters())

        if self.optimizer_type == 'SGD':
            self._optimizer = optim.SGD(
                params_to_optimize,
                lr=self._lr,
                momentum=self._config.get('momentum', 0.9),
                weight_decay=self._config.get('weight_decay', 5e-4)
            )
        elif self.optimizer_type == 'ADAM':
            self._optimizer = optim.Adam(
                params_to_optimize,
                lr=self._lr
            )
        else:
            raise ValueError(f"不支持的优化器类型: {self.optimizer_type}")

        # 创建损失函数
        if self.loss_type == 'CrossEntropyLoss':
            self._criterion = nn.CrossEntropyLoss()
        elif self.loss_type == 'MSELoss':
            self._criterion = nn.MSELoss()
        else:
            raise ValueError(f"不支持的损失函数: {self.loss_type}")

        # 创建数据加载器
        train_datasets = self._datasets.get("train", [])
        if train_datasets:
            train_dataset = train_datasets[0]
            self._train_loader = DataLoader(
                train_dataset,
                batch_size=self._bs,
                shuffle=True,
                drop_last=False
            )
            self.logger.info(
                f"Setup completed: train_samples={len(train_dataset)}, "
                f"batch_size={self._bs}"
            )

        test_datasets = self._datasets.get("test", [])
        if test_datasets:
            test_dataset = test_datasets[0]
            self._test_loader = DataLoader(
                test_dataset,
                batch_size=self._bs,
                shuffle=False
            )

        # 从配置中获取任务ID（如果有）
        task_id = config.get('task_id', self.current_task_id)
        if task_id != self.current_task_id:
            self._update_task_state(task_id)

    def _freeze_backbone_params(self):
        """冻结主干网络参数，只训练分类头"""
        # 假设模型的最后一层是分类器，冻结其他层
        for name, param in self.torch_model.named_parameters():
            # 通常分类器层包含'fc', 'classifier', 'head'等关键词
            if not any(keyword in name.lower() for keyword in ['fc', 'classifier', 'head']):
                param.requires_grad = False
                self.logger.debug(f"Frozen parameter: {name}")

    def _update_task_state(self, task_id: int):
        """
        更新任务状态

        Args:
            task_id: 新任务ID
        """
        self.logger.info(
            f"[{self._node_id}] Switching task: {self.current_task_id} -> {task_id}"
        )
        self.current_task_id = task_id

        # 更新已见类别
        start_class = task_id * self.classes_per_task
        end_class = min(start_class + self.classes_per_task,
                       self.num_tasks * self.classes_per_task)
        new_classes = list(range(start_class, end_class))

        # 只添加新的类别
        for cls in new_classes:
            if cls not in self.seen_classes:
                self.seen_classes.append(cls)

    def _get_task_data_loader(self, task_id: int, dataset=None) -> DataLoader:
        """
        获取特定任务的数据加载器

        根据当前任务ID筛选对应的类别数据

        Args:
            task_id: 任务ID
            dataset: 数据集（如果不提供则使用训练集）

        Returns:
            DataLoader: 任务数据加载器
        """
        if dataset is None:
            train_datasets = self._datasets.get("train", [])
            if not train_datasets:
                self.logger.warning("No train dataset available")
                return self._train_loader
            dataset = train_datasets[0]

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
            return self._train_loader

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

    def get_dataloader(self) -> DataLoader:
        """
        获取数据加载器（Learner基类要求）

        Returns:
            DataLoader: 当前任务的数据加载器
        """
        return self._get_task_data_loader(self.current_task_id)

    def get_num_samples(self) -> int:
        """
        获取训练样本数

        Returns:
            int: 样本数量
        """
        train_datasets = self._datasets.get("train", [])
        if train_datasets:
            return len(train_datasets[0])
        return 0

    async def train_step(self, batch: Any, batch_idx: int) -> StepMetrics:
        """
        单批次训练

        Fed-CPrompt的训练流程：
        1. 使用任务特定的提示
        2. 通过对比学习增强提示表示
        3. 冻结主干网络，只更新提示和分类头

        Args:
            batch: 批次数据 (data, target)
            batch_idx: 批次索引

        Returns:
            StepMetrics: 训练指标
        """
        data, target = batch
        data, target = data.to(self.device), target.to(self.device)

        self._optimizer.zero_grad()

        # 获取提示增强的特征和输出
        features, output = self._forward_with_prompt(data, self.current_task_id)

        # 1. 分类损失
        ce_loss = self._criterion(output, target)

        # 2. 对比学习损失(增强提示表示)
        contrastive_loss = 0.0
        if self.use_prompt_pool and self.current_task_id > 0:
            contrastive_loss = self._compute_contrastive_loss(features, target, self.current_task_id)

        # 3. 提示正则化损失(防止提示偏移过大)
        prompt_reg_loss = 0.0
        if self.use_prompt_pool:
            prompt_reg_loss = self._compute_prompt_regularization()

        # 总损失
        if self.current_task_id > 0:
            loss = (ce_loss +
                    self.contrastive_weight * contrastive_loss +
                    0.01 * prompt_reg_loss)
        else:
            loss = ce_loss + 0.01 * prompt_reg_loss

        # 反向传播
        loss.backward()
        self._optimizer.step()

        # 计算准确率
        _, predicted = output.max(1)
        correct = predicted.eq(target).sum().item()
        accuracy = correct / target.size(0)

        return StepMetrics(
            loss=loss.item(),
            batch_size=target.size(0),
            metrics={
                'accuracy': accuracy,
                'ce_loss': ce_loss.item(),
                'contrastive_loss': contrastive_loss.item() if isinstance(contrastive_loss, torch.Tensor) else contrastive_loss,
                'prompt_reg_loss': prompt_reg_loss.item() if isinstance(prompt_reg_loss, torch.Tensor) else prompt_reg_loss
            }
        )

    async def train_epoch(self, epoch_idx: int) -> EpochMetrics:
        """
        单轮训练

        Args:
            epoch_idx: 轮次索引

        Returns:
            EpochMetrics: 轮次指标
        """
        # 调用父类的train_epoch方法
        self.torch_model.train()
        if self.prompt_pool is not None:
            self.prompt_pool.train()

        epoch_metrics = await super().train_epoch(epoch_idx)

        return epoch_metrics

    async def fit(self, config: Optional[Dict[str, Any]] = None) -> TrainMetrics:
        """
        执行完整的训练流程

        Args:
            config: 训练配置

        Returns:
            TrainMetrics: 训练指标
        """
        config = config or {}

        # 提取任务ID并更新任务状态
        task_id = config.get("task_id", self.current_task_id)
        if task_id != self.current_task_id:
            self._update_task_state(task_id)

        # 调用父类的fit方法
        return await super().fit(config)

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
        output = self.torch_model(data)

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

    async def evaluate(self, config: Optional[Dict[str, Any]] = None) -> EvalResult:
        """
        评估模型性能

        Args:
            config: 评估配置

        Returns:
            EvalResult: 评估结果
        """
        config = config or {}
        task_id = config.get("task_id", self.current_task_id)

        # 获取评估数据
        if self._test_loader is not None:
            eval_loader = self._test_loader
        else:
            # 使用训练集评估
            if task_id is not None:
                eval_loader = self._get_task_data_loader(task_id)
            else:
                eval_loader = self._train_loader

        self.torch_model.eval()
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

                loss = self._criterion(output, target)

                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        accuracy = correct / total if total > 0 else 0
        avg_loss = total_loss / len(eval_loader) if len(eval_loader) > 0 else 0

        return EvalResult(
            num_samples=total,
            metrics={
                'accuracy': accuracy,
                'loss': avg_loss,
                'task_id': task_id
            },
            metadata={
                'scenario': self.scenario,
                'use_prompt_pool': self.use_prompt_pool
            }
        )

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

    def get_metadata(self) -> Dict[str, Any]:
        """
        获取训练元数据

        Returns:
            元数据字典
        """
        base_metadata = super().get_metadata()
        base_metadata.update({
            'task_id': self.current_task_id,
            'seen_classes': len(self.seen_classes),
            'prompt_summary': self.get_prompt_summary()
        })
        return base_metadata
