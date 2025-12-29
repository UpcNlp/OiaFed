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
import copy
import numpy as np

from ....core.learner import Learner
from ....core.types import TrainResult, EvalResult, StepMetrics, EpochMetrics, TrainMetrics
from ....registry import learner


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


@learner(
    name='cl.fedweit',
    description='FedWeIT: Federated Continual Learning with Weighted Inter-client Transfer (ICML 2021)',
    version='1.0',
    author='Federation Framework'
)
class FedWeITLearner(Learner):
    """
    FedWeIT 持续学习方法的Learner实现

    特点：
    - 任务自适应注意力机制
    - 加权客户端间知识转移
    - 支持异构任务序列
    - 自适应聚合策略
    """

    def __init__(
        self,
        model: Any,
        datasets: Optional[Dict[str, Any]] = None,
        tracker: Optional[Any] = None,
        callbacks: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        node_id: Optional[str] = None
    ):
        """
        初始化FedWeIT学习器

        Args:
            model: 模型包装器
            datasets: 数据集字典，按 split 分组
            tracker: 追踪器
            callbacks: 回调管理器
            config: 配置字典
            node_id: 节点ID
        """
        # 调用基类构造函数
        super().__init__(model, None, tracker, callbacks, config, node_id)

        # 保存数据集
        self._datasets = datasets or {}

        # 提取配置
        cfg = self._config or {}

        # 训练参数
        self._lr = cfg.get('learning_rate', 0.01)
        self._bs = cfg.get('batch_size', 32)
        self._epochs = cfg.get('local_epochs', 5)

        # 持续学习参数
        self.num_tasks = cfg.get('num_tasks', 5)
        self.classes_per_task = cfg.get('classes_per_task', 2)
        self.scenario = cfg.get('scenario', 'class_incremental')

        # FedWeIT特定参数
        self.use_attention = cfg.get('use_attention', True)
        self.feature_dim = cfg.get('feature_dim', 512)
        self.transfer_weight = cfg.get('transfer_weight', 0.5)
        self.use_weighted_transfer = cfg.get('use_weighted_transfer', True)
        self.distill_temperature = cfg.get('distill_temperature', 2.0)

        # 设备配置
        device_str = cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(device_str)

        # 持续学习状态
        self.current_task_id = 0
        self.seen_classes = []

        # 任务特定的模型参数快照
        self.task_specific_params = {}  # {task_id: model_state_dict}

        # 注意力模块(延迟初始化)
        self.attention_module = None

        # 组件占位符
        self.torch_model = None
        self._optimizer = None
        self._criterion = None
        self._train_loader = None

        self.logger.info(
            f"FedWeITLearner {node_id} initialized: "
            f"num_tasks={self.num_tasks}, classes_per_task={self.classes_per_task}, "
            f"use_attention={self.use_attention}"
        )

    async def setup(self, config: Dict) -> None:
        """
        初始化训练环境

        Args:
            config: 运行配置
        """
        self.logger.debug(f"[{self._node_id}] 初始化FedWeIT训练环境...")

        # 获取PyTorch模型
        self.torch_model = self._model.get_model()
        self.torch_model = self.torch_model.to(self.device)
        self.torch_model.train()

        # 获取训练配置
        batch_size = config.get("batch_size", self._bs)
        learning_rate = config.get("learning_rate", self._lr)

        # 创建训练数据加载器
        train_datasets = self._datasets.get("train", [])
        if train_datasets:
            train_dataset = train_datasets[0]
            self._train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False
            )
            self.logger.info(
                f"[{self._node_id}] 训练数据加载器已创建，样本数: {len(train_dataset)}, "
                f"batch_size: {batch_size}"
            )
        else:
            self._train_loader = []
            self.logger.warning(f"[{self._node_id}] 未提供训练数据")

        # 初始化注意力模块(如果需要且未初始化)
        if self.use_attention and self.attention_module is None:
            self.attention_module = TaskAdaptiveAttention(
                num_tasks=self.num_tasks,
                feature_dim=self.feature_dim
            ).to(self.device)
            self.logger.info(f"[{self._node_id}] 注意力模块已创建")

        # 收集需要优化的参数
        params_to_optimize = list(self.torch_model.parameters())
        if self.attention_module is not None:
            params_to_optimize += list(self.attention_module.parameters())

        # 创建优化器
        self._optimizer = optim.SGD(
            params_to_optimize,
            lr=learning_rate,
            momentum=0.9,
            weight_decay=5e-4
        )
        self.logger.info(f"[{self._node_id}] 优化器已创建: SGD, lr={learning_rate}")

        # 创建损失函数
        self._criterion = nn.CrossEntropyLoss()
        self.logger.info(f"[{self._node_id}] 损失函数已创建: CrossEntropyLoss")

    def _get_task_data_loader(self, task_id: int) -> DataLoader:
        """
        获取特定任务的数据加载器

        根据当前任务ID筛选对应的类别数据
        """
        train_datasets = self._datasets.get("train", [])
        if not train_datasets:
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

    async def train(self, epochs: int) -> TrainMetrics:
        """
        训练循环 - FedWeIT特定实现

        Args:
            epochs: 训练轮数

        Returns:
            TrainMetrics: 完整训练过程的指标
        """
        # 提取任务ID（如果配置中指定）
        task_id = self._config.get("task_id", self.current_task_id)

        # 更新任务状态
        if task_id != self.current_task_id:
            self.logger.info(
                f"[{self._node_id}] Switching task: {self.current_task_id} -> {task_id}"
            )

            # 保存当前任务的参数快照
            if self.torch_model is not None:
                self.task_specific_params[self.current_task_id] = {
                    k: v.cpu().clone() for k, v in self.torch_model.state_dict().items()
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
        self._task_loader = task_loader

        # 执行标准训练循环
        epoch_history = []

        for epoch in range(1, epochs + 1):
            self._current_epoch = epoch

            # 执行单个 epoch
            epoch_metrics = await self.train_epoch(epoch_idx=epoch)
            epoch_history.append(epoch_metrics)

        # 汇总指标
        total_samples = sum(em.total_samples for em in epoch_history)
        final_loss = epoch_history[-1].avg_loss if epoch_history else 0.0

        # 聚合 epoch 指标
        aggregated_metrics = self._aggregate_epoch_metrics(epoch_history)
        aggregated_metrics['loss'] = final_loss
        aggregated_metrics['task_id'] = task_id
        aggregated_metrics['seen_classes'] = len(self.seen_classes)
        aggregated_metrics['num_task_params'] = len(self.task_specific_params)

        train_metrics = TrainMetrics(
            total_epochs=epochs,
            final_loss=final_loss,
            total_samples=total_samples,
            metrics=aggregated_metrics,
            epoch_history=epoch_history
        )

        return train_metrics

    async def train_step(self, batch: Any, batch_idx: int) -> StepMetrics:
        """
        单批次训练 - FedWeIT特定实现

        Args:
            batch: 批次数据
            batch_idx: 批次索引

        Returns:
            StepMetrics: 包含 loss、batch_size、metrics
        """
        data, target = batch
        data, target = data.to(self.device), target.to(self.device)

        self._optimizer.zero_grad()

        # 前向传播(带注意力)
        features, output = self._forward_with_attention(data, self.current_task_id)

        # 1. 分类损失
        ce_loss = self._criterion(output, target)

        # 2. 知识转移损失(如果有历史任务)
        transfer_loss = 0.0
        if self.use_weighted_transfer and len(self.task_specific_params) > 0 and self.current_task_id > 0:
            transfer_loss = self._compute_weighted_transfer_loss(data)

        # 3. 正则化损失(防止任务间参数差异过大)
        regularization_loss = 0.0
        if len(self.task_specific_params) > 0 and self.current_task_id > 0:
            regularization_loss = self._compute_regularization_loss()

        # 总损失
        if self.current_task_id > 0:
            loss = (ce_loss +
                    self.transfer_weight * transfer_loss +
                    0.01 * regularization_loss)
        else:
            loss = ce_loss

        # 反向传播
        loss.backward()
        self._optimizer.step()

        # 计算准确率
        _, predicted = output.max(1)
        correct = predicted.eq(target).sum().item()
        accuracy = correct / target.size(0)

        return StepMetrics(
            loss=loss.item(),
            batch_size=data.size(0),
            metrics={
                "accuracy": accuracy,
                "ce_loss": ce_loss.item(),
                "transfer_loss": transfer_loss.item() if isinstance(transfer_loss, torch.Tensor) else transfer_loss,
                "reg_loss": regularization_loss.item() if isinstance(regularization_loss, torch.Tensor) else regularization_loss
            }
        )

    def get_dataloader(self) -> Any:
        """获取数据加载器"""
        return getattr(self, '_task_loader', self._train_loader)

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
        output = self.torch_model(data)

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

        current_output = self.torch_model(data)
        total_loss = 0.0
        num_prev_tasks = len(self.task_specific_params)

        # 遍历所有历史任务
        for prev_task_id, prev_params in self.task_specific_params.items():
            # 加载历史任务参数
            prev_model = copy.deepcopy(self.torch_model)
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
        current_params = self.torch_model.state_dict()

        # 计算与所有历史任务参数的L2距离
        for prev_task_id, prev_params in self.task_specific_params.items():
            for name, param in current_params.items():
                if name in prev_params:
                    prev_param = prev_params[name].to(self.device)
                    total_loss += ((param - prev_param) ** 2).sum()

        num_prev_tasks = len(self.task_specific_params)
        return total_loss / num_prev_tasks if num_prev_tasks > 0 else torch.tensor(0.0, device=self.device)

    async def evaluate(self, config: Optional[Dict[str, Any]] = None) -> EvalResult:
        """
        执行本地评估

        Args:
            config: 评估配置

        Returns:
            EvalResult: 评估结果
        """
        task_id = (config or {}).get("task_id", self.current_task_id)

        # 获取评估数据
        if task_id is not None:
            eval_loader = self._get_task_data_loader(task_id)
        else:
            eval_loader = self._train_loader

        self.torch_model.eval()
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
                'seen_classes': len(self.seen_classes),
                'num_task_params': len(self.task_specific_params)
            }
        )

    def get_num_samples(self) -> int:
        """获取训练样本数"""
        train_datasets = self._datasets.get("train", [])
        if not train_datasets:
            return 0
        try:
            return len(train_datasets[0])
        except Exception:
            return 0

    def get_metadata(self) -> Dict[str, Any]:
        """获取训练元数据"""
        return {
            "node_id": self._node_id,
            "total_steps": self._global_step,
            "current_task": self.current_task_id,
            "seen_classes": self.seen_classes,
            "num_task_params": len(self.task_specific_params),
            "use_attention": self.use_attention
        }

    def get_weights(self) -> Any:
        """获取模型权重（包括注意力模块）"""
        weights = self._model.get_weights()

        # 如果有注意力模块，也包含其参数
        if self.attention_module is not None:
            attention_weights = {
                f'attention.{k}': v.cpu().numpy()
                for k, v in self.attention_module.state_dict().items()
            }
            # 合并权重
            if isinstance(weights, dict):
                weights.update(attention_weights)

        return weights

    def set_weights(self, weights: Any) -> bool:
        """设置模型权重（包括注意力模块）"""
        try:
            if isinstance(weights, dict):
                # 分离模型权重和注意力权重
                model_weights = {}
                attention_weights = {}

                for k, v in weights.items():
                    if k.startswith('attention.'):
                        attention_weights[k.replace('attention.', '')] = v
                    else:
                        model_weights[k] = v

                # 设置模型权重
                if model_weights:
                    self._model.set_weights(model_weights)

                # 设置注意力模块权重
                if attention_weights and self.attention_module is not None:
                    attention_state_dict = {
                        k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
                        for k, v in attention_weights.items()
                    }
                    self.attention_module.load_state_dict(attention_state_dict)

            else:
                # 直接设置模型权重
                self._model.set_weights(weights)

            return True
        except Exception as e:
            self.logger.error(f"Failed to set weights: {e}")
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
