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
import copy
import numpy as np

from ....core.learner import Learner
from ....core.types import EpochMetrics, EvalResult
from ....registry import learner


@learner('cl.lga', description='LGA: Layerwise Gradient Accumulation for Federated Continual Learning (TPAMI 2023)')
class LGALearner(Learner):
    """
    LGA (Layerwise Gradient Accumulation) 持续学习方法的Learner实现

    特点：
    - 逐层梯度累积策略
    - 基于重要性的参数保护
    - 选择性参数更新
    - 高效的通信和存储
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
        super().__init__(model, None, tracker, callbacks, config, node_id)

        self._datasets = datasets or {}

        # 从配置提取参数
        self.learning_rate = self._config.get('learning_rate', 0.01)
        self.batch_size = self._config.get('batch_size', 32)
        self.local_epochs = self._config.get('local_epochs', 5)
        self.momentum = self._config.get('momentum', 0.9)
        self.weight_decay = self._config.get('weight_decay', 5e-4)
        self.optimizer_type = self._config.get('optimizer', 'SGD').upper()
        self.loss_type = self._config.get('loss', 'CrossEntropyLoss')

        # 持续学习参数
        self.num_tasks = self._config.get('num_tasks', 5)
        self.classes_per_task = self._config.get('classes_per_task', 2)
        self.scenario = self._config.get('scenario', 'class_incremental')

        # LGA特定参数
        self.accumulation_steps = self._config.get('accumulation_steps', 4)
        self.use_layerwise_accumulation = self._config.get('use_layerwise_accumulation', True)
        self.importance_method = self._config.get('importance_method', 'fisher')
        self.protection_threshold = self._config.get('protection_threshold', 0.1)
        self.selective_update = self._config.get('selective_update', True)
        self.update_ratio = self._config.get('update_ratio', 0.8)

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
        self.torch_model = None
        self._optimizer = None
        self._criterion = None
        self._train_loader = None

        self.logger.info(
            f"LGALearner {node_id} 初始化完成: "
            f"num_tasks={self.num_tasks}, classes_per_task={self.classes_per_task}, "
            f"accumulation_steps={self.accumulation_steps}, importance_method={self.importance_method}"
        )

    async def setup(self, config: Dict) -> None:
        """初始化训练环境"""
        # 获取PyTorch模型
        self.torch_model = self._model.get_model()
        self.torch_model.to(self.device)

        # 创建优化器
        if self.optimizer_type == 'SGD':
            self._optimizer = optim.SGD(
                self.torch_model.parameters(),
                lr=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_type == 'ADAM':
            self._optimizer = optim.Adam(
                self.torch_model.parameters(),
                lr=self.learning_rate
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
            self._train_loader = DataLoader(
                train_datasets[0],
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=False
            )

        self.logger.info(
            f"Setup完成: train_samples={len(train_datasets[0]) if train_datasets else 0}"
        )

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
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False
        )

    async def train_epoch(self, epoch: int) -> EpochMetrics:
        """
        执行LGA方法的单轮训练

        LGA的训练流程：
        1. 逐层累积梯度
        2. 计算参数重要性
        3. 选择性更新参数(保护重要参数)
        4. 更新重要性分数
        """
        # 从配置中获取task_id
        task_id = self._config.get('task_id', self.current_task_id)

        # 更新任务状态
        if task_id != self.current_task_id:
            self.logger.info(
                f"[{self._node_id}] Switching task: {self.current_task_id} -> {task_id}"
            )

            # 保存当前任务的参数快照
            if self.torch_model is not None:
                self.old_task_params[self.current_task_id] = {
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

        # 训练
        self.torch_model.train()

        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        # 重置梯度累积
        if self.use_layerwise_accumulation:
            self._reset_accumulated_gradients()

        for batch_idx, (data, target) in enumerate(task_loader):
            data, target = data.to(self.device), target.to(self.device)

            # 前向传播
            output = self.torch_model(data)

            # 1. 当前任务的分类损失
            ce_loss = self._criterion(output, target)

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
                    self._optimizer.step()

                self._optimizer.zero_grad()

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
            f"  [{self._node_id}] Task {task_id} Epoch {epoch}: "
            f"Loss={avg_epoch_loss:.4f}, Acc={epoch_accuracy:.4f}"
        )

        # 训练完成后，更新参数重要性
        await self._update_parameter_importance(task_loader)

        epoch_metrics = EpochMetrics(
            epoch=epoch,
            avg_loss=avg_epoch_loss,
            total_samples=epoch_total,
            metrics={
                'accuracy': epoch_accuracy,
                'task_id': task_id,
                'seen_classes': len(self.seen_classes),
                'num_protected_params': self._count_protected_parameters()
            }
        )

        # 增加全局 epoch 计数器（用于 MLflow step）
        self._global_epoch_counter += 1

        # 触发 epoch 结束回调
        if self._callbacks:
            await self._callbacks.on_epoch_end(self, epoch, epoch_metrics)

        return epoch_metrics

    def _reset_accumulated_gradients(self):
        """重置梯度累积"""
        self.accumulated_gradients = {}
        for name, param in self.torch_model.named_parameters():
            if param.requires_grad:
                self.accumulated_gradients[name] = torch.zeros_like(param.data)

    def _accumulate_gradients(self):
        """累积当前batch的梯度"""
        for name, param in self.torch_model.named_parameters():
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
        current_params = self.torch_model.state_dict()

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
        for name, param in self.torch_model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if name in self.parameter_importance:
                    importance = self.parameter_importance[name].to(self.device)

                    # 创建更新掩码：重要性低于阈值的参数允许更新
                    update_mask = (importance < self.protection_threshold).float()

                    # 应用掩码到梯度
                    param.grad.data *= update_mask

        # 执行参数更新
        self._optimizer.step()

    async def _update_parameter_importance(self, train_loader: DataLoader):
        """
        更新参数重要性

        使用Fisher信息矩阵或梯度幅度估计参数重要性

        Args:
            train_loader: 训练数据加载器
        """
        self.torch_model.eval()

        if self.importance_method == 'fisher':
            await self._compute_fisher_importance(train_loader)
        elif self.importance_method == 'gradient_magnitude':
            await self._compute_gradient_importance(train_loader)
        else:
            self.logger.warning(f"Unknown importance method: {self.importance_method}")

        self.torch_model.train()

    async def _compute_fisher_importance(self, train_loader: DataLoader):
        """
        使用Fisher信息矩阵计算参数重要性

        Args:
            train_loader: 训练数据加载器
        """
        fisher = {}
        for name, param in self.torch_model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param.data)

        self.torch_model.zero_grad()

        # 采样一部分数据计算Fisher信息
        num_samples = min(100, len(train_loader.dataset))
        sample_count = 0

        for data, target in train_loader:
            if sample_count >= num_samples:
                break

            data, target = data.to(self.device), target.to(self.device)

            output = self.torch_model(data)
            loss = self._criterion(output, target)

            loss.backward()

            for name, param in self.torch_model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.data ** 2

            self.torch_model.zero_grad()
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

    async def _compute_gradient_importance(self, train_loader: DataLoader):
        """
        使用梯度幅度计算参数重要性

        Args:
            train_loader: 训练数据加载器
        """
        grad_magnitude = {}
        for name, param in self.torch_model.named_parameters():
            if param.requires_grad:
                grad_magnitude[name] = torch.zeros_like(param.data)

        self.torch_model.zero_grad()

        # 采样一部分数据计算梯度幅度
        num_samples = min(100, len(train_loader.dataset))
        sample_count = 0

        for data, target in train_loader:
            if sample_count >= num_samples:
                break

            data, target = data.to(self.device), target.to(self.device)

            output = self.torch_model(data)
            loss = self._criterion(output, target)

            loss.backward()

            for name, param in self.torch_model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad_magnitude[name] += param.grad.data.abs()

            self.torch_model.zero_grad()
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

    async def evaluate_model(self, config: Optional[Dict] = None) -> EvalResult:
        """评估模型性能"""
        task_id = (config or {}).get("task_id", self.current_task_id)

        # 获取评估数据
        if task_id is not None:
            eval_loader = self._get_task_data_loader(task_id)
        else:
            eval_loader = self._train_loader

        self.torch_model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for data, target in eval_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.torch_model(data)
                loss = self._criterion(output, target)

                total_loss += loss.item() * data.size(0)
                _, predicted = output.max(1)
                total_samples += target.size(0)
                total_correct += predicted.eq(target).sum().item()

        accuracy = total_correct / total_samples if total_samples > 0 else 0
        avg_loss = total_loss / total_samples if total_samples > 0 else 0

        return EvalResult(
            num_samples=total_samples,
            metrics={
                'accuracy': accuracy,
                'loss': avg_loss,
                'task_id': task_id
            }
        )

    async def get_weights(self) -> Dict[str, Any]:
        """获取模型权重"""
        return {
            name: param.data.clone()
            for name, param in self.torch_model.state_dict().items()
        }

    async def set_weights(self, weights: Dict[str, Any]):
        """设置模型权重"""
        # 转换为torch tensor
        torch_weights = {}
        for k, v in weights.items():
            if torch.is_tensor(v):
                torch_weights[k] = v
            else:
                torch_weights[k] = torch.from_numpy(v)

        self.torch_model.load_state_dict(torch_weights)
        self.logger.debug(f"[{self._node_id}] Model weights updated")

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
