"""
FedKNOW: Federated Continual Learning with Signature Task Knowledge Integration
INFOCOM 2023

Paper: https://ieeexplore.ieee.org/document/10228927
GitHub: https://github.com/LINC-BIT/FedKNOW

核心思想：
1. 从每个任务提取"签名任务知识"(Signature Task Knowledge)
2. 使用知识集成机制避免灾难性遗忘
3. 支持多种网络架构：LeNet, AlexNet, VGG, ResNet等
4. 在本地维护任务知识库
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


@learner('cl.fedknow', description='FedKNOW: Federated Continual Learning with Signature Task Knowledge Integration (INFOCOM 2023)')
class FedKNOWLearner(Learner):
    """
    FedKNOW 持续学习方法的Learner实现

    特点：
    - 提取和保存签名任务知识(Signature Task Knowledge)
    - 知识集成机制(Knowledge Integration)
    - 支持Task-Incremental和Class-Incremental场景
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
        初始化 FedKNOW Learner

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
        self._lr = self._config.get('learning_rate', 0.01)
        self._bs = self._config.get('batch_size', 32)
        self._epochs = self._config.get('local_epochs', 5)
        self.momentum = self._config.get('momentum', 0.9)
        self.weight_decay = self._config.get('weight_decay', 5e-4)
        self.optimizer_type = self._config.get('optimizer', 'SGD').upper()
        self.loss_type = self._config.get('loss', 'CrossEntropyLoss')

        # 持续学习参数
        self.num_tasks = self._config.get('num_tasks', 5)
        self.classes_per_task = self._config.get('classes_per_task', 2)
        self.scenario = self._config.get('scenario', 'class_incremental')

        # FedKNOW特定参数
        self.signature_ratio = self._config.get('signature_ratio', 0.1)  # 签名样本比例
        self.knowledge_weight = self._config.get('knowledge_weight', 0.5)  # 知识蒸馏权重
        self.distill_temperature = self._config.get('distill_temperature', 2.0)
        self.integration_method = self._config.get('integration_method', 'weighted_sum')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 持续学习状态
        self.current_task_id = 0
        self.seen_classes = []

        # 知识库（存储每个任务的签名知识）
        self.task_knowledge = {}  # {task_id: {'model': model_copy, 'data_indices': indices}}

        # 组件占位符
        self.torch_model = None
        self._optimizer = None
        self._criterion = None
        self._train_loader = None
        self._test_loader = None

        self.logger.info(
            f"FedKNOWLearner {node_id} initialized: "
            f"num_tasks={self.num_tasks}, classes_per_task={self.classes_per_task}, "
            f"signature_ratio={self.signature_ratio}"
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

        # 创建优化器
        if self.optimizer_type == 'SGD':
            self._optimizer = optim.SGD(
                self.torch_model.parameters(),
                lr=self._lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_type == 'ADAM':
            self._optimizer = optim.Adam(
                self.torch_model.parameters(),
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

        Args:
            batch: 批次数据 (data, target)
            batch_idx: 批次索引

        Returns:
            StepMetrics: 训练指标
        """
        data, target = batch
        data, target = data.to(self.device), target.to(self.device)

        self._optimizer.zero_grad()

        # 前向传播
        output = self.torch_model(data)

        # 1. 当前任务的分类损失
        ce_loss = self._criterion(output, target)

        # 2. 知识集成损失（如果有历史任务知识）
        integration_loss = 0.0
        if self.task_knowledge and self.current_task_id > 0:
            integration_loss = self._compute_knowledge_integration_loss(data, output)

        # 总损失
        if self.current_task_id > 0 and self.task_knowledge:
            loss = ce_loss + self.knowledge_weight * integration_loss
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
            batch_size=target.size(0),
            metrics={
                'accuracy': accuracy,
                'ce_loss': ce_loss.item(),
                'integration_loss': integration_loss.item() if isinstance(integration_loss, torch.Tensor) else integration_loss
            }
        )

    async def train_epoch(self, epoch_idx: int) -> EpochMetrics:
        """
        单轮训练

        在训练完成后提取签名知识

        Args:
            epoch_idx: 轮次索引

        Returns:
            EpochMetrics: 轮次指标
        """
        # 调用父类的train_epoch方法
        self.torch_model.train()
        epoch_metrics = await super().train_epoch(epoch_idx)

        # 在最后一个epoch后提取签名任务知识
        # 这里我们检查是否是最后一个epoch（通过检查当前epoch是否等于配置的epochs）
        total_epochs = self._config.get('epochs', self._epochs)
        if epoch_idx == total_epochs:
            task_loader = self._get_task_data_loader(self.current_task_id)
            await self._extract_signature_knowledge(task_loader, self.current_task_id)

        return epoch_metrics

    def _compute_knowledge_integration_loss(
        self,
        data: torch.Tensor,
        current_output: torch.Tensor
    ) -> torch.Tensor:
        """
        计算知识集成损失

        将历史任务的知识集成到当前模型中
        使用温度缩放的KL散度

        Args:
            data: 输入数据
            current_output: 当前模型输出

        Returns:
            torch.Tensor: 知识集成损失
        """
        if not self.task_knowledge:
            return torch.tensor(0.0, device=self.device)

        total_loss = 0.0
        num_tasks = len(self.task_knowledge)

        # 遍历所有历史任务
        for tid, knowledge in self.task_knowledge.items():
            teacher_model = knowledge['model']
            teacher_model.eval()

            with torch.no_grad():
                teacher_output = teacher_model(data)

            # 使用温度缩放的KL散度
            teacher_probs = F.softmax(teacher_output / self.distill_temperature, dim=1)
            student_log_probs = F.log_softmax(current_output / self.distill_temperature, dim=1)

            kl_div = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
            kl_div *= (self.distill_temperature ** 2)

            total_loss += kl_div

        # 平均损失
        return total_loss / num_tasks if num_tasks > 0 else torch.tensor(0.0, device=self.device)

    async def _extract_signature_knowledge(self, train_loader: DataLoader, task_id: int):
        """
        提取签名任务知识

        从当前任务中选择代表性样本，保存为签名知识

        Args:
            train_loader: 训练数据加载器
            task_id: 任务ID
        """
        if self.torch_model is None:
            return

        # 保存当前任务的模型副本
        model_copy = copy.deepcopy(self.torch_model)
        model_copy.eval()

        # 提取签名数据索引（基于不确定性采样）
        dataset = train_loader.dataset
        num_signature = max(1, int(len(dataset) * self.signature_ratio))

        # 计算样本的不确定性
        uncertainties = []
        self.torch_model.eval()

        with torch.no_grad():
            for idx in range(len(dataset)):
                data, _ = dataset[idx]
                data = data.unsqueeze(0).to(self.device)
                output = self.torch_model(data)
                probs = F.softmax(output, dim=1)

                # 使用熵作为不确定性度量
                entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
                uncertainties.append((idx, entropy))

        self.torch_model.train()

        # 选择不确定性最高的样本作为签名数据
        uncertainties.sort(key=lambda x: x[1], reverse=True)
        signature_indices = [idx for idx, _ in uncertainties[:num_signature]]

        # 保存到知识库
        self.task_knowledge[task_id] = {
            'model': model_copy,
            'data_indices': signature_indices,
            'num_samples': len(signature_indices)
        }

        self.logger.info(
            f"[{self._node_id}] Extracted signature knowledge for Task {task_id}: "
            f"{len(signature_indices)} samples (based on uncertainty)"
        )

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

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in eval_loader:
                data, target = data.to(self.device), target.to(self.device)

                # 根据场景选择预测方式
                if self.scenario == 'class_incremental' and self.task_knowledge:
                    # CIL场景：使用知识集成预测
                    output = self._integrated_prediction(data)
                else:
                    # TIL场景：直接使用当前模型
                    output = self.torch_model(data)

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
                'num_task_knowledge': len(self.task_knowledge)
            }
        )

    def _integrated_prediction(self, data: torch.Tensor) -> torch.Tensor:
        """
        集成所有任务知识进行预测

        用于Class-Incremental场景（不知道任务ID）

        Args:
            data: 输入数据

        Returns:
            torch.Tensor: 集成后的预测输出
        """
        if not self.task_knowledge:
            # 如果没有历史知识，使用当前模型
            return self.torch_model(data)

        if self.integration_method == 'weighted_sum':
            # 加权求和所有任务的输出
            outputs = []

            # 包含当前模型
            outputs.append(self.torch_model(data))

            # 包含所有历史任务模型
            for tid, knowledge in self.task_knowledge.items():
                teacher_model = knowledge['model']
                teacher_model.eval()
                with torch.no_grad():
                    output = teacher_model(data)
                outputs.append(output)

            # 简单平均
            integrated_output = torch.stack(outputs).mean(dim=0)
            return integrated_output

        elif self.integration_method == 'ensemble':
            # 集成投票
            predictions = []

            # 当前模型的预测
            predictions.append(self.torch_model(data).argmax(dim=1))

            # 历史任务模型的预测
            for tid, knowledge in self.task_knowledge.items():
                teacher_model = knowledge['model']
                teacher_model.eval()
                with torch.no_grad():
                    pred = teacher_model(data).argmax(dim=1)
                predictions.append(pred)

            # 多数投票
            predictions = torch.stack(predictions)
            # 返回众数
            mode_result = torch.mode(predictions, dim=0)
            final_pred = mode_result.values

            # 转换为one-hot输出
            num_classes = self.num_tasks * self.classes_per_task
            output = torch.zeros(data.size(0), num_classes, device=self.device)
            output.scatter_(1, final_pred.unsqueeze(1), 1.0)
            return output

        else:
            # 默认使用当前模型
            return self.torch_model(data)

    def get_knowledge_summary(self) -> Dict[str, Any]:
        """
        获取知识库摘要

        Returns:
            知识库统计信息
        """
        return {
            'num_tasks': len(self.task_knowledge),
            'current_task': self.current_task_id,
            'total_signature_samples': sum(
                k['num_samples'] for k in self.task_knowledge.values()
            ),
            'tasks': list(self.task_knowledge.keys()),
            'integration_method': self.integration_method,
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
            'num_task_knowledge': len(self.task_knowledge),
            'knowledge_summary': self.get_knowledge_summary()
        })
        return base_metadata
