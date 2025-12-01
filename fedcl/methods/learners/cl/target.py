"""
TARGET: Federated Class-Continual Learning via Exemplar-Free Distillation
ICCV 2023

Paper: https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_TARGET_Federated_Class-Continual_Learning_via_Exemplar-Free_Distillation_ICCV_2023_paper.pdf
GitHub: https://github.com/zj-jayzhang/Federated-Class-Continual-Learning

核心思想：
1. 无需存储历史样本（Exemplar-Free）
2. 使用知识蒸馏从旧模型转移知识到新模型
3. 专注于Class-Incremental Learning场景
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from typing import Dict, Any, Optional
from datetime import datetime
import copy
import numpy as np

from .._decorators import learner
from fedcl.learner.base_learner import BaseLearner
from fedcl.types import TrainingResponse, EvaluationResult, ModelData
from fedcl.methods.models.base import get_weights_as_dict, set_weights_from_dict


@learner('cl', 'TARGET', description='TARGET: Federated Class-Continual Learning via Exemplar-Free Distillation (ICCV 2023)')
class TARGETLearner(BaseLearner):
    """
    TARGET 持续学习方法的Learner实现

    特点：
    - 支持Class-Incremental Learning (CIL)
    - 使用知识蒸馏防止遗忘
    - 无需存储历史样本
    """

    def __init__(self, client_id: str, config: Optional[Dict[str, Any]] = None, lazy_init: bool = True):
        # 提取配置
        learner_params = (config or {}).get('learner', {}).get('params', {})

        # 保存TARGET特定配置
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

        # 知识蒸馏参数
        self.use_distillation = learner_params.get('use_distillation', True)
        self.distill_temperature = learner_params.get('distill_temperature', 2.0)
        self.distill_weight = learner_params.get('distill_weight', 1.0)

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
        self.previous_model = None  # 保存上一个任务的模型用于蒸馏

        # 组件占位符
        self._model = None
        self._optimizer = None
        self._criterion = None
        self._train_loader = None

        self.logger.info(
            f"TARGETLearner {client_id} initialized: "
            f"num_tasks={self.num_tasks}, classes_per_task={self.classes_per_task}"
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
            return self.train_loader  # 回退到完整数据

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
        执行TARGET方法的本地训练
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
            # 保存当前模型作为previous_model
            if self._model is not None:
                self.previous_model = copy.deepcopy(self._model)
                self.previous_model.eval()

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

        # Task 0: 标准训练
        if task_id == 0:
            return await self._train_first_task(task_loader, num_epochs, task_id, start_time)

        # Task > 0: 使用合成数据进行知识蒸馏
        else:
            return await self._train_with_distillation(task_loader, num_epochs, task_id, round_number, start_time)

    async def _train_first_task(self, task_loader, num_epochs, task_id, start_time):
        """训练第一个任务（无需蒸馏）"""
        self.model.train()

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

                # 前向传播
                output = self.model(data)

                # 分类损失
                loss = self.criterion(output, target)

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

        # 返回结果
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

    async def _train_with_distillation(self, task_loader, num_epochs, task_id, round_number, start_time):
        """使用合成数据进行知识蒸馏训练（Task > 0）"""
        # 加载合成数据
        syn_loader = self._load_synthetic_data(task_id)

        if syn_loader is None:
            self.logger.warning(
                f"[{self.client_id}] No synthetic data found for task {task_id}, "
                f"falling back to standard training"
            )
            return await self._train_first_task(task_loader, num_epochs, task_id, start_time)

        self.model.train()
        if self.previous_model is not None:
            self.previous_model.eval()

        # 计算已见类别数
        known_classes = task_id * self.classes_per_task

        total_loss = 0.0
        total_ce_loss = 0.0
        total_kd_loss = 0.0
        total_accuracy = 0.0
        total_samples = 0

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_ce_loss = 0.0
            epoch_kd_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            # 使用zip同时迭代新任务数据和合成数据
            syn_iter = iter(syn_loader)

            for batch_idx, (data, target) in enumerate(task_loader):
                data, target = data.to(self.device), target.to(self.device)

                # 获取一个batch的合成数据
                try:
                    syn_data = next(syn_iter)
                    syn_data = syn_data.to(self.device)
                except StopIteration:
                    # 重新开始synthetic data迭代
                    syn_iter = iter(syn_loader)
                    syn_data = next(syn_iter)
                    syn_data = syn_data.to(self.device)

                self.optimizer.zero_grad()

                # 1. 新任务数据的CE损失
                output = self.model(data)

                # 将标签映射到相对任务的类别索引（用于计算CE loss）
                # 例如：Task 1的类别[5,6,7,8,9] -> [0,1,2,3,4]（相对索引）
                fake_targets = target - known_classes

                # 只对新任务的logits计算CE loss
                ce_loss = F.cross_entropy(output[:, known_classes:], fake_targets)

                # 2. 合成数据的KD损失（针对旧任务）
                if self.previous_model is not None:
                    s_out = self.model(syn_data)
                    with torch.no_grad():
                        t_out = self.previous_model(syn_data)

                    # 只对旧任务的logits计算KD loss
                    kd_loss = self._compute_kd_loss(
                        s_out[:, :known_classes],
                        t_out[:, :known_classes],
                        temperature=self.distill_temperature
                    )
                else:
                    kd_loss = torch.tensor(0.0, device=self.device)

                # 总损失
                loss = ce_loss + self.distill_weight * kd_loss

                # 反向传播
                loss.backward()
                self.optimizer.step()

                # 统计
                epoch_loss += loss.item()
                epoch_ce_loss += ce_loss.item()
                epoch_kd_loss += kd_loss.item() if isinstance(kd_loss, torch.Tensor) else kd_loss

                _, predicted = output.max(1)
                epoch_total += target.size(0)
                epoch_correct += predicted.eq(target).sum().item()

            # Epoch统计
            epoch_accuracy = epoch_correct / epoch_total if epoch_total > 0 else 0
            avg_epoch_loss = epoch_loss / len(task_loader) if len(task_loader) > 0 else 0
            avg_ce_loss = epoch_ce_loss / len(task_loader) if len(task_loader) > 0 else 0
            avg_kd_loss = epoch_kd_loss / len(task_loader) if len(task_loader) > 0 else 0

            self.logger.info(
                f"  [{self.client_id}] Task {task_id} Epoch {epoch+1}/{num_epochs}: "
                f"Loss={avg_epoch_loss:.4f}, CE={avg_ce_loss:.4f}, KD={avg_kd_loss:.4f}, Acc={epoch_accuracy:.4f}"
            )

            total_loss += epoch_loss
            total_ce_loss += epoch_ce_loss
            total_kd_loss += epoch_kd_loss
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

        # 返回结果
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
                'samples_used': total_samples,
                'ce_loss': total_ce_loss / (num_epochs * len(task_loader)),
                'kd_loss': total_kd_loss / (num_epochs * len(task_loader))
            },
            execution_time=(datetime.now() - start_time).total_seconds()
        )

        # 触发回调
        for callback in self._callbacks.get('after_train', []):
            callback(response)

        return response

    def _load_synthetic_data(self, task_id: int) -> Optional[DataLoader]:
        """
        从磁盘加载合成数据

        Args:
            task_id: 当前任务ID（用于查找上一个任务的合成数据）

        Returns:
            DataLoader or None
        """
        import os
        from torchvision import transforms
        from .target_generator import UnlabeledImageDataset

        # 从配置获取save_dir
        save_dir = self.config.get('trainer', {}).get('params', {}).get('save_dir', 'run/target_synthetic_data')

        # 加载前一个任务的合成数据
        prev_task_id = task_id - 1
        if prev_task_id < 0:
            return None

        data_dir = os.path.join(save_dir, f"task_{prev_task_id}")

        if not os.path.exists(data_dir):
            self.logger.warning(f"Synthetic data directory not found: {data_dir}")
            return None

        # 检查目录是否有文件
        files = [f for f in os.listdir(data_dir) if f.endswith('.png')]
        if not files:
            self.logger.warning(f"No synthetic images found in: {data_dir}")
            return None

        self.logger.info(f"Loading {len(files)} synthetic images from {data_dir}")

        # 确定数据集的归一化参数
        dataset_name = self.config.get('dataset', {}).get('name', 'MNIST')
        if dataset_name.upper() == 'CIFAR100':
            data_normalize = dict(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
        else:  # MNIST或其他
            data_normalize = dict(mean=(0.1307,), std=(0.3081,))

        # 创建transform
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(**data_normalize)
        ])

        # 创建dataset
        syn_dataset = UnlabeledImageDataset(
            data_dir,
            transform=transform,
            nums=None  # 使用所有生成的数据
        )

        # 创建DataLoader
        batch_size = self.config.get('trainer', {}).get('params', {}).get(
            'data_generation', {}
        ).get('sample_batch_size', self._bs)

        syn_loader = DataLoader(
            syn_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )

        self.logger.info(f"Synthetic data loader created: {len(syn_dataset)} samples, batch_size={batch_size}")

        return syn_loader

    def _compute_kd_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                         temperature: float = 2.0) -> torch.Tensor:
        """
        计算KL散度知识蒸馏损失

        Args:
            student_logits: 学生模型的logits
            teacher_logits: 教师模型的logits
            temperature: 温度参数

        Returns:
            KL散度损失
        """
        q = F.log_softmax(student_logits / temperature, dim=1)
        p = F.softmax(teacher_logits / temperature, dim=1)
        kl_div = F.kl_div(q, p, reduction='batchmean') * (temperature ** 2)
        return kl_div

    def _compute_distillation_loss(self, data: torch.Tensor) -> torch.Tensor:
        """
        计算知识蒸馏损失

        使用温度缩放的KL散度
        """
        if self.previous_model is None:
            return torch.tensor(0.0, device=self.device)

        self.previous_model.eval()
        with torch.no_grad():
            teacher_output = self.previous_model(data)

        student_output = self.model(data)

        # 温度缩放的softmax
        teacher_probs = F.softmax(teacher_output / self.distill_temperature, dim=1)
        student_log_probs = F.log_softmax(student_output / self.distill_temperature, dim=1)

        # KL散度
        kl_div = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
        kl_div *= (self.distill_temperature ** 2)

        return kl_div

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

            set_weights_from_dict(self.model, weights)
            return True
        except Exception as e:
            self.logger.error(f"Failed to set model: {e}")
            return False
