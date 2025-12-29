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
import copy
import os

from ....core.learner import Learner
from ....core.types import EpochMetrics, EvalResult
from ....registry import learner


@learner('cl.target', description='TARGET: Federated Class-Continual Learning via Exemplar-Free Distillation (ICCV 2023)')
class TARGETLearner(Learner):
    """
    TARGET 持续学习方法的Learner实现

    特点：
    - 支持Class-Incremental Learning (CIL)
    - 使用知识蒸馏防止遗忘
    - 无需存储历史样本

    配置示例:
    {
        "learner": {
            "name": "cl.target",
            "learning_rate": 0.01,
            "batch_size": 32,
            "local_epochs": 5,
            "optimizer": "SGD",
            "momentum": 0.9,
            "num_tasks": 5,
            "classes_per_task": 2,
            "scenario": "class_incremental",
            "use_distillation": true,
            "distill_temperature": 2.0,
            "distill_weight": 1.0
        }
    }
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

        # 知识蒸馏参数
        self.use_distillation = self._config.get('use_distillation', True)
        self.distill_temperature = self._config.get('distill_temperature', 2.0)
        self.distill_weight = self._config.get('distill_weight', 1.0)

        # 数据生成配置（用于加载合成数据）
        self.save_dir = self._config.get('save_dir', 'run/target_synthetic_data')
        self.sample_batch_size = self._config.get('sample_batch_size', self.batch_size)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 持续学习状态
        self.current_task_id = 0
        self.seen_classes = []
        self.previous_model = None  # 保存上一个任务的模型用于蒸馏

        # 组件占位符
        self.torch_model = None
        self._optimizer = None
        self._criterion = None
        self._train_loader = None
        self._test_loader = None

        self.logger.info(
            f"TARGETLearner {node_id} 初始化完成: "
            f"num_tasks={self.num_tasks}, classes_per_task={self.classes_per_task}, "
            f"use_distillation={self.use_distillation}"
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

        test_datasets = self._datasets.get("test", [])
        if test_datasets:
            self._test_loader = DataLoader(
                test_datasets[0],
                batch_size=self.batch_size,
                shuffle=False
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
            return self._train_loader  # 回退到完整数据

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
        单轮训练 - TARGET训练循环

        根据当前任务ID决定训练策略：
        - Task 0: 标准训练
        - Task > 0: 使用合成数据进行知识蒸馏
        """
        # 从配置中获取task_id
        task_id = self._config.get('task_id', self.current_task_id)

        # 更新任务状态
        if task_id != self.current_task_id:
            self.logger.info(
                f"[{self._node_id}] Switching task: {self.current_task_id} -> {task_id}"
            )
            # 保存当前模型作为previous_model
            if self.torch_model is not None:
                self.previous_model = copy.deepcopy(self.torch_model)
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

        # Task 0: 标准训练
        if task_id == 0:
            return await self._train_first_task_epoch(epoch, task_loader, task_id)

        # Task > 0: 使用合成数据进行知识蒸馏
        else:
            return await self._train_with_distillation_epoch(epoch, task_loader, task_id)

    async def _train_first_task_epoch(self, epoch: int, task_loader: DataLoader, task_id: int) -> EpochMetrics:
        """训练第一个任务的单个epoch（无需蒸馏）"""
        self.torch_model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_idx, (data, target) in enumerate(task_loader):
            data, target = data.to(self.device), target.to(self.device)

            self._optimizer.zero_grad()

            # 前向传播
            output = self.torch_model(data)

            # 分类损失
            loss = self._criterion(output, target)

            # 反向传播
            loss.backward()
            self._optimizer.step()

            # 统计
            total_loss += loss.item() * data.size(0)
            _, predicted = output.max(1)
            total_samples += target.size(0)
            total_correct += predicted.eq(target).sum().item()

        # 计算平均值
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        avg_accuracy = total_correct / total_samples if total_samples > 0 else 0

        self.logger.info(
            f"  [{self._node_id}] Task {task_id} Epoch {epoch}: "
            f"Loss={avg_loss:.4f}, Acc={avg_accuracy:.4f}"
        )

        epoch_metrics = EpochMetrics(
            epoch=epoch,
            avg_loss=avg_loss,
            total_samples=total_samples,
            metrics={
                'accuracy': avg_accuracy,
                'task_id': task_id,
                'seen_classes': len(self.seen_classes)
            }
        )

        # 增加全局 epoch 计数器（用于 MLflow step）
        self._global_epoch_counter += 1

        # 触发 epoch 结束回调
        if self._callbacks:
            await self._callbacks.on_epoch_end(self, epoch, epoch_metrics)

        return epoch_metrics

    async def _train_with_distillation_epoch(self, epoch: int, task_loader: DataLoader, task_id: int) -> EpochMetrics:
        """使用合成数据进行知识蒸馏训练的单个epoch（Task > 0）"""
        # 加载合成数据
        syn_loader = self._load_synthetic_data(task_id)

        if syn_loader is None:
            self.logger.warning(
                f"[{self._node_id}] No synthetic data found for task {task_id}, "
                f"falling back to standard training"
            )
            return await self._train_first_task_epoch(epoch, task_loader, task_id)

        self.torch_model.train()
        if self.previous_model is not None:
            self.previous_model.eval()

        # 计算已见类别数
        known_classes = task_id * self.classes_per_task

        total_loss = 0.0
        total_ce_loss = 0.0
        total_kd_loss = 0.0
        total_correct = 0
        total_samples = 0

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

            self._optimizer.zero_grad()

            # 1. 新任务数据的CE损失
            output = self.torch_model(data)

            # 将标签映射到相对任务的类别索引（用于计算CE loss）
            # 例如：Task 1的类别[5,6,7,8,9] -> [0,1,2,3,4]（相对索引）
            fake_targets = target - known_classes

            # 只对新任务的logits计算CE loss
            ce_loss = F.cross_entropy(output[:, known_classes:], fake_targets)

            # 2. 合成数据的KD损失（针对旧任务）
            if self.previous_model is not None:
                s_out = self.torch_model(syn_data)
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
            self._optimizer.step()

            # 统计
            total_loss += loss.item() * data.size(0)
            total_ce_loss += ce_loss.item() * data.size(0)
            total_kd_loss += (kd_loss.item() if isinstance(kd_loss, torch.Tensor) else kd_loss) * data.size(0)

            _, predicted = output.max(1)
            total_samples += target.size(0)
            total_correct += predicted.eq(target).sum().item()

        # 计算平均值
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        avg_accuracy = total_correct / total_samples if total_samples > 0 else 0
        avg_ce_loss = total_ce_loss / total_samples if total_samples > 0 else 0
        avg_kd_loss = total_kd_loss / total_samples if total_samples > 0 else 0

        self.logger.info(
            f"  [{self._node_id}] Task {task_id} Epoch {epoch}: "
            f"Loss={avg_loss:.4f}, CE={avg_ce_loss:.4f}, KD={avg_kd_loss:.4f}, Acc={avg_accuracy:.4f}"
        )

        epoch_metrics = EpochMetrics(
            epoch=epoch,
            avg_loss=avg_loss,
            total_samples=total_samples,
            metrics={
                'accuracy': avg_accuracy,
                'task_id': task_id,
                'seen_classes': len(self.seen_classes),
                'ce_loss': avg_ce_loss,
                'kd_loss': avg_kd_loss
            }
        )

        # 增加全局 epoch 计数器（用于 MLflow step）
        self._global_epoch_counter += 1

        # 触发 epoch 结束回调
        if self._callbacks:
            await self._callbacks.on_epoch_end(self, epoch, epoch_metrics)

        return epoch_metrics

    def _load_synthetic_data(self, task_id: int) -> Optional[DataLoader]:
        """
        从磁盘加载合成数据

        Args:
            task_id: 当前任务ID（用于查找上一个任务的合成数据）

        Returns:
            DataLoader or None
        """
        from torchvision import transforms

        # 动态导入以避免循环依赖
        try:
            from methods.learners.cl.target_generator import UnlabeledImageDataset
        except ImportError:
            self.logger.error("Failed to import UnlabeledImageDataset from target_generator")
            return None

        # 加载前一个任务的合成数据
        prev_task_id = task_id - 1
        if prev_task_id < 0:
            return None

        data_dir = os.path.join(self.save_dir, f"task_{prev_task_id}")

        if not os.path.exists(data_dir):
            self.logger.warning(f"Synthetic data directory not found: {data_dir}")
            return None

        # 检查目录是否有文件
        files = [f for f in os.listdir(data_dir) if f.endswith('.png')]
        if not files:
            self.logger.warning(f"No synthetic images found in: {data_dir}")
            return None

        self.logger.info(f"Loading {len(files)} synthetic images from {data_dir}")

        # 确定数据集的归一化参数（从配置获取，如果有的话）
        dataset_name = self._config.get('dataset_name', 'MNIST')
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
        syn_loader = DataLoader(
            syn_dataset,
            batch_size=self.sample_batch_size,
            shuffle=True,
            num_workers=0
        )

        self.logger.info(
            f"Synthetic data loader created: {len(syn_dataset)} samples, "
            f"batch_size={self.sample_batch_size}"
        )

        return syn_loader

    def _compute_kd_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        temperature: float = 2.0
    ) -> torch.Tensor:
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

    async def evaluate(self, config: Optional[Dict] = None) -> EvalResult:
        """评估模型性能"""
        task_id = (config or {}).get("task_id", self.current_task_id)

        # 获取评估数据
        if task_id is not None:
            eval_loader = self._get_task_data_loader(task_id)
        else:
            eval_loader = self._test_loader if self._test_loader else self._train_loader

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

    async def train_step(self, batch: Any, batch_idx: int) -> Dict[str, Any]:
        """单批次训练"""
        data, target = batch
        data, target = data.to(self.device), target.to(self.device)

        self._optimizer.zero_grad()
        output = self.torch_model(data)
        loss = self._criterion(output, target)
        loss.backward()
        self._optimizer.step()

        _, predicted = output.max(1)
        correct = predicted.eq(target).sum().item()
        accuracy = correct / target.size(0)

        return {
            'loss': loss.item(),
            'batch_size': target.size(0),
            'accuracy': accuracy
        }

    def get_dataloader(self) -> DataLoader:
        """获取数据加载器"""
        return self._train_loader

    def get_num_samples(self) -> int:
        """获取训练样本数"""
        if hasattr(self, '_train_loader') and self._train_loader is not None:
            return len(self._train_loader.dataset)
        return 0

    def get_weights(self) -> Dict[str, Any]:
        """获取模型权重"""
        return {
            name: param.data.clone()
            for name, param in self.torch_model.state_dict().items()
        }

    def set_weights(self, weights: Dict[str, Any]) -> bool:
        """设置模型权重"""
        try:
            # 转换为torch tensor
            torch_weights = {}
            for k, v in weights.items():
                if torch.is_tensor(v):
                    torch_weights[k] = v
                else:
                    torch_weights[k] = torch.from_numpy(v)

            self.torch_model.load_state_dict(torch_weights)
            self.logger.debug(f"[{self._node_id}] Model weights updated")
            return True
        except Exception as e:
            self.logger.error(f"Failed to set weights: {e}")
            return False