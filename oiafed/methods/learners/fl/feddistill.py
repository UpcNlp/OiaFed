"""
FedDistill (Federated Knowledge Distillation) 学习器实现

从 methods/learners/fl/feddistill.py 迁移到新架构

基于知识蒸馏的联邦学习，使用全局模型作为教师模型来指导本地训练

核心思想：
- 使用全局模型的软标签（soft labels）来指导本地模型训练
- 结合硬标签（真实标签）和软标签进行训练
- 使用温度缩放的softmax来平滑预测分布
"""
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import copy

from ....core.learner import Learner
from ....core.types import StepMetrics, EpochMetrics, EvalResult
from ....registry import learner


@learner('fl.feddistill', description='FedDistill: Federated Learning with Knowledge Distillation')
class FedDistillLearner(Learner):
    """FedDistill学习器 - 使用知识蒸馏的联邦学习

    配置示例:
    {
        "learner": {
            "name": "fl.feddistill",
            "learning_rate": 0.01,
            "batch_size": 128,
            "local_epochs": 5,
            "optimizer": "SGD",
            "momentum": 0.9,
            "temperature": 3.0,  # 蒸馏温度
            "alpha": 0.5,  # 硬标签损失权重
            "beta": 0.5  # 软标签损失权重
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
        self.optimizer_type = self._config.get('optimizer', 'SGD').upper()

        # FedDistill特有参数
        self.temperature = self._config.get('temperature', 3.0)
        self.alpha = self._config.get('alpha', 0.5)  # 硬标签损失权重
        self.beta = self._config.get('beta', 0.5)   # 软标签损失权重

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 组件占位符
        self.torch_model = None
        self._optimizer = None
        self._criterion = None
        self._train_loader = None
        self._test_loader = None

        # 教师模型（全局模型）
        self.teacher_model = None
        self.round_number = 0

        self.logger.info(
            f"FedDistillLearner {node_id} 初始化完成 "
            f"(lr={self.learning_rate}, T={self.temperature}, alpha={self.alpha}, beta={self.beta})"
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
                momentum=self.momentum
            )
        elif self.optimizer_type == 'ADAM':
            self._optimizer = optim.Adam(
                self.torch_model.parameters(),
                lr=self.learning_rate
            )
        else:
            raise ValueError(f"不支持的优化器类型: {self.optimizer_type}")

        # 创建损失函数
        loss_name = self._config.get('loss', 'CrossEntropyLoss')
        if loss_name == 'CrossEntropyLoss':
            self._criterion = nn.CrossEntropyLoss()
        elif loss_name == 'MSELoss':
            self._criterion = nn.MSELoss()
        else:
            raise ValueError(f"不支持的损失函数: {loss_name}")

        # 创建数据加载器
        train_datasets = self._datasets.get("train", [])
        if train_datasets:
            self._train_loader = DataLoader(
                train_datasets[0],
                batch_size=self.batch_size,
                shuffle=True
            )

        test_datasets = self._datasets.get("test", [])
        if test_datasets:
            self._test_loader = DataLoader(
                test_datasets[0],
                batch_size=self.batch_size,
                shuffle=False
            )

        self.logger.info(f"Setup完成: train_samples={len(train_datasets[0]) if train_datasets else 0}")

    def distillation_loss(self, student_logits, teacher_logits, temperature):
        """
        计算蒸馏损失（KL散度）

        Args:
            student_logits: 学生模型的logits (batch_size, num_classes)
            teacher_logits: 教师模型的logits (batch_size, num_classes)
            temperature: 温度参数

        Returns:
            蒸馏损失值
        """
        # 使用温度缩放的softmax
        student_probs = F.log_softmax(student_logits / temperature, dim=1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=1)

        # KL散度
        kl_loss = F.kl_div(
            student_probs,
            teacher_probs,
            reduction='batchmean'
        )

        # 乘以温度的平方来调整梯度尺度
        return kl_loss * (temperature ** 2)

    async def train_epoch(self, epoch: int) -> EpochMetrics:
        """单轮训练 - FedDistill训练循环"""
        # 设置模型为训练模式
        self.torch_model.train()
        if self.teacher_model is not None:
            self.teacher_model.eval()

        total_loss = 0.0
        total_hard_loss = 0.0
        total_soft_loss = 0.0
        total_correct = 0
        total_samples = 0

        for data, target in self._train_loader:
            data, target = data.to(self.device), target.to(self.device)

            self._optimizer.zero_grad()

            # 学生模型前向传播
            student_logits = self.torch_model(data)

            # 硬标签损失（交叉熵）
            hard_loss = self._criterion(student_logits, target)

            # 软标签损失（蒸馏损失）
            soft_loss = torch.tensor(0.0, device=self.device)
            if self.teacher_model is not None and self.round_number > 1:
                with torch.no_grad():
                    teacher_logits = self.teacher_model(data)
                soft_loss = self.distillation_loss(
                    student_logits,
                    teacher_logits,
                    self.temperature
                )

            # 总损失
            loss = self.alpha * hard_loss + self.beta * soft_loss

            loss.backward()
            self._optimizer.step()

            total_loss += loss.item() * data.size(0)
            total_hard_loss += hard_loss.item() * data.size(0)
            total_soft_loss += soft_loss.item() * data.size(0)

            pred = student_logits.argmax(dim=1, keepdim=True)
            total_correct += pred.eq(target.view_as(pred)).sum().item()
            total_samples += data.size(0)

        avg_loss = total_loss / total_samples
        avg_accuracy = total_correct / total_samples

        self.logger.info(
            f"[{self._node_id}] Epoch {epoch}: "
            f"Loss={avg_loss:.4f} (Hard={total_hard_loss/total_samples:.4f}, "
            f"Soft={total_soft_loss/total_samples:.4f}), Acc={avg_accuracy:.4f}"
        )

        epoch_metrics = EpochMetrics(
            epoch=epoch,
            avg_loss=avg_loss,
            total_samples=total_samples,
            metrics={
                'accuracy': avg_accuracy,
                'hard_loss': total_hard_loss / total_samples,
                'soft_loss': total_soft_loss / total_samples
            }
        )

        # 增加全局 epoch 计数器（用于 MLflow step）
        self._global_epoch_counter += 1

        # 触发 epoch 结束回调
        if self._callbacks:
            await self._callbacks.on_epoch_end(self, epoch, epoch_metrics)

        return epoch_metrics

    async def fit(self, config: Optional[Dict] = None):
        """训练模型"""
        # 增加轮次计数
        self.round_number += 1

        # 调用父类的fit方法
        await super().fit(config)

    async def evaluate_model(self, config: Optional[Dict] = None) -> EvalResult:
        """评估模型"""
        self.torch_model.eval()

        loader = self._test_loader if self._test_loader else self._train_loader

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.torch_model(data)
                loss = self._criterion(output, target)

                total_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                total_correct += pred.eq(target.view_as(pred)).sum().item()
                total_samples += data.size(0)

        return EvalResult(
            num_samples=total_samples,
            metrics={
                'loss': total_loss / total_samples,
                'accuracy': total_correct / total_samples
            }
        )

    async def set_weights(self, weights: Dict[str, Any]):
        """设置模型权重 - 保存教师模型副本"""
        # 转换为torch tensor
        torch_weights = {}
        for k, v in weights.items():
            if torch.is_tensor(v):
                torch_weights[k] = v
            else:
                torch_weights[k] = torch.from_numpy(v)

        # 更新学生模型（当前模型）
        self.torch_model.load_state_dict(torch_weights)

        # 保存教师模型的副本
        self.teacher_model = copy.deepcopy(self.torch_model)
        self.teacher_model.eval()

        self.logger.debug(f"[{self._node_id}] FedDistill: Updated model and saved teacher model")

    async def get_weights(self) -> Dict[str, Any]:
        """获取模型权重"""
        return {name: param.data.clone() for name, param in self.torch_model.state_dict().items()}
