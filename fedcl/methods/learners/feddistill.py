"""
FedDistill (Federated Knowledge Distillation) 学习器实现
fedcl/methods/learners/feddistill.py

基于知识蒸馏的联邦学习，使用全局模型作为教师模型来指导本地训练

核心思想：
- 使用全局模型的软标签（soft labels）来指导本地模型训练
- 结合硬标签（真实标签）和软标签进行训练
- 使用温度缩放的softmax来平滑预测分布
"""
from typing import Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import copy

from fedcl.api.decorators import learner
from fedcl.learner.base_learner import BaseLearner
from fedcl.types import TrainingResponse
from fedcl.methods.models.base import get_weights_as_dict, set_weights_from_dict, get_param_count


@learner('FedDistill',
         description='FedDistill: Federated Learning with Knowledge Distillation',
         version='1.0',
         author='MOE-FedCL')
class FedDistillLearner(BaseLearner):
    """FedDistill学习器 - 使用知识蒸馏的联邦学习

    配置示例:
    {
        "learner": {
            "name": "FedDistill",
            "params": {
                "model": {"name": "SimpleCNN", "params": {"num_classes": 10}},
                "optimizer": {"type": "SGD", "lr": 0.01, "momentum": 0.9},
                "loss": "CrossEntropyLoss",
                "learning_rate": 0.01,
                "batch_size": 128,
                "local_epochs": 5,
                "temperature": 3.0,  # 蒸馏温度
                "alpha": 0.5,  # 硬标签损失权重
                "beta": 0.5  # 软标签损失权重
            }
        },
        "dataset": {...}
    }
    """

    def __init__(self, client_id: str, config: Dict[str, Any] = None, lazy_init: bool = True):
        learner_params = (config or {}).get('learner', {}).get('params', {})

        self._model_cfg = learner_params.get('model', {})
        if not self._model_cfg.get('name'):
            raise ValueError("必须在配置中指定模型名称")

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

        # FedDistill特有参数
        self.temperature = learner_params.get('temperature', 3.0)
        self.alpha = learner_params.get('alpha', 0.5)  # 硬标签损失权重
        self.beta = learner_params.get('beta', 0.5)   # 软标签损失权重

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

        self._model = None
        self._optimizer = None
        self._criterion = None
        self._train_loader = None

        # 教师模型（全局模型）
        self.teacher_model = None

        self.logger.info(
            f"FedDistillLearner {client_id} 初始化完成 "
            f"(model={self._model_cfg.get('name')}, "
            f"T={self.temperature}, alpha={self.alpha}, beta={self.beta})"
        )

    @property
    def model(self):
        if self._model is None:
            from fedcl.api.registry import registry
            model_name = self._model_cfg['name']
            model_params = self._model_cfg.get('params', {})

            model_class = registry.get_model(model_name)
            self._model = model_class(**model_params).to(self.device)
        return self._model

    @property
    def optimizer(self):
        if self._optimizer is None:
            opt_type = self._optimizer_cfg.get('type', 'SGD').upper()
            lr = self._optimizer_cfg.get('lr', self._lr)

            if opt_type == 'SGD':
                self._optimizer = optim.SGD(
                    self.model.parameters(),
                    lr=lr,
                    momentum=self._optimizer_cfg.get('momentum', 0.9)
                )
            elif opt_type == 'ADAM':
                self._optimizer = optim.Adam(self.model.parameters(), lr=lr)
            else:
                raise ValueError(f"不支持的优化器类型: {opt_type}")
        return self._optimizer

    @property
    def criterion(self):
        if self._criterion is None:
            if self._loss_cfg == 'CrossEntropyLoss':
                self._criterion = nn.CrossEntropyLoss()
            elif self._loss_cfg == 'MSELoss':
                self._criterion = nn.MSELoss()
            else:
                raise ValueError(f"不支持的损失函数: {self._loss_cfg}")
        return self._criterion

    @property
    def train_loader(self):
        if self._train_loader is None:
            dataset = self.dataset
            self._train_loader = DataLoader(
                dataset,
                batch_size=self._bs,
                shuffle=True
            )
        return self._train_loader

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

    async def train(self, params: Dict[str, Any]) -> TrainingResponse:
        """训练方法 - FedDistill训练循环"""
        num_epochs = params.get("num_epochs", self._epochs)
        round_number = params.get("round_number", 1)

        self.logger.info(
            f"  [{self.client_id}] Round {round_number}, "
            f"FedDistill Training {num_epochs} epochs (T={self.temperature})..."
        )

        # 设置模型为训练模式
        self.model.train()
        if self.teacher_model is not None:
            self.teacher_model.eval()

        total_loss = 0.0
        total_hard_loss = 0.0
        total_soft_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_hard_loss = 0.0
            epoch_soft_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0

            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()

                # 学生模型前向传播
                student_logits = self.model(data)

                # 硬标签损失（交叉熵）
                hard_loss = self.criterion(student_logits, target)

                # 软标签损失（蒸馏损失）
                soft_loss = torch.tensor(0.0, device=self.device)
                if self.teacher_model is not None and round_number > 1:
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
                self.optimizer.step()

                epoch_loss += loss.item() * data.size(0)
                epoch_hard_loss += hard_loss.item() * data.size(0)
                epoch_soft_loss += soft_loss.item() * data.size(0)

                pred = student_logits.argmax(dim=1, keepdim=True)
                epoch_correct += pred.eq(target.view_as(pred)).sum().item()
                epoch_samples += data.size(0)

            avg_epoch_loss = epoch_loss / epoch_samples
            epoch_accuracy = epoch_correct / epoch_samples
            self.logger.info(
                f"    [{self.client_id}] Epoch {epoch+1}: "
                f"Loss={avg_epoch_loss:.4f} (Hard={epoch_hard_loss/epoch_samples:.4f}, "
                f"Soft={epoch_soft_loss/epoch_samples:.4f}), Acc={epoch_accuracy:.4f}"
            )

            total_loss += epoch_loss
            total_hard_loss += epoch_hard_loss
            total_soft_loss += epoch_soft_loss
            correct_predictions += epoch_correct
            total_samples += epoch_samples

        # 计算平均值
        avg_loss = total_loss / total_samples
        accuracy = correct_predictions / total_samples

        # 获取模型权重
        model_weights = get_weights_as_dict(self.model)

        # 创建训练响应
        response = TrainingResponse(
            request_id="",
            client_id=self.client_id,
            success=True,
            result={
                "epochs_completed": num_epochs,
                "loss": avg_loss,
                "hard_loss": total_hard_loss / total_samples,
                "soft_loss": total_soft_loss / total_samples,
                "accuracy": accuracy,
                "samples_used": total_samples,
                "model_weights": model_weights
            },
            execution_time=0.0
        )

        self.logger.info(
            f"  [{self.client_id}] Round {round_number} completed: "
            f"Loss={avg_loss:.4f}, Acc={accuracy:.4f}"
        )
        return response

    async def set_model(self, model_data: Dict[str, Any]) -> bool:
        """设置模型数据 - 保存教师模型副本"""
        try:
            if "parameters" in model_data and "weights" in model_data["parameters"]:
                weights = model_data["parameters"]["weights"]
                torch_weights = {}
                for k, v in weights.items():
                    if torch.is_tensor(v):
                        torch_weights[k] = v
                    else:
                        torch_weights[k] = torch.from_numpy(v)

                # 更新学生模型（当前模型）
                set_weights_from_dict(self.model, torch_weights)

                # 保存教师模型的副本
                self.teacher_model = copy.deepcopy(self.model)
                self.teacher_model.eval()

                self.logger.info(f"  [{self.client_id}] Updated model and saved teacher model")
                return True
        except Exception as e:
            self.logger.exception(f"  [{self.client_id}] Failed to set model: {e}")
        return False

    async def get_model(self) -> Dict[str, Any]:
        """获取模型数据"""
        dataset = self.dataset
        return {
            "model_type": self._model_cfg['name'],
            "parameters": {"weights": get_weights_as_dict(self.model)},
            "metadata": {
                "client_id": self.client_id,
                "samples": len(dataset),
                "param_count": get_param_count(self.model)
            }
        }

    async def evaluate(self, model_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """评估方法"""
        self.model.eval()

        if model_data and "model_weights" in model_data:
            weights = model_data["model_weights"]
            torch_weights = {}
            for k, v in weights.items():
                if isinstance(v, torch.Tensor):
                    torch_weights[k] = v
                else:
                    torch_weights[k] = torch.from_numpy(v)
            set_weights_from_dict(self.model, torch_weights)

        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += data.size(0)

        return {
            "accuracy": correct / total,
            "loss": test_loss / total,
            "samples": total
        }

    def get_data_statistics(self) -> Dict[str, Any]:
        """获取数据统计"""
        dataset = self.dataset
        stats = {"total_samples": len(dataset)}
        if hasattr(dataset, 'get_statistics'):
            stats.update(dataset.get_statistics())
        return stats

    async def get_local_model(self) -> Dict[str, Any]:
        return await self.get_model()

    async def set_local_model(self, model_data: Dict[str, Any]) -> bool:
        return await self.set_model(model_data)
