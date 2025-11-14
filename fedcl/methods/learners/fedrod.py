"""
FedRoD (Federated Learning with Robust Disentangled Representation) 学习器实现
fedcl/methods/learners/fedrod.py

论文：On Bridging Generic and Personalized Federated Learning for Image Classification
作者：Hong-You Chen, Wei-Lun Chao
发表：ICLR 2022

FedRoD的核心思想：
- 将模型分为generic feature extractor（共享）和personalized head（个性化）
- 使用balanced softmax来处理类别不平衡问题
- 只有feature extractor参与联邦聚合
"""
from typing import Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from fedcl.api.decorators import learner
from fedcl.learner.base_learner import BaseLearner
from fedcl.types import TrainingResponse
from fedcl.methods.models.base import get_param_count


@learner('FedRoD',
         description='FedRoD: Federated Learning with Robust Disentangled Representation',
         version='1.0',
         author='MOE-FedCL')
class FedRoDLearner(BaseLearner):
    """FedRoD学习器 - 使用平衡softmax的个性化联邦学习

    配置示例:
    {
        "learner": {
            "name": "FedRoD",
            "params": {
                "model": {"name": "SimpleCNN", "params": {"num_classes": 10}},
                "optimizer": {"type": "SGD", "lr": 0.01, "momentum": 0.9},
                "loss": "CrossEntropyLoss",
                "learning_rate": 0.01,
                "batch_size": 128,
                "local_epochs": 5,
                "head_layer_names": ["fc2"],  # 个性化头部层的名称
                "use_balanced_softmax": true,  # 是否使用balanced softmax
                "balance_alpha": 0.5  # balanced softmax的alpha参数
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

        # FedRoD特有参数
        self.head_layer_names = learner_params.get(
            'head_layer_names',
            ['fc2', 'fc', 'classifier', 'head']
        )
        self.use_balanced_softmax = learner_params.get('use_balanced_softmax', True)
        self.balance_alpha = learner_params.get('balance_alpha', 0.5)

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

        # 用于balanced softmax的类别统计
        self.class_counts = None

        self.logger.info(
            f"FedRoDLearner {client_id} 初始化完成 "
            f"(model={self._model_cfg.get('name')}, "
            f"balanced_softmax={self.use_balanced_softmax})"
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

    def is_head_layer(self, param_name: str) -> bool:
        """判断参数是否属于头部层"""
        for layer_name in self.head_layer_names:
            if layer_name in param_name:
                return True
        return False

    def get_generic_parameters(self) -> Dict[str, torch.Tensor]:
        """获取generic feature extractor的参数（用于联邦聚合）"""
        generic_params = {}
        for name, param in self.model.named_parameters():
            if not self.is_head_layer(name):
                generic_params[name] = param.data.cpu().clone()
        return generic_params

    def get_head_parameters(self) -> Dict[str, torch.Tensor]:
        """获取personalized head的参数（不参与聚合）"""
        head_params = {}
        for name, param in self.model.named_parameters():
            if self.is_head_layer(name):
                head_params[name] = param.data.cpu().clone()
        return head_params

    def compute_class_counts(self):
        """计算本地数据集的类别分布"""
        if self.class_counts is not None:
            return

        num_classes = self._model_cfg.get('params', {}).get('num_classes', 10)
        counts = torch.zeros(num_classes, dtype=torch.long)

        for _, target in self.train_loader:
            for t in target:
                counts[t.item()] += 1

        self.class_counts = counts.float().to(self.device)
        self.logger.info(f"  [{self.client_id}] Class distribution: {counts.tolist()}")

    def balanced_softmax_loss(self, logits, targets):
        """
        计算balanced softmax损失

        Args:
            logits: 模型输出 (batch_size, num_classes)
            targets: 真实标签 (batch_size,)

        Returns:
            balanced softmax loss
        """
        if not self.use_balanced_softmax or self.class_counts is None:
            return self.criterion(logits, targets)

        # Balanced softmax: 调整logits by class frequency
        # adjusted_logits = logits + alpha * log(class_counts)
        log_counts = torch.log(self.class_counts + 1e-9)
        adjusted_logits = logits + self.balance_alpha * log_counts.unsqueeze(0)

        loss = F.cross_entropy(adjusted_logits, targets)
        return loss

    async def train(self, params: Dict[str, Any]) -> TrainingResponse:
        """训练方法 - 标准训练，但只上传generic parameters"""
        num_epochs = params.get("num_epochs", self._epochs)
        round_number = params.get("round_number", 1)

        self.logger.info(
            f"  [{self.client_id}] Round {round_number}, "
            f"FedRoD Training {num_epochs} epochs..."
        )

        # 计算类别分布（用于balanced softmax）
        if self.use_balanced_softmax:
            self.compute_class_counts()

        self.model.train()

        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0

            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data)

                # 使用balanced softmax loss
                loss = self.balanced_softmax_loss(output, target)

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                epoch_correct += pred.eq(target.view_as(pred)).sum().item()
                epoch_samples += data.size(0)

            avg_epoch_loss = epoch_loss / epoch_samples
            epoch_accuracy = epoch_correct / epoch_samples
            self.logger.info(
                f"    [{self.client_id}] Epoch {epoch+1}: "
                f"Loss={avg_epoch_loss:.4f}, Acc={epoch_accuracy:.4f}"
            )

            total_loss += epoch_loss
            correct_predictions += epoch_correct
            total_samples += epoch_samples

        # 计算平均值
        avg_loss = total_loss / total_samples
        accuracy = correct_predictions / total_samples

        # 只上传generic parameters
        generic_weights = self.get_generic_parameters()
        head_weights = self.get_head_parameters()

        self.logger.info(
            f"  [{self.client_id}] FedRoD: "
            f"Generic params: {len(generic_weights)}, "
            f"Head params: {len(head_weights)}"
        )

        # 创建训练响应
        response = TrainingResponse(
            request_id="",
            client_id=self.client_id,
            success=True,
            result={
                "epochs_completed": num_epochs,
                "loss": avg_loss,
                "accuracy": accuracy,
                "samples_used": total_samples,
                "model_weights": generic_weights  # 只上传generic parameters
            },
            execution_time=0.0
        )

        self.logger.info(
            f"  [{self.client_id}] Round {round_number} completed: "
            f"Loss={avg_loss:.4f}, Acc={accuracy:.4f}"
        )
        return response

    async def set_model(self, model_data: Dict[str, Any]) -> bool:
        """设置模型数据 - 只更新generic parameters"""
        try:
            if "parameters" in model_data and "weights" in model_data["parameters"]:
                weights = model_data["parameters"]["weights"]

                # 只更新generic parameters，保留personalized head
                state_dict = self.model.state_dict()
                updated_count = 0

                for name, value in weights.items():
                    if name in state_dict and not self.is_head_layer(name):
                        if not isinstance(value, torch.Tensor):
                            value = torch.from_numpy(value)
                        state_dict[name] = value.to(self.device)
                        updated_count += 1

                self.model.load_state_dict(state_dict, strict=True)

                self.logger.info(
                    f"  [{self.client_id}] FedRoD: Updated {updated_count} generic parameters, "
                    f"kept personalized head unchanged"
                )
                return True
        except Exception as e:
            self.logger.exception(f"  [{self.client_id}] Failed to set model: {e}")
        return False

    async def get_model(self) -> Dict[str, Any]:
        """获取模型数据 - 返回generic parameters"""
        dataset = self.dataset
        return {
            "model_type": self._model_cfg['name'],
            "parameters": {"weights": self.get_generic_parameters()},
            "metadata": {
                "client_id": self.client_id,
                "samples": len(dataset),
                "param_count": get_param_count(self.model),
                "generic_param_count": len(self.get_generic_parameters()),
                "head_param_count": len(self.get_head_parameters())
            }
        }

    async def evaluate(self, model_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """评估方法"""
        self.model.eval()

        if model_data and "model_weights" in model_data:
            await self.set_model({"parameters": {"weights": model_data["model_weights"]}})

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
