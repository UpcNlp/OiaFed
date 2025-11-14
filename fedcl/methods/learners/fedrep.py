"""
FedRep (Federated Learning via Representation Learning) 学习器实现
fedcl/methods/learners/fedrep.py

论文：Exploiting Shared Representations for Personalized Federated Learning
作者：Liam Collins et al.
发表：ICML 2021

FedRep的核心思想：
- 将模型分为representation layers（共享）和head layers（个性化）
- 训练分为两个阶段：
  1. 只训练representation layers（多个epoch）
  2. 只训练head layers（少量epoch）
- 只有representation layers参与联邦聚合
"""
from typing import Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from fedcl.api.decorators import learner
from fedcl.learner.base_learner import BaseLearner
from fedcl.types import TrainingResponse
from fedcl.methods.models.base import get_param_count


@learner('FedRep',
         description='FedRep: Federated Learning via Representation Learning',
         version='1.0',
         author='MOE-FedCL')
class FedRepLearner(BaseLearner):
    """FedRep学习器 - 两阶段训练：表示学习 + 头部微调

    配置示例:
    {
        "learner": {
            "name": "FedRep",
            "params": {
                "model": {"name": "SimpleCNN", "params": {"num_classes": 10}},
                "optimizer": {"type": "SGD", "lr": 0.01, "momentum": 0.9},
                "loss": "CrossEntropyLoss",
                "learning_rate": 0.01,
                "batch_size": 128,
                "local_epochs": 5,
                "head_epochs": 1,  # 头部训练的epoch数
                "head_layer_names": ["fc2"]  # 头部层的名称
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

        # FedRep特有参数
        self.head_epochs = learner_params.get('head_epochs', 1)
        self.head_layer_names = learner_params.get(
            'head_layer_names',
            ['fc2', 'fc', 'classifier', 'head']
        )

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
        self._optimizer_rep = None
        self._optimizer_head = None
        self._criterion = None
        self._train_loader = None

        self.logger.info(
            f"FedRepLearner {client_id} 初始化完成 "
            f"(model={self._model_cfg.get('name')}, "
            f"rep_epochs={self._epochs}, head_epochs={self.head_epochs})"
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

    def get_representation_parameters(self) -> Dict[str, torch.Tensor]:
        """获取representation layers的参数"""
        rep_params = {}
        for name, param in self.model.named_parameters():
            if not self.is_head_layer(name):
                rep_params[name] = param.data.cpu().clone()
        return rep_params

    def create_optimizers(self):
        """创建两个优化器：一个for representation，一个for head"""
        opt_type = self._optimizer_cfg.get('type', 'SGD').upper()
        lr = self._optimizer_cfg.get('lr', self._lr)

        # Representation optimizer
        rep_params = [p for n, p in self.model.named_parameters() if not self.is_head_layer(n)]
        # Head optimizer
        head_params = [p for n, p in self.model.named_parameters() if self.is_head_layer(n)]

        if opt_type == 'SGD':
            momentum = self._optimizer_cfg.get('momentum', 0.9)
            self._optimizer_rep = optim.SGD(rep_params, lr=lr, momentum=momentum)
            self._optimizer_head = optim.SGD(head_params, lr=lr, momentum=momentum)
        elif opt_type == 'ADAM':
            self._optimizer_rep = optim.Adam(rep_params, lr=lr)
            self._optimizer_head = optim.Adam(head_params, lr=lr)
        else:
            raise ValueError(f"不支持的优化器类型: {opt_type}")

    async def train(self, params: Dict[str, Any]) -> TrainingResponse:
        """训练方法 - 两阶段训练"""
        num_epochs = params.get("num_epochs", self._epochs)
        round_number = params.get("round_number", 1)

        self.logger.info(
            f"  [{self.client_id}] Round {round_number}, FedRep Training "
            f"(rep_epochs={num_epochs}, head_epochs={self.head_epochs})..."
        )

        # 创建优化器
        if self._optimizer_rep is None or self._optimizer_head is None:
            self.create_optimizers()

        self.model.train()

        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        # 阶段1: 训练representation layers
        self.logger.info(f"    [{self.client_id}] Phase 1: Training representation layers")
        for epoch in range(num_epochs):
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)

                self._optimizer_rep.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self._optimizer_rep.step()

                total_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                correct_predictions += pred.eq(target.view_as(pred)).sum().item()
                total_samples += data.size(0)

        # 阶段2: 训练head layers
        self.logger.info(f"    [{self.client_id}] Phase 2: Training head layers")
        for epoch in range(self.head_epochs):
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)

                self._optimizer_head.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self._optimizer_head.step()

                total_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                correct_predictions += pred.eq(target.view_as(pred)).sum().item()
                total_samples += data.size(0)

        # 计算平均值
        avg_loss = total_loss / total_samples
        accuracy = correct_predictions / total_samples

        # 只上传representation parameters
        rep_weights = self.get_representation_parameters()

        response = TrainingResponse(
            request_id="",
            client_id=self.client_id,
            success=True,
            result={
                "epochs_completed": num_epochs + self.head_epochs,
                "loss": avg_loss,
                "accuracy": accuracy,
                "samples_used": total_samples,
                "model_weights": rep_weights
            },
            execution_time=0.0
        )

        self.logger.info(
            f"  [{self.client_id}] Round {round_number} completed: "
            f"Loss={avg_loss:.4f}, Acc={accuracy:.4f}"
        )
        return response

    async def set_model(self, model_data: Dict[str, Any]) -> bool:
        """设置模型 - 只更新representation parameters"""
        try:
            if "parameters" in model_data and "weights" in model_data["parameters"]:
                weights = model_data["parameters"]["weights"]
                state_dict = self.model.state_dict()

                for name, value in weights.items():
                    if name in state_dict and not self.is_head_layer(name):
                        if not isinstance(value, torch.Tensor):
                            value = torch.from_numpy(value)
                        state_dict[name] = value.to(self.device)

                self.model.load_state_dict(state_dict, strict=True)
                return True
        except Exception as e:
            self.logger.exception(f"  [{self.client_id}] Failed to set model: {e}")
        return False

    async def get_model(self) -> Dict[str, Any]:
        dataset = self.dataset
        return {
            "model_type": self._model_cfg['name'],
            "parameters": {"weights": self.get_representation_parameters()},
            "metadata": {
                "client_id": self.client_id,
                "samples": len(dataset),
                "param_count": get_param_count(self.model)
            }
        }

    async def evaluate(self, model_data: Dict[str, Any] = None) -> Dict[str, Any]:
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

        return {"accuracy": correct / total, "loss": test_loss / total, "samples": total}

    def get_data_statistics(self) -> Dict[str, Any]:
        dataset = self.dataset
        stats = {"total_samples": len(dataset)}
        if hasattr(dataset, 'get_statistics'):
            stats.update(dataset.get_statistics())
        return stats

    async def get_local_model(self) -> Dict[str, Any]:
        return await self.get_model()

    async def set_local_model(self, model_data: Dict[str, Any]) -> bool:
        return await self.set_model(model_data)
