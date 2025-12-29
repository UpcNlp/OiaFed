"""
FedRep (Federated Learning via Representation Learning) 学习器实现

从 methods/learners/fl/fedrep.py 迁移到新架构

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
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ....core.learner import Learner
from ....core.types import StepMetrics, EpochMetrics, EvalResult
from ....registry import learner


@learner('fl.fedrep', description='FedRep: Federated Learning via Representation Learning')
class FedRepLearner(Learner):
    """FedRep学习器 - 两阶段训练：表示学习 + 头部微调

    配置示例:
    {
        "learner": {
            "name": "fl.fedrep",
            "learning_rate": 0.01,
            "batch_size": 128,
            "local_epochs": 5,
            "head_epochs": 1,  # 头部训练的epoch数
            "optimizer": "SGD",
            "momentum": 0.9,
            "head_layer_names": ["fc2"]  # 头部层的名称
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

        # FedRep特有参数
        self.head_epochs = self._config.get('head_epochs', 1)
        self.head_layer_names = self._config.get(
            'head_layer_names',
            ['fc2', 'fc', 'classifier', 'head']
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 组件占位符
        self.torch_model = None
        self._optimizer_rep = None
        self._optimizer_head = None
        self._criterion = None
        self._train_loader = None
        self._test_loader = None

        self.logger.info(
            f"FedRepLearner {node_id} 初始化完成 "
            f"(lr={self.learning_rate}, rep_epochs={self.local_epochs}, head_epochs={self.head_epochs})"
        )

    async def setup(self, config: Dict) -> None:
        """初始化训练环境"""
        # 获取PyTorch模型
        self.torch_model = self._model.get_model()
        self.torch_model.to(self.device)

        # 创建两个优化器：一个for representation，一个for head
        self.create_optimizers()

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

    def is_head_layer(self, param_name: str) -> bool:
        """判断参数是否属于头部层"""
        for layer_name in self.head_layer_names:
            if layer_name in param_name:
                return True
        return False

    def get_representation_parameters(self) -> Dict[str, torch.Tensor]:
        """获取representation layers的参数"""
        rep_params = {}
        for name, param in self.torch_model.named_parameters():
            if not self.is_head_layer(name):
                rep_params[name] = param.data.cpu().clone()
        return rep_params

    def create_optimizers(self):
        """创建两个优化器：一个for representation，一个for head"""
        # Representation optimizer
        rep_params = [p for n, p in self.torch_model.named_parameters() if not self.is_head_layer(n)]
        # Head optimizer
        head_params = [p for n, p in self.torch_model.named_parameters() if self.is_head_layer(n)]

        if self.optimizer_type == 'SGD':
            self._optimizer_rep = optim.SGD(rep_params, lr=self.learning_rate, momentum=self.momentum)
            self._optimizer_head = optim.SGD(head_params, lr=self.learning_rate, momentum=self.momentum)
        elif self.optimizer_type == 'ADAM':
            self._optimizer_rep = optim.Adam(rep_params, lr=self.learning_rate)
            self._optimizer_head = optim.Adam(head_params, lr=self.learning_rate)
        else:
            raise ValueError(f"不支持的优化器类型: {self.optimizer_type}")

    async def train_epoch(self, epoch: int) -> EpochMetrics:
        """单轮训练 - FedRep两阶段训练"""
        self.torch_model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        # 阶段1: 训练representation layers
        if epoch < self.local_epochs:
            self.logger.info(f"[{self._node_id}] Epoch {epoch}: Training representation layers")
            for data, target in self._train_loader:
                data, target = data.to(self.device), target.to(self.device)

                self._optimizer_rep.zero_grad()
                output = self.torch_model(data)
                loss = self._criterion(output, target)
                loss.backward()
                self._optimizer_rep.step()

                total_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                total_correct += pred.eq(target.view_as(pred)).sum().item()
                total_samples += data.size(0)
        else:
            # 阶段2: 训练head layers
            self.logger.info(f"[{self._node_id}] Epoch {epoch}: Training head layers")
            for data, target in self._train_loader:
                data, target = data.to(self.device), target.to(self.device)

                self._optimizer_head.zero_grad()
                output = self.torch_model(data)
                loss = self._criterion(output, target)
                loss.backward()
                self._optimizer_head.step()

                total_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                total_correct += pred.eq(target.view_as(pred)).sum().item()
                total_samples += data.size(0)

        avg_loss = total_loss / total_samples
        avg_accuracy = total_correct / total_samples

        self.logger.info(
            f"[{self._node_id}] Epoch {epoch}: "
            f"loss={avg_loss:.4f}, acc={avg_accuracy:.4f}"
        )

        epoch_metrics = EpochMetrics(
            epoch=epoch,
            avg_loss=avg_loss,
            total_samples=total_samples,
            metrics={'accuracy': avg_accuracy}
        )

        # 增加全局 epoch 计数器（用于 MLflow step）
        self._global_epoch_counter += 1

        # 触发 epoch 结束回调
        if self._callbacks:
            await self._callbacks.on_epoch_end(self, epoch, epoch_metrics)

        return epoch_metrics

    async def fit(self, config: Optional[Dict] = None):
        """训练模型 - 重写以实现两阶段训练"""
        total_epochs = self.local_epochs + self.head_epochs

        for epoch in range(total_epochs):
            epoch_metrics = await self.train_epoch(epoch)

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

    async def get_weights(self) -> Dict[str, Any]:
        """获取模型权重 - 只返回representation parameters"""
        return self.get_representation_parameters()

    async def set_weights(self, weights: Dict[str, Any]):
        """设置模型权重 - 只更新representation parameters"""
        state_dict = self.torch_model.state_dict()

        for name, value in weights.items():
            if name in state_dict and not self.is_head_layer(name):
                if not isinstance(value, torch.Tensor):
                    value = torch.from_numpy(value)
                state_dict[name] = value.to(self.device)

        self.torch_model.load_state_dict(state_dict, strict=True)
        self.logger.debug(f"[{self._node_id}] FedRep: Updated representation parameters")
