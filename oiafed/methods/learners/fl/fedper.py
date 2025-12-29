"""
FedPer (Federated Learning with Personalization Layers) 学习器实现

从 methods/learners/fl/fedper.py 迁移到新架构

论文：Federated Learning with Personalization Layers
作者：Manoj Ghuhan Arivazhagan et al.
发表：arXiv 2019

FedPer的核心思想：
- 将模型分为base layers（共享的特征提取器）和personalization layers（个性化分类器）
- 只有base layers参与联邦聚合
- 每个客户端保留自己的personalization layers
"""
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ....core.learner import Learner
from ....core.types import StepMetrics, EpochMetrics, EvalResult
from ....registry import learner


@learner('fl.fedper', description='FedPer: Federated Learning with Personalization Layers')
class FedPerLearner(Learner):
    """FedPer学习器 - 分离共享层和个性化层

    配置示例:
    {
        "learner": {
            "name": "fl.fedper",
            "learning_rate": 0.01,
            "batch_size": 128,
            "local_epochs": 5,
            "optimizer": "SGD",
            "momentum": 0.9,
            "personalization_layer_names": ["fc2"]  # 个性化层的名称
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

        # 保存datasets
        self._datasets = datasets or {}

        # 从配置提取参数
        self.learning_rate = self._config.get('learning_rate', 0.01)
        self.batch_size = self._config.get('batch_size', 32)
        self.local_epochs = self._config.get('local_epochs', 5)
        self.momentum = self._config.get('momentum', 0.9)
        self.optimizer_type = self._config.get('optimizer', 'SGD').upper()

        # FedPer特有参数：个性化层的名称
        self.personalization_layer_names = self._config.get(
            'personalization_layer_names',
            ['fc', 'fc2', 'classifier', 'head']  # 默认值
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 组件占位符
        self.torch_model = None
        self._optimizer = None
        self._criterion = None
        self._train_loader = None
        self._test_loader = None

        self.logger.info(
            f"FedPerLearner {node_id} 初始化完成 "
            f"(lr={self.learning_rate}, personalization_layers={self.personalization_layer_names})"
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

        self.logger.info(
            f"Setup完成: train_samples={len(train_datasets[0]) if train_datasets else 0}, "
            f"test_samples={len(test_datasets[0]) if test_datasets else 0}"
        )

    def is_personalization_layer(self, param_name: str) -> bool:
        """判断参数是否属于个性化层"""
        for layer_name in self.personalization_layer_names:
            if layer_name in param_name:
                return True
        return False

    def get_base_parameters(self) -> Dict[str, torch.Tensor]:
        """获取base layers的参数（用于联邦聚合）"""
        base_params = {}
        for name, param in self.torch_model.named_parameters():
            if not self.is_personalization_layer(name):
                base_params[name] = param.data.cpu().clone()
        return base_params

    def get_personalization_parameters(self) -> Dict[str, torch.Tensor]:
        """获取personalization layers的参数（不参与聚合）"""
        personal_params = {}
        for name, param in self.torch_model.named_parameters():
            if self.is_personalization_layer(name):
                personal_params[name] = param.data.cpu().clone()
        return personal_params

    async def train_step(self, batch, step: int) -> StepMetrics:
        """单步训练"""
        data, target = batch
        data, target = data.to(self.device), target.to(self.device)

        self._optimizer.zero_grad()
        output = self.torch_model(data)
        loss = self._criterion(output, target)
        loss.backward()
        self._optimizer.step()

        # 计算准确率
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        accuracy = correct / data.size(0)

        return StepMetrics(
            loss=loss.item(),
            batch_size=data.size(0),
            metrics={'accuracy': accuracy}
        )

    async def train_epoch(self, epoch: int) -> EpochMetrics:
        """单轮训练"""
        self.torch_model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for step, batch in enumerate(self._train_loader):
            step_metrics = await self.train_step(batch, step)

            total_loss += step_metrics.loss * step_metrics.batch_size
            total_correct += step_metrics.metrics.get('accuracy', 0) * step_metrics.batch_size
            total_samples += step_metrics.batch_size

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

    async def evaluate_model(self, config: Optional[Dict] = None) -> EvalResult:
        """评估模型"""
        self.torch_model.eval()

        # 使用测试数据或训练数据评估
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
        """获取模型权重 - 只返回base parameters"""
        base_weights = self.get_base_parameters()
        personal_weights = self.get_personalization_parameters()

        self.logger.debug(
            f"[{self._node_id}] FedPer: "
            f"Base params: {len(base_weights)}, "
            f"Personal params: {len(personal_weights)}"
        )

        return base_weights

    async def set_weights(self, weights: Dict[str, Any]):
        """设置模型权重 - 只更新base parameters"""
        # 只更新base parameters，保留personalization parameters
        state_dict = self.torch_model.state_dict()
        updated_count = 0

        for name, value in weights.items():
            if name in state_dict and not self.is_personalization_layer(name):
                if not isinstance(value, torch.Tensor):
                    value = torch.from_numpy(value)
                state_dict[name] = value.to(self.device)
                updated_count += 1

        self.torch_model.load_state_dict(state_dict, strict=True)

        self.logger.debug(
            f"[{self._node_id}] FedPer: Updated {updated_count} base parameters, "
            f"kept personalization layers unchanged"
        )
