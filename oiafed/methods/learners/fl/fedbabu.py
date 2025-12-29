"""
FedBABU (Federated Learning via Body and Bottom-up) 学习器实现

从 methods/learners/fl/fedbabu.py 迁移到新架构

论文：Towards Fair Federated Learning with Zero-Shot Data Augmentation
作者：Junyuan Hong et al.
发表：CVPR 2021 Workshop

FedBABU的核心思想：
- 将模型分为body (feature extractor)和head (classifier)
- 训练过程：前面的epoch正常训练，最后几个epoch冻结body只训练head
- 只有body参与联邦聚合，head是个性化的
"""
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ....core.learner import Learner
from ....core.types import StepMetrics, EpochMetrics, EvalResult
from ....registry import learner


@learner('fl.fedbabu', description='FedBABU: Federated Learning via Body and Bottom-up')
class FedBABULearner(Learner):
    """FedBABU学习器 - 冻结body，微调head

    配置示例:
    {
        "learner": {
            "name": "fl.fedbabu",
            "learning_rate": 0.01,
            "batch_size": 128,
            "local_epochs": 5,
            "finetune_epochs": 1,  # 最后几个epoch只训练head
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

        # FedBABU特有参数
        self.finetune_epochs = self._config.get('finetune_epochs', 1)
        self.head_layer_names = self._config.get(
            'head_layer_names',
            ['fc2', 'fc', 'classifier', 'head']
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 组件占位符
        self.torch_model = None
        self._optimizer = None
        self._head_optimizer = None
        self._criterion = None
        self._train_loader = None
        self._test_loader = None

        self.logger.info(
            f"FedBABULearner {node_id} 初始化完成 "
            f"(lr={self.learning_rate}, epochs={self.local_epochs}, finetune_epochs={self.finetune_epochs})"
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

    def is_head_layer(self, param_name: str) -> bool:
        """判断参数是否属于头部层"""
        for layer_name in self.head_layer_names:
            if layer_name in param_name:
                return True
        return False

    def get_body_parameters(self) -> Dict[str, torch.Tensor]:
        """获取body (feature extractor)的参数（用于联邦聚合）"""
        body_params = {}
        for name, param in self.torch_model.named_parameters():
            if not self.is_head_layer(name):
                body_params[name] = param.data.cpu().clone()
        return body_params

    def freeze_body(self):
        """冻结body layers"""
        for name, param in self.torch_model.named_parameters():
            if not self.is_head_layer(name):
                param.requires_grad = False

    def unfreeze_body(self):
        """解冻body layers"""
        for name, param in self.torch_model.named_parameters():
            if not self.is_head_layer(name):
                param.requires_grad = True

    async def train_epoch(self, epoch: int) -> EpochMetrics:
        """单轮训练 - FedBABU训练循环"""
        self.torch_model.train()

        # 判断是否是finetune阶段
        is_finetune = epoch >= (self.local_epochs - self.finetune_epochs)

        if is_finetune and epoch == (self.local_epochs - self.finetune_epochs):
            # 第一次进入finetune阶段，冻结body
            self.logger.info(f"[{self._node_id}] Entering finetune phase - freezing body")
            self.freeze_body()

            # 创建只包含head参数的优化器
            head_params = [p for n, p in self.torch_model.named_parameters() if self.is_head_layer(n)]
            if self.optimizer_type == 'SGD':
                self._head_optimizer = optim.SGD(head_params, lr=self.learning_rate, momentum=self.momentum)
            elif self.optimizer_type == 'ADAM':
                self._head_optimizer = optim.Adam(head_params, lr=self.learning_rate)

        # 选择优化器
        optimizer = self._head_optimizer if is_finetune else self._optimizer

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for data, target in self._train_loader:
            data, target = data.to(self.device), target.to(self.device)

            optimizer.zero_grad()
            output = self.torch_model(data)
            loss = self._criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            total_correct += pred.eq(target.view_as(pred)).sum().item()
            total_samples += data.size(0)

        avg_loss = total_loss / total_samples
        avg_accuracy = total_correct / total_samples

        phase = "Finetune" if is_finetune else "Full"
        self.logger.info(
            f"[{self._node_id}] Epoch {epoch} ({phase}): "
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
        """训练模型 - 重写以确保解冻body"""
        # 确保body未冻结
        self.unfreeze_body()

        # 调用父类的fit方法
        await super().fit(config)

        # 训练后解冻body供下一轮使用
        self.unfreeze_body()

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
        """获取模型权重 - 只返回body parameters"""
        return self.get_body_parameters()

    async def set_weights(self, weights: Dict[str, Any]):
        """设置模型权重 - 只更新body parameters"""
        state_dict = self.torch_model.state_dict()
        updated_count = 0

        for name, value in weights.items():
            if name in state_dict and not self.is_head_layer(name):
                if not isinstance(value, torch.Tensor):
                    value = torch.from_numpy(value)
                state_dict[name] = value.to(self.device)
                updated_count += 1

        self.torch_model.load_state_dict(state_dict, strict=True)

        self.logger.debug(
            f"[{self._node_id}] FedBABU: Updated {updated_count} body parameters, "
            f"kept personalized head unchanged"
        )
