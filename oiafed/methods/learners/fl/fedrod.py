"""
FedRoD (Federated Learning with Robust Disentangled Representation) 学习器实现

从 methods/learners/fl/fedrod.py 迁移到新架构

论文：On Bridging Generic and Personalized Federated Learning for Image Classification
作者：Hong-You Chen, Wei-Lun Chao
发表：ICLR 2022

FedRoD的核心思想：
- 将模型分为generic feature extractor（共享）和personalized head（个性化）
- 使用balanced softmax来处理类别不平衡问题
- 只有feature extractor参与联邦聚合
"""
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ....core.learner import Learner
from ....core.types import StepMetrics, EpochMetrics, EvalResult
from ....registry import learner


@learner('fl.fedrod', description='FedRoD: Federated Learning with Robust Disentangled Representation')
class FedRoDLearner(Learner):
    """FedRoD学习器 - 使用平衡softmax的个性化联邦学习

    配置示例:
    {
        "learner": {
            "name": "fl.fedrod",
            "learning_rate": 0.01,
            "batch_size": 128,
            "local_epochs": 5,
            "optimizer": "SGD",
            "momentum": 0.9,
            "head_layer_names": ["fc2"],  # 个性化头部层的名称
            "use_balanced_softmax": true,  # 是否使用balanced softmax
            "balance_alpha": 0.5  # balanced softmax的alpha参数
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

        # FedRoD特有参数
        self.head_layer_names = self._config.get(
            'head_layer_names',
            ['fc2', 'fc', 'classifier', 'head']
        )
        self.use_balanced_softmax = self._config.get('use_balanced_softmax', True)
        self.balance_alpha = self._config.get('balance_alpha', 0.5)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 组件占位符
        self.torch_model = None
        self._optimizer = None
        self._criterion = None
        self._train_loader = None
        self._test_loader = None

        # 用于balanced softmax的类别统计
        self.class_counts = None

        self.logger.info(
            f"FedRoDLearner {node_id} 初始化完成 "
            f"(lr={self.learning_rate}, balanced_softmax={self.use_balanced_softmax})"
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

    def get_generic_parameters(self) -> Dict[str, torch.Tensor]:
        """获取generic feature extractor的参数（用于联邦聚合）"""
        generic_params = {}
        for name, param in self.torch_model.named_parameters():
            if not self.is_head_layer(name):
                generic_params[name] = param.data.cpu().clone()
        return generic_params

    def compute_class_counts(self):
        """计算本地数据集的类别分布"""
        if self.class_counts is not None:
            return

        # 尝试从模型配置获取类别数
        num_classes = self._config.get('num_classes', 10)
        counts = torch.zeros(num_classes, dtype=torch.long)

        for _, target in self._train_loader:
            for t in target:
                counts[t.item()] += 1

        self.class_counts = counts.float().to(self.device)
        self.logger.info(f"[{self._node_id}] Class distribution: {counts.tolist()}")

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
            return self._criterion(logits, targets)

        # Balanced softmax: 调整logits by class frequency
        # adjusted_logits = logits + alpha * log(class_counts)
        log_counts = torch.log(self.class_counts + 1e-9)
        adjusted_logits = logits + self.balance_alpha * log_counts.unsqueeze(0)

        loss = F.cross_entropy(adjusted_logits, targets)
        return loss

    async def train_epoch(self, epoch: int) -> EpochMetrics:
        """单轮训练 - 使用balanced softmax"""
        # 计算类别分布（用于balanced softmax）
        if self.use_balanced_softmax and self.class_counts is None:
            self.compute_class_counts()

        self.torch_model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for data, target in self._train_loader:
            data, target = data.to(self.device), target.to(self.device)

            self._optimizer.zero_grad()
            output = self.torch_model(data)

            # 使用balanced softmax loss
            loss = self.balanced_softmax_loss(output, target)

            loss.backward()
            self._optimizer.step()

            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            total_correct += pred.eq(target.view_as(pred)).sum().item()
            total_samples += data.size(0)

        avg_loss = total_loss / total_samples
        avg_accuracy = total_correct / total_samples

        self.logger.info(
            f"[{self._node_id}] Epoch {epoch}: "
            f"Loss={avg_loss:.4f}, Acc={avg_accuracy:.4f}"
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
        """获取模型权重 - 只返回generic parameters"""
        return self.get_generic_parameters()

    async def set_weights(self, weights: Dict[str, Any]):
        """设置模型权重 - 只更新generic parameters"""
        # 只更新generic parameters，保留personalized head
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
            f"[{self._node_id}] FedRoD: Updated {updated_count} generic parameters, "
            f"kept personalized head unchanged"
        )
