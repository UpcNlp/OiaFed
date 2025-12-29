"""
GPFL (Generalized Personalized Federated Learning) 学习器实现

从 methods/learners/fl/gpfl.py 迁移到新架构

GPFL通过混合全局模型和本地模型来实现个性化

核心思想：
- 维护一个全局共享模型和一个本地个性化模型
- 使用混合系数alpha来组合两个模型的预测
- 通过训练自适应调整混合系数
"""
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import copy

from ....core.learner import Learner
from ....core.types import StepMetrics, EpochMetrics, EvalResult
from ....registry import learner


@learner('fl.gpfl', description='GPFL: Generalized Personalized Federated Learning')
class GPFLLearner(Learner):
    """GPFL学习器 - 泛化个性化联邦学习

    配置示例:
    {
        "learner": {
            "name": "fl.gpfl",
            "learning_rate": 0.01,
            "batch_size": 128,
            "local_epochs": 5,
            "optimizer": "SGD",
            "momentum": 0.9,
            "alpha": 0.5  # 全局模型和本地模型的混合系数（初始值）
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

        # GPFL特有参数
        self.alpha = self._config.get('alpha', 0.5)  # 混合系数

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 组件占位符
        self.torch_model = None  # 本地模型
        self.global_model = None  # 全局模型副本
        self._optimizer = None
        self._criterion = None
        self._train_loader = None
        self._test_loader = None

        # 可学习的混合系数
        self.alpha_param = nn.Parameter(torch.tensor([self.alpha]))

        self.logger.info(
            f"GPFLLearner {node_id} 初始化完成 "
            f"(lr={self.learning_rate}, alpha={self.alpha})"
        )

    async def setup(self, config: Dict) -> None:
        """初始化训练环境"""
        # 获取PyTorch模型（本地模型）
        self.torch_model = self._model.get_model()
        self.torch_model.to(self.device)

        # 创建全局模型副本
        self.global_model = copy.deepcopy(self.torch_model)
        self.global_model.eval()

        # 创建优化器（包含本地模型参数和混合系数）
        params = list(self.torch_model.parameters()) + [self.alpha_param]

        if self.optimizer_type == 'SGD':
            self._optimizer = optim.SGD(
                params,
                lr=self.learning_rate,
                momentum=self.momentum
            )
        elif self.optimizer_type == 'ADAM':
            self._optimizer = optim.Adam(params, lr=self.learning_rate)
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

    def mixed_prediction(self, data):
        """
        混合全局模型和本地模型的预测

        Args:
            data: 输入数据

        Returns:
            混合后的预测和alpha值
        """
        # 本地模型预测
        local_output = self.torch_model(data)

        # 全局模型预测
        if self.global_model is not None:
            with torch.no_grad():
                global_output = self.global_model(data)
        else:
            global_output = local_output

        # 使用sigmoid确保alpha在[0, 1]之间
        alpha = torch.sigmoid(self.alpha_param)

        # 混合预测: output = alpha * global + (1-alpha) * local
        mixed_output = alpha * global_output + (1 - alpha) * local_output

        return mixed_output, alpha.item()

    async def train_epoch(self, epoch: int) -> EpochMetrics:
        """单轮训练 - GPFL训练循环"""
        self.torch_model.train()
        if self.global_model is not None:
            self.global_model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        alpha_values = []

        for data, target in self._train_loader:
            data, target = data.to(self.device), target.to(self.device)

            self._optimizer.zero_grad()

            # 混合预测
            output, alpha_val = self.mixed_prediction(data)
            alpha_values.append(alpha_val)

            # 计算损失
            loss = self._criterion(output, target)

            loss.backward()
            self._optimizer.step()

            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            total_correct += pred.eq(target.view_as(pred)).sum().item()
            total_samples += data.size(0)

        avg_loss = total_loss / total_samples
        avg_accuracy = total_correct / total_samples
        avg_alpha = sum(alpha_values) / len(alpha_values) if alpha_values else 0.5

        self.logger.info(
            f"[{self._node_id}] Epoch {epoch}: "
            f"Loss={avg_loss:.4f}, Acc={avg_accuracy:.4f}, Alpha={avg_alpha:.4f}"
        )

        epoch_metrics = EpochMetrics(
            epoch=epoch,
            avg_loss=avg_loss,
            total_samples=total_samples,
            metrics={
                'accuracy': avg_accuracy,
                'alpha': avg_alpha
            }
        )

        # 增加全局 epoch 计数器（用于 MLflow step）
        self._global_epoch_counter += 1

        # 触发 epoch 结束回调
        if self._callbacks:
            await self._callbacks.on_epoch_end(self, epoch, epoch_metrics)

        return epoch_metrics

    async def evaluate_model(self, config: Optional[Dict] = None) -> EvalResult:
        """评估模型 - 使用混合预测"""
        self.torch_model.eval()
        if self.global_model is not None:
            self.global_model.eval()

        loader = self._test_loader if self._test_loader else self._train_loader

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                output, _ = self.mixed_prediction(data)
                loss = self._criterion(output, target)

                total_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                total_correct += pred.eq(target.view_as(pred)).sum().item()
                total_samples += data.size(0)

        return EvalResult(
            num_samples=total_samples,
            metrics={
                'loss': total_loss / total_samples,
                'accuracy': total_correct / total_samples,
                'alpha': torch.sigmoid(self.alpha_param).item()
            }
        )

    async def get_weights(self) -> Dict[str, Any]:
        """获取模型权重 - 返回本地模型"""
        return {name: param.data.clone() for name, param in self.torch_model.state_dict().items()}

    async def set_weights(self, weights: Dict[str, Any]):
        """设置模型权重 - 更新全局模型和本地模型"""
        # 转换为torch tensor
        torch_weights = {}
        for k, v in weights.items():
            if torch.is_tensor(v):
                torch_weights[k] = v
            else:
                torch_weights[k] = torch.from_numpy(v)

        # 更新全局模型
        self.global_model.load_state_dict(torch_weights)
        self.global_model.eval()

        # 如果是第一轮，同时初始化本地模型
        if self._optimizer is None or not hasattr(self, '_initialized'):
            self.torch_model.load_state_dict(torch_weights)
            self._initialized = True

        self.logger.debug(f"[{self._node_id}] GPFL: Updated global model")
