"""
GenericLearner - 通用联邦学习器（迁移到 src/ 架构）

从 methods/learners/fl/generic.py 迁移而来
保留原算法逻辑，适配 src/ 接口
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional

from ....core.learner import Learner
from ....core.types import TrainResult, EvalResult, StepMetrics, EpochMetrics, TrainMetrics, FitStatus
from ....registry import learner


@learner('fl.generic', description='通用联邦学习器 - 配置驱动')
class GenericLearner(Learner):
    """
    通用学习器 - 完全通过配置驱动

    从 methods/ 迁移而来，适配 src/ 架构：
    - 继承 src.core.learner.Learner
    - 实现 train_step() 而不是完整的 train()
    - 通过 self._model 和 self._data 访问模型和数据（依赖注入）
    - 返回 TrainResult 而不是 TrainingResponse
    """

    def __init__(
        self,
        model,              # src/ 传入的 Model 对象
        data,               # src/ 传入的 DataProvider
        tracker=None,
        callbacks=None,
        config=None,
        node_id=None
    ):
        # 调用父类
        super().__init__(model, data, tracker, callbacks, config, node_id)

        # 从配置中提取参数（保持 methods/ 的配置风格）
        self.learning_rate = self._config.get('learning_rate', 0.01)
        self.batch_size = self._config.get('batch_size', 32)
        self.local_epochs = self._config.get('local_epochs', 1)
        self.momentum = self._config.get('momentum', 0.9)
        self.optimizer_type = self._config.get('optimizer', {}).get('type', 'SGD').upper()

        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 训练组件（延迟初始化）
        self.torch_model = None
        self._optimizer = None
        self._criterion = None
        self._train_loader = None

        self.logger.info(f"GenericLearner {self._node_id} 初始化完成 (lr={self.learning_rate}, bs={self.batch_size})")

    async def setup(self, config: Dict) -> None:
        """初始化训练环境"""
        # 获取 PyTorch 模型（从 src/ 的 Model 对象）
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
        elif self.optimizer_type == 'ADAMW':
            self._optimizer = optim.AdamW(
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
        elif loss_name == 'NLLLoss':
            self._criterion = nn.NLLLoss()
        else:
            raise ValueError(f"不支持的损失函数: {loss_name}")

        # 创建数据加载器（从 src/ 的 DataProvider）
        train_dataset = self._data.get_train_data()
        self._train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        self.logger.info(
            f"Setup完成: samples={len(train_dataset)}, "
            f"batch_size={self.batch_size}, optimizer={self.optimizer_type}"
        )

    async def train_step(self, batch, step: int) -> StepMetrics:
        """
        单步训练（必须实现）

        这是 src/ 架构要求的接口
        """
        data, target = batch
        data, target = data.to(self.device), target.to(self.device)

        # 前向传播
        self._optimizer.zero_grad()
        output = self.torch_model(data)
        loss = self._criterion(output, target)

        # 反向传播
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
        """
        单轮训练（可选重写）

        重写以添加详细日志
        """
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
        """
        评估模型（必须实现）

        这是 src/ 架构要求的接口
        """
        self.torch_model.eval()

        # 获取测试数据
        test_dataset = self._data.get_test_data()
        if test_dataset is None:
            # 如果没有测试集，使用训练集评估
            test_dataset = self._data.get_train_data()

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for data, target in test_loader:
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

    # ==================== 以下是可选的辅助方法 ====================

    async def set_weights(self, weights: Dict[str, Any]):
        """
        设置模型权重（从服务端接收）

        虽然基类有默认实现，但为了兼容 methods/ 的格式，这里重写
        """
        # 转换为 torch tensor（兼容 numpy）
        torch_weights = {}
        for k, v in weights.items():
            if torch.is_tensor(v):
                torch_weights[k] = v
            else:
                import numpy as np
                if isinstance(v, np.ndarray):
                    torch_weights[k] = torch.from_numpy(v)
                else:
                    torch_weights[k] = v

        self.torch_model.load_state_dict(torch_weights)
        self.logger.debug(f"[{self._node_id}] 已更新模型权重")

    async def get_weights(self) -> Dict[str, Any]:
        """
        获取模型权重（发送给服务端）
        """
        return {name: param.data.clone() for name, param in self.torch_model.state_dict().items()}
