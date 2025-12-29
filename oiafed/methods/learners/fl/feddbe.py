"""
FedDBE (Federated Data-Free Knowledge Distillation based Ensemble) 学习器实现

从 methods/learners/fl/feddbe.py 迁移到新架构

FedDBE使用无数据知识蒸馏和模型集成来提高联邦学习性能

核心思想：
- 在服务器端集成多个客户端模型
- 使用生成的合成数据进行知识蒸馏
- 不需要访问原始训练数据
"""
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import copy

from ....core.learner import Learner
from ....core.types import StepMetrics, EpochMetrics, EvalResult
from ....registry import learner


@learner('fl.feddbe', description='FedDBE: Federated Data-Free Knowledge Distillation based Ensemble')
class FedDBELearner(Learner):
    """FedDBE学习器 - 无数据知识蒸馏集成

    配置示例:
    {
        "learner": {
            "name": "fl.feddbe",
            "learning_rate": 0.01,
            "batch_size": 128,
            "local_epochs": 5,
            "optimizer": "SGD",
            "momentum": 0.9,
            "ensemble_distill": false,  # 是否使用集成蒸馏（简化版本不使用）
            "temperature": 3.0  # 蒸馏温度
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

        # FedDBE特有参数
        self.ensemble_distill = self._config.get('ensemble_distill', False)
        self.temperature = self._config.get('temperature', 3.0)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 组件占位符
        self.torch_model = None
        self._optimizer = None
        self._criterion = None
        self._train_loader = None
        self._test_loader = None

        # 存储历史模型用于集成（简化版本）
        self.historical_models = []

        self.logger.info(
            f"FedDBELearner {node_id} 初始化完成 "
            f"(lr={self.learning_rate}, ensemble_distill={self.ensemble_distill})"
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

    def ensemble_prediction(self, data):
        """
        集成多个模型的预测

        Args:
            data: 输入数据

        Returns:
            集成后的预测
        """
        # 当前模型预测
        output = self.torch_model(data)

        # 如果有历史模型，进行集成
        if self.ensemble_distill and len(self.historical_models) > 0:
            ensemble_outputs = [output]

            with torch.no_grad():
                for hist_model in self.historical_models:
                    hist_model.eval()
                    hist_output = hist_model(data)
                    ensemble_outputs.append(hist_output)

            # 平均集成
            output = torch.stack(ensemble_outputs).mean(dim=0)

        return output

    async def train_epoch(self, epoch: int) -> EpochMetrics:
        """单轮训练 - FedDBE训练循环"""
        self.torch_model.train()

        total_loss = 0.0
        total_ce_loss = 0.0
        total_distill_loss = 0.0
        total_correct = 0
        total_samples = 0

        for data, target in self._train_loader:
            data, target = data.to(self.device), target.to(self.device)

            self._optimizer.zero_grad()

            # 前向传播
            output = self.torch_model(data)
            ce_loss = self._criterion(output, target)

            # 如果启用集成蒸馏，添加蒸馏损失
            distill_loss = torch.tensor(0.0, device=self.device)
            if self.ensemble_distill and len(self.historical_models) > 0:
                ensemble_output = self.ensemble_prediction(data)

                # KL散度蒸馏损失
                student_probs = F.log_softmax(output / self.temperature, dim=1)
                teacher_probs = F.softmax(ensemble_output / self.temperature, dim=1)

                distill_loss = F.kl_div(
                    student_probs,
                    teacher_probs.detach(),
                    reduction='batchmean'
                ) * (self.temperature ** 2)

                # 组合损失
                loss = 0.5 * ce_loss + 0.5 * distill_loss
            else:
                loss = ce_loss

            loss.backward()
            self._optimizer.step()

            total_loss += loss.item() * data.size(0)
            total_ce_loss += ce_loss.item() * data.size(0)
            total_distill_loss += distill_loss.item() * data.size(0)

            pred = output.argmax(dim=1, keepdim=True)
            total_correct += pred.eq(target.view_as(pred)).sum().item()
            total_samples += data.size(0)

        avg_loss = total_loss / total_samples
        avg_accuracy = total_correct / total_samples

        self.logger.info(
            f"[{self._node_id}] Epoch {epoch}: "
            f"Loss={avg_loss:.4f} (CE={total_ce_loss/total_samples:.4f}, "
            f"Distill={total_distill_loss/total_samples:.4f}), Acc={avg_accuracy:.4f}"
        )

        epoch_metrics = EpochMetrics(
            epoch=epoch,
            avg_loss=avg_loss,
            total_samples=total_samples,
            metrics={
                'accuracy': avg_accuracy,
                'ce_loss': total_ce_loss / total_samples,
                'distill_loss': total_distill_loss / total_samples
            }
        )

        # 增加全局 epoch 计数器（用于 MLflow step）
        self._global_epoch_counter += 1

        # 触发 epoch 结束回调
        if self._callbacks:
            await self._callbacks.on_epoch_end(self, epoch, epoch_metrics)

        return epoch_metrics

    async def fit(self, config: Optional[Dict] = None):
        """训练模型 - 保存历史模型"""
        # 调用父类的fit方法
        await super().fit(config)

        # 保存当前模型到历史（用于下一轮集成）
        if self.ensemble_distill:
            self.historical_models.append(copy.deepcopy(self.torch_model))
            # 限制历史模型数量
            if len(self.historical_models) > 3:
                self.historical_models.pop(0)

    async def evaluate_model(self, config: Optional[Dict] = None) -> EvalResult:
        """评估模型 - 使用集成预测"""
        self.torch_model.eval()

        loader = self._test_loader if self._test_loader else self._train_loader

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)

                # 使用集成预测
                output = self.ensemble_prediction(data)

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
                'num_historical_models': len(self.historical_models)
            }
        )

    async def get_weights(self) -> Dict[str, Any]:
        """获取模型权重"""
        return {name: param.data.clone() for name, param in self.torch_model.state_dict().items()}

    async def set_weights(self, weights: Dict[str, Any]):
        """设置模型权重"""
        # 转换为torch tensor
        torch_weights = {}
        for k, v in weights.items():
            if torch.is_tensor(v):
                torch_weights[k] = v
            else:
                torch_weights[k] = torch.from_numpy(v)

        self.torch_model.load_state_dict(torch_weights)

        self.logger.debug(f"[{self._node_id}] FedDBE: Updated model")
