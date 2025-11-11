"""
MNIST学习器
fedcl/methods/learners/mnist_learner.py
"""
from typing import Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from fedcl.api.decorators import learner
from fedcl.learner.base_learner import BaseLearner
from fedcl.types import TrainingResponse


@learner('MNISTLearner',
         description='MNIST数据集学习器',
         version='1.0',
         author='MOE-FedCL',
         dataset='MNIST')
class MNISTLearner(BaseLearner):
    """MNIST学习器 - 使用自动数据划分功能

    特性：
    - 支持配置驱动的数据集自动划分
    - 延迟加载模型、优化器和数据加载器
    - 完整的训练和评估功能
    """

    def __init__(self, client_id: str, config: Dict[str, Any] = None, lazy_init: bool = True):
        """初始化MNIST学习器

        Args:
            client_id: 客户端ID
            config: 配置字典（由ComponentBuilder.parse_config()生成）
            lazy_init: 是否延迟初始化组件
        """
        super().__init__(client_id, config, lazy_init)

        # 提取训练参数（从config.learner.params）
        if not hasattr(self, 'learning_rate'):
            self.learning_rate = 0.01
        if not hasattr(self, 'batch_size'):
            self.batch_size = 32
        if not hasattr(self, 'local_epochs'):
            self.local_epochs = 1

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 组件占位符（延迟加载）
        self._model = None
        self._optimizer = None
        self._criterion = None
        self._train_loader = None

        self.logger.info(f"MNISTLearner {client_id} 初始化完成 (lazy_init={lazy_init})")

    @property
    def model(self):
        """延迟加载模型"""
        if self._model is None:
            # 从注册表获取模型类
            from fedcl.api.registry import registry
            model_class = registry.get_model('MNIST_CNN')
            self._model = model_class(num_classes=10).to(self.device)
            self.logger.debug(f"Client {self.client_id}: 模型创建完成")
        return self._model

    @property
    def optimizer(self):
        """延迟加载优化器"""
        if self._optimizer is None:
            self._optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
            self.logger.debug(f"Client {self.client_id}: 优化器创建完成")
        return self._optimizer

    @property
    def criterion(self):
        """延迟加载损失函数"""
        if self._criterion is None:
            self._criterion = nn.CrossEntropyLoss()
        return self._criterion

    @property
    def train_loader(self):
        """延迟加载数据加载器"""
        if self._train_loader is None:
            # 直接使用 self.dataset（已经是划分后的子集）
            dataset = self.dataset

            self._train_loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True
            )
            self.logger.info(
                f"Client {self.client_id}: 数据加载器创建完成 "
                f"(samples={len(dataset)}, batch_size={self.batch_size})"
            )
        return self._train_loader

    async def train(self, params: Dict[str, Any]) -> TrainingResponse:
        """训练方法"""
        num_epochs = params.get("num_epochs", self.local_epochs)
        round_number = params.get("round_number", 1)

        self.logger.info(f"  [{self.client_id}] Round {round_number}, Training {num_epochs} epochs...")

        # 设置模型为训练模式
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
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                epoch_correct += pred.eq(target.view_as(pred)).sum().item()
                epoch_samples += data.size(0)

            avg_epoch_loss = epoch_loss / epoch_samples
            epoch_accuracy = epoch_correct / epoch_samples
            self.logger.info(f"    [{self.client_id}] Epoch {epoch+1}: Loss={avg_epoch_loss:.4f}, Acc={epoch_accuracy:.4f}")

            total_loss += epoch_loss
            correct_predictions += epoch_correct
            total_samples += epoch_samples

        # 计算平均值
        avg_loss = total_loss / total_samples
        accuracy = correct_predictions / total_samples

        # 获取模型权重（直接返回torch.Tensor，底层会自动转换）
        model_weights = self.model.get_weights_as_dict()

        # 创建训练响应
        response = TrainingResponse(
            request_id="",  # 会被stub填充
            client_id=self.client_id,
            success=True,
            result={
                "epochs_completed": num_epochs,
                "loss": avg_loss,
                "accuracy": accuracy,
                "samples_used": total_samples,
                "model_weights": model_weights  # 框架会自动序列化tensor
            },
            execution_time=0.0
        )

        self.logger.info(f"  [{self.client_id}] Round {round_number} completed: Loss={avg_loss:.4f}, Acc={accuracy:.4f}")
        return response

    async def evaluate(self, model_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """评估方法"""
        self.model.eval()

        # 如果提供了模型权重，先更新模型
        if model_data and "model_weights" in model_data:
            weights = model_data["model_weights"]
            # 将numpy数组转换为torch tensor
            torch_weights = {}
            for k, v in weights.items():
                if isinstance(v, np.ndarray):
                    torch_weights[k] = torch.from_numpy(v)
                else:
                    torch_weights[k] = v
            self.model.set_weights_from_dict(torch_weights)

        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.train_loader:  # 使用训练数据作为评估数据
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

    async def get_model(self) -> Dict[str, Any]:
        """获取模型数据

        直接返回torch.Tensor，框架会自动序列化
        """
        # 获取数据集（触发延迟加载）
        dataset = self.dataset
        return {
            "model_type": "mnist_cnn",
            "parameters": {"weights": self.model.get_weights_as_dict()},
            "metadata": {
                "client_id": self.client_id,
                "samples": len(dataset),
                "param_count": self.model.get_param_count()
            }
        }

    async def set_model(self, model_data: Dict[str, Any]) -> bool:
        """设置模型数据

        接受torch.Tensor或numpy数组，自动转换
        """
        try:
            if "parameters" in model_data and "weights" in model_data["parameters"]:
                weights = model_data["parameters"]["weights"]
                # 智能转换：支持numpy数组、torch.Tensor和dict
                torch_weights = {}
                for k, v in weights.items():
                    if isinstance(v, np.ndarray):
                        torch_weights[k] = torch.from_numpy(v)
                    elif torch.is_tensor(v):
                        torch_weights[k] = v
                    else:
                        torch_weights[k] = v
                self.model.set_weights_from_dict(torch_weights)
                return True
        except Exception as e:
            self.logger.exception(f"  [{self.client_id}] Failed to set model: {e}")
        return False

    def get_data_statistics(self) -> Dict[str, Any]:
        """获取数据统计"""
        # 获取数据集（触发延迟加载）
        dataset = self.dataset
        return {
            "total_samples": len(dataset),
            "num_classes": 10,
            "feature_dim": 784,
            "input_shape": (1, 28, 28)
        }

    async def get_local_model(self) -> Dict[str, Any]:
        return await self.get_model()

    async def set_local_model(self, model_data: Dict[str, Any]) -> bool:
        return await self.set_model(model_data)
