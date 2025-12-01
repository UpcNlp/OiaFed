"""
通用学习器 - 配置驱动的联邦学习客户端
fedcl/methods/learners/generic.py

完全通过配置文件驱动，无需为每个数据集编写专门的Learner。
"""
from typing import Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .._decorators import learner
from fedcl.learner.base_learner import BaseLearner
from fedcl.types import TrainingResponse
from fedcl.methods.models.base import (
    get_weights_as_dict,
    set_weights_from_dict,
    get_param_count
)


@learner('fl', 'Generic', description='通用联邦学习器 - 配置驱动')
class GenericLearner(BaseLearner):
    """通用学习器 - 完全通过配置驱动

    配置示例:
    {
        "learner": {
            "name": "Generic",
            "params": {
                "model": {"name": "MNIST_CNN"},
                "optimizer": {"type": "SGD", "lr": 0.01, "momentum": 0.9},
                "loss": "CrossEntropyLoss",
                "learning_rate": 0.01,
                "batch_size": 32,
                "local_epochs": 1
            }
        },
        "dataset": {
            "name": "MNIST",
            "params": {...},
            "partition": {...}
        }
    }
    """

    def __init__(self, client_id: str, config: Dict[str, Any] = None, lazy_init: bool = True):
        # 先提取配置，避免与property冲突
        learner_params = (config or {}).get('learner', {}).get('params', {})

        # 保存模型、优化器、损失函数配置（使用不同的变量名）
        self._model_cfg = learner_params.get('model', {})
        if not self._model_cfg.get('name'):
            raise ValueError("必须在配置中指定模型名称: learner.params.model.name")

        self._optimizer_cfg = learner_params.get('optimizer', {
            'type': 'SGD',
            'lr': learner_params.get('learning_rate', 0.01),
            'momentum': 0.9
        })

        self._loss_cfg = learner_params.get('loss', 'CrossEntropyLoss')

        # 训练参数
        self._lr = learner_params.get('learning_rate', 0.01)
        self._bs = learner_params.get('batch_size', 32)
        self._epochs = learner_params.get('local_epochs', 1)

        # 创建一个不包含冲突字段的config副本传递给父类
        clean_config = config.copy() if config else {}
        if 'learner' in clean_config and 'params' in clean_config['learner']:
            clean_params = clean_config['learner']['params'].copy()
            # 移除会与property冲突的字段
            clean_params.pop('model', None)
            clean_params.pop('optimizer', None)
            clean_params.pop('loss', None)
            clean_config['learner'] = clean_config['learner'].copy()
            clean_config['learner']['params'] = clean_params

        # 然后调用父类的__init__
        super().__init__(client_id, clean_config, lazy_init)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 组件占位符（延迟加载）
        self._model = None
        self._optimizer = None
        self._criterion = None
        self._train_loader = None

        self.logger.info(f"GenericLearner {client_id} 初始化完成 (model={self._model_cfg.get('name')})")

    @property
    def model(self):
        """延迟加载模型"""
        if self._model is None:
            from fedcl.api.registry import registry
            model_name = self._model_cfg['name']
            model_params = self._model_cfg.get('params', {})

            model_class = registry.get_model(model_name)
            self._model = model_class(**model_params).to(self.device)
            self.logger.debug(f"Client {self.client_id}: 模型 {model_name} 创建完成")
        return self._model

    @property
    def optimizer(self):
        """延迟加载优化器"""
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
                self._optimizer = optim.Adam(
                    self.model.parameters(),
                    lr=lr,
                    betas=self._optimizer_cfg.get('betas', (0.9, 0.999))
                )
            elif opt_type == 'ADAMW':
                self._optimizer = optim.AdamW(
                    self.model.parameters(),
                    lr=lr,
                    betas=self._optimizer_cfg.get('betas', (0.9, 0.999))
                )
            else:
                raise ValueError(f"不支持的优化器类型: {opt_type}")

            self.logger.debug(f"Client {self.client_id}: 优化器 {opt_type} 创建完成")
        return self._optimizer

    @property
    def criterion(self):
        """延迟加载损失函数"""
        if self._criterion is None:
            if isinstance(self._loss_cfg, str):
                loss_name = self._loss_cfg
                if loss_name == 'CrossEntropyLoss':
                    self._criterion = nn.CrossEntropyLoss()
                elif loss_name == 'MSELoss':
                    self._criterion = nn.MSELoss()
                elif loss_name == 'NLLLoss':
                    self._criterion = nn.NLLLoss()
                elif loss_name == 'BCELoss':
                    self._criterion = nn.BCELoss()
                elif loss_name == 'BCEWithLogitsLoss':
                    self._criterion = nn.BCEWithLogitsLoss()
                else:
                    raise ValueError(f"不支持的损失函数: {loss_name}")
            else:
                # 如果配置是dict，可以包含参数
                raise NotImplementedError("暂不支持带参数的损失函数配置")

            self.logger.debug(f"Client {self.client_id}: 损失函数创建完成")
        return self._criterion

    @property
    def train_loader(self):
        """延迟加载数据加载器"""
        if self._train_loader is None:
            dataset = self.dataset  # 触发数据集的延迟加载和自动划分
            self._train_loader = DataLoader(
                dataset,
                batch_size=self._bs,
                shuffle=True
            )
            self.logger.info(
                f"Client {self.client_id}: 数据加载器创建完成 "
                f"(samples={len(dataset)}, batch_size={self._bs})"
            )
        return self._train_loader

    async def train(self, params: Dict[str, Any]) -> TrainingResponse:
        """训练方法 - 标准PyTorch训练循环"""
        num_epochs = params.get("num_epochs", self._epochs)
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

        # 获取模型权重
        model_weights = get_weights_as_dict(self.model)

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
                "model_weights": model_weights
            },
            execution_time=0.0
        )

        self.logger.info(
            f"  [{self.client_id}] Round {round_number} completed: "
            f"Loss={avg_loss:.4f}, Acc={accuracy:.4f}"
        )
        return response

    async def evaluate(self, model_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """评估方法"""
        self.model.eval()

        # 如果提供了模型权重，先更新模型
        if model_data and "model_weights" in model_data:
            weights = model_data["model_weights"]
            torch_weights = {}
            for k, v in weights.items():
                if isinstance(v, torch.Tensor):
                    torch_weights[k] = v
                else:
                    torch_weights[k] = torch.from_numpy(v)
            set_weights_from_dict(self.model, torch_weights)

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

    async def get_model(self) -> Dict[str, Any]:
        """获取模型数据"""
        dataset = self.dataset
        return {
            "model_type": self._model_cfg['name'],
            "parameters": {"weights": get_weights_as_dict(self.model)},
            "metadata": {
                "client_id": self.client_id,
                "samples": len(dataset),
                "param_count": get_param_count(self.model)
            }
        }

    async def set_model(self, model_data: Dict[str, Any]) -> bool:
        """设置模型数据"""
        try:
            if "parameters" in model_data and "weights" in model_data["parameters"]:
                weights = model_data["parameters"]["weights"]
                torch_weights = {}
                for k, v in weights.items():
                    if torch.is_tensor(v):
                        torch_weights[k] = v
                    else:
                        torch_weights[k] = torch.from_numpy(v)
                set_weights_from_dict(self.model, torch_weights)
                return True
        except Exception as e:
            self.logger.exception(f"  [{self.client_id}] Failed to set model: {e}")
        return False

    def get_data_statistics(self) -> Dict[str, Any]:
        """获取数据统计"""
        dataset = self.dataset
        # 尝试获取数据集的统计信息
        stats = {
            "total_samples": len(dataset),
        }

        # 如果数据集有额外的统计方法，调用它
        if hasattr(dataset, 'get_statistics'):
            stats.update(dataset.get_statistics())

        return stats

    async def get_local_model(self) -> Dict[str, Any]:
        return await self.get_model()

    async def set_local_model(self, model_data: Dict[str, Any]) -> bool:
        return await self.set_model(model_data)
