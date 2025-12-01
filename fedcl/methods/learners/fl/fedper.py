"""
FedPer (Federated Learning with Personalization Layers) 学习器实现
fedcl/methods/learners/fedper.py

论文：Federated Learning with Personalization Layers
作者：Manoj Ghuhan Arivazhagan et al.
发表：arXiv 2019

FedPer的核心思想：
- 将模型分为base layers（共享的特征提取器）和personalization layers（个性化分类器）
- 只有base layers参与联邦聚合
- 每个客户端保留自己的personalization layers
"""
from typing import Dict, Any, List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .._decorators import learner
from fedcl.learner.base_learner import BaseLearner
from fedcl.types import TrainingResponse
from fedcl.methods.models.base import get_param_count


@learner('fl', 'FedPer', description='FedPer: Federated Learning with Personalization Layers')
class FedPerLearner(BaseLearner):
    """FedPer学习器 - 分离共享层和个性化层

    配置示例:
    {
        "learner": {
            "name": "FedPer",
            "params": {
                "model": {"name": "SimpleCNN", "params": {"num_classes": 10}},
                "optimizer": {"type": "SGD", "lr": 0.01, "momentum": 0.9},
                "loss": "CrossEntropyLoss",
                "learning_rate": 0.01,
                "batch_size": 128,
                "local_epochs": 5,
                "personalization_layer_names": ["fc2"]  # 个性化层的名称
            }
        },
        "dataset": {...}
    }
    """

    def __init__(self, client_id: str, config: Dict[str, Any] = None, lazy_init: bool = True):
        # 提取配置
        learner_params = (config or {}).get('learner', {}).get('params', {})

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
        self._epochs = learner_params.get('local_epochs', 5)

        # FedPer特有参数：个性化层的名称
        self.personalization_layer_names = learner_params.get(
            'personalization_layer_names',
            ['fc', 'fc2', 'classifier', 'head']  # 默认值
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

        # 调用父类
        super().__init__(client_id, clean_config, lazy_init)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 组件占位符
        self._model = None
        self._optimizer = None
        self._criterion = None
        self._train_loader = None

        self.logger.info(
            f"FedPerLearner {client_id} 初始化完成 "
            f"(model={self._model_cfg.get('name')}, "
            f"personalization_layers={self.personalization_layer_names})"
        )

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
                else:
                    raise ValueError(f"不支持的损失函数: {loss_name}")
            self.logger.debug(f"Client {self.client_id}: 损失函数创建完成")
        return self._criterion

    @property
    def train_loader(self):
        """延迟加载数据加载器"""
        if self._train_loader is None:
            dataset = self.dataset
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

    def is_personalization_layer(self, param_name: str) -> bool:
        """判断参数是否属于个性化层"""
        for layer_name in self.personalization_layer_names:
            if layer_name in param_name:
                return True
        return False

    def get_base_parameters(self) -> Dict[str, torch.Tensor]:
        """获取base layers的参数（用于联邦聚合）"""
        base_params = {}
        for name, param in self.model.named_parameters():
            if not self.is_personalization_layer(name):
                base_params[name] = param.data.cpu().clone()
        return base_params

    def get_personalization_parameters(self) -> Dict[str, torch.Tensor]:
        """获取personalization layers的参数（不参与聚合）"""
        personal_params = {}
        for name, param in self.model.named_parameters():
            if self.is_personalization_layer(name):
                personal_params[name] = param.data.cpu().clone()
        return personal_params

    async def train(self, params: Dict[str, Any]) -> TrainingResponse:
        """训练方法 - 标准训练，但只上传base parameters"""
        num_epochs = params.get("num_epochs", self._epochs)
        round_number = params.get("round_number", 1)

        self.logger.info(
            f"  [{self.client_id}] Round {round_number}, "
            f"FedPer Training {num_epochs} epochs..."
        )

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

        # 只上传base parameters
        base_weights = self.get_base_parameters()
        personal_weights = self.get_personalization_parameters()

        self.logger.info(
            f"  [{self.client_id}] FedPer: "
            f"Base params: {len(base_weights)}, "
            f"Personal params: {len(personal_weights)}"
        )

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
                "model_weights": base_weights  # 只上传base parameters
            },
            execution_time=0.0
        )

        self.logger.info(
            f"  [{self.client_id}] Round {round_number} completed: "
            f"Loss={avg_loss:.4f}, Acc={accuracy:.4f}"
        )
        return response

    async def set_model(self, model_data: Dict[str, Any]) -> bool:
        """设置模型数据 - 只更新base parameters"""
        try:
            if "parameters" in model_data and "weights" in model_data["parameters"]:
                weights = model_data["parameters"]["weights"]

                # 只更新base parameters，保留personalization parameters
                state_dict = self.model.state_dict()
                updated_count = 0

                for name, value in weights.items():
                    if name in state_dict and not self.is_personalization_layer(name):
                        if not isinstance(value, torch.Tensor):
                            value = torch.from_numpy(value)
                        state_dict[name] = value.to(self.device)
                        updated_count += 1

                self.model.load_state_dict(state_dict, strict=True)

                self.logger.info(
                    f"  [{self.client_id}] FedPer: Updated {updated_count} base parameters, "
                    f"kept personalization layers unchanged"
                )
                return True
        except Exception as e:
            self.logger.exception(f"  [{self.client_id}] Failed to set model: {e}")
        return False

    async def get_model(self) -> Dict[str, Any]:
        """获取模型数据 - 返回base parameters"""
        dataset = self.dataset
        return {
            "model_type": self._model_cfg['name'],
            "parameters": {"weights": self.get_base_parameters()},
            "metadata": {
                "client_id": self.client_id,
                "samples": len(dataset),
                "param_count": get_param_count(self.model),
                "base_param_count": len(self.get_base_parameters()),
                "personal_param_count": len(self.get_personalization_parameters())
            }
        }

    async def evaluate(self, model_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """评估方法"""
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

        return {
            "accuracy": correct / total,
            "loss": test_loss / total,
            "samples": total
        }

    def get_data_statistics(self) -> Dict[str, Any]:
        """获取数据统计"""
        dataset = self.dataset
        stats = {"total_samples": len(dataset)}
        if hasattr(dataset, 'get_statistics'):
            stats.update(dataset.get_statistics())
        return stats

    async def get_local_model(self) -> Dict[str, Any]:
        return await self.get_model()

    async def set_local_model(self, model_data: Dict[str, Any]) -> bool:
        return await self.set_model(model_data)
