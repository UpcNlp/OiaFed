"""
FedBABU (Federated Learning via Body and Bottom-up) 学习器实现
fedcl/methods/learners/fedbabu.py

论文：Towards Fair Federated Learning with Zero-Shot Data Augmentation
作者：Junyuan Hong et al.
发表：CVPR 2021 Workshop

FedBABU的核心思想：
- 将模型分为body (feature extractor)和head (classifier)
- 训练过程：前面的epoch正常训练，最后几个epoch冻结body只训练head
- 只有body参与联邦聚合，head是个性化的
"""
from typing import Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .._decorators import learner
from fedcl.learner.base_learner import BaseLearner
from fedcl.types import TrainingResponse
from fedcl.methods.models.base import get_param_count


@learner('fl', 'FedBABU', description='FedBABU: Federated Learning via Body and Bottom-up')
class FedBABULearner(BaseLearner):
    """FedBABU学习器 - 冻结body，微调head

    配置示例:
    {
        "learner": {
            "name": "FedBABU",
            "params": {
                "model": {"name": "SimpleCNN", "params": {"num_classes": 10}},
                "optimizer": {"type": "SGD", "lr": 0.01, "momentum": 0.9},
                "loss": "CrossEntropyLoss",
                "learning_rate": 0.01,
                "batch_size": 128,
                "local_epochs": 5,
                "finetune_epochs": 1,  # 最后几个epoch只训练head
                "head_layer_names": ["fc2"]  # 头部层的名称
            }
        },
        "dataset": {...}
    }
    """

    def __init__(self, client_id: str, config: Dict[str, Any] = None, lazy_init: bool = True):
        learner_params = (config or {}).get('learner', {}).get('params', {})

        self._model_cfg = learner_params.get('model', {})
        if not self._model_cfg.get('name'):
            raise ValueError("必须在配置中指定模型名称")

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

        # FedBABU特有参数
        self.finetune_epochs = learner_params.get('finetune_epochs', 1)
        self.head_layer_names = learner_params.get(
            'head_layer_names',
            ['fc2', 'fc', 'classifier', 'head']
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

        super().__init__(client_id, clean_config, lazy_init)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._model = None
        self._optimizer = None
        self._criterion = None
        self._train_loader = None

        self.logger.info(
            f"FedBABULearner {client_id} 初始化完成 "
            f"(model={self._model_cfg.get('name')}, "
            f"epochs={self._epochs}, finetune_epochs={self.finetune_epochs})"
        )

    @property
    def model(self):
        if self._model is None:
            from fedcl.api.registry import registry
            model_name = self._model_cfg['name']
            model_params = self._model_cfg.get('params', {})

            model_class = registry.get_model(model_name)
            self._model = model_class(**model_params).to(self.device)
        return self._model

    @property
    def optimizer(self):
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
                self._optimizer = optim.Adam(self.model.parameters(), lr=lr)
            else:
                raise ValueError(f"不支持的优化器类型: {opt_type}")
        return self._optimizer

    @property
    def criterion(self):
        if self._criterion is None:
            if self._loss_cfg == 'CrossEntropyLoss':
                self._criterion = nn.CrossEntropyLoss()
            elif self._loss_cfg == 'MSELoss':
                self._criterion = nn.MSELoss()
            else:
                raise ValueError(f"不支持的损失函数: {self._loss_cfg}")
        return self._criterion

    @property
    def train_loader(self):
        if self._train_loader is None:
            dataset = self.dataset
            self._train_loader = DataLoader(
                dataset,
                batch_size=self._bs,
                shuffle=True
            )
        return self._train_loader

    def is_head_layer(self, param_name: str) -> bool:
        """判断参数是否属于头部层"""
        for layer_name in self.head_layer_names:
            if layer_name in param_name:
                return True
        return False

    def get_body_parameters(self) -> Dict[str, torch.Tensor]:
        """获取body (feature extractor)的参数（用于联邦聚合）"""
        body_params = {}
        for name, param in self.model.named_parameters():
            if not self.is_head_layer(name):
                body_params[name] = param.data.cpu().clone()
        return body_params

    def get_head_parameters(self) -> Dict[str, torch.Tensor]:
        """获取head (classifier)的参数（不参与聚合）"""
        head_params = {}
        for name, param in self.model.named_parameters():
            if self.is_head_layer(name):
                head_params[name] = param.data.cpu().clone()
        return head_params

    def freeze_body(self):
        """冻结body layers"""
        for name, param in self.model.named_parameters():
            if not self.is_head_layer(name):
                param.requires_grad = False

    def unfreeze_body(self):
        """解冻body layers"""
        for name, param in self.model.named_parameters():
            if not self.is_head_layer(name):
                param.requires_grad = True

    async def train(self, params: Dict[str, Any]) -> TrainingResponse:
        """训练方法 - FedBABU训练循环"""
        num_epochs = params.get("num_epochs", self._epochs)
        round_number = params.get("round_number", 1)

        self.logger.info(
            f"  [{self.client_id}] Round {round_number}, FedBABU Training "
            f"({num_epochs - self.finetune_epochs} full + {self.finetune_epochs} head-only epochs)..."
        )

        # 确保body未冻结
        self.unfreeze_body()

        self.model.train()

        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        # 阶段1: 训练整个模型 (body + head)
        full_train_epochs = num_epochs - self.finetune_epochs
        if full_train_epochs > 0:
            self.logger.info(f"    [{self.client_id}] Phase 1: Training full model")
            for epoch in range(full_train_epochs):
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

        # 阶段2: 冻结body，只训练head
        if self.finetune_epochs > 0:
            self.logger.info(f"    [{self.client_id}] Phase 2: Fine-tuning head only (body frozen)")
            self.freeze_body()

            # 重新创建优化器，只包含head参数
            head_params = [p for n, p in self.model.named_parameters() if self.is_head_layer(n)]
            opt_type = self._optimizer_cfg.get('type', 'SGD').upper()
            lr = self._optimizer_cfg.get('lr', self._lr)

            if opt_type == 'SGD':
                head_optimizer = optim.SGD(
                    head_params,
                    lr=lr,
                    momentum=self._optimizer_cfg.get('momentum', 0.9)
                )
            elif opt_type == 'ADAM':
                head_optimizer = optim.Adam(head_params, lr=lr)
            else:
                raise ValueError(f"不支持的优化器类型: {opt_type}")

            for epoch in range(self.finetune_epochs):
                epoch_loss = 0.0
                epoch_correct = 0
                epoch_samples = 0

                for batch_idx, (data, target) in enumerate(self.train_loader):
                    data, target = data.to(self.device), target.to(self.device)

                    head_optimizer.zero_grad()
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    loss.backward()
                    head_optimizer.step()

                    epoch_loss += loss.item() * data.size(0)
                    pred = output.argmax(dim=1, keepdim=True)
                    epoch_correct += pred.eq(target.view_as(pred)).sum().item()
                    epoch_samples += data.size(0)

                avg_epoch_loss = epoch_loss / epoch_samples
                epoch_accuracy = epoch_correct / epoch_samples
                self.logger.info(
                    f"    [{self.client_id}] Finetune Epoch {epoch+1}: "
                    f"Loss={avg_epoch_loss:.4f}, Acc={epoch_accuracy:.4f}"
                )

                total_loss += epoch_loss
                correct_predictions += epoch_correct
                total_samples += epoch_samples

            # 解冻body供下一轮使用
            self.unfreeze_body()

        # 计算平均值
        avg_loss = total_loss / total_samples
        accuracy = correct_predictions / total_samples

        # 只上传body parameters
        body_weights = self.get_body_parameters()
        head_weights = self.get_head_parameters()

        self.logger.info(
            f"  [{self.client_id}] FedBABU: "
            f"Body params: {len(body_weights)}, "
            f"Head params: {len(head_weights)}"
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
                "model_weights": body_weights  # 只上传body parameters
            },
            execution_time=0.0
        )

        self.logger.info(
            f"  [{self.client_id}] Round {round_number} completed: "
            f"Loss={avg_loss:.4f}, Acc={accuracy:.4f}"
        )
        return response

    async def set_model(self, model_data: Dict[str, Any]) -> bool:
        """设置模型数据 - 只更新body parameters"""
        try:
            if "parameters" in model_data and "weights" in model_data["parameters"]:
                weights = model_data["parameters"]["weights"]

                # 只更新body parameters，保留personalized head
                state_dict = self.model.state_dict()
                updated_count = 0

                for name, value in weights.items():
                    if name in state_dict and not self.is_head_layer(name):
                        if not isinstance(value, torch.Tensor):
                            value = torch.from_numpy(value)
                        state_dict[name] = value.to(self.device)
                        updated_count += 1

                self.model.load_state_dict(state_dict, strict=True)

                self.logger.info(
                    f"  [{self.client_id}] FedBABU: Updated {updated_count} body parameters, "
                    f"kept personalized head unchanged"
                )
                return True
        except Exception as e:
            self.logger.exception(f"  [{self.client_id}] Failed to set model: {e}")
        return False

    async def get_model(self) -> Dict[str, Any]:
        """获取模型数据 - 返回body parameters"""
        dataset = self.dataset
        return {
            "model_type": self._model_cfg['name'],
            "parameters": {"weights": self.get_body_parameters()},
            "metadata": {
                "client_id": self.client_id,
                "samples": len(dataset),
                "param_count": get_param_count(self.model),
                "body_param_count": len(self.get_body_parameters()),
                "head_param_count": len(self.get_head_parameters())
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
