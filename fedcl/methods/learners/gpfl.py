"""
GPFL (Generalized Personalized Federated Learning) 学习器实现
fedcl/methods/learners/gpfl.py

GPFL通过混合全局模型和本地模型来实现个性化

核心思想：
- 维护一个全局共享模型和一个本地个性化模型
- 使用混合系数alpha来组合两个模型的预测
- 通过训练自适应调整混合系数
"""
from typing import Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import copy

from fedcl.api.decorators import learner
from fedcl.learner.base_learner import BaseLearner
from fedcl.types import TrainingResponse
from fedcl.methods.models.base import get_weights_as_dict, set_weights_from_dict, get_param_count


@learner('GPFL',
         description='GPFL: Generalized Personalized Federated Learning',
         version='1.0',
         author='MOE-FedCL')
class GPFLLearner(BaseLearner):
    """GPFL学习器 - 泛化个性化联邦学习

    配置示例:
    {
        "learner": {
            "name": "GPFL",
            "params": {
                "model": {"name": "SimpleCNN", "params": {"num_classes": 10}},
                "optimizer": {"type": "SGD", "lr": 0.01, "momentum": 0.9},
                "loss": "CrossEntropyLoss",
                "learning_rate": 0.01,
                "batch_size": 128,
                "local_epochs": 5,
                "alpha": 0.5  # 全局模型和本地模型的混合系数（初始值）
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

        # GPFL特有参数
        self.alpha = learner_params.get('alpha', 0.5)  # 混合系数

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

        self._model = None  # 本地模型
        self._global_model = None  # 全局模型副本
        self._optimizer = None
        self._criterion = None
        self._train_loader = None

        # 可学习的混合系数
        self.alpha_param = nn.Parameter(torch.tensor([self.alpha]))

        self.logger.info(
            f"GPFLLearner {client_id} 初始化完成 "
            f"(model={self._model_cfg.get('name')}, alpha={self.alpha})"
        )

    @property
    def model(self):
        """本地模型"""
        if self._model is None:
            from fedcl.api.registry import registry
            model_name = self._model_cfg['name']
            model_params = self._model_cfg.get('params', {})

            model_class = registry.get_model(model_name)
            self._model = model_class(**model_params).to(self.device)
        return self._model

    @property
    def global_model(self):
        """全局模型"""
        if self._global_model is None:
            from fedcl.api.registry import registry
            model_name = self._model_cfg['name']
            model_params = self._model_cfg.get('params', {})

            model_class = registry.get_model(model_name)
            self._global_model = model_class(**model_params).to(self.device)
        return self._global_model

    @property
    def optimizer(self):
        if self._optimizer is None:
            opt_type = self._optimizer_cfg.get('type', 'SGD').upper()
            lr = self._optimizer_cfg.get('lr', self._lr)

            # 优化器包含本地模型参数和混合系数
            params = list(self.model.parameters()) + [self.alpha_param]

            if opt_type == 'SGD':
                self._optimizer = optim.SGD(
                    params,
                    lr=lr,
                    momentum=self._optimizer_cfg.get('momentum', 0.9)
                )
            elif opt_type == 'ADAM':
                self._optimizer = optim.Adam(params, lr=lr)
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

    def mixed_prediction(self, data):
        """
        混合全局模型和本地模型的预测

        Args:
            data: 输入数据

        Returns:
            混合后的预测
        """
        # 本地模型预测
        local_output = self.model(data)

        # 全局模型预测
        if self._global_model is not None:
            with torch.no_grad():
                global_output = self.global_model(data)
        else:
            global_output = local_output

        # 使用sigmoid确保alpha在[0, 1]之间
        alpha = torch.sigmoid(self.alpha_param)

        # 混合预测: output = alpha * global + (1-alpha) * local
        mixed_output = alpha * global_output + (1 - alpha) * local_output

        return mixed_output, alpha.item()

    async def train(self, params: Dict[str, Any]) -> TrainingResponse:
        """训练方法 - GPFL训练循环"""
        num_epochs = params.get("num_epochs", self._epochs)
        round_number = params.get("round_number", 1)

        self.logger.info(
            f"  [{self.client_id}] Round {round_number}, "
            f"GPFL Training {num_epochs} epochs..."
        )

        self.model.train()
        if self._global_model is not None:
            self.global_model.eval()

        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        alpha_values = []

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0

            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()

                # 混合预测
                output, alpha_val = self.mixed_prediction(data)
                alpha_values.append(alpha_val)

                # 计算损失
                loss = self.criterion(output, target)

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                epoch_correct += pred.eq(target.view_as(pred)).sum().item()
                epoch_samples += data.size(0)

            avg_epoch_loss = epoch_loss / epoch_samples
            epoch_accuracy = epoch_correct / epoch_samples
            avg_alpha = sum(alpha_values) / len(alpha_values) if alpha_values else 0.5

            self.logger.info(
                f"    [{self.client_id}] Epoch {epoch+1}: "
                f"Loss={avg_epoch_loss:.4f}, Acc={epoch_accuracy:.4f}, Alpha={avg_alpha:.4f}"
            )

            total_loss += epoch_loss
            correct_predictions += epoch_correct
            total_samples += epoch_samples

        # 计算平均值
        avg_loss = total_loss / total_samples
        accuracy = correct_predictions / total_samples
        final_alpha = sum(alpha_values) / len(alpha_values) if alpha_values else 0.5

        # 获取本地模型权重（上传用于聚合）
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
                "model_weights": model_weights,
                "alpha": final_alpha  # 记录混合系数
            },
            execution_time=0.0
        )

        self.logger.info(
            f"  [{self.client_id}] Round {round_number} completed: "
            f"Loss={avg_loss:.4f}, Acc={accuracy:.4f}, Alpha={final_alpha:.4f}"
        )
        return response

    async def set_model(self, model_data: Dict[str, Any]) -> bool:
        """设置模型数据 - 更新全局模型和本地模型"""
        try:
            if "parameters" in model_data and "weights" in model_data["parameters"]:
                weights = model_data["parameters"]["weights"]
                torch_weights = {}
                for k, v in weights.items():
                    if torch.is_tensor(v):
                        torch_weights[k] = v
                    else:
                        torch_weights[k] = torch.from_numpy(v)

                # 更新全局模型
                set_weights_from_dict(self.global_model, torch_weights)
                self.global_model.eval()

                # 同时初始化本地模型（如果是第一轮）
                if self._optimizer is None:
                    set_weights_from_dict(self.model, torch_weights)

                self.logger.info(f"  [{self.client_id}] GPFL: Updated global model")
                return True
        except Exception as e:
            self.logger.exception(f"  [{self.client_id}] Failed to set model: {e}")
        return False

    async def get_model(self) -> Dict[str, Any]:
        """获取模型数据 - 返回本地模型"""
        dataset = self.dataset
        return {
            "model_type": self._model_cfg['name'],
            "parameters": {"weights": get_weights_as_dict(self.model)},
            "metadata": {
                "client_id": self.client_id,
                "samples": len(dataset),
                "param_count": get_param_count(self.model),
                "alpha": torch.sigmoid(self.alpha_param).item()
            }
        }

    async def evaluate(self, model_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """评估方法 - 使用混合预测"""
        self.model.eval()
        if self._global_model is not None:
            self.global_model.eval()

        if model_data and "model_weights" in model_data:
            weights = model_data["model_weights"]
            torch_weights = {}
            for k, v in weights.items():
                if isinstance(v, torch.Tensor):
                    torch_weights[k] = v
                else:
                    torch_weights[k] = torch.from_numpy(v)
            set_weights_from_dict(self.global_model, torch_weights)

        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                output, _ = self.mixed_prediction(data)
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
