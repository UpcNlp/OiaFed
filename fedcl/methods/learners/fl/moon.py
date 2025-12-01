"""
MOON (Model-Contrastive Federated Learning) 学习器实现
fedcl/methods/learners/moon.py

论文：Model-Contrastive Federated Learning
作者：Qinbin Li et al.
发表：CVPR 2021

MOON通过对比学习来改进联邦学习，拉近本地模型与全局模型的表示，
同时推远与之前本地模型的表示。
"""
from typing import Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import copy

from .._decorators import learner
from fedcl.learner.base_learner import BaseLearner
from fedcl.types import TrainingResponse
from fedcl.methods.models.base import (
    get_weights_as_dict,
    set_weights_from_dict,
    get_param_count
)


@learner('fl', 'MOON', description='MOON: Model-Contrastive Federated Learning')
class MOONLearner(BaseLearner):
    """MOON学习器 - 使用对比学习改进联邦学习

    配置示例:
    {
        "learner": {
            "name": "MOON",
            "params": {
                "model": {"name": "SimpleCNN", "params": {"num_classes": 10}},
                "optimizer": {"type": "SGD", "lr": 0.01, "momentum": 0.9},
                "loss": "CrossEntropyLoss",
                "learning_rate": 0.01,
                "batch_size": 128,
                "local_epochs": 5,
                "temperature": 0.5,  # 对比损失的温度参数
                "mu": 1.0  # 对比损失的权重
            }
        },
        "dataset": {
            "name": "CIFAR10",
            "params": {...},
            "partition": {...}
        }
    }
    """

    def __init__(self, client_id: str, config: Dict[str, Any] = None, lazy_init: bool = True):
        # 先提取配置
        learner_params = (config or {}).get('learner', {}).get('params', {})

        # 保存模型、优化器、损失函数配置
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

        # MOON特有参数
        self.temperature = learner_params.get('temperature', 0.5)
        self.mu = learner_params.get('mu', 1.0)

        # 创建一个不包含冲突字段的config副本传递给父类
        clean_config = config.copy() if config else {}
        if 'learner' in clean_config and 'params' in clean_config['learner']:
            clean_params = clean_config['learner']['params'].copy()
            clean_params.pop('model', None)
            clean_params.pop('optimizer', None)
            clean_params.pop('loss', None)
            clean_config['learner'] = clean_config['learner'].copy()
            clean_config['learner']['params'] = clean_params

        # 调用父类的__init__
        super().__init__(client_id, clean_config, lazy_init)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 组件占位符
        self._model = None
        self._optimizer = None
        self._criterion = None
        self._train_loader = None

        # MOON特有：保存全局模型和之前的本地模型
        self.global_model = None
        self.previous_model = None

        self.logger.info(
            f"MOONLearner {client_id} 初始化完成 "
            f"(model={self._model_cfg.get('name')}, temperature={self.temperature}, mu={self.mu})"
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
                raise NotImplementedError("暂不支持带参数的损失函数配置")

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

    def contrastive_loss(self, z, z_global, z_prev):
        """
        计算MOON的对比损失

        Args:
            z: 当前模型的特征表示 (batch_size, feature_dim)
            z_global: 全局模型的特征表示 (batch_size, feature_dim)
            z_prev: 之前本地模型的特征表示 (batch_size, feature_dim)

        Returns:
            对比损失值
        """
        # L2归一化
        z = F.normalize(z, dim=1)
        z_global = F.normalize(z_global, dim=1)
        z_prev = F.normalize(z_prev, dim=1)

        # 计算相似度
        sim_global = torch.exp(torch.sum(z * z_global, dim=1) / self.temperature)
        sim_prev = torch.exp(torch.sum(z * z_prev, dim=1) / self.temperature)

        # 对比损失：-log(sim_global / (sim_global + sim_prev))
        loss = -torch.log(sim_global / (sim_global + sim_prev))

        return loss.mean()

    def get_features(self, model, data):
        """
        获取模型的特征表示（倒数第二层的输出）

        Args:
            model: 模型
            data: 输入数据

        Returns:
            特征表示
        """
        # 尝试找到最后的全连接层
        # 常见的名称：fc, fc2, classifier, head
        last_fc_candidates = ['fc2', 'fc', 'classifier', 'head']

        last_fc_name = None
        for name in last_fc_candidates:
            if hasattr(model, name):
                last_fc_name = name
                break

        if last_fc_name:
            # 保存原始的全连接层
            original_fc = getattr(model, last_fc_name)
            # 临时替换为Identity
            setattr(model, last_fc_name, nn.Identity())
            # 前向传播获取特征
            features = model(data)
            # 恢复原始全连接层
            setattr(model, last_fc_name, original_fc)
            return features
        else:
            # 使用hook方式获取倒数第二层的输出
            features = []

            def hook_fn(module, input, output):
                features.append(output)

            # 注册hook到倒数第二个模块
            modules = list(model.children())
            if len(modules) >= 2:
                # 在倒数第二个模块上注册hook
                handle = modules[-2].register_forward_hook(hook_fn)
                _ = model(data)
                handle.remove()

                if features:
                    # 如果特征是多维的，展平
                    feat = features[0]
                    if len(feat.shape) > 2:
                        feat = torch.flatten(feat, 1)
                    return feat

            # 如果以上方法都失败，使用完整输出
            self.logger.warning(
                f"无法提取特征，使用模型完整输出。"
                f"建议：确保模型有 'fc', 'fc2', 'classifier' 或 'head' 属性"
            )
            output = model(data)
            # 如果输出是多维的，展平
            if len(output.shape) > 2:
                output = torch.flatten(output, 1)
            return output

    async def train(self, params: Dict[str, Any]) -> TrainingResponse:
        """训练方法 - MOON训练循环"""
        num_epochs = params.get("num_epochs", self._epochs)
        round_number = params.get("round_number", 1)

        self.logger.info(
            f"  [{self.client_id}] Round {round_number}, "
            f"MOON Training {num_epochs} epochs (mu={self.mu}, T={self.temperature})..."
        )

        # 保存当前模型为下一轮的previous model
        if self.previous_model is None and round_number > 1:
            self.previous_model = copy.deepcopy(self.model)

        # 设置模型为训练模式
        self.model.train()
        if self.global_model is not None:
            self.global_model.eval()
        if self.previous_model is not None:
            self.previous_model.eval()

        total_loss = 0.0
        total_ce_loss = 0.0
        total_con_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_ce_loss = 0.0
            epoch_con_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0

            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()

                # 前向传播
                output = self.model(data)
                ce_loss = self.criterion(output, target)

                # 计算对比损失（如果有全局模型和之前的模型）
                con_loss = torch.tensor(0.0, device=self.device)
                if self.global_model is not None and self.previous_model is not None and round_number > 1:
                    # 获取特征表示
                    z = self.get_features(self.model, data)
                    with torch.no_grad():
                        z_global = self.get_features(self.global_model, data)
                        z_prev = self.get_features(self.previous_model, data)
                    con_loss = self.contrastive_loss(z, z_global, z_prev)

                # 总损失
                loss = ce_loss + self.mu * con_loss

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * data.size(0)
                epoch_ce_loss += ce_loss.item() * data.size(0)
                epoch_con_loss += con_loss.item() * data.size(0)

                pred = output.argmax(dim=1, keepdim=True)
                epoch_correct += pred.eq(target.view_as(pred)).sum().item()
                epoch_samples += data.size(0)

            avg_epoch_loss = epoch_loss / epoch_samples
            epoch_accuracy = epoch_correct / epoch_samples
            self.logger.info(
                f"    [{self.client_id}] Epoch {epoch+1}: "
                f"Loss={avg_epoch_loss:.4f} (CE={epoch_ce_loss/epoch_samples:.4f}, "
                f"Con={epoch_con_loss/epoch_samples:.4f}), Acc={epoch_accuracy:.4f}"
            )

            total_loss += epoch_loss
            total_ce_loss += epoch_ce_loss
            total_con_loss += epoch_con_loss
            correct_predictions += epoch_correct
            total_samples += epoch_samples

        # 保存当前模型为下一轮的previous model
        self.previous_model = copy.deepcopy(self.model)

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
                "ce_loss": total_ce_loss / total_samples,
                "contrastive_loss": total_con_loss / total_samples,
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

    async def set_model(self, model_data: Dict[str, Any]) -> bool:
        """设置模型数据 - 保存为全局模型"""
        try:
            if "parameters" in model_data and "weights" in model_data["parameters"]:
                weights = model_data["parameters"]["weights"]
                torch_weights = {}
                for k, v in weights.items():
                    if torch.is_tensor(v):
                        torch_weights[k] = v
                    else:
                        torch_weights[k] = torch.from_numpy(v)

                # 更新当前模型
                set_weights_from_dict(self.model, torch_weights)

                # 保存全局模型的副本
                self.global_model = copy.deepcopy(self.model)
                self.global_model.eval()

                return True
        except Exception as e:
            self.logger.exception(f"  [{self.client_id}] Failed to set model: {e}")
        return False

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

    def get_data_statistics(self) -> Dict[str, Any]:
        """获取数据统计"""
        dataset = self.dataset
        stats = {
            "total_samples": len(dataset),
        }

        if hasattr(dataset, 'get_statistics'):
            stats.update(dataset.get_statistics())

        return stats

    async def get_local_model(self) -> Dict[str, Any]:
        return await self.get_model()

    async def set_local_model(self, model_data: Dict[str, Any]) -> bool:
        return await self.set_model(model_data)
