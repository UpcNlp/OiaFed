"""
FedProto (Federated Prototypical Learning) 学习器实现
fedcl/methods/learners/fedproto.py

论文：FedProto: Federated Prototype Learning across Heterogeneous Clients
作者：Yue Tan et al.
发表：AAAI 2022

FedProto的核心思想：
- 使用原型（prototypes）来表示每个类别的特征中心
- 客户端之间共享类别原型而不是模型参数
- 使用原型进行知识蒸馏，提高泛化性能
"""
from typing import Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import copy

from fedcl.api.decorators import learner
from fedcl.learner.base_learner import BaseLearner
from fedcl.types import TrainingResponse
from fedcl.methods.models.base import get_weights_as_dict, set_weights_from_dict, get_param_count


@learner('FedProto',
         description='FedProto: Federated Prototypical Learning',
         version='1.0',
         author='MOE-FedCL')
class FedProtoLearner(BaseLearner):
    """FedProto学习器 - 使用原型学习的联邦学习

    配置示例:
    {
        "learner": {
            "name": "FedProto",
            "params": {
                "model": {"name": "SimpleCNN", "params": {"num_classes": 10}},
                "optimizer": {"type": "SGD", "lr": 0.01, "momentum": 0.9},
                "loss": "CrossEntropyLoss",
                "learning_rate": 0.01,
                "batch_size": 128,
                "local_epochs": 5,
                "num_classes": 10,
                "lambda_proto": 1.0,  # 原型损失的权重
                "temperature": 0.5  # 原型对比的温度参数
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

        # FedProto特有参数
        self.num_classes = learner_params.get('num_classes', 10)
        self.lambda_proto = learner_params.get('lambda_proto', 1.0)
        self.temperature = learner_params.get('temperature', 0.5)

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

        # 原型相关
        self.local_prototypes = None  # 本地原型
        self.global_prototypes = None  # 全局原型

        self.logger.info(
            f"FedProtoLearner {client_id} 初始化完成 "
            f"(model={self._model_cfg.get('name')}, "
            f"lambda_proto={self.lambda_proto})"
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

    def get_features(self, model, data):
        """获取模型的特征表示（倒数第二层的输出）"""
        last_fc_candidates = ['fc2', 'fc', 'classifier', 'head']

        last_fc_name = None
        for name in last_fc_candidates:
            if hasattr(model, name):
                last_fc_name = name
                break

        if last_fc_name:
            original_fc = getattr(model, last_fc_name)
            setattr(model, last_fc_name, nn.Identity())
            features = model(data)
            setattr(model, last_fc_name, original_fc)
            return features
        else:
            # 使用hook方式获取倒数第二层的输出
            features = []

            def hook_fn(module, input, output):
                features.append(output)

            modules = list(model.children())
            if len(modules) >= 2:
                handle = modules[-2].register_forward_hook(hook_fn)
                _ = model(data)
                handle.remove()

                if features:
                    feat = features[0]
                    if len(feat.shape) > 2:
                        feat = torch.flatten(feat, 1)
                    return feat

            self.logger.warning(
                f"无法提取特征，使用模型完整输出。"
                f"建议：确保模型有 'fc', 'fc2', 'classifier' 或 'head' 属性"
            )
            output = model(data)
            if len(output.shape) > 2:
                output = torch.flatten(output, 1)
            return output

    def compute_prototypes(self):
        """计算本地原型"""
        self.model.eval()

        # 初始化原型存储
        proto_features = {i: [] for i in range(self.num_classes)}

        with torch.no_grad():
            for data, target in self.train_loader:
                data = data.to(self.device)
                features = self.get_features(self.model, data)

                # 按类别收集特征
                for i in range(len(target)):
                    label = target[i].item()
                    proto_features[label].append(features[i].cpu())

        # 计算每个类别的原型（平均特征）
        prototypes = {}
        for label in range(self.num_classes):
            if len(proto_features[label]) > 0:
                prototypes[label] = torch.stack(proto_features[label]).mean(dim=0)
            else:
                # 如果该类别没有样本，使用零向量
                feature_dim = next(iter(proto_features.values()))[0].shape[0] if any(proto_features.values()) else 128
                prototypes[label] = torch.zeros(feature_dim)

        self.local_prototypes = prototypes
        self.logger.info(f"  [{self.client_id}] Computed prototypes for {len([p for p in prototypes.values() if p.sum() != 0])} classes")

        return prototypes

    def prototype_loss(self, features, targets):
        """
        计算原型损失

        Args:
            features: 特征向量 (batch_size, feature_dim)
            targets: 真实标签 (batch_size,)

        Returns:
            原型损失值
        """
        if self.global_prototypes is None:
            return torch.tensor(0.0, device=self.device)

        batch_loss = 0.0
        valid_samples = 0

        for i in range(len(targets)):
            label = targets[i].item()
            feature = features[i]

            # 获取全局原型
            if label not in self.global_prototypes:
                continue

            proto = self.global_prototypes[label].to(self.device)

            # 计算与原型的距离（使用余弦相似度）
            feature_norm = F.normalize(feature.unsqueeze(0), dim=1)
            proto_norm = F.normalize(proto.unsqueeze(0), dim=1)

            similarity = torch.sum(feature_norm * proto_norm)

            # 负相似度作为损失（希望特征接近原型）
            batch_loss += -similarity
            valid_samples += 1

        if valid_samples > 0:
            return batch_loss / valid_samples
        else:
            return torch.tensor(0.0, device=self.device)

    async def train(self, params: Dict[str, Any]) -> TrainingResponse:
        """训练方法 - FedProto训练循环"""
        num_epochs = params.get("num_epochs", self._epochs)
        round_number = params.get("round_number", 1)

        self.logger.info(
            f"  [{self.client_id}] Round {round_number}, "
            f"FedProto Training {num_epochs} epochs..."
        )

        self.model.train()

        total_loss = 0.0
        total_ce_loss = 0.0
        total_proto_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_ce_loss = 0.0
            epoch_proto_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0

            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()

                # 前向传播
                output = self.model(data)
                ce_loss = self.criterion(output, target)

                # 计算原型损失
                proto_loss = torch.tensor(0.0, device=self.device)
                if self.global_prototypes is not None and round_number > 1:
                    features = self.get_features(self.model, data)
                    proto_loss = self.prototype_loss(features, target)

                # 总损失
                loss = ce_loss + self.lambda_proto * proto_loss

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * data.size(0)
                epoch_ce_loss += ce_loss.item() * data.size(0)
                epoch_proto_loss += proto_loss.item() * data.size(0)

                pred = output.argmax(dim=1, keepdim=True)
                epoch_correct += pred.eq(target.view_as(pred)).sum().item()
                epoch_samples += data.size(0)

            avg_epoch_loss = epoch_loss / epoch_samples
            epoch_accuracy = epoch_correct / epoch_samples
            self.logger.info(
                f"    [{self.client_id}] Epoch {epoch+1}: "
                f"Loss={avg_epoch_loss:.4f} (CE={epoch_ce_loss/epoch_samples:.4f}, "
                f"Proto={epoch_proto_loss/epoch_samples:.4f}), Acc={epoch_accuracy:.4f}"
            )

            total_loss += epoch_loss
            total_ce_loss += epoch_ce_loss
            total_proto_loss += epoch_proto_loss
            correct_predictions += epoch_correct
            total_samples += epoch_samples

        # 计算本地原型
        local_prototypes = self.compute_prototypes()

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
                "proto_loss": total_proto_loss / total_samples,
                "accuracy": accuracy,
                "samples_used": total_samples,
                "model_weights": model_weights,
                "prototypes": local_prototypes  # 上传本地原型
            },
            execution_time=0.0
        )

        self.logger.info(
            f"  [{self.client_id}] Round {round_number} completed: "
            f"Loss={avg_loss:.4f}, Acc={accuracy:.4f}"
        )
        return response

    async def set_model(self, model_data: Dict[str, Any]) -> bool:
        """设置模型数据和全局原型"""
        try:
            # 更新模型权重
            if "parameters" in model_data and "weights" in model_data["parameters"]:
                weights = model_data["parameters"]["weights"]
                torch_weights = {}
                for k, v in weights.items():
                    if torch.is_tensor(v):
                        torch_weights[k] = v
                    else:
                        torch_weights[k] = torch.from_numpy(v)

                set_weights_from_dict(self.model, torch_weights)

            # 更新全局原型
            if "prototypes" in model_data:
                self.global_prototypes = model_data["prototypes"]
                self.logger.info(
                    f"  [{self.client_id}] Updated global prototypes for "
                    f"{len(self.global_prototypes)} classes"
                )

            return True
        except Exception as e:
            self.logger.exception(f"  [{self.client_id}] Failed to set model: {e}")
        return False

    async def get_model(self) -> Dict[str, Any]:
        """获取模型数据和本地原型"""
        dataset = self.dataset
        return {
            "model_type": self._model_cfg['name'],
            "parameters": {"weights": get_weights_as_dict(self.model)},
            "prototypes": self.local_prototypes if self.local_prototypes else {},
            "metadata": {
                "client_id": self.client_id,
                "samples": len(dataset),
                "param_count": get_param_count(self.model)
            }
        }

    async def evaluate(self, model_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """评估方法"""
        self.model.eval()

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
        stats = {"total_samples": len(dataset)}
        if hasattr(dataset, 'get_statistics'):
            stats.update(dataset.get_statistics())
        return stats

    async def get_local_model(self) -> Dict[str, Any]:
        return await self.get_model()

    async def set_local_model(self, model_data: Dict[str, Any]) -> bool:
        return await self.set_model(model_data)
