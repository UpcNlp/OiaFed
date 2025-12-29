"""
FedProto (Federated Prototypical Learning) 学习器实现

从 methods/learners/fl/fedproto.py 迁移到新架构

论文：FedProto: Federated Prototype Learning across Heterogeneous Clients
作者：Yue Tan et al.
发表：AAAI 2022

FedProto的核心思想：
- 使用原型（prototypes）来表示每个类别的特征中心
- 客户端之间共享类别原型而不是模型参数
- 使用原型进行知识蒸馏，提高泛化性能
"""
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from ....core.learner import Learner
from ....core.types import StepMetrics, EpochMetrics, EvalResult
from ....registry import learner


@learner('fl.fedproto', description='FedProto: Federated Prototypical Learning')
class FedProtoLearner(Learner):
    """FedProto学习器 - 使用原型学习的联邦学习

    配置示例:
    {
        "learner": {
            "name": "fl.fedproto",
            "learning_rate": 0.01,
            "batch_size": 128,
            "local_epochs": 5,
            "optimizer": "SGD",
            "momentum": 0.9,
            "num_classes": 10,
            "lambda_proto": 1.0,  # 原型损失的权重
            "temperature": 0.5  # 原型对比的温度参数
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

        # FedProto特有参数
        self.num_classes = self._config.get('num_classes', 10)
        self.lambda_proto = self._config.get('lambda_proto', 1.0)
        self.temperature = self._config.get('temperature', 0.5)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 组件占位符
        self.torch_model = None
        self._optimizer = None
        self._criterion = None
        self._train_loader = None
        self._test_loader = None

        # 原型相关
        self.local_prototypes = None  # 本地原型
        self.global_prototypes = None  # 全局原型
        self.round_number = 0

        self.logger.info(
            f"FedProtoLearner {node_id} 初始化完成 "
            f"(lr={self.learning_rate}, num_classes={self.num_classes}, lambda_proto={self.lambda_proto})"
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
        self.torch_model.eval()

        # 初始化原型存储
        proto_features = {i: [] for i in range(self.num_classes)}

        with torch.no_grad():
            for data, target in self._train_loader:
                data = data.to(self.device)
                features = self.get_features(self.torch_model, data)

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
        self.logger.info(
            f"[{self._node_id}] Computed prototypes for "
            f"{len([p for p in prototypes.values() if p.sum() != 0])} classes"
        )

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

    async def train_epoch(self, epoch: int) -> EpochMetrics:
        """单轮训练 - FedProto训练循环"""
        self.torch_model.train()

        total_loss = 0.0
        total_ce_loss = 0.0
        total_proto_loss = 0.0
        total_correct = 0
        total_samples = 0

        for data, target in self._train_loader:
            data, target = data.to(self.device), target.to(self.device)

            self._optimizer.zero_grad()

            # 前向传播
            output = self.torch_model(data)
            ce_loss = self._criterion(output, target)

            # 计算原型损失
            proto_loss = torch.tensor(0.0, device=self.device)
            if self.global_prototypes is not None and self.round_number > 1:
                features = self.get_features(self.torch_model, data)
                proto_loss = self.prototype_loss(features, target)

            # 总损失
            loss = ce_loss + self.lambda_proto * proto_loss

            loss.backward()
            self._optimizer.step()

            total_loss += loss.item() * data.size(0)
            total_ce_loss += ce_loss.item() * data.size(0)
            total_proto_loss += proto_loss.item() * data.size(0)

            pred = output.argmax(dim=1, keepdim=True)
            total_correct += pred.eq(target.view_as(pred)).sum().item()
            total_samples += data.size(0)

        avg_loss = total_loss / total_samples
        avg_accuracy = total_correct / total_samples

        self.logger.info(
            f"[{self._node_id}] Epoch {epoch}: "
            f"Loss={avg_loss:.4f} (CE={total_ce_loss/total_samples:.4f}, "
            f"Proto={total_proto_loss/total_samples:.4f}), Acc={avg_accuracy:.4f}"
        )

        epoch_metrics = EpochMetrics(
            epoch=epoch,
            avg_loss=avg_loss,
            total_samples=total_samples,
            metrics={
                'accuracy': avg_accuracy,
                'ce_loss': total_ce_loss / total_samples,
                'proto_loss': total_proto_loss / total_samples
            }
        )

        # 增加全局 epoch 计数器（用于 MLflow step）
        self._global_epoch_counter += 1

        # 触发 epoch 结束回调
        if self._callbacks:
            await self._callbacks.on_epoch_end(self, epoch, epoch_metrics)

        return epoch_metrics

    async def fit(self, config: Optional[Dict] = None):
        """训练模型 - 重写以计算原型"""
        # 增加轮次计数
        self.round_number += 1

        # 调用父类的fit方法
        await super().fit(config)

        # 训练后计算本地原型
        self.compute_prototypes()

    async def evaluate_model(self, config: Optional[Dict] = None) -> EvalResult:
        """评估模型"""
        self.torch_model.eval()

        loader = self._test_loader if self._test_loader else self._train_loader

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for data, target in loader:
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

    async def get_weights(self) -> Dict[str, Any]:
        """获取模型权重和本地原型"""
        weights = {name: param.data.clone() for name, param in self.torch_model.state_dict().items()}

        # 添加原型信息
        if self.local_prototypes is not None:
            weights['_prototypes'] = self.local_prototypes

        return weights

    async def set_weights(self, weights: Dict[str, Any]):
        """设置模型权重和全局原型"""
        # 提取原型信息
        if '_prototypes' in weights:
            self.global_prototypes = weights.pop('_prototypes')
            self.logger.info(
                f"[{self._node_id}] Updated global prototypes for "
                f"{len(self.global_prototypes)} classes"
            )

        # 更新模型权重
        torch_weights = {}
        for k, v in weights.items():
            if torch.is_tensor(v):
                torch_weights[k] = v
            else:
                torch_weights[k] = torch.from_numpy(v)

        self.torch_model.load_state_dict(torch_weights, strict=False)
