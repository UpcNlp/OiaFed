"""
FedCP (Federated Contrastive Personalization) 学习器实现

从 methods/learners/fl/fedcp.py 迁移到新架构

FedCP结合了对比学习和个性化联邦学习

核心思想：
- 使用对比学习来学习更好的特征表示
- 分离共享特征提取器和个性化分类头
- 通过对比损失拉近同类样本，推远不同类样本
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


@learner('fl.fedcp', description='FedCP: Federated Contrastive Personalization')
class FedCPLearner(Learner):
    """FedCP学习器 - 对比个性化联邦学习

    配置示例:
    {
        "learner": {
            "name": "fl.fedcp",
            "learning_rate": 0.01,
            "batch_size": 128,
            "local_epochs": 5,
            "optimizer": "SGD",
            "momentum": 0.9,
            "head_layer_names": ["fc2"],  # 个性化头部层的名称
            "lambda_contrast": 0.1,  # 对比损失权重
            "temperature": 0.5  # 对比学习温度
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

        # FedCP特有参数
        self.head_layer_names = self._config.get(
            'head_layer_names',
            ['fc2', 'fc', 'classifier', 'head']
        )
        self.lambda_contrast = self._config.get('lambda_contrast', 0.1)
        self.temperature = self._config.get('temperature', 0.5)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 组件占位符
        self.torch_model = None
        self._optimizer = None
        self._criterion = None
        self._train_loader = None
        self._test_loader = None

        self.logger.info(
            f"FedCPLearner {node_id} 初始化完成 "
            f"(lr={self.learning_rate}, lambda_contrast={self.lambda_contrast})"
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

    def is_head_layer(self, param_name: str) -> bool:
        """判断参数是否属于头部层"""
        for layer_name in self.head_layer_names:
            if layer_name in param_name:
                return True
        return False

    def get_base_parameters(self) -> Dict[str, torch.Tensor]:
        """获取base layers的参数（用于联邦聚合）"""
        base_params = {}
        for name, param in self.torch_model.named_parameters():
            if not self.is_head_layer(name):
                base_params[name] = param.data.cpu().clone()
        return base_params

    def get_features(self, model, data):
        """获取模型的特征表示"""
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
            # fallback
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

            output = model(data)
            if len(output.shape) > 2:
                output = torch.flatten(output, 1)
            return output

    def contrastive_loss(self, features, labels):
        """
        计算监督对比损失

        Args:
            features: 特征向量 (batch_size, feature_dim)
            labels: 真实标签 (batch_size,)

        Returns:
            对比损失值
        """
        # L2归一化
        features = F.normalize(features, dim=1)

        # 计算相似度矩阵
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # 创建标签mask：同类为1，不同类为0
        batch_size = labels.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(self.device)

        # 排除自己与自己的相似度
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(self.device),
            0
        )
        mask = mask * logits_mask

        # 计算对比损失
        exp_logits = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True))

        # 计算平均损失（只对同类样本）
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)
        loss = -mean_log_prob_pos.mean()

        return loss

    async def train_epoch(self, epoch: int) -> EpochMetrics:
        """单轮训练 - FedCP训练循环"""
        self.torch_model.train()

        total_loss = 0.0
        total_ce_loss = 0.0
        total_contrast_loss = 0.0
        total_correct = 0
        total_samples = 0

        for data, target in self._train_loader:
            data, target = data.to(self.device), target.to(self.device)

            self._optimizer.zero_grad()

            # 前向传播
            output = self.torch_model(data)
            ce_loss = self._criterion(output, target)

            # 计算对比损失
            features = self.get_features(self.torch_model, data)
            contrast_loss = self.contrastive_loss(features, target)

            # 总损失
            loss = ce_loss + self.lambda_contrast * contrast_loss

            loss.backward()
            self._optimizer.step()

            total_loss += loss.item() * data.size(0)
            total_ce_loss += ce_loss.item() * data.size(0)
            total_contrast_loss += contrast_loss.item() * data.size(0)

            pred = output.argmax(dim=1, keepdim=True)
            total_correct += pred.eq(target.view_as(pred)).sum().item()
            total_samples += data.size(0)

        avg_loss = total_loss / total_samples
        avg_accuracy = total_correct / total_samples

        self.logger.info(
            f"[{self._node_id}] Epoch {epoch}: "
            f"Loss={avg_loss:.4f} (CE={total_ce_loss/total_samples:.4f}, "
            f"Contrast={total_contrast_loss/total_samples:.4f}), Acc={avg_accuracy:.4f}"
        )

        epoch_metrics = EpochMetrics(
            epoch=epoch,
            avg_loss=avg_loss,
            total_samples=total_samples,
            metrics={
                'accuracy': avg_accuracy,
                'ce_loss': total_ce_loss / total_samples,
                'contrast_loss': total_contrast_loss / total_samples
            }
        )

        # 增加全局 epoch 计数器（用于 MLflow step）
        self._global_epoch_counter += 1

        # 触发 epoch 结束回调
        if self._callbacks:
            await self._callbacks.on_epoch_end(self, epoch, epoch_metrics)

        return epoch_metrics

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
        """获取模型权重 - 只返回base parameters"""
        return self.get_base_parameters()

    async def set_weights(self, weights: Dict[str, Any]):
        """设置模型权重 - 只更新base parameters"""
        # 只更新base parameters，保留personalized head
        state_dict = self.torch_model.state_dict()
        updated_count = 0

        for name, value in weights.items():
            if name in state_dict and not self.is_head_layer(name):
                if not isinstance(value, torch.Tensor):
                    value = torch.from_numpy(value)
                state_dict[name] = value.to(self.device)
                updated_count += 1

        self.torch_model.load_state_dict(state_dict, strict=True)

        self.logger.debug(
            f"[{self._node_id}] FedCP: Updated {updated_count} base parameters, "
            f"kept personalized head unchanged"
        )
