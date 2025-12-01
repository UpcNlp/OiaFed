"""
FedCP (Federated Contrastive Personalization) 学习器实现
fedcl/methods/learners/fedcp.py

FedCP结合了对比学习和个性化联邦学习

核心思想：
- 使用对比学习来学习更好的特征表示
- 分离共享特征提取器和个性化分类头
- 通过对比损失拉近同类样本，推远不同类样本
"""
from typing import Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from .._decorators import learner
from fedcl.learner.base_learner import BaseLearner
from fedcl.types import TrainingResponse
from fedcl.methods.models.base import get_param_count


@learner('fl', 'FedCP', description='FedCP: Federated Contrastive Personalization')
class FedCPLearner(BaseLearner):
    """FedCP学习器 - 对比个性化联邦学习

    配置示例:
    {
        "learner": {
            "name": "FedCP",
            "params": {
                "model": {"name": "SimpleCNN", "params": {"num_classes": 10}},
                "optimizer": {"type": "SGD", "lr": 0.01, "momentum": 0.9},
                "loss": "CrossEntropyLoss",
                "learning_rate": 0.01,
                "batch_size": 128,
                "local_epochs": 5,
                "head_layer_names": ["fc2"],  # 个性化头部层的名称
                "lambda_contrast": 0.1,  # 对比损失权重
                "temperature": 0.5  # 对比学习温度
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

        # FedCP特有参数
        self.head_layer_names = learner_params.get(
            'head_layer_names',
            ['fc2', 'fc', 'classifier', 'head']
        )
        self.lambda_contrast = learner_params.get('lambda_contrast', 0.1)
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

        self.logger.info(
            f"FedCPLearner {client_id} 初始化完成 "
            f"(model={self._model_cfg.get('name')}, "
            f"lambda_contrast={self.lambda_contrast})"
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

    def get_base_parameters(self) -> Dict[str, torch.Tensor]:
        """获取base layers的参数（用于联邦聚合）"""
        base_params = {}
        for name, param in self.model.named_parameters():
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

    async def train(self, params: Dict[str, Any]) -> TrainingResponse:
        """训练方法 - FedCP训练循环"""
        num_epochs = params.get("num_epochs", self._epochs)
        round_number = params.get("round_number", 1)

        self.logger.info(
            f"  [{self.client_id}] Round {round_number}, "
            f"FedCP Training {num_epochs} epochs..."
        )

        self.model.train()

        total_loss = 0.0
        total_ce_loss = 0.0
        total_contrast_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_ce_loss = 0.0
            epoch_contrast_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0

            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()

                # 前向传播
                output = self.model(data)
                ce_loss = self.criterion(output, target)

                # 计算对比损失
                features = self.get_features(self.model, data)
                contrast_loss = self.contrastive_loss(features, target)

                # 总损失
                loss = ce_loss + self.lambda_contrast * contrast_loss

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * data.size(0)
                epoch_ce_loss += ce_loss.item() * data.size(0)
                epoch_contrast_loss += contrast_loss.item() * data.size(0)

                pred = output.argmax(dim=1, keepdim=True)
                epoch_correct += pred.eq(target.view_as(pred)).sum().item()
                epoch_samples += data.size(0)

            avg_epoch_loss = epoch_loss / epoch_samples
            epoch_accuracy = epoch_correct / epoch_samples
            self.logger.info(
                f"    [{self.client_id}] Epoch {epoch+1}: "
                f"Loss={avg_epoch_loss:.4f} (CE={epoch_ce_loss/epoch_samples:.4f}, "
                f"Contrast={epoch_contrast_loss/epoch_samples:.4f}), Acc={epoch_accuracy:.4f}"
            )

            total_loss += epoch_loss
            total_ce_loss += epoch_ce_loss
            total_contrast_loss += epoch_contrast_loss
            correct_predictions += epoch_correct
            total_samples += epoch_samples

        # 计算平均值
        avg_loss = total_loss / total_samples
        accuracy = correct_predictions / total_samples

        # 只上传base parameters
        base_weights = self.get_base_parameters()

        # 创建训练响应
        response = TrainingResponse(
            request_id="",
            client_id=self.client_id,
            success=True,
            result={
                "epochs_completed": num_epochs,
                "loss": avg_loss,
                "ce_loss": total_ce_loss / total_samples,
                "contrast_loss": total_contrast_loss / total_samples,
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

                # 只更新base parameters，保留personalized head
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
                    f"  [{self.client_id}] FedCP: Updated {updated_count} base parameters"
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
                "param_count": get_param_count(self.model)
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
