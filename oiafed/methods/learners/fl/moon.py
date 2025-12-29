"""
MOON (Model-Contrastive Federated Learning) 学习器实现

从 methods/learners/fl/moon.py 迁移到新架构

论文：Model-Contrastive Federated Learning
作者：Qinbin Li et al.
发表：CVPR 2021

MOON通过对比学习来改进联邦学习，拉近本地模型与全局模型的表示，
同时推远与之前本地模型的表示。
"""
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import copy

from ....core.learner import Learner
from ....core.types import StepMetrics, EpochMetrics, EvalResult
from ....registry import learner


@learner('fl.moon', description='MOON: Model-Contrastive Federated Learning')
class MOONLearner(Learner):
    """MOON学习器 - 使用对比学习改进联邦学习

    配置示例:
    {
        "learner": {
            "name": "fl.moon",
            "learning_rate": 0.01,
            "batch_size": 128,
            "local_epochs": 5,
            "optimizer": "SGD",
            "momentum": 0.9,
            "temperature": 0.5,  # 对比损失的温度参数
            "mu": 1.0  # 对比损失的权重
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

        # MOON特有参数
        self.temperature = self._config.get('temperature', 0.5)
        self.mu = self._config.get('mu', 1.0)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 组件占位符
        self.torch_model = None
        self._optimizer = None
        self._criterion = None
        self._train_loader = None
        self._test_loader = None

        # MOON特有：保存全局模型和之前的本地模型
        self.global_model = None
        self.previous_model = None
        self.round_number = 0

        self.logger.info(
            f"MOONLearner {node_id} 初始化完成 "
            f"(lr={self.learning_rate}, temperature={self.temperature}, mu={self.mu})"
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
        elif self.optimizer_type == 'ADAMW':
            self._optimizer = optim.AdamW(
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
        elif loss_name == 'NLLLoss':
            self._criterion = nn.NLLLoss()
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
                handle = modules[-2].register_forward_hook(hook_fn)
                _ = model(data)
                handle.remove()

                if features:
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
            if len(output.shape) > 2:
                output = torch.flatten(output, 1)
            return output

    async def train_step(self, batch, step: int) -> StepMetrics:
        """单步训练 - 包含对比损失"""
        data, target = batch
        data, target = data.to(self.device), target.to(self.device)

        self._optimizer.zero_grad()

        # 前向传播
        output = self.torch_model(data)
        ce_loss = self._criterion(output, target)

        # 计算对比损失（如果有全局模型和之前的模型）
        con_loss = torch.tensor(0.0, device=self.device)
        if self.global_model is not None and self.previous_model is not None and self.round_number > 1:
            # 获取特征表示
            z = self.get_features(self.torch_model, data)
            with torch.no_grad():
                z_global = self.get_features(self.global_model, data)
                z_prev = self.get_features(self.previous_model, data)
            con_loss = self.contrastive_loss(z, z_global, z_prev)

        # 总损失
        loss = ce_loss + self.mu * con_loss

        loss.backward()
        self._optimizer.step()

        # 计算准确率
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        accuracy = correct / data.size(0)

        return StepMetrics(
            loss=loss.item(),
            batch_size=data.size(0),
            metrics={
                'accuracy': accuracy,
                'ce_loss': ce_loss.item(),
                'contrastive_loss': con_loss.item()
            }
        )

    async def train_epoch(self, epoch: int) -> EpochMetrics:
        """单轮训练"""
        self.torch_model.train()
        if self.global_model is not None:
            self.global_model.eval()
        if self.previous_model is not None:
            self.previous_model.eval()

        total_loss = 0.0
        total_ce_loss = 0.0
        total_con_loss = 0.0
        total_correct = 0
        total_samples = 0

        for step, batch in enumerate(self._train_loader):
            step_metrics = await self.train_step(batch, step)

            total_loss += step_metrics.loss * step_metrics.batch_size
            total_ce_loss += step_metrics.metrics.get('ce_loss', 0) * step_metrics.batch_size
            total_con_loss += step_metrics.metrics.get('contrastive_loss', 0) * step_metrics.batch_size
            total_correct += step_metrics.metrics.get('accuracy', 0) * step_metrics.batch_size
            total_samples += step_metrics.batch_size

        avg_loss = total_loss / total_samples
        avg_accuracy = total_correct / total_samples

        self.logger.info(
            f"[{self._node_id}] Epoch {epoch}: "
            f"Loss={avg_loss:.4f} (CE={total_ce_loss/total_samples:.4f}, "
            f"Con={total_con_loss/total_samples:.4f}), Acc={avg_accuracy:.4f}"
        )

        epoch_metrics = EpochMetrics(
            epoch=epoch,
            avg_loss=avg_loss,
            total_samples=total_samples,
            metrics={
                'accuracy': avg_accuracy,
                'ce_loss': total_ce_loss / total_samples,
                'contrastive_loss': total_con_loss / total_samples
            }
        )

        # 增加全局 epoch 计数器（用于 MLflow step）
        self._global_epoch_counter += 1

        # 触发 epoch 结束回调
        if self._callbacks:
            await self._callbacks.on_epoch_end(self, epoch, epoch_metrics)

        return epoch_metrics

    async def fit(self, config: Optional[Dict] = None):
        """训练模型 - 重写以保存previous model"""
        # 增加轮次计数
        self.round_number += 1

        # 保存当前模型为下一轮的previous model
        if self.round_number > 1:
            self.previous_model = copy.deepcopy(self.torch_model)
            self.previous_model.eval()

        # 调用父类的fit方法
        await super().fit(config)

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

    async def set_weights(self, weights: Dict[str, Any]):
        """设置模型权重 - 保存为全局模型"""
        # 转换为torch tensor
        torch_weights = {}
        for k, v in weights.items():
            if torch.is_tensor(v):
                torch_weights[k] = v
            else:
                torch_weights[k] = torch.from_numpy(v)

        # 更新当前模型
        self.torch_model.load_state_dict(torch_weights)

        # 保存全局模型的副本
        self.global_model = copy.deepcopy(self.torch_model)
        self.global_model.eval()

        self.logger.debug(f"[{self._node_id}] MOON: Updated model and saved global model copy")

    async def get_weights(self) -> Dict[str, Any]:
        """获取模型权重"""
        return {name: param.data.clone() for name, param in self.torch_model.state_dict().items()}
