"""
SplitNN: Split Learning for Vertical Federated Learning
MIT Media Lab, 2018

Paper: Split Learning for Health: Distributed Deep Learning without Sharing Raw Data
       https://arxiv.org/abs/1812.00564

核心思想：
1. 模型垂直分割：不同参与方持有模型的不同层
2. 前向传播：客户端计算到切分点，将激活值（smashed data）发送给服务端
3. 反向传播：服务端计算梯度，将切分点梯度返回给客户端
4. 数据隐私：原始数据不离开本地，只传输中间表示

适用场景：
- 纵向联邦学习（VFL）：不同参与方持有相同样本的不同特征
- 医疗数据：多个医院联合训练，数据不出本地
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List, Tuple
import copy

from ....core.learner import Learner
from ....core.types import EpochMetrics, EvalResult, StepMetrics, TrainResult
from ....registry import learner


class SplitModel(nn.Module):
    """
    可分割的模型包装器
    
    将模型分割成 client_model 和 server_model 两部分
    
    支持两种分割方式:
    1. 如果模型有 features/classifier 属性（如 VGG），直接使用
    2. 否则按 split_layer 索引分割，并自动处理 flatten
    """
    
    def __init__(self, full_model: nn.Module, split_layer: int = -1):
        """
        Args:
            full_model: 完整模型
            split_layer: 切分位置（从0开始的层索引，-1表示自动选择）
                        对于CNN，建议设为卷积层数量（在全连接层之前分割）
        """
        super().__init__()
        
        self._needs_flatten = False
        
        # 获取所有层
        if hasattr(full_model, 'features') and hasattr(full_model, 'classifier'):
            # 类似 VGG 结构：features（卷积）+ classifier（全连接）
            self.client_layers = full_model.features
            self.server_layers = full_model.classifier
            self._needs_flatten = True
        else:
            # 通用处理：分析模型结构
            layers = list(full_model.children())
            
            if split_layer == -1:
                # 自动选择：找到第一个 Linear 层之前的位置
                split_layer = self._find_split_point(layers)
            
            self.client_layers = nn.Sequential(*layers[:split_layer])
            self.server_layers = nn.Sequential(*layers[split_layer:])
            
            # 检查是否需要在中间 flatten
            self._needs_flatten = self._check_needs_flatten(layers, split_layer)
    
    def _find_split_point(self, layers: list) -> int:
        """找到卷积层和全连接层之间的分割点"""
        for i, layer in enumerate(layers):
            if isinstance(layer, nn.Linear):
                return i
        # 如果没找到 Linear，默认中间分割
        return len(layers) // 2
    
    def _check_needs_flatten(self, layers: list, split_layer: int) -> bool:
        """检查是否需要在分割点 flatten"""
        if split_layer >= len(layers):
            return False
        
        # 如果分割后第一层是 Linear，需要 flatten
        first_server_layer = layers[split_layer]
        if isinstance(first_server_layer, nn.Linear):
            return True
        
        # 如果是 Sequential，检查其第一个子层
        if isinstance(first_server_layer, nn.Sequential):
            children = list(first_server_layer.children())
            if children and isinstance(children[0], nn.Linear):
                return True
        
        return False
    
    def client_forward(self, x: torch.Tensor) -> torch.Tensor:
        """客户端前向传播"""
        return self.client_layers(x)
    
    def server_forward(self, smashed_data: torch.Tensor) -> torch.Tensor:
        """服务端前向传播"""
        return self.server_layers(smashed_data)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """完整前向传播（用于评估）"""
        smashed = self.client_forward(x)
        if self._needs_flatten and len(smashed.shape) > 2:
            smashed = smashed.view(smashed.size(0), -1)
        return self.server_forward(smashed)


@learner('vfl.splitnn', description='SplitNN: Split Learning for Vertical Federated Learning (MIT 2018)')
class SplitNNLearner(Learner):
    """
    SplitNN 客户端学习器
    
    实现分割学习的客户端逻辑：
    1. 本地数据前向传播到切分点
    2. 发送 smashed data 给服务端
    3. 接收梯度并完成本地反向传播
    
    配置示例:
    {
        "learner": {
            "name": "vfl.splitnn",
            "split_layer": 2,
            "learning_rate": 0.01,
            "batch_size": 64,
            "local_epochs": 1
        }
    }
    """
    
    def __init__(
        self,
        model,
        datasets=None,
        tracker=None,
        callbacks=None,
        config=None,
        node_id=None
    ):
        super().__init__(model, None, tracker, callbacks, config, node_id)
        
        self._datasets = datasets or {}
        
        # 配置参数
        self.split_layer = self._config.get('split_layer', -1)
        self.learning_rate = self._config.get('learning_rate', 0.01)
        self.batch_size = self._config.get('batch_size', 64)
        self.local_epochs = self._config.get('local_epochs', 1)
        self.momentum = self._config.get('momentum', 0.9)
        self.weight_decay = self._config.get('weight_decay', 5e-4)
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 组件
        self.split_model: Optional[SplitModel] = None
        self.client_optimizer: Optional[optim.Optimizer] = None
        self._train_loader: Optional[DataLoader] = None
        self._test_loader: Optional[DataLoader] = None
        
        # 训练状态
        self._smashed_data: Optional[torch.Tensor] = None
        self._labels: Optional[torch.Tensor] = None
        
        self.logger.info(
            f"SplitNNLearner {node_id} 初始化: split_layer={self.split_layer}"
        )
    
    async def setup(self, config: Dict) -> None:
        """初始化训练环境"""
        # 获取原始模型
        if hasattr(self._model, 'get_model'):
            base_model = self._model.get_model()
        else:
            base_model = self._model
        
        # 创建分割模型
        self.split_model = SplitModel(base_model, self.split_layer)
        self.split_model.to(self.device)
        
        # 只为客户端层创建优化器
        self.client_optimizer = optim.SGD(
            self.split_model.client_layers.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )
        
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
        
        self.logger.info(f"SplitNN setup 完成: device={self.device}")
    
    async def client_forward(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        客户端前向传播
        
        Args:
            batch: (data, labels) 元组
            
        Returns:
            smashed_data: 切分点的激活值（需要发送给服务端）
        """
        data, labels = batch
        data = data.to(self.device)
        labels = labels.to(self.device)
        
        # 保存标签用于后续
        self._labels = labels
        
        # 客户端前向传播
        self.split_model.client_layers.train()
        smashed_data = self.split_model.client_forward(data)
        
        # 展平（如果需要）
        if len(smashed_data.shape) > 2:
            smashed_data = smashed_data.view(smashed_data.size(0), -1)
        
        # 保存用于反向传播
        self._smashed_data = smashed_data
        
        # 返回时 detach，因为要发送给服务端
        return smashed_data.detach().clone()
    
    async def client_backward(self, grad_smashed: torch.Tensor) -> None:
        """
        客户端反向传播
        
        Args:
            grad_smashed: 服务端返回的切分点梯度
        """
        if self._smashed_data is None:
            raise RuntimeError("Must call client_forward before client_backward")
        
        grad_smashed = grad_smashed.to(self.device)
        
        # 客户端反向传播
        self.client_optimizer.zero_grad()
        
        # 使用保存的 smashed_data 进行反向传播
        self._smashed_data.backward(grad_smashed)
        
        # 更新客户端参数
        self.client_optimizer.step()
        
        # 清理
        self._smashed_data = None
        self._labels = None
    
    async def train_step(self, batch: Any, batch_idx: int) -> StepMetrics:
        """
        单步训练（用于端到端测试）
        
        在实际 VFL 场景中，train_step 会被分解为：
        1. client_forward -> 发送 smashed_data
        2. [服务端处理]
        3. client_backward <- 接收 grad_smashed
        """
        data, labels = batch
        data = data.to(self.device)
        labels = labels.to(self.device)
        
        # 完整前向传播（测试用）
        self.split_model.train()
        output = self.split_model(data)
        
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, labels)
        
        # 反向传播
        self.client_optimizer.zero_grad()
        loss.backward()
        self.client_optimizer.step()
        
        # 计算准确率
        _, predicted = output.max(1)
        correct = predicted.eq(labels).sum().item()
        accuracy = correct / labels.size(0)
        
        return StepMetrics(
            loss=loss.item(),
            batch_size=data.size(0),
            metrics={'accuracy': accuracy}
        )
    
    async def train_epoch(self, epoch_idx: int) -> EpochMetrics:
        """单轮训练"""
        if self._train_loader is None:
            raise RuntimeError("Train loader not initialized. Call setup() first.")
        
        self.split_model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, (data, labels) in enumerate(self._train_loader):
            metrics = await self.train_step((data, labels), batch_idx)
            total_loss += metrics.loss * metrics.batch_size
            total_correct += metrics.metrics['accuracy'] * metrics.batch_size
            total_samples += metrics.batch_size
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        avg_accuracy = total_correct / total_samples if total_samples > 0 else 0
        
        return EpochMetrics(
            epoch=epoch_idx,
            avg_loss=avg_loss,
            total_samples=total_samples,
            metrics={'accuracy': avg_accuracy}
        )
    
    async def evaluate(self, config: Optional[Dict] = None) -> EvalResult:
        """评估模型"""
        if self._test_loader is None:
            return EvalResult(num_samples=0, metrics={})
        
        self.split_model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, labels in self._test_loader:
                data = data.to(self.device)
                labels = labels.to(self.device)
                
                output = self.split_model(data)
                loss = criterion(output, labels)
                
                total_loss += loss.item() * data.size(0)
                _, predicted = output.max(1)
                total_correct += predicted.eq(labels).sum().item()
                total_samples += labels.size(0)
        
        return EvalResult(
            num_samples=total_samples,
            metrics={
                'accuracy': total_correct / total_samples if total_samples > 0 else 0,
                'loss': total_loss / total_samples if total_samples > 0 else 0
            }
        )
    
    def get_client_weights(self) -> Dict[str, torch.Tensor]:
        """获取客户端模型权重"""
        return {
            f"client.{name}": param.data.clone()
            for name, param in self.split_model.client_layers.state_dict().items()
        }
    
    def set_client_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        """设置客户端模型权重"""
        state_dict = {}
        for key, value in weights.items():
            # 移除 "client." 前缀
            clean_key = key.replace("client.", "")
            state_dict[clean_key] = value
        
        self.split_model.client_layers.load_state_dict(state_dict)
    
    def get_weights(self) -> Dict[str, Any]:
        """获取完整模型权重"""
        return {
            name: param.data.clone()
            for name, param in self.split_model.state_dict().items()
        }
    
    def set_weights(self, weights: Dict[str, Any]) -> bool:
        """设置完整模型权重"""
        try:
            torch_weights = {}
            for k, v in weights.items():
                if torch.is_tensor(v):
                    torch_weights[k] = v
                else:
                    torch_weights[k] = torch.from_numpy(v)
            
            self.split_model.load_state_dict(torch_weights)
            return True
        except Exception as e:
            self.logger.error(f"设置权重失败: {e}")
            return False
    
    def get_dataloader(self) -> DataLoader:
        """获取数据加载器"""
        return self._train_loader
    
    def get_num_samples(self) -> int:
        """获取样本数"""
        train_datasets = self._datasets.get("train", [])
        if train_datasets:
            return len(train_datasets[0])
        return 0
    
    async def teardown(self) -> None:
        """清理资源"""
        self._smashed_data = None
        self._labels = None


@learner('vfl.splitnn_server', description='SplitNN Server: Server-side model for Split Learning')
class SplitNNServerLearner(Learner):
    """
    SplitNN 服务端学习器
    
    实现分割学习的服务端逻辑：
    1. 接收客户端的 smashed data
    2. 完成服务端前向传播和损失计算
    3. 反向传播并返回切分点梯度给客户端
    """
    
    def __init__(
        self,
        model,
        datasets=None,
        tracker=None,
        callbacks=None,
        config=None,
        node_id=None
    ):
        super().__init__(model, None, tracker, callbacks, config, node_id)
        
        self._datasets = datasets or {}
        
        # 配置参数
        self.split_layer = self._config.get('split_layer', -1)
        self.learning_rate = self._config.get('learning_rate', 0.01)
        self.momentum = self._config.get('momentum', 0.9)
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 组件
        self.split_model: Optional[SplitModel] = None
        self.server_optimizer: Optional[optim.Optimizer] = None
        self.criterion = nn.CrossEntropyLoss()
        
        self.logger.info(f"SplitNNServerLearner {node_id} 初始化")
    
    async def setup(self, config: Dict) -> None:
        """初始化训练环境"""
        if hasattr(self._model, 'get_model'):
            base_model = self._model.get_model()
        else:
            base_model = self._model
        
        self.split_model = SplitModel(base_model, self.split_layer)
        self.split_model.to(self.device)
        
        # 只为服务端层创建优化器
        self.server_optimizer = optim.SGD(
            self.split_model.server_layers.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum
        )
        
        self.logger.info(f"SplitNN Server setup 完成")
    
    async def server_forward_backward(
        self,
        smashed_data: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, float, float]:
        """
        服务端前向和反向传播
        
        Args:
            smashed_data: 客户端发送的切分点激活值
            labels: 真实标签
            
        Returns:
            grad_smashed: 切分点梯度（需要返回给客户端）
            loss: 损失值
            accuracy: 准确率
        """
        smashed_data = smashed_data.to(self.device)
        labels = labels.to(self.device)
        
        # 需要梯度
        smashed_data.requires_grad_(True)
        
        # 服务端前向传播
        self.split_model.server_layers.train()
        output = self.split_model.server_forward(smashed_data)
        
        # 计算损失
        loss = self.criterion(output, labels)
        
        # 计算准确率
        _, predicted = output.max(1)
        accuracy = predicted.eq(labels).sum().item() / labels.size(0)
        
        # 反向传播
        self.server_optimizer.zero_grad()
        loss.backward()
        
        # 获取切分点梯度
        grad_smashed = smashed_data.grad.clone()
        
        # 更新服务端参数
        self.server_optimizer.step()
        
        return grad_smashed, loss.item(), accuracy
    
    def get_server_weights(self) -> Dict[str, torch.Tensor]:
        """获取服务端模型权重"""
        return {
            f"server.{name}": param.data.clone()
            for name, param in self.split_model.server_layers.state_dict().items()
        }
    
    def set_server_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        """设置服务端模型权重"""
        state_dict = {}
        for key, value in weights.items():
            clean_key = key.replace("server.", "")
            state_dict[clean_key] = value
        
        self.split_model.server_layers.load_state_dict(state_dict)
    
    def get_weights(self) -> Dict[str, Any]:
        """获取完整模型权重"""
        return {
            name: param.data.clone()
            for name, param in self.split_model.state_dict().items()
        }
    
    def set_weights(self, weights: Dict[str, Any]) -> bool:
        """设置完整模型权重"""
        try:
            torch_weights = {}
            for k, v in weights.items():
                if torch.is_tensor(v):
                    torch_weights[k] = v
                else:
                    torch_weights[k] = torch.from_numpy(v)
            
            self.split_model.load_state_dict(torch_weights)
            return True
        except Exception as e:
            return False
    
    async def train_step(self, batch: Any, batch_idx: int) -> StepMetrics:
        """占位方法"""
        raise NotImplementedError("SplitNN Server uses server_forward_backward instead")
    
    def get_dataloader(self):
        return None
    
    def get_num_samples(self) -> int:
        return 0
    
    async def teardown(self) -> None:
        pass