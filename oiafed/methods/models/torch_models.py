"""
内置模型实现
"""

from typing import Any, Dict, List, Optional
import pickle

from ...core.model import Model
from ...registry import register
from ...infra import get_module_logger

logger = get_module_logger(__name__)


@register("model.torch.wrapper")
class TorchModelWrapper(Model):
    """
    PyTorch 模型包装器
    
    将 PyTorch nn.Module 包装为 Model 接口
    """
    
    def __init__(self, model: Any, device: str = "cpu"):
        """
        初始化
        
        Args:
            model: PyTorch nn.Module
            device: 设备
        """
        self._model = model
        self._device = device
        
        try:
            import torch
            self._model = self._model.to(device)
        except ImportError:
            pass
    
    def get_weights(self) -> List[Any]:
        """获取模型权重"""
        try:
            import torch
            return [p.detach().cpu().numpy() for p in self._model.parameters()]
        except ImportError:
            return []
    
    def set_weights(self, weights: List[Any]) -> None:
        """设置模型权重"""
        try:
            import torch
            import numpy as np
            
            for param, w in zip(self._model.parameters(), weights):
                if isinstance(w, np.ndarray):
                    param.data = torch.from_numpy(w).to(self._device)
                else:
                    param.data = torch.tensor(w).to(self._device)
        except ImportError:
            pass
    
    def serialize(self) -> bytes:
        """序列化模型"""
        return pickle.dumps(self.get_weights())
    
    @classmethod
    def deserialize(cls, data: bytes) -> "TorchModelWrapper":
        """反序列化模型"""
        weights = pickle.loads(data)
        # 注意：这需要一个空模型来设置权重
        # 实际使用时需要提供模型结构
        raise NotImplementedError("Need model structure to deserialize")
    
    def train_mode(self) -> None:
        """设置为训练模式"""
        self._model.train()
    
    def eval_mode(self) -> None:
        """设置为评估模式"""
        self._model.eval()
    
    def to_device(self, device: str) -> "TorchModelWrapper":
        """移动到设备"""
        try:
            import torch
            self._device = device
            self._model = self._model.to(device)
        except ImportError:
            pass
        return self
    
    def num_parameters(self) -> int:
        """获取参数数量"""
        try:
            return sum(p.numel() for p in self._model.parameters())
        except:
            return 0


@register("model.torch.mlp")
class MLPModel(Model):
    """
    多层感知机模型
    
    简单的全连接网络
    """
    
    def __init__(
        self,
        input_size: int = 784,
        hidden_sizes: List[int] = None,
        output_size: int = 10,
        activation: str = "relu",
        device: str = "cpu",
    ):
        """
        初始化
        
        Args:
            input_size: 输入维度
            hidden_sizes: 隐藏层大小列表
            output_size: 输出维度
            activation: 激活函数
            device: 设备
        """
        hidden_sizes = hidden_sizes or [128, 64]
        self._device = device
        
        try:
            import torch
            import torch.nn as nn
            
            layers = []
            prev_size = input_size
            
            for hidden_size in hidden_sizes:
                layers.append(nn.Linear(prev_size, hidden_size))
                if activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "tanh":
                    layers.append(nn.Tanh())
                prev_size = hidden_size
            
            layers.append(nn.Linear(prev_size, output_size))
            
            self._model = nn.Sequential(*layers).to(device)
            
        except ImportError:
            logger.warning("PyTorch not available, using dummy model")
            self._model = None
            self._weights = []
    
    def get_weights(self) -> List[Any]:
        """获取模型权重"""
        if self._model is None:
            return self._weights
        
        try:
            import torch
            return [p.detach().cpu().numpy() for p in self._model.parameters()]
        except:
            return []
    
    def set_weights(self, weights: List[Any]) -> None:
        """设置模型权重"""
        if self._model is None:
            self._weights = weights
            return
        
        try:
            import torch
            import numpy as np
            
            for param, w in zip(self._model.parameters(), weights):
                if isinstance(w, np.ndarray):
                    param.data = torch.from_numpy(w).to(self._device)
                else:
                    param.data = torch.tensor(w).to(self._device)
        except:
            pass
    
    def serialize(self) -> bytes:
        """序列化模型"""
        return pickle.dumps(self.get_weights())
    
    @classmethod
    def deserialize(cls, data: bytes) -> "MLPModel":
        """反序列化模型"""
        model = cls()
        weights = pickle.loads(data)
        model.set_weights(weights)
        return model
    
    def train_mode(self) -> None:
        """设置为训练模式"""
        if self._model is not None:
            self._model.train()
    
    def eval_mode(self) -> None:
        """设置为评估模式"""
        if self._model is not None:
            self._model.eval()


@register("model.torch.cnn")
class CNNModel(Model):
    """
    卷积神经网络模型
    
    简单的 CNN 用于图像分类
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        device: str = "cpu",
    ):
        """
        初始化
        
        Args:
            in_channels: 输入通道数
            num_classes: 类别数
            device: 设备
        """
        self._device = device
        
        try:
            import torch
            import torch.nn as nn
            
            self._model = nn.Sequential(
                nn.Conv2d(in_channels, 32, 3, 1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, 1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(9216, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes),
            ).to(device)
            
        except ImportError:
            logger.warning("PyTorch not available, using dummy model")
            self._model = None
            self._weights = []
    
    def get_weights(self) -> List[Any]:
        """获取模型权重"""
        if self._model is None:
            return self._weights
        
        try:
            import torch
            return [p.detach().cpu().numpy() for p in self._model.parameters()]
        except:
            return []
    
    def set_weights(self, weights: List[Any]) -> None:
        """设置模型权重"""
        if self._model is None:
            self._weights = weights
            return
        
        try:
            import torch
            import numpy as np
            
            for param, w in zip(self._model.parameters(), weights):
                if isinstance(w, np.ndarray):
                    param.data = torch.from_numpy(w).to(self._device)
                else:
                    param.data = torch.tensor(w).to(self._device)
        except:
            pass
    
    def serialize(self) -> bytes:
        """序列化模型"""
        return pickle.dumps(self.get_weights())
    
    @classmethod
    def deserialize(cls, data: bytes) -> "CNNModel":
        """反序列化模型"""
        model = cls()
        weights = pickle.loads(data)
        model.set_weights(weights)
        return model
    
    def train_mode(self) -> None:
        """设置为训练模式"""
        if self._model is not None:
            self._model.train()
    
    def eval_mode(self) -> None:
        """设置为评估模式"""
        if self._model is not None:
            self._model.eval()


@register("model.numpy.linear")
class NumpyLinearModel(Model):
    """
    NumPy 线性模型
    
    不依赖深度学习框架的简单线性模型
    """
    
    def __init__(
        self,
        input_size: int = 784,
        output_size: int = 10,
    ):
        """
        初始化
        
        Args:
            input_size: 输入维度
            output_size: 输出维度
        """
        try:
            import numpy as np
            
            # 初始化权重
            self._W = np.random.randn(input_size, output_size) * 0.01
            self._b = np.zeros(output_size)
            
        except ImportError:
            self._W = [[0.0] * output_size for _ in range(input_size)]
            self._b = [0.0] * output_size
    
    def get_weights(self) -> List[Any]:
        """获取模型权重"""
        return [self._W, self._b]
    
    def set_weights(self, weights: List[Any]) -> None:
        """设置模型权重"""
        if len(weights) >= 2:
            self._W = weights[0]
            self._b = weights[1]
    
    def serialize(self) -> bytes:
        """序列化模型"""
        return pickle.dumps(self.get_weights())
    
    @classmethod
    def deserialize(cls, data: bytes) -> "NumpyLinearModel":
        """反序列化模型"""
        model = cls()
        weights = pickle.loads(data)
        model.set_weights(weights)
        return model
    
    def forward(self, x: Any) -> Any:
        """前向传播"""
        try:
            import numpy as np
            return np.dot(x, self._W) + self._b
        except:
            return x
