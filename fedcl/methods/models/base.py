"""
联邦学习模型基类
fedcl/methods/models/base.py

提供模型统一管理接口，方便模型注册和使用。
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple


class FederatedModel(nn.Module):
    """
    联邦学习模型基类（可选）

    提供一些联邦学习特定的便利方法。
    用户可以选择：
    1. 直接继承 nn.Module（推荐）
    2. 继承 FederatedModel 获得额外功能

    额外功能：
    - 模型信息查询（参数量、模型大小等）
    - 权重序列化/反序列化辅助
    - 模型元数据管理
    """

    def __init__(self):
        super().__init__()

        # 模型元数据
        self._metadata = {
            'model_name': self.__class__.__name__,
            'task_type': None,  # 'classification', 'regression', 'generation'
            'input_shape': None,
            'output_shape': None,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播（子类必须实现）

        Args:
            x: 输入张量

        Returns:
            输出张量
        """
        raise NotImplementedError("Subclass must implement forward()")

    # ==================== 模型信息查询 ====================

    def get_param_count(self) -> int:
        """
        获取模型总参数量

        Returns:
            总参数数量
        """
        return sum(p.numel() for p in self.parameters())

    def get_trainable_param_count(self) -> int:
        """
        获取可训练参数数量

        Returns:
            可训练参数数量
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_size(self, unit: str = 'MB') -> float:
        """
        获取模型大小

        Args:
            unit: 单位，可选 'KB', 'MB', 'GB'

        Returns:
            模型大小（浮点数）
        """
        param_size = sum(p.nelement() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in self.buffers())
        total_bytes = param_size + buffer_size

        units = {
            'B': 1,
            'KB': 1024,
            'MB': 1024 ** 2,
            'GB': 1024 ** 3,
        }

        if unit not in units:
            raise ValueError(f"Invalid unit: {unit}. Choose from {list(units.keys())}")

        return total_bytes / units[unit]

    def get_layer_info(self) -> Dict[str, Dict[str, Any]]:
        """
        获取每一层的详细信息

        Returns:
            Dict[layer_name, layer_info]
        """
        layer_info = {}

        for name, module in self.named_modules():
            if len(list(module.children())) == 0:  # 叶子节点
                num_params = sum(p.numel() for p in module.parameters())
                layer_info[name] = {
                    'type': module.__class__.__name__,
                    'num_params': num_params,
                    'trainable': any(p.requires_grad for p in module.parameters()),
                }

        return layer_info

    # ==================== 权重管理 ====================

    def get_weights_as_dict(self) -> Dict[str, torch.Tensor]:
        """
        获取模型权重（返回字典格式）

        Returns:
            {param_name: tensor}
        """
        return {k: v.cpu().clone() for k, v in self.state_dict().items()}

    def set_weights_from_dict(self, weights: Dict[str, torch.Tensor], strict: bool = True):
        """
        从字典设置模型权重

        Args:
            weights: 权重字典
            strict: 是否严格匹配所有参数
        """
        # 将权重移到正确的设备
        device = next(self.parameters()).device
        device_weights = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in weights.items()
        }

        self.load_state_dict(device_weights, strict=strict)

    def get_weight_norm(self) -> float:
        """
        计算模型权重的L2范数

        Returns:
            权重范数
        """
        total_norm = 0.0
        for param in self.parameters():
            total_norm += param.norm(2).item() ** 2
        return total_norm ** 0.5

    # ==================== 元数据管理 ====================

    def set_metadata(self, **kwargs):
        """
        设置模型元数据

        Args:
            **kwargs: 元数据键值对
        """
        self._metadata.update(kwargs)

    def get_metadata(self) -> Dict[str, Any]:
        """
        获取模型元数据

        Returns:
            元数据字典
        """
        return self._metadata.copy()

    # ==================== 模型摘要 ====================

    def summary(self, input_shape: Optional[Tuple[int, ...]] = None) -> str:
        """
        生成模型摘要信息

        Args:
            input_shape: 输入形状（不包括batch维度）

        Returns:
            模型摘要字符串
        """
        lines = []
        lines.append("=" * 80)
        lines.append(f"Model: {self._metadata['model_name']}")
        lines.append("=" * 80)

        # 参数信息
        lines.append(f"Total parameters: {self.get_param_count():,}")
        lines.append(f"Trainable parameters: {self.get_trainable_param_count():,}")
        lines.append(f"Model size: {self.get_model_size('MB'):.2f} MB")

        # 输入输出形状
        if input_shape:
            lines.append(f"Input shape: {input_shape}")
        if self._metadata.get('output_shape'):
            lines.append(f"Output shape: {self._metadata['output_shape']}")

        lines.append("=" * 80)

        # 层级信息
        lines.append("\nLayer information:")
        lines.append("-" * 80)
        layer_info = self.get_layer_info()
        for name, info in layer_info.items():
            lines.append(f"{name:40s} {info['type']:20s} {info['num_params']:>12,}")
        lines.append("=" * 80)

        return "\n".join(lines)

    def __repr__(self) -> str:
        """字符串表示"""
        return (f"{self.__class__.__name__}("
                f"params={self.get_param_count():,}, "
                f"size={self.get_model_size('MB'):.2f}MB)")


# ==================== 辅助函数 ====================

def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """
    统计模型参数数量

    Args:
        model: PyTorch模型
        trainable_only: 是否只统计可训练参数

    Returns:
        参数数量
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def get_model_size_mb(model: nn.Module) -> float:
    """
    获取模型大小（MB）

    Args:
        model: PyTorch模型

    Returns:
        模型大小（MB）
    """
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)


def get_weights_as_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    获取模型权重（返回字典格式）- 兼容任何 nn.Module

    Args:
        model: PyTorch模型（nn.Module或FederatedModel）

    Returns:
        {param_name: tensor} - 权重在CPU上
    """
    # 优先使用模型自带的方法（如果有）
    if hasattr(model, 'get_weights_as_dict'):
        return model.get_weights_as_dict()

    # 否则使用标准的state_dict
    return {k: v.cpu().clone() for k, v in model.state_dict().items()}


def set_weights_from_dict(model: nn.Module, weights: Dict[str, torch.Tensor], strict: bool = True):
    """
    从字典设置模型权重 - 兼容任何 nn.Module

    Args:
        model: PyTorch模型（nn.Module或FederatedModel）
        weights: 权重字典
        strict: 是否严格匹配所有参数
    """
    # 优先使用模型自带的方法（如果有）
    if hasattr(model, 'set_weights_from_dict'):
        model.set_weights_from_dict(weights, strict)
        return

    # 否则使用标准的load_state_dict
    # 将权重移到正确的设备
    try:
        device = next(model.parameters()).device
    except StopIteration:
        # 模型没有参数，使用CPU
        device = torch.device('cpu')

    device_weights = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in weights.items()
    }

    model.load_state_dict(device_weights, strict=strict)


def get_param_count(model: nn.Module) -> int:
    """
    获取模型总参数量 - 兼容任何 nn.Module

    Args:
        model: PyTorch模型（nn.Module或FederatedModel）

    Returns:
        总参数数量
    """
    # 优先使用模型自带的方法（如果有）
    if hasattr(model, 'get_param_count'):
        return model.get_param_count()

    # 否则使用标准方法
    return count_parameters(model)


def print_model_summary(model: nn.Module, input_shape: Optional[Tuple[int, ...]] = None):
    """
    打印模型摘要

    Args:
        model: PyTorch模型
        input_shape: 输入形状（不包括batch维度）
    """
    if isinstance(model, FederatedModel):
        print(model.summary(input_shape))
    else:
        # 对于标准nn.Module，提供简单摘要
        print("=" * 80)
        print(f"Model: {model.__class__.__name__}")
        print("=" * 80)
        print(f"Total parameters: {count_parameters(model):,}")
        print(f"Trainable parameters: {count_parameters(model, trainable_only=True):,}")
        print(f"Model size: {get_model_size_mb(model):.2f} MB")
        if input_shape:
            print(f"Input shape: {input_shape}")
        print("=" * 80)
