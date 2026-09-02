"""
FedEMoE 辅助函数模块。
"""

import torch
import torch.nn as nn
import numpy as np
import random
import os
from typing import Dict, Any, Optional, List
import copy


def set_seed(seed: int):
    """
    设置随机种子以确保可复现性。
    
    参数:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 启用确定性行为（可能会降低训练速度）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device: str = "cuda") -> torch.device:
    """
    获取 PyTorch 设备。
    
    参数:
        device: 设备字符串 ('cuda' 或 'cpu')
    
    返回:
        torch.device 对象
    """
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


class AverageMeter:
    """计算并存储平均值和当前值。"""
    
    def __init__(self, name: str = ""):
        """
        初始化。
        
        参数:
            name: 指标名称
        """
        self.name = name
        self.reset()
    
    def reset(self):
        """重置所有计数器。"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        """
        更新指标。
        
        参数:
            val: 当前值
            n: 样本数
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0
    
    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"


def model_to_vector(model: nn.Module) -> torch.Tensor:
    """
    将模型参数展平为单个向量。
    
    参数:
        model: PyTorch 模型
    
    返回:
        包含所有参数的一维张量
    """
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def vector_to_model(vector: torch.Tensor, model: nn.Module) -> nn.Module:
    """
    从向量加载参数到模型。
    
    参数:
        vector: 包含所有参数的一维张量
        model: 目标 PyTorch 模型
    
    返回:
        加载参数后的模型
    """
    pointer = 0
    for p in model.parameters():
        num_params = p.numel()
        p.data.copy_(vector[pointer:pointer + num_params].view_as(p))
        pointer += num_params
    return model


def average_models(models: List[nn.Module], weights: Optional[List[float]] = None) -> nn.Module:
    """
    对多个模型取平均。
    
    参数:
        models: 要平均的模型列表
        weights: 每个模型的可选权重（默认：等权重）
    
    返回:
        平均后的模型（第一个模型的副本，带有平均权重）
    """
    if len(models) == 0:
        raise ValueError("无法对空模型列表取平均")
    
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    
    # 归一化权重
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    # 创建第一个模型的副本
    avg_model = copy.deepcopy(models[0])
    
    with torch.no_grad():
        # 1. 聚合可训练参数
        for p in avg_model.parameters():
            p.data.zero_()
        
        for model, weight in zip(models, weights):
            for avg_p, p in zip(avg_model.parameters(), model.parameters()):
                avg_p.data.add_(weight * p.data)
        
        # 2. 聚合缓冲区 (buffers)，特别是 BatchNorm 的 running_mean 和 running_var
        avg_buffers = dict(avg_model.named_buffers())
        
        # 初始化需要聚合的缓冲区为零
        for name, buf in avg_buffers.items():
            if 'running_mean' in name or 'running_var' in name:
                buf.data.zero_()
        
        # 加权聚合缓冲区
        for model, weight in zip(models, weights):
            model_buffers = dict(model.named_buffers())
            for name, buf_avg in avg_buffers.items():
                if 'running_mean' in name or 'running_var' in name:
                    if name in model_buffers:
                        buf_avg.data.add_(weight * model_buffers[name].data)
    
    return avg_model


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    save_path: str,
    metrics: Optional[Dict[str, float]] = None
):
    """
    保存训练检查点。
    
    参数:
        model: 要保存的模型
        optimizer: 要保存的优化器状态
        epoch: 当前轮数
        save_path: 保存路径
        metrics: 可选的指标字典
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    
    if metrics is not None:
        checkpoint["metrics"] = metrics
    
    torch.save(checkpoint, save_path)


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    load_path: str,
    device: torch.device
) -> Dict[str, Any]:
    """
    加载训练检查点。
    
    参数:
        model: 要加载权重的模型
        optimizer: 可选的要加载状态的优化器
        load_path: 检查点路径
        device: 加载到的设备
    
    返回:
        检查点字典
    """
    checkpoint = torch.load(load_path, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return checkpoint


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    统计模型参数数量。
    
    参数:
        model: PyTorch 模型
        trainable_only: 如果为 True，只统计可训练参数
    
    返回:
        参数数量
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def print_model_summary(model: nn.Module, input_size: tuple = None):
    """
    打印模型摘要。
    
    参数:
        model: PyTorch 模型
        input_size: 可选的输入尺寸元组（用于形状推断）
    """
    print("=" * 60)
    print("模型摘要")
    print("=" * 60)
    
    total_params = 0
    trainable_params = 0
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # 只显示叶子模块
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                print(f"{name}: {module.__class__.__name__} - {params:,} 参数")
                total_params += params
                trainable_params += sum(p.numel() for p in module.parameters() if p.requires_grad)
    
    print("=" * 60)
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print("=" * 60)


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """从优化器获取当前学习率。"""
    for param_group in optimizer.param_groups:
        return param_group['lr']
    return 0.0


def adjust_learning_rate(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    initial_lr: float,
    decay_rate: float = 0.1,
    decay_epochs: List[int] = None
):
    """
    根据轮数调整学习率。
    
    参数:
        optimizer: 要调整的优化器
        epoch: 当前轮数
        initial_lr: 初始学习率
        decay_rate: 学习率乘以的衰减率
        decay_epochs: 进行衰减的轮数列表
    """
    if decay_epochs is None:
        decay_epochs = [100, 150]
    
    lr = initial_lr
    for milestone in decay_epochs:
        if epoch >= milestone:
            lr *= decay_rate
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def create_optimizer(
    model: nn.Module,
    optimizer_name: str,
    lr: float,
    momentum: float = 0.9,
    weight_decay: float = 1e-4
) -> torch.optim.Optimizer:
    """
    创建优化器。
    
    参数:
        model: 要优化的模型
        optimizer_name: 优化器名称 ('sgd' 或 'adam')
        lr: 学习率
        momentum: 动量（用于 SGD）
        weight_decay: 权重衰减
    
    返回:
        优化器实例
    """
    if optimizer_name.lower() == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
    elif optimizer_name.lower() == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"未知的优化器: {optimizer_name}")
