"""
模型工具函数

设计原则：
1. 直接使用 PyTorch nn.Module，不重复定义基类
2. 只提供必要的辅助函数（权重转换等）
3. 支持 Hugging Face Transformers 模型
"""

from typing import Any, List, TYPE_CHECKING

if TYPE_CHECKING:
    import torch.nn as nn


def get_model_weights(model: "nn.Module") -> List[Any]:
    """
    获取 PyTorch 模型权重（numpy 数组格式）

    Args:
        model: PyTorch nn.Module

    Returns:
        权重列表（numpy 数组）

    Example:
        import torch.nn as nn
        from federation.core.model import get_model_weights

        model = nn.Linear(10, 2)
        weights = get_model_weights(model)
    """
    import torch

    return [
        p.detach().cpu().numpy().copy()
        for p in model.parameters()
    ]


def set_model_weights(model: "nn.Module", weights: List[Any]) -> None:
    """
    设置 PyTorch 模型权重（从 numpy 数组）

    Args:
        model: PyTorch nn.Module
        weights: numpy 数组列表

    Example:
        from federation.core.model import set_model_weights

        set_model_weights(model, weights)
    """
    import torch

    device = next(model.parameters()).device

    with torch.no_grad():
        for param, w in zip(model.parameters(), weights):
            param.copy_(torch.from_numpy(w).to(device))


def clone_model_weights(model: "nn.Module") -> List[Any]:
    """
    克隆模型权重

    Args:
        model: PyTorch nn.Module

    Returns:
        权重深拷贝

    Example:
        from federation.core.model import clone_model_weights

        cloned_weights = clone_model_weights(model)
    """
    import copy
    return copy.deepcopy(get_model_weights(model))


def get_num_parameters(model: "nn.Module") -> int:
    """
    获取模型参数总数

    Args:
        model: PyTorch nn.Module

    Returns:
        参数数量

    Example:
        from federation.core.model import get_num_parameters

        num_params = get_num_parameters(model)
        print(f"Model has {num_params:,} parameters")
    """
    return sum(p.numel() for p in model.parameters())


# ========== Hugging Face Transformers 支持 ==========

def get_transformers_model_weights(model) -> List[Any]:
    """
    获取 Transformers 模型权重

    Args:
        model: Hugging Face Transformers 模型（PreTrainedModel）

    Returns:
        权重列表（numpy 数组）

    Example:
        from transformers import BertModel
        from federation.core.model import get_transformers_model_weights

        model = BertModel.from_pretrained('bert-base-uncased')
        weights = get_transformers_model_weights(model)
    """
    # Transformers 模型本质上也是 nn.Module
    return get_model_weights(model)


def set_transformers_model_weights(model, weights: List[Any]) -> None:
    """
    设置 Transformers 模型权重

    Args:
        model: Hugging Face Transformers 模型
        weights: numpy 数组列表

    Example:
        from federation.core.model import set_transformers_model_weights

        set_transformers_model_weights(model, weights)
    """
    set_model_weights(model, weights)
