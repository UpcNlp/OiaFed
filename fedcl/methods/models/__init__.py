"""
联邦学习模型模块
fedcl/methods/models/__init__.py

提供：
1. 可选基类：FederatedModel
2. 辅助函数：参数统计、模型摘要等
3. 常见模型实现（将在后续添加）
"""

from .base import (
    FederatedModel,
    count_parameters,
    get_model_size_mb,
    print_model_summary,
)

__all__ = [
    # 基类
    'FederatedModel',

    # 辅助函数
    'count_parameters',
    'get_model_size_mb',
    'print_model_summary',
]
