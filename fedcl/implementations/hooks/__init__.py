# fedcl/implementations/hooks/__init__.py
"""
钩子实现模块

提供各种钩子的具体实现，用于实验监控、日志记录和结果保存。

可用钩子:
- WandBHook: Weights & Biases集成
- TensorBoardHook: TensorBoard日志记录
- CustomHook: 自定义钩子示例
"""

from .wandb_hook import WandBHook
from .tensorboard_hook import TensorBoardHook
from .custom_hook import CustomHook

__all__ = [
    "WandBHook",
    "TensorBoardHook", 
    "CustomHook"
]
