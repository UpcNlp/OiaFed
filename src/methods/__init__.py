"""
内置组件

自动注册到 Registry
"""

from . import aggregators
from . import trainers
from . import learners
from . import models
from . import datasets

# 导入以触发注册
from ..callback import builtin as callbacks
from ..callback import tracker_sync  # 新增
from ..tracker import mlflow_tracker

__all__ = [
    "aggregators",
    "trainers",
    "learners",
    "models",
    "datasets",
]
