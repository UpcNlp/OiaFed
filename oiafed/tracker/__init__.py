"""
Tracker 模块

提供实验追踪功能
"""

from .base import Tracker
from .composite import CompositeTracker

# MLflow 是可选依赖
try:
    from .mlflow_tracker import MLflowTracker
except ImportError:
    MLflowTracker = None  # mlflow 未安装

__all__ = [
    "Tracker",
    "MLflowTracker",
    "CompositeTracker",
]