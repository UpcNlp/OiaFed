"""
联邦学习 Logger 系统
fedcl/loggers/__init__.py

设计参考 PyTorch Lightning 的 Logger 模式：
- Logger 和 Callback 职责分离
- Logger 专门负责实验记录（metrics, params, artifacts）
- 支持组合多个 logger（MLflow, JSON, TensorBoard, W&B 等）
- 与训练框架解耦，可以独立使用

新版本：
- 引入 ExperimentTracker 抽象基类（统一接口）
- MLflowTracker 替代 MLflowLogger（支持嵌套 runs、角色管理）
- 保持向后兼容（MLflowLogger 作为别名）
"""

from .base_logger import Logger
from .json_logger import JSONLogger
from .base_tracker import ExperimentTracker, TrackerRole

__all__ = ['Logger', 'JSONLogger', 'ExperimentTracker', 'TrackerRole']

# 可选的 MLflow Tracker（如果已安装）
try:
    from .mlflow_tracker import MLflowTracker, MLflowLogger
    __all__.extend(['MLflowTracker', 'MLflowLogger'])
except ImportError:
    pass
