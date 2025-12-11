"""
实验管理模块
fedcl/experiment/

新架构（参考 PyTorch Lightning）：
- Logger 系统：独立的实验记录（MLflow, JSON, TensorBoard 等）
- Callback 系统：训练流程控制
- 职责分离：Logger 只负责记录，Callback 负责流程

使用示例：
    >>> from fedcl.loggers import MLflowLogger, JSONLogger
    >>> from fedcl.experiment import create_callbacks
    >>>
    >>> # 创建 logger（可以组合多个）
    >>> loggers = [
    >>>     MLflowLogger(experiment_name="my_exp"),
    >>>     JSONLogger(save_dir="results/"),
    >>> ]
    >>>
    >>> # 创建回调函数
    >>> callbacks = create_callbacks(loggers)
    >>> trainer.add_callback('after_round', callbacks['round_callback'])
    >>>
    >>> # 训练完成后
    >>> for logger in loggers:
    >>>     logger.finalize()
"""

# 导入回调函数
from .callbacks import create_callbacks

# 导入工具函数
from .utils import (
    generate_experiment_id,
    generate_experiment_id_with_params,
    parse_experiment_id
)

# 导入批量实验运行器
from .batch_runner import (
    BatchExperimentRunner,
    create_grid_search_experiments,
    create_algorithm_comparison_experiments
)

__all__ = [
    # 回调系统
    'create_callbacks',

    # 工具函数
    'generate_experiment_id',
    'generate_experiment_id_with_params',
    'parse_experiment_id',

    # 批量实验
    'BatchExperimentRunner',
    'create_grid_search_experiments',
    'create_algorithm_comparison_experiments',
]

__version__ = '2.0.0'  # 新架构版本
