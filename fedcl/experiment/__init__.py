"""
实验管理模块
fedcl/experiment/

提供实验记录、回调函数和工具函数

主要功能：
- Recorder: 实验结果记录器（支持JSON和MLflow两种后端）
- create_callbacks: 创建预定义的回调函数
- generate_experiment_id: 生成可读的实验ID
- BatchExperimentRunner: 批量实验运行器

使用示例：
    >>> from fedcl.experiment import Recorder, create_callbacks
    >>>
    >>> # 创建记录器（默认使用JSON）
    >>> recorder = Recorder.initialize("my_exp", "server", "server_0")
    >>> recorder.start_run({"mode": "memory"})
    >>>
    >>> # 创建回调函数
    >>> callbacks = create_callbacks(recorder)
    >>> trainer.add_callback('after_round', callbacks['round_callback'])
    >>>
    >>> # 训练完成后保存结果
    >>> recorder.finish(status="COMPLETED")

切换到MLflow后端：
    >>> import os
    >>> os.environ['FEDCL_RECORDER_BACKEND'] = 'mlflow'  # 在导入前设置
    >>> from fedcl.experiment import Recorder  # 自动使用 MLflow

    或者直接导入：
    >>> from fedcl.experiment import MLflowRecorder
    >>> recorder = MLflowRecorder.initialize("my_exp", "server", "server_0")
"""

import os

# 支持的后端
_SUPPORTED_BACKENDS = ['json', 'mlflow']

# 默认使用 JSON 后端
_DEFAULT_BACKEND = 'json'

# 从环境变量读取后端配置
_RECORDER_BACKEND = os.environ.get('FEDCL_RECORDER_BACKEND', _DEFAULT_BACKEND).lower()

# 导入基础模块
from .callbacks import create_callbacks
from .utils import (
    generate_experiment_id,
    generate_experiment_id_with_params,
    parse_experiment_id
)
from .batch_runner import (
    BatchExperimentRunner,
    create_grid_search_experiments,
    create_algorithm_comparison_experiments
)

# 导入 Recorder（根据配置选择后端）
from .recorder import Recorder as JSONRecorder

try:
    from .mlflow_recorder import MLflowRecorder
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False
    MLflowRecorder = None

# 根据配置选择默认的 Recorder
if _RECORDER_BACKEND == 'mlflow':
    if _MLFLOW_AVAILABLE:
        Recorder = MLflowRecorder
        print(f"[fedcl.experiment] Using MLflow backend for experiment tracking")
    else:
        print(f"[fedcl.experiment] Warning: MLflow backend requested but not available. "
              f"Falling back to JSON. Install: pip install mlflow")
        Recorder = JSONRecorder
else:
    Recorder = JSONRecorder

__all__ = [
    # 默认 Recorder（根据配置）
    'Recorder',

    # 可以明确选择的 Recorder
    'JSONRecorder',
    'MLflowRecorder',

    # 回调和工具
    'create_callbacks',
    'generate_experiment_id',
    'generate_experiment_id_with_params',
    'parse_experiment_id',

    # 批量实验
    'BatchExperimentRunner',
    'create_grid_search_experiments',
    'create_algorithm_comparison_experiments',
]

__version__ = '1.0.0'
