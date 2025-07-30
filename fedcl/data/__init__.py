# fedcl/data/__init__.py
"""
数据管理模块

提供联邦学习所需的数据处理、分割、加载等功能。
"""

from .dataset import Dataset, DatasetError
from .dataloader import DataLoader, DataLoaderError  
from .task import Task, TaskError, TaskType
from .task_generator import TaskGenerator, TaskGeneratorError, TaskValidationError
from .results import TaskResults, ExperimentResults, ResultsError
from .split_strategy import (
    SplitStrategy,
    IIDSplitStrategy, 
    NonIIDSplitStrategy,
    SplitStatistics,
    SplitStrategyError,
    DataSplitValidationError
)
from .data_processor import (
    DataProcessor,
    DataProcessorError,
    DataTransformError,
    DataValidationError,
    DataAugmentationError
)
from .dataset_manager import (
    DatasetManager,
    DatasetManagerError,
    DatasetNotFoundError,
    DatasetCacheError,
    DatasetValidationManager
)

__all__ = [
    'Dataset',
    'DatasetError',
    'DataLoader', 
    'DataLoaderError',
    'Task',
    'TaskError',
    'TaskType',
    'TaskGenerator',
    'TaskGeneratorError',
    'TaskValidationError',
    'TaskResults',
    'ExperimentResults',
    'ResultsError',
    'SplitStrategy',
    'IIDSplitStrategy',
    'NonIIDSplitStrategy', 
    'SplitStatistics',
    'SplitStrategyError',
    'DataSplitValidationError',
    'DataProcessor',
    'DataProcessorError',
    'DataTransformError',
    'DataValidationError',
    'DataAugmentationError',
    'DatasetManager',
    'DatasetManagerError',
    'DatasetNotFoundError',
    'DatasetCacheError',
    'DatasetValidationManager'
]