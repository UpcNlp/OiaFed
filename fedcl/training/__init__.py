# fedcl/training/__init__.py
"""
Training module for FedCL framework.

This module provides training engines and related utilities for federated continual learning.
The evaluation engine works with existing Hook system to avoid functionality duplication.
"""

from .training_engine import TrainingEngine
from .evaluation_engine import EvaluationEngine

__all__ = [
    "TrainingEngine",
    "EvaluationEngine"
]
