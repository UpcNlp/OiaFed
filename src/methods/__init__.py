"""
内置组件

自动注册到 Registry
"""

from . import aggregators
from . import trainers
from . import learners
from . import models
from . import datasets

__all__ = [
    "aggregators",
    "trainers",
    "learners",
    "models",
    "datasets",
]
