"""
组件注册系统
"""

from .registry import (
    Registry,
    registry,
    register,
    get_component,
    create_component,
    list_components,
)

from .decorators import (
    learner,
    trainer,
    aggregator,
    dataset,
    model,
    callback,
    get_component_metadata,
    list_components_with_metadata,
)

__all__ = [
    # Registry
    "Registry",
    "registry",
    "register",
    "get_component",
    "create_component",
    "list_components",
    # Decorators
    "learner",
    "trainer",
    "aggregator",
    "dataset",
    "model",
    "callback",
    "get_component_metadata",
    "list_components_with_metadata",
]
