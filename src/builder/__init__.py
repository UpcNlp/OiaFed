"""
通用组件构建器

提供灵活的组件初始化机制，避免大量 if-else
"""

from .builder import ComponentBuilder, BuildHook

__all__ = ["ComponentBuilder", "BuildHook"]
