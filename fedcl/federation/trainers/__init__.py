# fedcl/federation/trainers/__init__.py
"""
联邦学习训练器模块

包含本地训练器和分布式训练器实现。
"""

from .local_trainer import LocalTrainer

__all__ = [
    'LocalTrainer'
]
