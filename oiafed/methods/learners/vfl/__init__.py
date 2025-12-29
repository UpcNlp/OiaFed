"""
VFL (Vertical Federated Learning) 纵向联邦学习算法

实现的算法:
- SplitNN: Split Learning for Health (MIT Media Lab, 2018)
"""

from .splitnn import SplitNNLearner, SplitNNServerLearner, SplitModel

__all__ = [
    'SplitNNLearner',
    'SplitNNServerLearner', 
    'SplitModel',
]