"""
预设联邦训练器

本模块实现了各种经典和特殊的联邦训练协调器。
所有训练器都继承自AbstractFederationTrainer，负责协调整个联邦学习流程。
"""

from .personalized_federation_trainer import PersonalizedFederationTrainer
from .standard_federation_trainer import StandardFederationTrainer

__all__ = [
    "PersonalizedFederationTrainer",
    "StandardFederationTrainer"
]