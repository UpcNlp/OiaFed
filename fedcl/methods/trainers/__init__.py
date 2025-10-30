"""
预设联邦训练器

本模块实现了各种经典和特殊的联邦训练协调器。
所有训练器都继承自AbstractFederationTrainer，负责协调整个联邦学习流程。
"""

# from .personalized_federation_trainer import PersonalizedFederationTrainer  # File deleted
from .default import StandardFederationTrainer
# from .test import SimpleTrainer  # Temporarily commented out due to broken import

__all__ = [
    # "PersonalizedFederationTrainer",
    "StandardFederationTrainer",
    # "SimpleTrainer"
]