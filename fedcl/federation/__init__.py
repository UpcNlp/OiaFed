"""
MOE-FedCL 联邦学习协调器模块
fedcl/federation/__init__.py
"""

from .server import FederationServer
from .client import FederationClient

# 初始化器
from .communication_initializer import CommunicationInitializer
from .business_initializer import BusinessInitializer

# 组件数据类
from .components import (
    CommunicationComponents,
    ServerBusinessComponents,
    ClientBusinessComponents
)

__all__ = [
    # 核心类
    'FederationServer',
    'FederationClient',

    # 初始化器
    'CommunicationInitializer',
    'BusinessInitializer',

    # 组件数据类
    'CommunicationComponents',
    'ServerBusinessComponents',
    'ClientBusinessComponents',
]
