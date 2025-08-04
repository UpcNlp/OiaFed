# fedcl/federation/state/__init__.py
"""
联邦学习状态管理模块

提供联邦学习过程中的状态管理功能，包括：
- 状态枚举定义
- 状态管理器
- 状态转换控制
- 状态汇报机制
"""

from .state_enums import ServerState, ClientState, AuxiliaryState
from .state_manager import StateManager, StateCallback
from .state_reporter import StateReporter

__all__ = [
    'ServerState',
    'ClientState', 
    'AuxiliaryState',
    'StateManager',
    'StateCallback',
    'StateReporter'
]
