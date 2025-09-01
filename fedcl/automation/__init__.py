# fedcl/automation/__init__.py
"""
自动化处理层

实现通信、同步、故障处理的完全自动化，让用户完全感受不到分布式的存在。
第3阶段实施完成。
"""

# 从comm模块导入通信相关功能
from ..comm import (
    MemoryTransport,
    ProcessTransport,
    NetworkTransport
)

from .model_sync import (
    ModelSynchronizer,
    SyncMode,
    SyncConfig,
    ClientUpdate
)

from .data_manager import (
    AutoDataManager,
    DataDistributionType,
    DataConfig,
    DataPartition,
    FederatedDataLoader,
    IIDPartitioner,
    NonIIDLabelPartitioner,
    NonIIDQuantityPartitioner
)

from .failure_recovery import (
    FailureRecoveryManager,
    CheckpointManager,
    FailureDetector,
    FailureType,
    FailureEvent,
    Checkpoint
)

__all__ = [
    # 通信传输层
    "MemoryTransport",
    "ProcessTransport", 
    "NetworkTransport",
    
    # 模型同步
    "ModelSynchronizer",
    "SyncMode",
    "SyncConfig",
    "ClientUpdate",
    
    # 数据管理
    "AutoDataManager",
    "DataDistributionType",
    "DataConfig",
    "DataPartition",
    "FederatedDataLoader",
    "IIDPartitioner",
    "NonIIDLabelPartitioner",
    "NonIIDQuantityPartitioner",
    
    # 故障恢复
    "FailureRecoveryManager",
    "CheckpointManager",
    "FailureDetector",
    "FailureType",
    "FailureEvent",
    "Checkpoint",
]