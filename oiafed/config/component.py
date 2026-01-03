"""
组件配置

包含 trainer、learner、aggregator、model、dataset、callback 等组件的配置类及解析方法。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ==================== 组件配置 ====================

@dataclass
class ComponentConfig:
    """
    通用组件配置
    
    用于 trainer, learner, aggregator, model 等组件。
    
    Attributes:
        type: 组件类型（如 "federated.trainer.fedavg"）
        args: 组件参数
    """
    type: str
    args: Optional[Dict[str, Any]] = None
    
    def get_args(self) -> Dict[str, Any]:
        """获取参数字典"""
        return self.args or {}


def parse_component_config(data: Optional[Dict[str, Any]]) -> Optional[ComponentConfig]:
    """解析组件配置"""
    if not data:
        return None
    return ComponentConfig(
        type=data.get("type", ""),
        args=data.get("args"),
    )


# ==================== 数据集配置 ====================

@dataclass
class DatasetConfig:
    """
    数据集配置
    
    Attributes:
        type: 数据集类型
        split: 数据集划分（train/test/valid）
        args: 数据集参数
        partition: 分区配置（用于联邦学习数据划分）
    """
    type: str
    split: str = "train"
    args: Optional[Dict[str, Any]] = None
    partition: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        # 自动将 split 注入 args
        if self.args is None:
            self.args = {}
        if "split" not in self.args:
            self.args["split"] = self.split
    
    def get_args(self) -> Dict[str, Any]:
        return self.args.copy() if self.args else {}


def parse_dataset_config(data: Dict[str, Any]) -> DatasetConfig:
    """解析数据集配置"""
    return DatasetConfig(
        type=data["type"],
        split=data.get("split", "train"),
        args=data.get("args"),
        partition=data.get("partition"),
    )


def parse_datasets_config(data: Optional[List[Dict[str, Any]]]) -> List[DatasetConfig]:
    """解析数据集列表配置"""
    if not data:
        return []
    return [parse_dataset_config(d) for d in data]


# ==================== 回调配置 ====================

@dataclass
class CallbackConfig:
    """
    回调配置
    
    Attributes:
        type: 回调类型
        args: 回调参数
    """
    type: str
    args: Optional[Dict[str, Any]] = None
    
    def get_args(self) -> Dict[str, Any]:
        """获取参数字典"""
        return self.args or {}


def parse_callback_config(data: Dict[str, Any]) -> CallbackConfig:
    """解析回调配置"""
    return CallbackConfig(
        type=data.get("type", ""),
        args=data.get("args"),
    )


def parse_callbacks_config(data: Optional[List[Dict[str, Any]]]) -> List[CallbackConfig]:
    """解析回调列表配置"""
    if not data:
        return []
    return [parse_callback_config(c) for c in data]


# ==================== 导出 ====================

__all__ = [
    # 组件配置
    "ComponentConfig",
    "parse_component_config",
    # 数据集配置
    "DatasetConfig",
    "parse_dataset_config",
    "parse_datasets_config",
    # 回调配置
    "CallbackConfig",
    "parse_callback_config",
    "parse_callbacks_config",
]
