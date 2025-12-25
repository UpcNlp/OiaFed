"""
联邦学习核心类型定义
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List
from enum import Enum


class FitStatus(Enum):
    """训练状态"""
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


class NodeState(Enum):
    """节点状态枚举"""
    IDLE = "idle"                      # 空闲，等待任务
    TRAINING = "training"              # 正在训练
    EVALUATING = "evaluating"          # 正在评估
    AGGREGATING = "aggregating"        # 正在聚合（Trainer）
    UNHEALTHY = "unhealthy"            # 网络失联
    SHUTTING_DOWN = "shutting_down"    # 关闭中


# ==================== 分层训练指标类型 ====================

@dataclass
class StepMetrics:
    """单个训练步骤的指标"""
    loss: float                                      # 损失值
    batch_size: int                                  # 批次大小
    metrics: Dict[str, float] = field(default_factory=dict)  # 额外指标（如准确率）


@dataclass
class EpochMetrics:
    """单个 epoch 的指标"""
    epoch: int                                       # Epoch 索引
    avg_loss: float                                  # 平均损失
    total_samples: int                               # 总样本数
    metrics: Dict[str, float] = field(default_factory=dict)  # 聚合指标


@dataclass
class TrainMetrics:
    """完整训练的指标"""
    total_epochs: int                                # 总轮数
    final_loss: float                                # 最终损失
    total_samples: int                               # 总样本数
    metrics: Dict[str, float] = field(default_factory=dict)      # 聚合指标
    epoch_history: List[EpochMetrics] = field(default_factory=list)  # Epoch 历史


@dataclass
class TrainResult:
    """训练结果（联邦学习接口）"""

    weights: Any                                    # 更新后的模型权重
    num_samples: int                                # 训练样本数
    metrics: TrainMetrics                           # 训练指标（分层）
    metadata: Dict[str, Any] = field(default_factory=dict)   # 其他元数据
    status: FitStatus = FitStatus.SUCCESS           # 训练状态

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "weights": self.weights,
            "num_samples": self.num_samples,
            "metrics": self._metrics_to_dict(self.metrics),
            "metadata": self.metadata,
            "status": self.status.value,
        }

    def _metrics_to_dict(self, metrics: TrainMetrics) -> Dict[str, Any]:
        """将 TrainMetrics 转换为字典"""
        return {
            "total_epochs": metrics.total_epochs,
            "final_loss": metrics.final_loss,
            "total_samples": metrics.total_samples,
            "metrics": metrics.metrics,
            "epoch_history": [
                {
                    "epoch": em.epoch,
                    "avg_loss": em.avg_loss,
                    "total_samples": em.total_samples,
                    "metrics": em.metrics,
                }
                for em in metrics.epoch_history
            ]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainResult":
        """从字典创建"""
        status = data.get("status", "success")
        if isinstance(status, str):
            status = FitStatus(status)

        # 处理 metrics 字段（兼容旧格式）
        metrics_data = data.get("metrics", {})
        if isinstance(metrics_data, dict) and "total_epochs" in metrics_data:
            # 新格式：TrainMetrics
            epoch_history = [
                EpochMetrics(
                    epoch=em["epoch"],
                    avg_loss=em["avg_loss"],
                    total_samples=em["total_samples"],
                    metrics=em.get("metrics", {})
                )
                for em in metrics_data.get("epoch_history", [])
            ]
            metrics = TrainMetrics(
                total_epochs=metrics_data["total_epochs"],
                final_loss=metrics_data["final_loss"],
                total_samples=metrics_data["total_samples"],
                metrics=metrics_data.get("metrics", {}),
                epoch_history=epoch_history
            )
        else:
            # 旧格式：Dict[str, float]，转换为 TrainMetrics
            metrics = TrainMetrics(
                total_epochs=1,
                final_loss=metrics_data.get("loss", 0.0),
                total_samples=data.get("num_samples", 0),
                metrics=metrics_data,
                epoch_history=[]
            )

        return cls(
            weights=data.get("weights"),
            num_samples=data.get("num_samples", 0),
            metrics=metrics,
            metadata=data.get("metadata", {}),
            status=status,
        )


@dataclass
class EvalResult:
    """评估结果"""
    
    num_samples: int                                # 评估样本数
    metrics: Dict[str, float] = field(default_factory=dict)  # 评估指标
    metadata: Dict[str, Any] = field(default_factory=dict)   # 其他元数据
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "num_samples": self.num_samples,
            "metrics": self.metrics,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvalResult":
        """从字典创建"""
        return cls(
            num_samples=data.get("num_samples", 0),
            metrics=data.get("metrics", {}),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ClientUpdate:
    """客户端更新（用于聚合）"""
    
    client_id: str                                  # 客户端 ID
    weights: Any                                    # 模型权重
    num_samples: int                                # 样本数
    metrics: Dict[str, float] = field(default_factory=dict)  # 指标
    metadata: Dict[str, Any] = field(default_factory=dict)   # 元数据
    
    @classmethod
    def from_result(cls, client_id: str, result: TrainResult) -> "ClientUpdate":
        """从 TrainResult 创建"""
        # 从 TrainMetrics 中提取聚合的指标
        if isinstance(result.metrics, TrainMetrics):
            metrics_dict = result.metrics.metrics  # 使用聚合后的指标
        else:
            # 兼容旧格式（直接是 Dict[str, float]）
            metrics_dict = result.metrics if isinstance(result.metrics, dict) else {}

        return cls(
            client_id=client_id,
            weights=result.weights,
            num_samples=result.num_samples,
            metrics=metrics_dict,
            metadata=result.metadata,
        )


@dataclass
class ClientInfo:
    """客户端信息"""

    client_id: str                                  # 客户端 ID
    capabilities: Dict[str, Any] = field(default_factory=dict)  # 能力信息
    data_info: Dict[str, Any] = field(default_factory=dict)     # 数据信息
    metadata: Dict[str, Any] = field(default_factory=dict)      # 其他元数据

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "client_id": self.client_id,
            "capabilities": self.capabilities,
            "data_info": self.data_info,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClientInfo":
        """从字典创建"""
        return cls(
            client_id=data.get("client_id", ""),
            capabilities=data.get("capabilities", {}),
            data_info=data.get("data_info", {}),
            metadata=data.get("metadata", {}),
        )


# ==================== 服务端轮次类型 ====================

@dataclass
class RoundMetrics:
    """单个联邦学习轮次的指标"""
    round_num: int                                   # 轮次编号
    num_clients: int                                 # 参与客户端数量
    total_samples: int                               # 总样本数
    metrics: Dict[str, float] = field(default_factory=dict)  # 聚合的客户端指标
    
    def to_dict(self):
        return {
            "round_num": self.round_num,
            "num_clients": self.num_clients,
            "total_samples": self.total_samples,
            "metrics": self.metrics,
        }


@dataclass
class RoundResult:
    """单个联邦学习轮次的结果"""
    round_num: int                                   # 轮次编号
    updates: List[ClientUpdate]                      # 客户端更新列表
    aggregated_weights: Any                          # 聚合后的权重
    metrics: RoundMetrics                            # 轮次指标
    metadata: Dict[str, Any] = field(default_factory=dict)  # 其他元数据
    
    def to_dict(self):
        return {
            "round_num": self.round_num,
            "updates": [update.__dict__ for update in self.updates],
            "aggregated_weights": self.aggregated_weights,
            "metrics": {
                "round_num": self.metrics.round_num,
                "num_clients": self.metrics.num_clients,
                "total_samples": self.metrics.total_samples,
                "metrics": self.metrics.metrics,
            },
            "metadata": self.metadata,
        }


@dataclass
class FitConfig:
    """训练配置"""
    
    epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 0.01
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
        }
        result.update(self.extra)
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FitConfig":
        """从字典创建"""
        known_keys = {"epochs", "batch_size", "learning_rate"}
        extra = {k: v for k, v in data.items() if k not in known_keys}
        return cls(
            epochs=data.get("epochs", 1),
            batch_size=data.get("batch_size", 32),
            learning_rate=data.get("learning_rate", 0.01),
            extra=extra,
        )


@dataclass
class EvalConfig:
    """评估配置"""
    
    batch_size: int = 32
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {"batch_size": self.batch_size}
        result.update(self.extra)
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvalConfig":
        """从字典创建"""
        extra = {k: v for k, v in data.items() if k != "batch_size"}
        return cls(
            batch_size=data.get("batch_size", 32),
            extra=extra,
        )
