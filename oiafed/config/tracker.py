"""
追踪器配置

包含 MLflow、WandB、TensorBoard 等追踪后端的配置类及解析方法。

结构：
    TrackerConfig
    └── TrackerBackendConfig
        ├── MLflowConfig
        ├── WandbConfig
        └── TensorBoardConfig
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


# ==================== MLflow 配置 ====================

@dataclass
class MLflowConfig:
    """MLflow 追踪配置"""
    tracking_uri: str = "http://localhost:5000"
    experiment_name: Optional[str] = None  # None 则使用 exp_name
    run_name: Optional[str] = None         # None 则使用全局 run_name
    
    # 同步字段（从 GlobalConfig 同步）
    _exp_name: Optional[str] = field(default=None, repr=False, compare=False)
    _run_name: Optional[str] = field(default=None, repr=False, compare=False)
    
    def get_experiment_name(self) -> str:
        """获取实验名称，优先使用显式设置的值"""
        return self.experiment_name or self._exp_name or "default"
    
    def get_run_name(self) -> Optional[str]:
        """获取运行名称，优先使用显式设置的值"""
        return self.run_name or self._run_name
    
    def sync_from_global(self, exp_name: str, run_name: Optional[str]):
        """从全局配置同步"""
        self._exp_name = exp_name
        self._run_name = run_name


def parse_mlflow_config(data: Optional[Dict[str, Any]]) -> MLflowConfig:
    """解析 MLflow 配置"""
    if not data:
        return MLflowConfig()
    return MLflowConfig(
        tracking_uri=data.get("tracking_uri", "http://localhost:5000"),
        experiment_name=data.get("experiment_name"),
        run_name=data.get("run_name"),
    )


# ==================== WandB 配置 ====================

@dataclass
class WandbConfig:
    """Weights & Biases 追踪配置"""
    project: str = "federated-learning"
    entity: Optional[str] = None
    name: Optional[str] = None  # None 则使用全局 run_name
    tags: List[str] = field(default_factory=list)
    
    # 同步字段
    _exp_name: Optional[str] = field(default=None, repr=False, compare=False)
    _run_name: Optional[str] = field(default=None, repr=False, compare=False)
    
    def get_name(self) -> Optional[str]:
        """获取运行名称"""
        return self.name or self._run_name
    
    def sync_from_global(self, exp_name: str, run_name: Optional[str]):
        """从全局配置同步"""
        self._exp_name = exp_name
        self._run_name = run_name


def parse_wandb_config(data: Optional[Dict[str, Any]]) -> WandbConfig:
    """解析 WandB 配置"""
    if not data:
        return WandbConfig()
    return WandbConfig(
        project=data.get("project", "federated-learning"),
        entity=data.get("entity"),
        name=data.get("name"),
        tags=data.get("tags", []),
    )


# ==================== TensorBoard 配置 ====================

@dataclass
class TensorBoardConfig:
    """TensorBoard 追踪配置"""
    log_dir: Optional[str] = None  # None 则自动生成
    
    # 同步字段
    _log_dir: Optional[str] = field(default=None, repr=False, compare=False)
    _exp_name: Optional[str] = field(default=None, repr=False, compare=False)
    _run_name: Optional[str] = field(default=None, repr=False, compare=False)
    
    def get_log_dir(self) -> str:
        """获取日志目录"""
        if self.log_dir:
            return self.log_dir
        # 自动生成：{log_dir}/{exp_name}/{run_name}/tensorboard
        parts = [self._log_dir or "./logs"]
        if self._exp_name:
            parts.append(self._exp_name)
        if self._run_name:
            parts.append(self._run_name)
        parts.append("tensorboard")
        return "/".join(parts)
    
    def sync_from_global(self, log_dir: str, exp_name: str, run_name: Optional[str]):
        """从全局配置同步"""
        self._log_dir = log_dir
        self._exp_name = exp_name
        self._run_name = run_name


def parse_tensorboard_config(data: Optional[Dict[str, Any]]) -> TensorBoardConfig:
    """解析 TensorBoard 配置"""
    if not data:
        return TensorBoardConfig()
    return TensorBoardConfig(
        log_dir=data.get("log_dir"),
    )


# ==================== Backend 配置 ====================

@dataclass
class TrackerBackendConfig:
    """
    单个 Tracker Backend 配置
    
    Attributes:
        type: backend 类型（file, mlflow, wandb, tensorboard）
        args: 配置参数字典
    """
    type: str
    args: Optional[Dict[str, Any]] = None
    
    def get_args(self) -> Dict[str, Any]:
        """获取参数字典"""
        return self.args or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于 YAML 序列化）"""
        result = {"type": self.type}
        if self.args:
            result["args"] = self.args
        return result


def parse_tracker_backend_config(data: Dict[str, Any]) -> TrackerBackendConfig:
    """解析单个 backend 配置"""
    return TrackerBackendConfig(
        type=data.get("type", "file"),
        args=data.get("args"),
    )


# ==================== Tracker 配置 ====================

@dataclass
class TrackerConfig:
    """
    训练追踪配置
    
    Attributes:
        enabled: 是否启用追踪
        tracking_dir: 追踪目录（相对于日志目录）
        backends: backend 配置列表
        
        # 同步字段（从 GlobalConfig 同步）
        _exp_name: 实验名称
        _run_name: 运行名称
        _log_dir: 日志目录
    """
    enabled: bool = True
    tracking_dir: str = "tracking"
    backends: Optional[List[Union[Dict[str, Any], TrackerBackendConfig]]] = None
    
    # 同步字段
    _exp_name: Optional[str] = field(default=None, repr=False, compare=False)
    _run_name: Optional[str] = field(default=None, repr=False, compare=False)
    _log_dir: Optional[str] = field(default=None, repr=False, compare=False)
    
    @property
    def exp_name(self) -> Optional[str]:
        return self._exp_name
    
    @property
    def run_name(self) -> Optional[str]:
        return self._run_name
    
    @property
    def log_dir(self) -> Optional[str]:
        return self._log_dir
    
    def get_tracking_path(self) -> str:
        """
        获取完整的追踪路径
        
        Returns:
            格式：{log_dir}/{exp_name}/{run_name}/{tracking_dir}/
        """
        parts = [self._log_dir or "./logs"]
        if self._exp_name:
            parts.append(self._exp_name)
        if self._run_name:
            parts.append(self._run_name)
        parts.append(self.tracking_dir)
        return "/".join(parts)
    
    def get_backends(self) -> List[TrackerBackendConfig]:
        """获取标准化的 backend 列表"""
        if not self.backends:
            return []
        
        result = []
        for backend in self.backends:
            if isinstance(backend, TrackerBackendConfig):
                result.append(backend)
            elif isinstance(backend, dict):
                result.append(parse_tracker_backend_config(backend))
        return result
    
    def sync_from_global(self, log_dir: str, exp_name: str, run_name: Optional[str]):
        """从全局配置同步"""
        self._log_dir = log_dir
        self._exp_name = exp_name
        self._run_name = run_name


def parse_tracker_config(data: Optional[Dict[str, Any]]) -> TrackerConfig:
    """解析追踪配置"""
    if not data:
        return TrackerConfig()
    
    backends = None
    if data.get("backends") is not None:
        backends = [
            parse_tracker_backend_config(b) if isinstance(b, dict) else b
            for b in data["backends"]
        ]
    
    return TrackerConfig(
        enabled=data.get("enabled", True),
        tracking_dir=data.get("tracking_dir", "tracking"),
        backends=backends,
    )


# ==================== 导出 ====================

__all__ = [
    # MLflow
    "MLflowConfig",
    "parse_mlflow_config",
    # WandB
    "WandbConfig",
    "parse_wandb_config",
    # TensorBoard
    "TensorBoardConfig",
    "parse_tensorboard_config",
    # Backend
    "TrackerBackendConfig",
    "parse_tracker_backend_config",
    # Tracker
    "TrackerConfig",
    "parse_tracker_config",
]
