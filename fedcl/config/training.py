"""
训练配置类定义（简化版）
统一的训练配置，自动根据字段推断角色
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from .base import BaseConfig


@dataclass
class TrainingConfig(BaseConfig):
    """
    训练配置（统一的配置类）

    角色自动推断：
    - 有 trainer/aggregator → 服务端配置
    - 有 learner/dataset → 客户端配置
    - 两者都有 → 本地模式配置
    """

    # ========== 组件导入 ==========
    components: List[str] = field(default_factory=list)  # 要导入的模块列表

    # ========== 训练参数（服务端）==========
    max_rounds: Optional[int] = None                    # 最大训练轮数
    min_clients: Optional[int] = None                   # 最少客户端数量

    # ========== 服务端组件（Optional）==========
    trainer: Optional[Dict[str, Any]] = None            # 训练器配置
    aggregator: Optional[Dict[str, Any]] = None         # 聚合器配置
    evaluator: Optional[Dict[str, Any]] = None          # 评估器配置
    global_model: Optional[Dict[str, Any]] = None       # 全局模型配置

    # ========== 客户端组件（Optional）==========
    learner: Optional[Dict[str, Any]] = None            # 学习器配置
    dataset: Optional[Dict[str, Any]] = None            # 数据集配置
    local_model: Optional[Dict[str, Any]] = None        # 本地模型配置

    # ========== 训练策略配置 ==========
    training_strategy: Dict[str, Any] = field(default_factory=lambda: {
        "early_stopping": {
            "enabled": False,
            "patience": 10,
            "metric": "accuracy",
            "mode": "max"
        },
        "lr_scheduler": {
            "enabled": False,
            "type": "step",
            "params": {}
        }
    })

    # ========== 检查点配置 ==========
    checkpoint: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "save_frequency": 10,
        "save_path": "./checkpoints",
        "keep_last_n": 5
    })

    def infer_role(self) -> str:
        """
        自动推断配置角色

        Returns:
            "server" / "client" / "local" / "unknown"
        """
        has_server_components = any([self.trainer, self.aggregator])
        has_client_components = any([self.learner, self.dataset])

        if has_server_components and has_client_components:
            return "local"
        elif has_server_components:
            return "server"
        elif has_client_components:
            return "client"
        else:
            return "unknown"

    def is_server_config(self) -> bool:
        """是否是服务端配置"""
        return self.infer_role() in ["server", "local"]

    def is_client_config(self) -> bool:
        """是否是客户端配置"""
        return self.infer_role() in ["client", "local"]
