# fedcl/implementations/hooks/tensorboard_hook.py
"""
TensorBoard 钩子实现
"""

from typing import Dict, Any, Optional
from omegaconf import DictConfig
from loguru import logger
import os

from ...core.hook import Hook
from ...core.execution_context import ExecutionContext
from ...registry import registry

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    logger.warning("tensorboard not available, TensorBoardHook will be disabled")


@registry.hook("tensorboard", metadata={
    "description": "TensorBoard Logging Hook",
    "requires": ["torch"]
})
class TensorBoardHook(Hook):
    """TensorBoard日志钩子"""
    
    def __init__(self, context: ExecutionContext, config: DictConfig):
        super().__init__(context, config)
        self.enabled = TENSORBOARD_AVAILABLE and config.get("enabled", True)
        
        if self.enabled:
            log_dir = config.get("log_dir", "logs/tensorboard")
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None
            
    def on_round_end(self, round_id: int, metrics: Dict[str, Any]) -> None:
        """记录轮次指标"""
        if self.enabled and self.writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(key, value, round_id)
                    
    def on_experiment_end(self, results: Dict[str, Any]) -> None:
        """关闭writer"""
        if self.enabled and self.writer:
            self.writer.close()
