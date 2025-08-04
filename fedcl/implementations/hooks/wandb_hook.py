# fedcl/implementations/hooks/wandb_hook.py
"""
Weights & Biases 钩子实现
"""

from typing import Dict, Any, Optional
from omegaconf import DictConfig
from loguru import logger

from ...core.hook import Hook
from ...core.execution_context import ExecutionContext
from ...registry import registry

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not available, WandBHook will be disabled")


@registry.hook("wandb", metadata={
    "description": "Weights & Biases Integration Hook",
    "requires": ["wandb"]
})
class WandBHook(Hook):
    """WandB集成钩子"""
    
    def __init__(self, context: ExecutionContext, config: DictConfig):
        super().__init__(context, config)
        self.enabled = WANDB_AVAILABLE and config.get("enabled", True)
        self.project_name = config.get("project_name", "fedcl-experiment")
        
        if self.enabled:
            wandb.init(project=self.project_name, config=dict(config))
            
    def on_experiment_start(self, experiment_id: str, config: DictConfig) -> None:
        """实验开始时的回调"""
        if self.enabled:
            wandb.log({"experiment_start": True, "experiment_id": experiment_id})
            
    def on_round_end(self, round_id: int, metrics: Dict[str, Any]) -> None:
        """轮次结束时的回调"""
        if self.enabled:
            wandb.log({"round": round_id, **metrics})
            
    def on_experiment_end(self, results: Dict[str, Any]) -> None:
        """实验结束时的回调"""
        if self.enabled:
            wandb.log({"experiment_end": True, **results})
            wandb.finish()
