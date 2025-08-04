# fedcl/implementations/hooks/custom_hook.py
"""
自定义钩子示例实现
"""

from typing import Dict, Any
from omegaconf import DictConfig
from loguru import logger

from ...core.hook import Hook
from ...core.execution_context import ExecutionContext
from ...registry import registry


@registry.hook("custom", metadata={
    "description": "Custom Hook Example"
})
class CustomHook(Hook):
    """自定义钩子示例"""
    
    def __init__(self, context: ExecutionContext, config: DictConfig):
        super().__init__(context, config)
        self.log_frequency = config.get("log_frequency", 10)
        
    def on_experiment_start(self, experiment_id: str, config: DictConfig) -> None:
        """实验开始回调"""
        logger.debug(f"Custom hook: Experiment {experiment_id} started")
        
    def on_round_end(self, round_id: int, metrics: Dict[str, Any]) -> None:
        """轮次结束回调"""
        if round_id % self.log_frequency == 0:
            logger.debug(f"Custom hook: Round {round_id} metrics: {metrics}")
            
    def on_experiment_end(self, results: Dict[str, Any]) -> None:
        """实验结束回调"""
        logger.debug(f"Custom hook: Experiment completed with results: {results}")
