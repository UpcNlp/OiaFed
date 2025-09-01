# fedcl/transparent/base_federation_engine.py
"""
底层联邦通信引擎

负责处理真伪联邦切换、通信管理、资源协调等底层细节
用户无需关心这些实现，专注于联邦训练逻辑
"""

import time
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass
from omegaconf import DictConfig
from loguru import logger

from .mode_detector import ModeDetector, ExecutionMode
from .strategy_selector import StrategySelector, ExecutionStrategy


@dataclass
class TrainingResult:
    """训练结果"""
    total_rounds: int
    final_metrics: Dict[str, float]
    round_history: List[Dict[str, Any]]
    client_results: Dict[str, Any]
    execution_mode: str
    training_time: float
    primary_metric: str = "accuracy"
    global_model_path: Optional[str] = None
    custom_results: Dict[str, Any] = None


@dataclass
class EvaluationResult:
    """评估结果"""
    metrics: Dict[str, float]
    task_metrics: Dict[str, Dict[str, float]]
    metadata: Dict[str, Any]
    evaluation_time: float
    primary_metric: str = "accuracy"


class BaseFederationEngine:
    """
    底层联邦通信引擎
    
    负责：
    1. 自动检测运行环境（真联邦/伪联邦/本地模拟）
    2. 透明地处理通信和资源协调
    3. 为上层训练器提供统一的联邦执行接口
    """
    
    def __init__(self, config: Union[Dict[str, Any], DictConfig]):
        """初始化底层联邦引擎"""
        self.config = config if isinstance(config, DictConfig) else DictConfig(config)
        self.logger = logger.bind(component="BaseFederationEngine")
        
        # 解析配置
        self.primary_metric = self.config.get("primary_metric", "accuracy")
        self.evaluation_metrics = self.config.get("evaluation_metrics", ["accuracy", "loss"])
        
        # 初始化核心组件
        self._mode_detector = ModeDetector()
        self._strategy_selector = StrategySelector()
        
        # 当前执行状态
        self._current_execution_mode = None
        self._current_strategy = None
        
        self.logger.info("✅ 底层联邦引擎初始化完成")
    
    def detect_execution_mode(self) -> ExecutionMode:
        """检测执行模式"""
        mode = self._mode_detector.detect_mode()
        self._current_execution_mode = mode
        self.logger.info(f"🔍 检测到执行模式: {mode.value}")
        return mode
    
    def select_strategy(self, mode: ExecutionMode) -> ExecutionStrategy:
        """选择执行策略"""
        strategy = self._strategy_selector.select_strategy(mode)
        self._current_strategy = strategy
        self.logger.info(f"🎯 选择执行策略: {strategy.value}")
        return strategy
    
    def create_federation_context(self, num_rounds: int, **kwargs):
        """创建联邦执行上下文（简化版本）"""
        # 检测执行模式
        mode = self.detect_execution_mode()
        
        # 选择执行策略
        strategy = self.select_strategy(mode)
        
        # 创建简化的上下文
        context = {
            "mode": mode,
            "strategy": strategy,
            "num_rounds": num_rounds,
            "learner": kwargs.get("learner") or self.config.get("learner", "simple_learner"),
            "aggregator": kwargs.get("aggregator") or self.config.get("aggregator", "fedavg"),
            "num_clients": kwargs.get("num_clients") or self.config.get("num_clients", 3),
            "dataset": kwargs.get("dataset") or self.config.get("dataset"),
            "global_model": kwargs.get("global_model"),
            "node_role": kwargs.get("node_role", "auto"),
            **kwargs
        }
        
        self.logger.info(f"📋 创建联邦上下文 - 模式: {mode.value}, 策略: {strategy.value}")
        return context
    
    def start_federation(self, federation_context):
        """启动联邦实现（简化版本）"""
        mode = federation_context["mode"]
        strategy = federation_context["strategy"]
        
        self.logger.info(f"🚀 启动联邦实现 - 模式: {mode.value}, 策略: {strategy.value}")
        return federation_context
    
    def stop_federation(self, federation_context):
        """停止联邦实现（简化版本）"""
        self.logger.info("🔌 联邦实现已停止")
    
    def get_execution_mode(self) -> Optional[str]:
        """获取当前执行模式"""
        if self._current_execution_mode:
            return self._current_execution_mode.value
        return None
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """获取执行统计信息"""
        stats = {
            "current_mode": self.get_execution_mode(),
            "current_strategy": self._current_strategy.value if self._current_strategy else None,
            "mode_detector_stats": self._mode_detector.get_detection_stats() if self._mode_detector else {},
            "strategy_selector_stats": self._strategy_selector.get_selection_stats() if self._strategy_selector else {}
        }
        
        return stats
    
    def cleanup(self):
        """清理资源"""
        try:
            self._current_execution_mode = None
            self._current_strategy = None
            
            self.logger.info("🧹 底层联邦引擎资源清理完成")
            
        except Exception as e:
            self.logger.error(f"清理资源失败: {e}")