# fedcl/fl/abstract_trainer.py
"""
抽象联邦训练器 - 精简版

定义联邦训练的核心接口，用户可以继承这个抽象类来实现自定义的联邦训练逻辑
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import time
from loguru import logger

from ..transparent.base_federation_engine import BaseFederationEngine, TrainingResult, EvaluationResult


class AbstractFederationTrainer(ABC):
    """
    抽象联邦训练器 - 精简版
    
    核心设计原则：
    1. 只定义必要的抽象接口
    2. 提供少量通用工具方法
    3. 让子类专注于业务逻辑实现
    4. 避免过度抽象和复杂实现
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logger.bind(component=self.__class__.__name__)
        
        # 初始化底层联邦引擎
        self.federation_engine = BaseFederationEngine(config)
        
        # 训练状态
        self.round_history: List[Dict[str, Any]] = []
        self.global_state: Dict[str, Any] = {}
        
        self.logger.info(f"✅ {self.__class__.__name__} 初始化完成")
    
    @abstractmethod
    def train(self, num_rounds: int, **kwargs) -> TrainingResult:
        """
        执行联邦训练 - 子类必须实现
        
        这是用户自定义的联邦训练业务逻辑，每个trainer可以有不同的实现。
        
        Args:
            num_rounds: 训练轮次
            **kwargs: 训练参数
            
        Returns:
            TrainingResult: 训练结果
        """
        pass
    
    @abstractmethod
    def evaluate(self, test_data: Optional[Any] = None, **kwargs) -> EvaluationResult:
        """
        执行模型评估 - 子类必须实现
        
        Args:
            test_data: 测试数据
            **kwargs: 评估参数
            
        Returns:
            EvaluationResult: 评估结果
        """
        pass
    
    # ================ 通用工具方法 ================
    
    def build_training_result(self, num_rounds: int, training_time: float, execution_mode: str = "unknown") -> TrainingResult:
        """
        构建训练结果 - 通用工具方法
        
        Args:
            num_rounds: 训练轮次
            training_time: 训练时间
            execution_mode: 执行模式
            
        Returns:
            TrainingResult: 训练结果
        """
        if not self.round_history:
            raise ValueError("没有记录到任何训练轮次")
        
        final_metrics = {}
        
        # 获取最后一轮的所有数值指标
        for k, v in self.round_history[-1].items():
            if k not in ["round", "participants", "num_participants"] and isinstance(v, (int, float)):
                final_metrics[k] = v
        
        primary_metric = self.config.get("primary_metric", "accuracy")
        
        return TrainingResult(
            total_rounds=num_rounds,
            final_metrics=final_metrics,
            round_history=self.round_history.copy(),
            client_results={},
            execution_mode=execution_mode,
            training_time=training_time,
            primary_metric=primary_metric,
            custom_results=self.global_state.copy()
        )
    
    def add_round_result(self, round_num: int, result: Dict[str, Any]) -> None:
        """
        添加轮次结果到历史记录
        
        Args:
            round_num: 轮次编号
            result: 轮次结果
        """
        self.round_history.append({
            "round": round_num,
            **result
        })
    
    def update_global_state(self, key: str, value: Any) -> None:
        """更新全局状态"""
        self.global_state[key] = value
    
    def get_global_state(self, key: str, default: Any = None) -> Any:
        """获取全局状态"""
        return self.global_state.get(key, default)
    
    def cleanup(self):
        """清理资源"""
        self.federation_engine.cleanup()
        self.round_history.clear()
        self.global_state.clear()
        self.logger.info("🧹 联邦训练器资源清理完成")