# fedcl/core/base_learner.py
"""
BaseLearner抽象基类模块

提供所有学习器实现的基础接口定义，包括任务训练、评估、钩子机制和状态管理。
支持持续学习场景下的模型训练和知识保持。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from loguru import logger

from .execution_context import ExecutionContext
from ..data.results import TaskResults
from .exceptions import LearnerError, ModelStateError, ConfigurationError


class BaseLearner(ABC):
    """
    学习器抽象基类
    
    定义了所有持续学习算法的基础接口，包括任务训练、评估、状态管理等功能。
    子类需要实现具体的学习算法逻辑，如L2P、EWC、DDDR等。
    
    Attributes:
        context: 执行上下文，提供配置和状态管理
        config: 学习器配置参数
        model: 神经网络模型
        optimizer: 优化器
        device: 计算设备
        current_task_id: 当前任务ID
    """
    
    def __init__(self, context: ExecutionContext, config: DictConfig) -> None:
        """
        初始化学习器
        
        Args:
            context: 执行上下文对象
            config: 学习器配置参数
            
        Raises:
            ConfigurationError: 配置参数无效时抛出
        """
        if not isinstance(context, ExecutionContext):
            raise ConfigurationError("Invalid execution context provided")
            
        if not isinstance(config, DictConfig):
            raise ConfigurationError("Invalid configuration provided")
            
        self.context = context
        self.config = config
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_task_id: Optional[int] = None
        
        logger.info(f"Initialized {self.__class__.__name__} with device: {self.device}")
    
    @abstractmethod
    def train_task(self, task_data: DataLoader) -> TaskResults:
        """
        训练单个任务
        
        执行单个持续学习任务的训练过程，包括模型参数更新、
        知识保持策略应用等。
        
        Args:
            task_data: 任务训练数据加载器
            
        Returns:
            TaskResults: 包含训练结果、指标和模型状态的结果对象
            
        Raises:
            LearnerError: 训练过程中出现错误时抛出
        """
        pass
    
    @abstractmethod
    def evaluate_task(self, task_data: DataLoader) -> Dict[str, float]:
        """
        评估单个任务
        
        对指定任务数据进行模型评估，计算相关性能指标。
        
        Args:
            task_data: 任务评估数据加载器
            
        Returns:
            Dict[str, float]: 评估指标字典，键为指标名称，值为指标值
            
        Raises:
            LearnerError: 评估过程中出现错误时抛出
        """
        pass
    
    def before_task_hook(self, task_id: int, task_data: DataLoader) -> None:
        """
        任务开始前的钩子方法
        
        在任务训练开始前执行的预处理逻辑，子类可重写以实现
        特定的初始化操作。
        
        Args:
            task_id: 任务ID
            task_data: 任务数据加载器
        """
        self.current_task_id = task_id
        logger.debug(f"Starting task {task_id}")
    
    def after_task_hook(self, task_id: int, results: TaskResults) -> None:
        """
        任务结束后的钩子方法
        
        在任务训练结束后执行的后处理逻辑，子类可重写以实现
        特定的清理或保存操作。
        
        Args:
            task_id: 任务ID
            results: 任务训练结果
        """
        logger.debug(f"Completed task {task_id} with results: {results.get_summary()}")
    
    def before_epoch_hook(self, epoch: int) -> None:
        """
        轮次开始前的钩子方法
        
        在每个训练轮次开始前执行的逻辑，子类可重写。
        
        Args:
            epoch: 轮次编号
        """
        logger.debug(f"Starting epoch {epoch}")
    
    def after_epoch_hook(self, epoch: int, metrics: Dict[str, float]) -> None:
        """
        轮次结束后的钩子方法
        
        在每个训练轮次结束后执行的逻辑，子类可重写。
        
        Args:
            epoch: 轮次编号
            metrics: 该轮次的评估指标
        """
        logger.debug(f"Completed epoch {epoch} with metrics: {metrics}")
    
    def get_model(self) -> nn.Module:
        """
        获取当前模型
        
        Returns:
            nn.Module: 当前的神经网络模型
            
        Raises:
            ModelStateError: 模型未初始化时抛出
        """
        if self.model is None:
            raise ModelStateError("Model not initialized")
        return self.model
    
    def set_model(self, model: nn.Module) -> None:
        """
        设置模型
        
        Args:
            model: 要设置的神经网络模型
            
        Raises:
            ModelStateError: 模型无效时抛出
        """
        if not isinstance(model, nn.Module):
            raise ModelStateError("Invalid model provided")
            
        self.model = model
        self.model.to(self.device)
        logger.info(f"Model set and moved to device: {self.device}")
    
    def get_optimizer(self) -> torch.optim.Optimizer:
        """
        获取优化器
        
        Returns:
            torch.optim.Optimizer: 当前的优化器
            
        Raises:
            ModelStateError: 优化器未初始化时抛出
        """
        if self.optimizer is None:
            raise ModelStateError("Optimizer not initialized")
        return self.optimizer
    
    def set_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        """
        设置优化器
        
        Args:
            optimizer: 要设置的优化器
            
        Raises:
            ModelStateError: 优化器无效时抛出
        """
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise ModelStateError("Invalid optimizer provided")
            
        self.optimizer = optimizer
        logger.info(f"Optimizer set: {optimizer.__class__.__name__}")
    
    def get_learning_rate(self) -> float:
        """
        获取学习率
        
        Returns:
            float: 当前学习率
        """
        return self.context.get_config("learning_rate", 0.001)
    
    def get_batch_size(self) -> int:
        """
        获取批次大小
        
        Returns:
            int: 当前批次大小
        """
        return self.context.get_config("batch_size", 32)
    
    def get_device(self) -> torch.device:
        """
        获取计算设备
        
        Returns:
            torch.device: 当前计算设备
        """
        return self.device
    
    def save_learner_state(self) -> Dict[str, Any]:
        """
        保存学习器状态
        
        保存模型参数、优化器状态和配置信息，用于检查点恢复。
        
        Returns:
            Dict[str, Any]: 包含完整学习器状态的字典
            
        Raises:
            ModelStateError: 模型或优化器未初始化时抛出
        """
        if self.model is None:
            raise ModelStateError("Cannot save state: model not initialized")
            
        if self.optimizer is None:
            raise ModelStateError("Cannot save state: optimizer not initialized")
        
        state = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "config": self.config,
            "current_task_id": self.current_task_id,
            "device": str(self.device),
            "learner_class": self.__class__.__name__
        }
        
        logger.info("Learner state saved successfully")
        return state
    
    def load_learner_state(self, state: Dict[str, Any]) -> None:
        """
        加载学习器状态
        
        从保存的状态字典恢复模型参数、优化器状态等。
        
        Args:
            state: 包含学习器状态的字典
            
        Raises:
            ModelStateError: 状态加载失败时抛出
        """
        try:
            if "model_state" not in state or "optimizer_state" not in state:
                raise ModelStateError("Invalid state dictionary: missing required keys")
                
            if self.model is None:
                raise ModelStateError("Cannot load state: model not initialized")
                
            if self.optimizer is None:
                raise ModelStateError("Cannot load state: optimizer not initialized")
            
            self.model.load_state_dict(state["model_state"])
            self.optimizer.load_state_dict(state["optimizer_state"])
            
            if "current_task_id" in state:
                self.current_task_id = state["current_task_id"]
                
            logger.info("Learner state loaded successfully")
            
        except Exception as e:
            raise ModelStateError(f"Failed to load learner state: {str(e)}")
    
    def reset_for_new_task(self, task_id: int) -> None:
        """
        为新任务重置学习器状态
        
        在开始新任务前重置必要的内部状态，子类可重写以实现
        特定的重置逻辑。
        
        Args:
            task_id: 新任务的ID
        """
        self.current_task_id = task_id
        logger.info(f"Reset learner for new task: {task_id}")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        获取内存使用情况
        
        Returns:
            Dict[str, float]: 内存使用统计信息
        """
        if torch.cuda.is_available():
            return {
                "gpu_allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
                "gpu_cached": torch.cuda.memory_reserved() / 1024**3,      # GB
                "gpu_max_allocated": torch.cuda.max_memory_allocated() / 1024**3  # GB
            }
        return {"cpu_memory": "N/A"}
    
    def cleanup(self) -> None:
        """
        清理资源
        
        释放模型、优化器等占用的资源，在学习器生命周期结束时调用。
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.model = None
        self.optimizer = None
        logger.info("Learner resources cleaned up")
    
    def __repr__(self) -> str:
        """字符串表示"""
        return (f"{self.__class__.__name__}("
                f"task_id={self.current_task_id}, "
                f"device={self.device}, "
                f"model={'initialized' if self.model else 'None'})")