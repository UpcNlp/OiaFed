# fedcl/core/base_learner.py
"""
BaseLearner抽象基类模块

提供所有学习器实现的基础接口定义，包括任务训练、评估、钩子机制和状态管理。
支持3种灵活的模型初始化方式：直接构建法、辅助模型法、工厂函数法。
支持持续学习场景下的模型训练和知识保持。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from loguru import logger

from .execution_context import ExecutionContext
from ..data.results import TaskResults
from ..exceptions import LearnerError, ModelStateError, ConfigurationError
from ..utils.improved_logging_manager import get_component_logger


class BaseLearner(ABC):
    """
    学习器抽象基类
    
    定义了所有持续学习算法的基础接口，包括任务训练、评估、状态管理等功能。
    支持3种灵活的模型初始化方式，子类需要实现具体的学习算法逻辑。
    
    模型初始化方式：
    1. 直接构建法：子类实现_create_default_model()方法
    2. 辅助模型法：通过auxiliary_models参数传入预创建的模型
    3. 工厂函数法：通过model_factory配置自定义模型创建函数
    
    Attributes:
        context: 执行上下文，提供配置和状态管理
        config: 学习器配置参数
        model: 神经网络模型
        optimizer: 优化器
        device: 计算设备
        current_task_id: 当前任务ID
    """
    
    def __init__(self, context: ExecutionContext, config: DictConfig, **kwargs) -> None:
        """
        初始化学习器
        
        Args:
            context: 执行上下文对象
            config: 学习器配置参数
            **kwargs: 额外参数，支持：
                - model: 直接传入的模型实例
                - auxiliary_models: 辅助模型字典
                
        Raises:
            ConfigurationError: 配置参数无效时抛出
        """
        if not isinstance(context, ExecutionContext):
            raise ConfigurationError("Invalid execution context provided")
        if not isinstance(config, DictConfig):
            raise ConfigurationError("Invalid configuration provided")
            
        self.context = context
        self.config = config
        client_id = context.get_state("client_info").get("client_id", "unknown")
        self.logger = get_component_logger( "client",client_id)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_task_id: Optional[int] = None
        
        # 简化的模型初始化逻辑
        self.model = self._initialize_model(**kwargs)
        if self.model:
            self.model = self.model.to(self.device)
            
        self.optimizer: Optional[torch.optim.Optimizer] = None
        
        self.logger.debug(f"Initialized {self.__class__.__name__} with device: {self.device}")

    def _initialize_model(self, **kwargs) -> nn.Module:
        """
        简化的模型初始化逻辑
        
        支持2种初始化方式（按优先级）：
        1. 直接传入模型实例 (model=xxx)
        2. 使用辅助模型 (从auxiliary_models中获取)
        3. 子类构建模型 (_create_default_model方法)
        
        Args:
            **kwargs: 可能包含model、auxiliary_models等参数
            
        Returns:
            初始化后的模型
            
        Raises:
            NotImplementedError: 当所有初始化方式都不可用时抛出
        """
        # 方式1: 直接传入模型实例（最高优先级）
        if 'model' in kwargs and kwargs['model'] is not None:
            logger.debug("Using directly passed model instance")
            return kwargs['model']
        
        # 方式2: 从辅助模型中获取
        auxiliary_models = kwargs.get('auxiliary_models', {})
        model_name = self.config.get('model_name')
        if model_name and auxiliary_models:
            model = self._get_auxiliary_model(auxiliary_models, model_name)
            if model is not None:
                return model
        
        # 方式3: 子类构建模型
        logger.debug("Creating model using _create_default_model method")
        return self._create_default_model()
    
    def _get_auxiliary_model(self, auxiliary_models: Dict[str, Any], model_name: str) -> Optional[nn.Module]:
        """
        从辅助模型中获取模型
        
        Args:
            auxiliary_models: 辅助模型字典
            model_name: 模型名称
            
        Returns:
            模型实例或None
        """
        if model_name not in auxiliary_models:
            logger.warning(f"Auxiliary model '{model_name}' not found")
            return None
        
        model_info = auxiliary_models[model_name]
        
        # 支持不同格式
        if isinstance(model_info, nn.Module):
            logger.debug(f"Using auxiliary model instance: {model_name}")
            return model_info
        elif isinstance(model_info, dict) and 'model' in model_info:
            logger.debug(f"Using auxiliary model from dict: {model_name}")
            return model_info['model']
        else:
            logger.warning(f"Invalid auxiliary model format for: {model_name}")
            return None
    
    
    @abstractmethod
    def _create_default_model(self) -> nn.Module:
        """
        创建默认模型（由子类实现）
        
        子类必须实现此方法来提供默认的模型创建逻辑。
        
        Returns:
            默认模型实例
        """
        pass
    
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
    
    def get_model_state(self) -> Dict[str, torch.Tensor]:
        """
        获取模型状态字典
        
        Returns:
            Dict[str, torch.Tensor]: 模型的状态字典
            
        Raises:
            ModelStateError: 模型未初始化时抛出
        """
        if self.model is None:
            raise ModelStateError("Model not initialized")
        return self.model.state_dict()
    
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
        logger.debug(f"Model set and moved to device: {self.device}")
    
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
        logger.debug(f"Optimizer set: {optimizer.__class__.__name__}")
    
    def get_upload_parameters(self) -> Dict[str, Any]:
        """
        获取要上传的参数（联邦学习聚合用）
        
        用户可重写此方法来自定义上传参数的选择策略。
        支持三种预设模式：
        1. full_model: 上传全部模型权重（默认）
        2. last_layer_output: 上传模型最后一层的输出特征
        3. custom: 用户自定义参数选择
        
        Returns:
            Dict[str, Any]: 要上传的参数字典
        """
        if self.model is None:
            raise ModelStateError("Model not initialized")
            
        # 从配置中获取上传策略
        upload_strategy = self.config.get('aggregation', {}).get('upload_strategy', 'full_model')
        
        if upload_strategy == "full_model":
            return self.model.state_dict()
            
        elif upload_strategy == "last_layer_output":
            return self._get_last_layer_output()
            
        elif upload_strategy == "custom":
            # 检查是否有指定的参数路径
            upload_params = self.config.get('aggregation', {}).get('upload_params', [])
            if upload_params:
                return self._get_parameters_by_paths(upload_params)
            else:
                # 调用用户自定义方法
                return self._custom_parameter_selection()
        else:
            logger.warning(f"Unknown upload strategy: {upload_strategy}, using full_model")
            return self.model.state_dict()
    
    def _get_last_layer_output(self) -> Dict[str, Any]:
        """
        获取最后一层输出特征（预设模式）
        
        Returns:
            Dict[str, Any]: 包含最后一层特征的字典
        """
        try:
            # 获取模型的最后一层
            last_layer_name = None
            last_layer = None
            
            for name, module in self.model.named_modules():
                if len(list(module.children())) == 0:  # 叶子节点
                    last_layer_name = name
                    last_layer = module
            
            if last_layer is not None:
                # 如果是线性层，返回权重和偏置
                if isinstance(last_layer, nn.Linear):
                    return {
                        "last_layer_weight": last_layer.weight.data.clone(),
                        "last_layer_bias": last_layer.bias.data.clone() if last_layer.bias is not None else None,
                        "layer_name": last_layer_name
                    }
                else:
                    # 其他类型的层，返回所有参数
                    return {
                        f"last_layer_{name}": param.data.clone() 
                        for name, param in last_layer.named_parameters()
                    }
            else:
                logger.warning("Could not find last layer, falling back to full model")
                return self.model.state_dict()
                
        except Exception as e:
            logger.error(f"Error getting last layer output: {e}")
            return self.model.state_dict()
    
    def _get_parameters_by_paths(self, param_paths: List[str]) -> Dict[str, torch.Tensor]:
        """
        根据参数路径获取指定参数
        
        Args:
            param_paths: 参数路径列表，如 ["conv1.weight", "fc.bias"]
            
        Returns:
            Dict[str, torch.Tensor]: 指定参数字典
        """
        state_dict = self.model.state_dict()
        selected_params = {}
        
        for path in param_paths:
            if path in state_dict:
                selected_params[path] = state_dict[path].clone()
            else:
                logger.warning(f"Parameter path not found: {path}")
        
        return selected_params
    
    def _custom_parameter_selection(self) -> Dict[str, Any]:
        """
        用户自定义参数选择逻辑
        
        子类可以重写此方法来实现自定义的参数选择策略。
        默认实现返回全部模型参数。
        
        Returns:
            Dict[str, Any]: 自定义选择的参数
        """
        logger.debug("Using default parameter selection (full model)")
        return self.model.state_dict()

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
        
        logger.debug("Learner state saved successfully")
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
                
            logger.debug("Learner state loaded successfully")
            
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
        logger.debug(f"Reset learner for new task: {task_id}")
    
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
        logger.debug("Learner resources cleaned up")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            Dict[str, Any]: 模型相关信息
        """
        if self.model is None:
            return {"model": "not_initialized"}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "model_class": type(self.model).__name__,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": str(self.device),
            "optimizer": type(self.optimizer).__name__ if self.optimizer else None,
            "current_task_id": self.current_task_id
        }
    
    def __repr__(self) -> str:
        """字符串表示"""
        return (f"{self.__class__.__name__}("
                f"task_id={self.current_task_id}, "
                f"device={self.device}, "
                f"model={'初始化完成' if self.model else 'None'})")