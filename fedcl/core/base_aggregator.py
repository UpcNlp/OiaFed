# fedcl/core/base_aggregator.py
"""
BaseAggregator抽象基类模块

提供联邦学习聚合器的基础接口定义，包括客户端更新聚合、权重计算、
部分参与处理等功能。支持多种聚合策略和安全聚合机制。
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import torch
from omegaconf import DictConfig
from loguru import logger

from .execution_context import ExecutionContext
from ..exceptions import AggregationError, ConfigurationError


class BaseAggregator(ABC):
    """
    聚合器抽象基类
    
    定义了联邦学习中聚合器的基础接口，负责将多个客户端的模型更新
    聚合为全局模型更新。支持加权聚合、部分客户端参与、安全聚合等功能。
    
    Attributes:
        context: 执行上下文，提供配置和状态管理
        config: 聚合器配置参数
        device: 计算设备
        aggregation_history: 聚合历史记录
        client_weights_cache: 客户端权重缓存
    """
    
    def __init__(self, context: ExecutionContext, config: DictConfig) -> None:
        """
        初始化聚合器
        
        Args:
            context: 执行上下文对象
            config: 聚合器配置参数
            
        Raises:
            ConfigurationError: 配置参数无效时抛出
        """
        if not isinstance(context, ExecutionContext):
            raise ConfigurationError("Invalid execution context provided")
            
        if not isinstance(config, DictConfig):
            raise ConfigurationError("Invalid configuration provided")
            
        self.context = context
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.aggregation_history: List[Dict[str, Any]] = []
        self.client_weights_cache: Dict[str, float] = {}
        
        logger.debug(f"Initialized {self.__class__.__name__} with device: {self.device}")
    
    @abstractmethod
    def aggregate(self, client_updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        聚合客户端更新
        
        将多个客户端的模型参数更新聚合为全局模型更新。这是聚合器的核心功能，
        不同的聚合算法（如FedAvg、FedProx等）需要实现不同的聚合逻辑。
        
        Args:
            client_updates: 客户端更新列表，每个元素是参数名到张量的映射
            
        Returns:
            Dict[str, torch.Tensor]: 聚合后的全局模型参数更新
            
        Raises:
            AggregationError: 聚合过程中出现错误时抛出
        """
        pass
    
    @abstractmethod
    def weight_updates(self, updates: List[Dict[str, torch.Tensor]]) -> List[float]:
        """
        计算客户端权重
        
        为每个客户端的更新计算聚合权重，通常基于数据量、模型质量等因素。
        权重将用于加权平均聚合。
        
        Args:
            updates: 客户端更新列表
            
        Returns:
            List[float]: 对应每个客户端的权重列表，权重和为1.0
            
        Raises:
            AggregationError: 权重计算失败时抛出
        """
        pass
    
    def pre_aggregate_hook(self, client_updates: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
        """
        聚合前处理钩子
        
        在执行聚合前对客户端更新进行预处理，如异常检测、格式转换等。
        子类可重写此方法实现特定的预处理逻辑。
        
        Args:
            client_updates: 原始客户端更新列表
            
        Returns:
            List[Dict[str, torch.Tensor]]: 预处理后的客户端更新列表
        """
        logger.debug(f"Pre-aggregation hook: processing {len(client_updates)} client updates")
        return client_updates
    
    def post_aggregate_hook(self, aggregated_update: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        聚合后处理钩子
        
        在聚合完成后对结果进行后处理，如噪声添加、约束应用等。
        子类可重写此方法实现特定的后处理逻辑。
        
        Args:
            aggregated_update: 聚合后的模型更新
            
        Returns:
            Dict[str, torch.Tensor]: 后处理后的模型更新
        """
        logger.debug("Post-aggregation hook: processing aggregated update")
        return aggregated_update
    
    def get_aggregation_stats(self) -> Dict[str, Any]:
        """
        获取聚合统计信息
        
        返回聚合过程的统计信息，如参与客户端数量、聚合轮次、性能指标等。
        
        Returns:
            Dict[str, Any]: 聚合统计信息字典
        """
        return {
            "total_轮次": len(self.aggregation_history),
            "device": str(self.device),
            "aggregator_type": self.__class__.__name__,
            "last_client_count": len(self.client_weights_cache) if self.client_weights_cache else 0
        }
    
    def supports_partial_participation(self) -> bool:
        """
        是否支持部分客户端参与
        
        指示此聚合器是否支持部分客户端参与训练，即并非所有客户端
        都需要在每轮中参与。
        
        Returns:
            bool: True表示支持部分参与，False表示需要全部参与
        """
        return True
    
    def adjust_for_missing_clients(self, missing_clients: List[str]) -> None:
        """
        调整缺失客户端的聚合策略
        
        当某些客户端未能参与当前轮次时，调整聚合策略以适应缺失的客户端。
        
        Args:
            missing_clients: 缺失客户端ID列表
        """
        if missing_clients:
            logger.warning(f"Adjusting aggregation for {len(missing_clients)} missing clients: {missing_clients}")
            
            # 从权重缓存中移除缺失的客户端
            for client_id in missing_clients:
                self.client_weights_cache.pop(client_id, None)
    
    def validate_client_updates(self, client_updates: List[Dict[str, torch.Tensor]]) -> bool:
        """
        验证客户端更新的有效性
        
        检查客户端更新是否包含必要的参数、形状是否一致等。
        
        Args:
            client_updates: 客户端更新列表
            
        Returns:
            bool: True表示所有更新都有效，False表示存在无效更新
            
        Raises:
            AggregationError: 发现严重的验证错误时抛出
        """
        if not client_updates:
            raise AggregationError("No client updates provided for aggregation")
        
        try:
            # 检查所有更新是否具有相同的参数键
            first_keys = set(client_updates[0].keys())
            for i, update in enumerate(client_updates[1:], 1):
                if set(update.keys()) != first_keys:
                    logger.error(f"Client {i} has different parameter keys than client 0")
                    return False
            
            # 检查参数形状是否一致
            for key in first_keys:
                first_shape = client_updates[0][key].shape
                for i, update in enumerate(client_updates[1:], 1):
                    if update[key].shape != first_shape:
                        logger.error(f"Parameter '{key}' shape mismatch: client 0 has {first_shape}, client {i} has {update[key].shape}")
                        return False
            
            logger.debug(f"Validated {len(client_updates)} client updates successfully")
            return True
            
        except Exception as e:
            raise AggregationError(f"Client update validation failed: {str(e)}")
    
    def compute_aggregation_weights(self, client_data_sizes: List[int]) -> List[float]:
        """
        基于数据大小计算聚合权重
        
        这是一个通用的权重计算方法，基于客户端的数据量大小。
        子类可以重写或扩展此方法。
        
        Args:
            client_data_sizes: 各客户端的数据量大小列表
            
        Returns:
            List[float]: 归一化后的权重列表
            
        Raises:
            AggregationError: 权重计算失败时抛出
        """
        if not client_data_sizes or any(size <= 0 for size in client_data_sizes):
            raise AggregationError("Invalid client data sizes for weight computation")
        
        total_size = sum(client_data_sizes)
        weights = [size / total_size for size in client_data_sizes]
        
        logger.debug(f"Computed aggregation weights based on data sizes: {weights}")
        return weights
    
    def apply_differential_privacy(self, aggregated_update: Dict[str, torch.Tensor], 
                                  noise_scale: float = 0.1) -> Dict[str, torch.Tensor]:
        """
        应用差分隐私噪声
        
        为聚合后的模型更新添加差分隐私噪声，保护客户端隐私。
        
        Args:
            aggregated_update: 聚合后的模型更新
            noise_scale: 噪声缩放因子
            
        Returns:
            Dict[str, torch.Tensor]: 添加噪声后的模型更新
        """
        if noise_scale <= 0:
            return aggregated_update
        
        noisy_update = {}
        for key, param in aggregated_update.items():
            noise = torch.normal(0, noise_scale, size=param.shape, device=param.device)
            noisy_update[key] = param + noise
        
        logger.debug(f"Applied differential privacy with noise scale: {noise_scale}")
        return noisy_update
    
    def record_aggregation_round(self, round_info: Dict[str, Any]) -> None:
        """
        记录聚合轮次信息
        
        将当前轮次的聚合信息添加到历史记录中，用于分析和调试。
        
        Args:
            round_info: 包含轮次信息的字典
        """
        round_info["timestamp"] = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        round_info["round_id"] = len(self.aggregation_history)
        
        self.aggregation_history.append(round_info)
        
        # 限制历史记录大小
        max_history = self.config.get("max_history_size", 100)
        if len(self.aggregation_history) > max_history:
            self.aggregation_history = self.aggregation_history[-max_history:]
        
        logger.debug(f"Recorded aggregation round {round_info['round_id']}")
    
    def get_client_contribution_stats(self) -> Dict[str, Dict[str, float]]:
        """
        获取客户端贡献统计
        
        分析历史聚合数据，计算各客户端的贡献统计信息。
        
        Returns:
            Dict[str, Dict[str, float]]: 客户端贡献统计，外层键为客户端ID，
                                       内层字典包含各种统计指标
        """
        stats = {}
        for round_info in self.aggregation_history:
            if "client_weights" in round_info:
                for client_id, weight in round_info["client_weights"].items():
                    if client_id not in stats:
                        stats[client_id] = {
                            "total_weight": 0.0,
                            "participation_count": 0,
                            "avg_weight": 0.0
                        }
                    stats[client_id]["total_weight"] += weight
                    stats[client_id]["participation_count"] += 1
        
        # 计算平均权重
        for client_id in stats:
            if stats[client_id]["participation_count"] > 0:
                stats[client_id]["avg_weight"] = (
                    stats[client_id]["total_weight"] / 
                    stats[client_id]["participation_count"]
                )
        
        return stats
    
    def reset_history(self) -> None:
        """
        重置聚合历史
        
        清空聚合历史记录和缓存，通常在开始新的实验时调用。
        """
        self.aggregation_history.clear()
        self.client_weights_cache.clear()
        logger.debug("Aggregation history reset")
    
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
            }
        return {"cpu_memory": "N/A"}
    
    def cleanup(self) -> None:
        """
        清理资源
        
        释放聚合器占用的资源，在聚合器生命周期结束时调用。
        """
        self.aggregation_history.clear()
        self.client_weights_cache.clear()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.debug("聚合器 resources cleaned up")
    
    def __repr__(self) -> str:
        """字符串表示"""
        return (f"{self.__class__.__name__}("
                f"device={self.device}, "
                f"轮次={len(self.aggregation_history)}, "
                f"partial_participation={self.supports_partial_participation()})")