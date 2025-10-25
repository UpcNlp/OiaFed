"""
MOE-FedCL 客户端学习器抽象基类
moe_fedcl/learner/base_learner.py
"""

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Optional, List

from ..exceptions import ValidationError
from ..types import ModelData, TrainingResult, EvaluationResult


class BaseLearner(ABC):
    """客户端学习器抽象基类 - 用户继承实现本地训练逻辑"""
    
    def __init__(self, 
                 client_id: str,
                 local_data: Any = None,
                 model_config: Dict[str, Any] = None,
                 training_config: Dict[str, Any] = None):
        """
        初始化学习器
        
        Args:
            client_id: 客户端唯一标识
            local_data: 本地数据集
            model_config: 模型配置
            training_config: 训练配置
        """
        self.client_id = client_id
        self.local_data = local_data
        self.model_config = model_config or {}
        self.training_config = training_config or {}
        
        # 内部状态
        self._local_model: Optional[ModelData] = None
        self._training_history: List[Dict[str, Any]] = []
        self._is_initialized = False
        self._lock = asyncio.Lock()
        
        # 统计信息
        self.training_count = 0
        self.evaluation_count = 0
        self.last_training_time: Optional[datetime] = None
        self.last_evaluation_time: Optional[datetime] = None
    
    # ==================== 核心训练方法 (用户必须实现) ====================
    
    @abstractmethod
    async def train(self, training_params: Dict[str, Any]) -> TrainingResult:
        """执行本地训练
        
        Args:
            training_params: 训练参数，可能包含：
                - global_model: 全局模型参数
                - epochs: 训练轮数
                - learning_rate: 学习率
                - batch_size: 批次大小
                - 其他算法特定参数
                
        Returns:
            TrainingResult: 训练结果，应包含：
                - model_update: 模型更新/参数差异
                - loss: 训练损失
                - metrics: 训练指标
                - samples_count: 训练样本数
                - training_time: 训练时间
                
        Raises:
            TrainingError: 训练过程中的错误
        """
        pass
    
    @abstractmethod
    async def evaluate(self, evaluation_params: Dict[str, Any]) -> EvaluationResult:
        """执行本地评估
        
        Args:
            evaluation_params: 评估参数，可能包含：
                - model: 要评估的模型
                - test_data: 测试数据集
                - metrics: 评估指标列表
                
        Returns:
            EvaluationResult: 评估结果，应包含：
                - accuracy: 准确率
                - loss: 测试损失
                - metrics: 各项评估指标
                - samples_count: 测试样本数
                - evaluation_time: 评估时间
                
        Raises:
            TrainingError: 评估过程中的错误
        """
        pass
    
    @abstractmethod
    async def get_local_model(self) -> ModelData:
        """获取本地模型参数
        
        Returns:
            ModelData: 本地模型参数
        """
        pass
    
    @abstractmethod
    async def set_local_model(self, model_data: ModelData) -> bool:
        """设置本地模型参数
        
        Args:
            model_data: 模型参数数据
            
        Returns:
            bool: 设置是否成功
        """
        pass
    
    # ==================== 数据管理方法 (可选实现) ====================
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """获取数据统计信息
        
        Returns:
            Dict[str, Any]: 数据统计，可能包含：
                - total_samples: 总样本数
                - classes: 类别信息
                - data_distribution: 数据分布
                - data_quality: 数据质量指标
        """
        if self.local_data is None:
            return {"total_samples": 0, "message": "No local data available"}
        
        # 默认实现，用户可以重写
        try:
            if hasattr(self.local_data, '__len__'):
                return {
                    "total_samples": len(self.local_data),
                    "data_type": type(self.local_data).__name__,
                    "available": True
                }
            else:
                return {
                    "total_samples": "unknown",
                    "data_type": type(self.local_data).__name__,
                    "available": True
                }
        except Exception as e:
            return {
                "total_samples": 0,
                "error": str(e),
                "available": False
            }
    
    def prepare_training_data(self, batch_size: int = 32) -> Any:
        """准备训练数据
        
        Args:
            batch_size: 批次大小
            
        Returns:
            Any: 训练数据加载器或处理后的数据
        """
        # 默认实现，直接返回本地数据
        return self.local_data
    
    def prepare_evaluation_data(self) -> Any:
        """准备评估数据
        
        Returns:
            Any: 评估数据
        """
        # 默认实现，直接返回本地数据
        return self.local_data
    
    # ==================== 模型管理方法 (框架提供默认实现) ====================
    
    async def save_model(self, model_path: str) -> bool:
        """保存模型到文件
        
        Args:
            model_path: 模型保存路径
            
        Returns:
            bool: 保存是否成功
        """
        try:
            async with self._lock:
                model_data = await self.get_local_model()
                
                # 这里可以使用pickle、torch.save、tensorflow.save等
                # 默认实现使用简单的序列化
                import pickle
                with open(model_path, 'wb') as f:
                    pickle.dump(model_data, f)
                
                return True
                
        except Exception as e:
            print(f"Save model failed: {e}")
            return False
    
    async def load_model(self, model_path: str) -> Optional[ModelData]:
        """从文件加载模型
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            Optional[ModelData]: 加载的模型数据，失败返回None
        """
        try:
            import pickle
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # 设置到本地模型
            success = await self.set_local_model(model_data)
            return model_data if success else None
            
        except Exception as e:
            print(f"Load model failed: {e}")
            return None
    
    async def get_model_parameters(self) -> Dict[str, Any]:
        """获取模型参数摘要
        
        Returns:
            Dict[str, Any]: 模型参数摘要信息
        """
        try:
            model_data = await self.get_local_model()
            
            if isinstance(model_data, dict):
                return {
                    "parameter_count": len(model_data),
                    "parameter_keys": list(model_data.keys()),
                    "model_size_bytes": len(str(model_data)),
                    "has_model": True
                }
            else:
                return {
                    "model_type": type(model_data).__name__,
                    "model_size_bytes": len(str(model_data)) if model_data else 0,
                    "has_model": model_data is not None
                }
                
        except Exception as e:
            return {
                "error": str(e),
                "has_model": False
            }
    
    # ==================== 生命周期方法 (框架提供) ====================
    
    async def initialize(self) -> bool:
        """初始化学习器
        
        Returns:
            bool: 初始化是否成功
        """
        try:
            async with self._lock:
                if self._is_initialized:
                    return True
                
                # 执行初始化逻辑
                await self._perform_initialization()
                
                self._is_initialized = True
                return True
                
        except Exception as e:
            print(f"Learner initialization failed: {e}")
            return False
    
    async def _perform_initialization(self):
        """执行具体的初始化逻辑 - 子类可重写"""
        # 默认初始化逻辑
        if self.local_data is None:
            print(f"Warning: No local data provided for client {self.client_id}")
        
        # 验证配置
        await self._validate_configuration()
        
        print(f"BaseLearner {self.client_id} initialized successfully")
    
    async def _validate_configuration(self):
        """验证配置 - 子类可重写"""
        if not self.client_id:
            raise ValidationError("Client ID cannot be empty")
        
        if not isinstance(self.model_config, dict):
            raise ValidationError("Model config must be a dictionary")
        
        if not isinstance(self.training_config, dict):
            raise ValidationError("Training config must be a dictionary")
    
    async def cleanup(self) -> None:
        """清理学习器资源"""
        async with self._lock:
            self._local_model = None
            self._training_history.clear()
            self._is_initialized = False
        
        print(f"BaseLearner {self.client_id} cleaned up")
    
    def get_learner_info(self) -> Dict[str, Any]:
        """获取学习器信息
        
        Returns:
            Dict[str, Any]: 学习器基本信息
        """
        return {
            "client_id": self.client_id,
            "is_initialized": self._is_initialized,
            "training_count": self.training_count,
            "evaluation_count": self.evaluation_count,
            "last_training_time": self.last_training_time.isoformat() if self.last_training_time else None,
            "last_evaluation_time": self.last_evaluation_time.isoformat() if self.last_evaluation_time else None,
            "data_statistics": self.get_data_statistics(),
            "model_config": self.model_config,
            "training_config": self.training_config,
            "has_local_model": self._local_model is not None
        }
    
    # ==================== 内部工具方法 ====================
    
    async def _record_training(self, training_params: Dict[str, Any], result: TrainingResult):
        """记录训练历史"""
        async with self._lock:
            self.training_count += 1
            self.last_training_time = datetime.now()
            
            history_entry = {
                "timestamp": self.last_training_time.isoformat(),
                "training_params": training_params,
                "result": result,
                "training_index": self.training_count
            }
            
            self._training_history.append(history_entry)
            
            # 限制历史记录长度
            if len(self._training_history) > 100:
                self._training_history = self._training_history[-100:]
    
    async def _record_evaluation(self, evaluation_params: Dict[str, Any], result: EvaluationResult):
        """记录评估历史"""
        async with self._lock:
            self.evaluation_count += 1
            self.last_evaluation_time = datetime.now()
    
    def get_training_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取训练历史
        
        Args:
            limit: 返回的历史记录数量限制
            
        Returns:
            List[Dict[str, Any]]: 最近的训练历史
        """
        return self._training_history[-limit:] if self._training_history else []
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标
        
        Returns:
            Dict[str, Any]: 性能统计指标
        """
        return {
            "total_training_sessions": self.training_count,
            "total_evaluation_sessions": self.evaluation_count,
            "average_training_interval": self._calculate_average_interval(),
            "is_active": self._is_recently_active(),
            "uptime_seconds": self._calculate_uptime()
        }
    
    def _calculate_average_interval(self) -> Optional[float]:
        """计算平均训练间隔"""
        if len(self._training_history) < 2:
            return None
        
        intervals = []
        for i in range(1, len(self._training_history)):
            prev_time = datetime.fromisoformat(self._training_history[i-1]["timestamp"])
            curr_time = datetime.fromisoformat(self._training_history[i]["timestamp"])
            intervals.append((curr_time - prev_time).total_seconds())
        
        return sum(intervals) / len(intervals) if intervals else None
    
    def _is_recently_active(self, threshold_minutes: int = 30) -> bool:
        """检查是否最近活跃"""
        if not self.last_training_time and not self.last_evaluation_time:
            return False
        
        last_activity = max(
            filter(None, [self.last_training_time, self.last_evaluation_time])
        )
        
        threshold = datetime.now().timestamp() - (threshold_minutes * 60)
        return last_activity.timestamp() > threshold
    
    def _calculate_uptime(self) -> float:
        """计算运行时间"""
        if not self._training_history:
            return 0.0
        
        start_time = datetime.fromisoformat(self._training_history[0]["timestamp"])
        return (datetime.now() - start_time).total_seconds()