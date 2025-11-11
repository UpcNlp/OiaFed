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
from ..utils.auto_logger import get_train_logger

class BaseLearner(ABC):
    """客户端学习器抽象基类 - 用户继承实现本地训练逻辑

    使用统一的组件初始化策略：
    1. 接收包含组件类引用和参数的配置字典
    2. 支持延迟加载（默认）或立即初始化
    3. 用户可覆盖默认创建方法
    """

    def __init__(self,
                 client_id: str,
                 config: Optional[Dict[str, Any]] = None,
                 lazy_init: bool = True):
        """
        初始化学习器

        Args:
            client_id: 客户端唯一标识
            config: 组件配置字典，包含类引用和参数
                   由ComponentBuilder.parse_config()生成
            lazy_init: 是否延迟初始化组件（默认True）
        """
        self.client_id = client_id
        self.config = config or {}
        self.lazy_init = lazy_init
        self.logger = get_train_logger(client_id)

        # 提取learner自己的配置参数
        learner_config = self.config.get('learner') or {}
        learner_params = learner_config.get('params', {})

        # 应用learner参数到实例属性
        for key, value in learner_params.items():
            setattr(self, key, value)

        # 向后兼容：保留旧字段
        self.model_config = (self.config.get('local_model') or {}).get('params', {})
        self.training_config = learner_params

        # 内部状态
        self._local_model: Optional[ModelData] = None
        self._training_history: List[Dict[str, Any]] = []
        self._is_initialized = False
        self._lock = asyncio.Lock()

        # 组件占位符（延迟加载）
        self._dataset = None

        # 统计信息
        self.training_count = 0
        self.evaluation_count = 0
        self.last_training_time: Optional[datetime] = None
        self.last_evaluation_time: Optional[datetime] = None

        # 如果不延迟初始化，立即创建所有组件
        if not self.lazy_init:
            self._initialize_all_components()

        self.logger.info(f"BaseLearner {client_id} initialized (lazy_init={self.lazy_init})")
    
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

    # ==================== 组件管理方法 (统一初始化策略) ====================

    def _initialize_all_components(self):
        """立即初始化所有组件"""
        _ = self.dataset
        if 'local_model' in self.config:
            _ = self.local_model
        self.logger.info(f"Learner {self.client_id}: All components initialized")

    @property
    def dataset(self):
        """延迟加载数据集"""
        if self._dataset is None:
            self._dataset = self._create_component('dataset')
            # 向后兼容：设置local_data
            self.local_data = self._dataset
            self.logger.debug(f"Learner {self.client_id}: Dataset created")
        return self._dataset

    @property
    def local_model(self):
        """延迟加载/获取本地模型"""
        # 注意：local_model可能通过set_local_model设置
        # 所以这里只在没有_local_model时才创建
        if self._local_model is None and 'local_model' in self.config:
            self._local_model = self._create_component('local_model')
            self.logger.debug(f"Learner {self.client_id}: Local model created")
        return self._local_model

    def _create_component(self, component_name: str):
        """
        通用组件创建方法（基类实现）

        工作流程：
        1. Builder 已经通过装饰器/注册表找到了类（component_config['class']）
        2. 这里只负责用参数实例化这个类
        3. 支持延迟加载：只在访问 @property 时才调用此方法

        优先级：
        1. 配置中的类 + 参数（Builder 已通过注册表解析）
        2. 子类的默认创建方法
        3. 抛出异常

        Args:
            component_name: 组件名称（如 'dataset', 'local_model'）

        Returns:
            创建的组件实例
        """
        # 特殊处理：dataset 需要支持自动划分
        if component_name == 'dataset':
            return self._create_dataset()

        component_config = self.config.get(component_name)

        # 优先使用配置中的类（Builder 通过注册表找到的）
        if component_config and 'class' in component_config:
            component_class = component_config['class']  # Builder 已从注册表获取
            component_params = component_config.get('params', {})

            # 注入 client_id（如果组件需要）
            import inspect
            if 'client_id' in inspect.signature(component_class.__init__).parameters:
                component_params['client_id'] = self.client_id

            self.logger.debug(
                f"Learner {self.client_id}: Creating {component_name} "
                f"from config: {component_class.__name__} (来自注册表)"
            )

            return component_class(**component_params)

        # 回退到默认创建方法（用户自定义）
        default_method = getattr(self, f'_create_default_{component_name}', None)
        if default_method and callable(default_method):
            self.logger.debug(
                f"Learner {self.client_id}: Creating {component_name} "
                f"using default method"
            )
            return default_method()

        raise ValueError(
            f"组件 '{component_name}' 未在配置中指定，"
            f"且子类未提供 _create_default_{component_name}() 方法"
        )

    def _create_dataset(self):
        """
        创建数据集（支持自动划分）

        优先级:
        1. 配置中的数据集 + partition配置 → 自动划分
        2. 配置中的数据集（无partition） → 完整数据集
        3. 子类的 _create_default_dataset() → 用户自定义

        Returns:
            Dataset: 数据集实例（可能是完整数据集或划分后的子集）
        """
        dataset_config = self.config.get('dataset')

        # 策略1: 使用配置中的数据集
        if dataset_config:
            # 如果有 'class' 字段，直接使用（已解析的类）
            if 'class' in dataset_config:
                dataset_class = dataset_config['class']
                params = dataset_config.get('params', {})
            # 如果有 'name' 字段，从注册表获取类（YAML配置格式）
            elif 'name' in dataset_config:
                dataset_name = dataset_config['name']
                # 从注册表获取数据集类
                from ..api.registry import registry
                dataset_class = registry.get_dataset(dataset_name)
                if not dataset_class:
                    raise ValueError(f"Dataset '{dataset_name}' not found in registry")
                params = dataset_config.get('params', {})
            else:
                # dataset_config存在但没有class或name字段
                dataset_class = None
                params = {}

            if dataset_class:
                # 注入 client_id（如果数据集需要）
                import inspect
                if 'client_id' in inspect.signature(dataset_class.__init__).parameters:
                    params['client_id'] = self.client_id

                self.logger.debug(
                    f"Learner {self.client_id}: Creating dataset "
                    f"from config: {dataset_class.__name__}"
                )

                # 实例化完整数据集
                full_dataset = dataset_class(**params)

                # 检查是否有partition配置
                # partition配置在原始配置中，不在ComponentBuilder解析后的配置中
                partition_config = dataset_config.get('partition')

                if partition_config:
                    # 执行自动划分
                    return self._partition_dataset(full_dataset, partition_config)
                else:
                    # 返回完整数据集（向后兼容）
                    self.logger.debug(
                        f"Learner {self.client_id}: No partition config, "
                        f"using full dataset ({len(full_dataset)} samples)"
                    )
                    return full_dataset

        # 策略2: 调用子类自定义方法
        default_method = getattr(self, '_create_default_dataset', None)
        if default_method and callable(default_method):
            self.logger.debug(
                f"Learner {self.client_id}: Creating dataset "
                f"using default method"
            )
            return default_method()

        # 策略3: 无法创建，抛出异常
        raise ValueError(
            "数据集未在配置中指定，"
            "且子类未提供 _create_default_dataset() 方法"
        )

    def _partition_dataset(self, full_dataset, partition_config: Dict[str, Any]):
        """
        对完整数据集进行划分，返回当前客户端的数据子集

        支持三种联邦学习模式（Memory/Process/Network）使用统一的划分策略。
        通过确定性算法保证相同配置下不同客户端得到相同的划分结果。

        Args:
            full_dataset: FederatedDataset实例或普通Dataset实例
            partition_config: 划分配置字典
                {
                    'strategy': 'dirichlet',    # 划分策略
                    'num_clients': 10,          # 总客户端数
                    'seed': 42,                 # 随机种子
                    'params': {'alpha': 0.5}    # 策略特定参数
                }

        Returns:
            Dataset: 当前客户端的数据子集

        Raises:
            ValueError: 如果配置无效或数据集不支持划分
        """
        strategy = partition_config.get('strategy', 'iid')
        num_clients = partition_config.get('num_clients')
        seed = partition_config.get('seed', 42)
        strategy_params = partition_config.get('params', {})

        if num_clients is None:
            raise ValueError("partition配置中必须指定 num_clients")

        # 从client_id中提取客户端索引
        client_idx = self._extract_client_index(self.client_id, num_clients)

        self.logger.info(
            f"Client {self.client_id}: 开始数据集自动划分 "
            f"(strategy={strategy}, client_idx={client_idx}/{num_clients}, seed={seed})"
        )

        # 检查数据集是否支持get_client_partition方法（FederatedDataset）
        if hasattr(full_dataset, 'get_client_partition'):
            # 使用FederatedDataset的get_client_partition方法
            client_dataset = full_dataset.get_client_partition(
                client_id=client_idx,
                num_clients=num_clients,
                strategy=strategy,
                seed=seed,
                **strategy_params
            )
            self.logger.info(
                f"Client {self.client_id}: 数据集划分完成 "
                f"(samples={len(client_dataset)}, strategy={strategy})"
            )
            return client_dataset

        else:
            # 如果不是FederatedDataset，尝试手动创建partitioner
            self.logger.warning(
                f"Client {self.client_id}: 数据集不是FederatedDataset类型，"
                f"尝试使用通用划分器"
            )

            try:
                from ..methods.datasets.partition import create_partitioner
                from torch.utils.data import Subset

                # 创建划分器
                partitioner = create_partitioner(strategy, seed=seed)

                # 执行划分获取所有客户端的索引
                all_indices = partitioner.partition(
                    full_dataset, num_clients, **strategy_params
                )

                # 获取当前客户端的索引
                if client_idx not in all_indices:
                    raise ValueError(
                        f"client_idx={client_idx} 不在划分结果中 "
                        f"(有效范围: 0-{num_clients-1})"
                    )

                client_indices = all_indices[client_idx]
                client_dataset = Subset(full_dataset, client_indices)

                self.logger.info(
                    f"Client {self.client_id}: 数据集划分完成（通用方式） "
                    f"(samples={len(client_dataset)}, strategy={strategy})"
                )
                return client_dataset

            except ImportError as e:
                raise ValueError(
                    f"数据集不支持自动划分，且无法导入通用划分器: {e}"
                )

    def _extract_client_index(self, client_id: str, num_clients: int) -> int:
        """
        从client_id字符串中提取客户端索引

        支持多种client_id格式：
        - "client_0", "client_1" → 0, 1
        - "memory_client_0" → 0
        - "process_client_123_8001" → 123
        - 配置中直接指定 client_index

        Args:
            client_id: 客户端ID字符串
            num_clients: 总客户端数（用于验证范围）

        Returns:
            int: 客户端索引 (0-based)

        Raises:
            ValueError: 如果无法从client_id解析索引
        """
        # 方法1: 从配置中获取（最可靠）
        if 'client_index' in self.config:
            idx = self.config['client_index']
            if not isinstance(idx, int):
                raise ValueError(f"配置中的client_index必须是整数，got {type(idx)}")
            if 0 <= idx < num_clients:
                return idx
            else:
                raise ValueError(
                    f"配置中的client_index={idx}超出范围[0, {num_clients})"
                )

        # 方法2: 从client_id解析（通用模式）
        import re

        # 尝试匹配 "client_数字" 或 "xxx_client_数字" 模式
        match = re.search(r'client[_-](\d+)', client_id, re.IGNORECASE)
        if match:
            idx = int(match.group(1))
            if 0 <= idx < num_clients:
                self.logger.debug(
                    f"Client {self.client_id}: 从client_id解析得到client_index={idx}"
                )
                return idx
            else:
                raise ValueError(
                    f"从client_id='{client_id}'解析得到的索引{idx}超出范围[0, {num_clients})"
                )

        # 方法3: 无法解析，抛出异常
        raise ValueError(
            f"无法从client_id='{client_id}'中提取客户端索引。"
            f"请在配置中指定'client_index'或使用标准命名格式（如'client_0', 'client_1'）"
        )

    # 子类可以覆盖这些方法提供默认实现
    def _create_default_dataset(self):
        """子类可覆盖：提供默认数据集"""
        raise NotImplementedError(
            "必须在配置中指定 dataset 或覆盖 _create_default_dataset()"
        )

    def _create_default_local_model(self):
        """子类可覆盖：提供默认本地模型"""
        # local_model是可选的，可以通过set_local_model设置
        return None

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
        try:
            # 尝试获取dataset（会触发延迟加载）
            local_data = self.dataset
            if local_data is None:
                return {"total_samples": 0, "message": "No local data available"}
        except Exception:
            # 如果没有配置dataset，尝试使用local_data（向后兼容）
            local_data = getattr(self, 'local_data', None)
            if local_data is None:
                return {"total_samples": 0, "message": "No local data available"}

        # 默认实现，用户可以重写
        try:
            if hasattr(local_data, '__len__'):
                return {
                    "total_samples": len(local_data),
                    "data_type": type(local_data).__name__,
                    "available": True
                }
            else:
                return {
                    "total_samples": "unknown",
                    "data_type": type(local_data).__name__,
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
            self.logger.exception(f"Save model failed: {e}")
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
            self.logger.exception(f"Load model failed: {e}")
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
            self.logger.exception(f"Learner initialization failed: {e}")
            return False
    
    async def _perform_initialization(self):
        """执行具体的初始化逻辑 - 子类可重写"""
        # 默认初始化逻辑
        # 兼容两种方式：新的统一初始化（dataset）和旧的方式（local_data）
        try:
            # 尝试访问 dataset（会触发延迟加载）
            _ = self.dataset
        except Exception:
            # 如果没有 dataset，尝试检查 local_data（向后兼容）
            if not hasattr(self, 'local_data') or self.local_data is None:
                self.logger.warning(f"Warning: No local data provided for client {self.client_id}")

        # 验证配置
        await self._validate_configuration()

        self.logger.info(f"BaseLearner {self.client_id} initialized successfully")
    
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
        
        self.logger.debug(f"BaseLearner {self.client_id} cleaned up")
    
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