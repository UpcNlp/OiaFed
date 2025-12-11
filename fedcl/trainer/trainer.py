"""
MOE-FedCL 服务端训练器抽象基类
moe_fedcl/trainer/base_trainer.py
"""

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
from logging import Logger
from ..communication.layer_event import ProxyManagerEventHandler
from ..exceptions import FederationError
from ..learner.proxy import LearnerProxy
from ..types import ModelData, EvaluationResult, RoundResult
from ..utils.auto_logger import get_train_logger, get_sys_logger

class FederationResult:
    """联邦学习训练结果"""

    def __init__(self):
        self.success = False
        self.total_rounds = 0
        self.completed_rounds = 0
        self.final_accuracy = 0.0
        self.final_loss = float('inf')
        self.total_time = 0.0
        self.convergence_round = None
        self.best_model: Optional[ModelData] = None
        self.training_history: List[RoundResult] = []
        self.error_message: Optional[str] = None
        self.termination_reason = "unknown"


class TrainingConfig:
    """训练配置"""
    
    def __init__(self,
                 max_rounds: int = 100,
                 min_clients: int = 2,
                 client_selection_ratio: float = 1.0,
                 round_timeout: float = 300.0,
                 client_timeout: float = 120.0,
                 convergence_threshold: float = 0.001,
                 patience: int = 10,
                 save_checkpoints: bool = True,
                 checkpoint_interval: int = 10):
        self.max_rounds = max_rounds
        self.min_clients = min_clients
        self.client_selection_ratio = client_selection_ratio
        self.round_timeout = round_timeout
        self.client_timeout = client_timeout
        self.convergence_threshold = convergence_threshold
        self.patience = patience
        self.save_checkpoints = save_checkpoints
        self.checkpoint_interval = checkpoint_interval


class TrainingStatus:
    """训练状态"""
    
    def __init__(self):
        self.current_round = 0
        self.total_rounds = 0
        self.is_training = False
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.selected_clients: List[str] = []
        self.active_clients: List[str] = []
        self.failed_clients: List[str] = []
        self.round_results: List[RoundResult] = []
        self.best_accuracy = 0.0
        self.patience_counter = 0
        self.convergence_history: List[float] = []


class RoundStatistics:
    """轮次统计"""
    
    def __init__(self):
        self.round_number = 0
        self.participants_count = 0
        self.successful_participants = 0
        self.failed_participants = 0
        self.average_training_time = 0.0
        self.average_loss = 0.0
        self.average_accuracy = 0.0
        self.model_size_bytes = 0
        self.communication_time = 0.0
        self.aggregation_time = 0.0
        self.total_round_time = 0.0
        self.convergence_metric = 0.0


class ClientStatistics:
    """客户端统计"""
    
    def __init__(self, client_id: str):
        self.client_id = client_id
        self.participation_count = 0
        self.successful_rounds = 0
        self.failed_rounds = 0
        self.average_training_time = 0.0
        self.total_samples_trained = 0
        self.last_participation_time: Optional[datetime] = None
        self.connection_stability = 1.0  # 0.0-1.0
        self.performance_score = 0.0


class ProxyManager:
    """代理管理器 - 负责LearnerProxy的生命周期管理"""
    
    def __init__(self, trainer: 'BaseTrainer', server_id: Optional[str] = None):
        """初始化代理管理器

        Args:
            trainer: BaseTrainer 实例
            server_id: 服务器节点ID（用于日志归属）
        """
        self.trainer = trainer
        self.proxies: Dict[str, LearnerProxy] = {}
        self._lock = asyncio.Lock()

        # 导入日志记录器
        from ..utils.auto_logger import get_logger
        # ProxyManager 是 Server 端组件，使用 server_id 的运行日志
        if server_id:
            self.logger = get_logger("runtime", server_id)
        else:
            # 向后兼容：如果没有 server_id，使用通用的 server
            self.logger = get_logger("runtime", "server")
    
    async def on_proxy_ready(self, client_id: str, proxy: LearnerProxy):
        """接收业务通信层创建的代理"""
        self.logger.info(f" [代理管理器] 收到代理就绪通知: {client_id}")
        
        async with self._lock:
            self.proxies[client_id] = proxy
            
            # 更新trainer的客户端统计
            self.trainer.client_statistics[client_id] = ClientStatistics(client_id)
            
            self.logger.debug(f"[代理管理器] 学习器代理已注册: {client_id}, 当前总数: {len(self.proxies)}")
            self.logger.info(f"[代理管理器] 可用客户端列表: {list(self.proxies.keys())}")
    
    async def on_proxy_disconnected(self, client_id: str):
        """处理代理断开"""
        self.logger.warning(f"[代理管理器] 收到代理断开通知: {client_id}")
        
        async with self._lock:
            if client_id in self.proxies:
                del self.proxies[client_id]
                
                # 清理trainer的客户端统计
                if client_id in self.trainer.client_statistics:
                    del self.trainer.client_statistics[client_id]
                
                self.logger.debug(f"🗑️ [代理管理器] 学习器代理已移除: {client_id}, 剩余数量: {len(self.proxies)}")
    
    def get_proxy(self, client_id: str) -> Optional[LearnerProxy]:
        """获取指定客户端的代理"""
        return self.proxies.get(client_id)
    
    def get_all_proxies(self) -> Dict[str, LearnerProxy]:
        """获取所有代理"""
        return self.proxies.copy()
    
    def get_available_clients(self) -> List[str]:
        """获取可用客户端列表"""
        available_clients = []
        self.logger.debug(f"[代理管理器] 检查可用客户端，总代理数: {len(self.proxies)}")
        
        for client_id, proxy in self.proxies.items():
            if proxy.is_client_ready():
                available_clients.append(client_id)
                self.logger.debug(f"[代理管理器] 客户端[{client_id}]可用")
            else:
                self.logger.warning(f"[代理管理器] 客户端[{client_id}]不可用")
        
        self.logger.info(f"[代理管理器] 可用客户端总数: {len(available_clients)}/{len(self.proxies)}")
        return available_clients


class BaseTrainer(ABC):
    """服务端训练器抽象基类 - 用户继承实现联邦学习算法

    使用统一的组件初始化策略：
    1. 接收包含组件类引用和参数的配置字典
    2. 支持延迟加载（默认）或立即初始化
    3. 用户可覆盖默认创建方法
    """

    def __init__(self,
                 config: Optional[Dict[str, Any]] = None,
                 lazy_init: bool = True,
                 logger: Optional[Logger] = None,
                 server_id: Optional[str] = None):
        """
        初始化训练器

        Args:
            config: 组件配置字典，包含类引用和参数
                   由ComponentBuilder.parse_config()生成
            lazy_init: 是否延迟初始化组件（默认True）
            logger: 日志记录器
            server_id: 服务器节点ID（用于日志归属和组件初始化）
        """
        self.config = config or {}
        self.lazy_init = lazy_init
        self.server_id = server_id  # 保存 server_id

        # 如果提供了 server_id，使用对应的训练日志；否则使用默认的 server
        if server_id:
            self.logger = logger if logger else get_train_logger(server_id)
        else:
            self.logger = logger if logger else get_train_logger("server")

        # 提取trainer自己的配置参数
        trainer_config = self.config.get('trainer', {})
        trainer_params = trainer_config.get('params', {})

        # 应用trainer参数到实例属性
        for key, value in trainer_params.items():
            setattr(self, key, value)

        # 如果没有从参数设置，使用默认值
        if not hasattr(self, 'max_rounds'):
            self.max_rounds = 100
        if not hasattr(self, 'min_clients'):
            self.min_clients = 2

        # 创建TrainingConfig（向后兼容）
        self.training_config = TrainingConfig(
            max_rounds=getattr(self, 'max_rounds', 100),
            min_clients=getattr(self, 'min_clients', 2),
            client_selection_ratio=getattr(self, 'client_selection_ratio', 1.0)
        )

        #  自动实例化代理管理器（用户无感知），传递 server_id
        self._proxy_manager = ProxyManager(self, server_id=server_id)

        # 创建事件处理器，用于接收业务层的代理创建事件，传递 server_id
        self._proxy_event_handler = ProxyManagerEventHandler(self._proxy_manager, server_id=server_id)

        # learner_proxies变成代理管理器的代理属性
        self.learner_proxies = self._proxy_manager.proxies

        # 组件占位符（延迟加载）
        self._aggregator = None
        self._global_model = None
        self._evaluator = None

        # 如果不延迟初始化，立即创建所有组件
        if not self.lazy_init:
            self._initialize_all_components()

        self.logger.info(f"BaseTrainer initialized (lazy_init={self.lazy_init})")

        # 训练状态
        self.training_status = TrainingStatus()
        self.training_status.total_rounds = self.training_config.max_rounds
        
        # 统计信息
        self.round_statistics: Dict[int, RoundStatistics] = {}
        self.client_statistics: Dict[str, ClientStatistics] = {}
        
        # 回调函数
        self.round_callbacks: List[Callable] = []
        self.training_callbacks: List[Callable] = []

        # 回调机制（用于实验记录等扩展功能）
        self._callbacks: Dict[str, List[Callable]] = {
            'before_round': [],
            'after_round': [],
            'after_evaluation': []
        }

        # 内部状态
        self._lock = asyncio.Lock()
        self._best_model: Optional[ModelData] = None
        self._checkpoint_models: Dict[int, ModelData] = {}
    
    # ==================== 核心训练方法 (用户必须实现) ====================
    
    @abstractmethod
    async def train_round(self, round_num: int, client_ids: List[str]) -> RoundResult:
        """执行一轮联邦训练
        
        Args:
            round_num: 当前轮次编号
            client_ids: 参与训练的客户端ID列表
            
        Returns:
            RoundResult: 轮次训练结果，应包含：
                - participants: 参与客户端列表
                - successful_clients: 成功的客户端列表
                - failed_clients: 失败的客户端列表
                - aggregated_model: 聚合后的模型
                - round_metrics: 轮次指标（损失、准确率等）
                - training_time: 训练总时间
                
        Raises:
            TrainingError: 训练过程中的错误
        
        使用示例:
            # 并发调用多个客户端训练
            tasks = []
            for client_id in client_ids:
                task = self.learner_proxies[client_id].train({
                    "global_model": self.global_model,
                    "epochs": 5,
                    "learning_rate": 0.01
                })
                tasks.append((client_id, task))
            
            # 收集结果
            client_results = {}
            for client_id, task in tasks:
                try:
                    result = await task
                    client_results[client_id] = result
                except Exception as e:
                    print(f"Client {client_id} failed: {e}")
            
            # 聚合模型
            aggregated_model = await self.aggregate_models(client_results)
            
            return {
                "participants": client_ids,
                "successful_clients": list(client_results.keys()),
                "aggregated_model": aggregated_model,
                "round_metrics": {"avg_loss": 0.1, "avg_accuracy": 0.9}
            }
        """
        pass
    
    @abstractmethod
    async def aggregate_models(self, client_results: Dict[str, Any]) -> ModelData:
        """聚合客户端模型
        
        Args:
            client_results: 客户端训练结果 {client_id: training_result}
            
        Returns:
            ModelData: 聚合后的全局模型
            
        Raises:
            TrainingError: 聚合过程中的错误
        
        使用示例:
            # FedAvg算法示例
            model_updates = []
            total_samples = 0
            
            for client_id, result in client_results.items():
                model_update = result.get("model_update", {})
                samples_count = result.get("samples_count", 1)
                
                model_updates.append((model_update, samples_count))
                total_samples += samples_count
            
            # 加权平均
            aggregated_model = {}
            for layer_name in model_updates[0][0].keys():
                weighted_sum = 0
                for model_update, samples_count in model_updates:
                    weight = samples_count / total_samples
                    weighted_sum += model_update[layer_name] * weight
                aggregated_model[layer_name] = weighted_sum
            
            return aggregated_model
        """
        pass
    
    @abstractmethod
    async def evaluate_global_model(self) -> EvaluationResult:
        """评估全局模型
        
        Returns:
            EvaluationResult: 评估结果，应包含：
                - accuracy: 准确率
                - loss: 损失值
                - metrics: 其他评估指标
                - samples_count: 评估样本数
                
        Raises:
            TrainingError: 评估过程中的错误
        
        使用示例:
            # 选择部分客户端进行评估
            eval_clients = self.select_evaluation_clients()
            
            eval_results = []
            for client_id in eval_clients:
                try:
                    result = await self.learner_proxies[client_id].evaluate({
                        "model": self.global_model
                    })
                    eval_results.append(result)
                except Exception as e:
                    print(f"Evaluation on client {client_id} failed: {e}")
            
            # 计算全局评估指标
            if eval_results:
                avg_accuracy = sum(r.get("accuracy", 0) for r in eval_results) / len(eval_results)
                avg_loss = sum(r.get("loss", 0) for r in eval_results) / len(eval_results)
                
                return {
                    "accuracy": avg_accuracy,
                    "loss": avg_loss,
                    "participants": len(eval_results)
                }
            
            return {"accuracy": 0.0, "loss": float('inf')}
        """
        pass
    
    @abstractmethod
    def should_stop_training(self, round_num: int, round_result: RoundResult) -> bool:
        """判断是否应该停止训练
        
        Args:
            round_num: 当前轮次
            round_result: 轮次结果
            
        Returns:
            bool: 是否应该停止训练
        
        使用示例:
            # 检查收敛条件
            if round_num >= self.training_config.max_rounds:
                return True
            
            # 检查准确率收敛
            current_accuracy = round_result.get("round_metrics", {}).get("avg_accuracy", 0)
            if abs(current_accuracy - self.training_status.best_accuracy) < self.training_config.convergence_threshold:
                self.training_status.patience_counter += 1
                if self.training_status.patience_counter >= self.training_config.patience:
                    return True
            else:
                self.training_status.patience_counter = 0
                if current_accuracy > self.training_status.best_accuracy:
                    self.training_status.best_accuracy = current_accuracy
            
            return False
        """
        pass

    # ==================== 组件管理方法 (统一初始化策略) ====================

    def _initialize_all_components(self):
        """立即初始化所有组件"""
        # 触发所有property，强制创建实例
        _ = self.aggregator
        _ = self.global_model
        if 'evaluator' in self.config:
            _ = self.evaluator

        self.logger.info("All trainer components initialized")

    @property
    def aggregator(self):
        """延迟加载聚合器"""
        if self._aggregator is None:
            self._aggregator = self._create_component('aggregator')
            self.logger.debug("Aggregator created")
        return self._aggregator

    @property
    def global_model(self):
        """延迟加载全局模型"""
        if self._global_model is None:
            self._global_model = self._create_component('global_model')
            self.logger.debug("Global model created")
        return self._global_model

    @property
    def evaluator(self):
        """延迟加载评估器（可选）"""
        if self._evaluator is None and 'evaluator' in self.config:
            self._evaluator = self._create_component('evaluator')
            self.logger.debug("Evaluator created")
        return self._evaluator

    def _create_component(self, component_name: str):
        """
        通用组件创建方法（基类实现）

        优先级：
        1. 配置中的类 + 参数
        2. 子类的默认创建方法
        3. 抛出异常

        Args:
            component_name: 组件名称

        Returns:
            创建的组件实例
        """
        component_config = self.config.get(component_name)

        # 优先使用配置
        if component_config and 'class' in component_config:
            component_class = component_config['class']
            component_params = component_config.get('params', {})

            self.logger.debug(
                f"Creating {component_name} from config: "
                f"{component_class.__name__}({component_params})"
            )

            return component_class(**component_params)

        # 回退到默认创建方法
        default_method = getattr(self, f'_create_default_{component_name}', None)
        if default_method and callable(default_method):
            self.logger.debug(f"Creating {component_name} using default method")
            return default_method()

        # 都没有则抛出异常
        raise ValueError(
            f"组件 '{component_name}' 未在配置中指定，"
            f"且子类未提供 _create_default_{component_name}() 方法"
        )

    # 子类可以覆盖这些方法提供默认实现
    def _create_default_aggregator(self):
        """子类可覆盖：提供默认聚合器"""
        raise NotImplementedError(
            "必须在配置中指定 aggregator 或覆盖 _create_default_aggregator()"
        )

    def _create_default_global_model(self):
        """子类可覆盖：提供默认全局模型"""
        raise NotImplementedError(
            "必须在配置中指定 global_model 或覆盖 _create_default_global_model()"
        )

    def _create_default_evaluator(self):
        """子类可覆盖：提供默认评估器（可选）"""
        return None  # 评估器是可选的，默认返回None

    # ==================== 状态管理方法 (框架提供) ====================
    
    def get_training_status(self) -> Dict[str, Any]:
        """获取训练状态
        
        Returns:
            Dict[str, Any]: 训练状态信息
        """
        return {
            "current_round": self.training_status.current_round,
            "total_rounds": self.training_status.total_rounds,
            "is_training": self.training_status.is_training,
            "start_time": self.training_status.start_time.isoformat() if self.training_status.start_time else None,
            "end_time": self.training_status.end_time.isoformat() if self.training_status.end_time else None,
            "selected_clients": self.training_status.selected_clients,
            "active_clients": self.training_status.active_clients,
            "failed_clients": self.training_status.failed_clients,
            "best_accuracy": self.training_status.best_accuracy,
            "patience_counter": self.training_status.patience_counter,
            "progress": self.training_status.current_round / self.training_status.total_rounds if self.training_status.total_rounds > 0 else 0
        }
    
    def get_round_statistics(self, round_num: int = None) -> Optional[Dict[str, Any]]:
        """获取轮次统计
        
        Args:
            round_num: 轮次编号，None表示获取最新轮次
            
        Returns:
            Optional[Dict[str, Any]]: 轮次统计信息
        """
        if round_num is None:
            round_num = self.training_status.current_round
        
        if round_num in self.round_statistics:
            stats = self.round_statistics[round_num]
            return {
                "round_number": stats.round_number,
                "participants_count": stats.participants_count,
                "successful_participants": stats.successful_participants,
                "failed_participants": stats.failed_participants,
                "success_rate": stats.successful_participants / max(stats.participants_count, 1),
                "average_training_time": stats.average_training_time,
                "average_loss": stats.average_loss,
                "average_accuracy": stats.average_accuracy,
                "model_size_bytes": stats.model_size_bytes,
                "communication_time": stats.communication_time,
                "aggregation_time": stats.aggregation_time,
                "total_round_time": stats.total_round_time,
                "convergence_metric": stats.convergence_metric
            }
        
        return None
    
    def get_client_statistics(self, client_id: str = None) -> Dict[str, Any]:
        """获取客户端统计
        
        Args:
            client_id: 客户端ID，None表示获取所有客户端
            
        Returns:
            Dict[str, Any]: 客户端统计信息
        """
        if client_id:
            if client_id in self.client_statistics:
                stats = self.client_statistics[client_id]
                return {
                    "client_id": stats.client_id,
                    "participation_count": stats.participation_count,
                    "successful_rounds": stats.successful_rounds,
                    "failed_rounds": stats.failed_rounds,
                    "success_rate": stats.successful_rounds / max(stats.participation_count, 1),
                    "average_training_time": stats.average_training_time,
                    "total_samples_trained": stats.total_samples_trained,
                    "last_participation_time": stats.last_participation_time.isoformat() if stats.last_participation_time else None,
                    "connection_stability": stats.connection_stability,
                    "performance_score": stats.performance_score
                }
            return {}
        else:
            # 返回所有客户端统计
            all_stats = {}
            for cid, stats in self.client_statistics.items():
                all_stats[cid] = {
                    "participation_count": stats.participation_count,
                    "success_rate": stats.successful_rounds / max(stats.participation_count, 1),
                    "average_training_time": stats.average_training_time,
                    "connection_stability": stats.connection_stability,
                    "performance_score": stats.performance_score
                }
            return all_stats
    
    async def save_checkpoint(self, checkpoint_path: str) -> bool:
        """保存检查点
        
        Args:
            checkpoint_path: 检查点保存路径
            
        Returns:
            bool: 保存是否成功
        """
        try:
            checkpoint_data = {
                "round_number": self.training_status.current_round,
                "global_model": self.global_model,
                "training_status": self.get_training_status(),
                "round_statistics": {k: vars(v) for k, v in self.round_statistics.items()},
                "client_statistics": {k: vars(v) for k, v in self.client_statistics.items()},
                "best_model": self._best_model,
                "timestamp": datetime.now().isoformat()
            }
            
            import pickle
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)

            self.logger.debug(f"Checkpoint saved: {checkpoint_path}")
            return True
            
        except Exception as e:
            self.logger.exception(f"Failed to save checkpoint: {e}")
            return False
    
    async def load_checkpoint(self, checkpoint_path: str) -> bool:
        """加载检查点
        
        Args:
            checkpoint_path: 检查点文件路径
            
        Returns:
            bool: 加载是否成功
        """
        try:
            import pickle
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            # 恢复状态
            self.training_status.current_round = checkpoint_data["round_number"]
            self.global_model = checkpoint_data["global_model"]
            self._best_model = checkpoint_data.get("best_model")
            
            self.logger.debug(f"Checkpoint loaded: {checkpoint_path}")
            return True
            
        except Exception as e:
            self.logger.exception(f"Failed to load checkpoint: {e}")
            return False
    
    # ==================== 客户端管理方法 ====================
    
    def select_clients_for_round(self, round_num: int) -> List[str]:
        """选择参与该轮训练的客户端
        
        Args:
            round_num: 轮次编号
            
        Returns:
            List[str]: 选中的客户端ID列表
        """
        available_clients = []
        
        # 检查客户端可用性
        available_clients = self.get_available_clients()
        
        # 检查最小客户端数量
        if len(available_clients) < self.training_config.min_clients:
            raise FederationError(f"Insufficient clients available: {len(available_clients)} < {self.training_config.min_clients}")
        
        # 按比例选择客户端
        selection_count = max(
            self.training_config.min_clients,
            int(len(available_clients) * self.training_config.client_selection_ratio)
        )
        
        # 可以在这里实现不同的选择策略：随机选择、基于性能选择、基于数据分布选择等
        import random
        selected_clients = random.sample(available_clients, min(selection_count, len(available_clients)))
        
        return selected_clients
    
    async def check_client_readiness(self, client_ids: List[str]) -> Dict[str, bool]:
        """检查客户端就绪状态（真正的并发版本）

        Args:
            client_ids: 要检查的客户端ID列表

        Returns:
            Dict[str, bool]: 客户端就绪状态映射
        """
        readiness = {}

        # 创建所有ping任务（带超时）
        ping_tasks = {}
        for client_id in client_ids:
            if client_id in self.learner_proxies:
                # 在创建时包装超时
                ping_tasks[client_id] = asyncio.wait_for(
                    self.learner_proxies[client_id].ping(),
                    timeout=5.0
                )

        # 真正的并发等待所有任务
        results = await asyncio.gather(
            *ping_tasks.values(),
            return_exceptions=True  # 捕获异常而不是中断其他任务
        )

        # 处理结果
        for client_id, result in zip(ping_tasks.keys(), results):
            if isinstance(result, Exception):
                self.logger.exception(f"Ping failed for {client_id}: {type(result).__name__}: {result}")
                readiness[client_id] = False
            else:
                readiness[client_id] = True

        return readiness
    
    def get_available_clients(self) -> List[str]:
        """获取可用客户端列表
        
        Returns:
            List[str]: 可用的客户端ID列表
        """
        return self._proxy_manager.get_available_clients()
    
    def is_client_ready(self, client_id: str) -> bool:
        """检查客户端是否就绪
        
        Args:
            client_id: 客户端ID
            
        Returns:
            bool: 客户端是否就绪
        """
        proxy = self._proxy_manager.get_proxy(client_id)
        return proxy is not None and proxy.is_client_ready()
    
    # ==================== 统计更新方法 ====================
    
    async def _update_round_statistics(self, round_num: int, round_result: RoundResult, start_time: datetime):
        """更新轮次统计"""
        async with self._lock:
            stats = RoundStatistics()
            stats.round_number = round_num
            stats.total_round_time = (datetime.now() - start_time).total_seconds()
            
            # 从轮次结果中提取统计信息
            participants = round_result.get("participants", [])
            successful = round_result.get("successful_clients", [])
            failed = round_result.get("failed_clients", [])
            
            stats.participants_count = len(participants)
            stats.successful_participants = len(successful)
            stats.failed_participants = len(failed)
            
            # 计算平均指标
            round_metrics = round_result.get("round_metrics", {})
            stats.average_loss = round_metrics.get("avg_loss", 0.0)
            stats.average_accuracy = round_metrics.get("avg_accuracy", 0.0)
            stats.convergence_metric = round_metrics.get("convergence", 0.0)
            
            self.round_statistics[round_num] = stats
    
    async def _update_client_statistics(self, client_id: str, success: bool, training_time: float, samples_count: int = 0):
        """更新客户端统计"""
        async with self._lock:
            if client_id not in self.client_statistics:
                self.client_statistics[client_id] = ClientStatistics(client_id)
            
            stats = self.client_statistics[client_id]
            stats.participation_count += 1
            stats.last_participation_time = datetime.now()
            
            if success:
                stats.successful_rounds += 1
                # 更新平均训练时间
                if stats.successful_rounds == 1:
                    stats.average_training_time = training_time
                else:
                    stats.average_training_time = (
                        (stats.average_training_time * (stats.successful_rounds - 1) + training_time) / 
                        stats.successful_rounds
                    )
                stats.total_samples_trained += samples_count
            else:
                stats.failed_rounds += 1
            
            # 更新连接稳定性评分
            stats.connection_stability = stats.successful_rounds / stats.participation_count
            
            # 更新性能评分（可以根据具体需求调整计算方式）
            stats.performance_score = (
                stats.connection_stability * 0.4 +
                (1.0 / max(stats.average_training_time, 0.1)) * 0.3 +  # 训练速度
                (stats.total_samples_trained / 10000.0) * 0.3  # 数据贡献
            )
    
    # ==================== 生命周期方法 (框架提供) ====================
    
    async def initialize(self) -> bool:
        """初始化训练器
        
        Returns:
            bool: 初始化是否成功
        """
        try:
            # 检查客户端连接 (允许为0，在训练时再检查)
            self.logger.debug("Checking client connections...")
            available_clients = self.get_available_clients()
            self.logger.debug(f"Found {len(available_clients)} available clients: {available_clients}")
            
            # 初始化全局模型
            if self.global_model is None:
                raise FederationError("Global model not provided")
            
            # 执行用户自定义初始化
            await self._perform_custom_initialization()
            
            self.logger.debug("BaseTrainer initialized successfully")
            return True
            
        except Exception as e:
            self.logger.exception(f"Trainer initialization failed: {e}")
            return False
    
    async def _perform_custom_initialization(self):
        """执行自定义初始化 - 子类可重写"""
        pass
    
    async def cleanup(self) -> None:
        """清理训练器资源"""
        async with self._lock:
            # 重置训练状态
            self.training_status = TrainingStatus()
            
            # 清理统计信息
            self.round_statistics.clear()
            for stats in self.client_statistics.values():
                stats.__init__(stats.client_id)  # 重置统计
            
            # 清理回调
            self.round_callbacks.clear()
            self.training_callbacks.clear()
        
        self.logger.debug("BaseTrainer cleaned up")
    
    async def handle_client_failure(self, client_id: str) -> None:
        """处理客户端故障
        
        Args:
            client_id: 故障的客户端ID
        """
        self.logger.debug(f"Handling client failure: {client_id}")
        
        # 更新失败统计
        await self._update_client_statistics(client_id, False, 0.0)
        
        # 从活跃客户端列表中移除
        if client_id in self.training_status.active_clients:
            self.training_status.active_clients.remove(client_id)
        
        # 添加到失败客户端列表
        if client_id not in self.training_status.failed_clients:
            self.training_status.failed_clients.append(client_id)
        
        # 可以在这里添加客户端恢复逻辑
        # 例如：尝试重新连接、从备用客户端列表中选择等
    
    # ==================== 回调管理 ====================
    
    def register_round_callback(self, callback: Callable) -> str:
        """注册轮次回调
        
        Args:
            callback: 回调函数，签名为 callback(round_num: int, round_result: RoundResult)
            
        Returns:
            str: 回调ID
        """
        callback_id = f"round_callback_{len(self.round_callbacks)}"
        self.round_callbacks.append((callback_id, callback))
        return callback_id
    
    def register_training_callback(self, callback: Callable) -> str:
        """注册训练回调
        
        Args:
            callback: 回调函数，签名为 callback(event: str, data: Any)
            
        Returns:
            str: 回调ID
        """
        callback_id = f"training_callback_{len(self.training_callbacks)}"
        self.training_callbacks.append((callback_id, callback))
        return callback_id
    
    def unregister_callback(self, callback_id: str) -> bool:
        """取消注册回调
        
        Args:
            callback_id: 回调ID
            
        Returns:
            bool: 是否成功取消
        """
        # 检查轮次回调
        for i, (cid, callback) in enumerate(self.round_callbacks):
            if cid == callback_id:
                del self.round_callbacks[i]
                return True
        
        # 检查训练回调
        for i, (cid, callback) in enumerate(self.training_callbacks):
            if cid == callback_id:
                del self.training_callbacks[i]
                return True
        
        return False
    
    async def _trigger_round_callbacks(self, round_num: int, round_result: RoundResult):
        """触发轮次回调"""
        for callback_id, callback in self.round_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(round_num, round_result)
                else:
                    callback(round_num, round_result)
            except Exception as e:
                self.logger.debug(f"Round callback {callback_id} error: {e}")
    
    async def _trigger_training_callbacks(self, event: str, data: Any):
        """触发训练回调"""
        for callback_id, callback in self.training_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event, data)
                else:
                    callback(event, data)
            except Exception as e:
                self.logger.debug(f"Training callback {callback_id} error: {e}")

    # ==================== 回调机制（用于实验记录） ====================

    def add_callback(self, event: str, callback: Callable):
        """添加回调函数（用于实验记录等扩展功能）

        Args:
            event: 事件名称（'before_round', 'after_round', 'after_evaluation'）
            callback: 回调函数
        """
        if event in self._callbacks:
            self._callbacks[event].append(callback)
        else:
            self.logger.warning(f"Unknown callback event: {event}")

    def _trigger_callbacks(self, event: str, *args, **kwargs):
        """触发回调（内部使用）

        Args:
            event: 事件名称
            *args, **kwargs: 传递给回调的参数
        """
        # 调试日志：输出判断条件和callback队列
        self.logger.info(f"[Callback Debug] 触发事件: {event}")
        self.logger.info(f"[Callback Debug] 判断条件: event='{event}', _callbacks 是否有此事件: {event in self._callbacks}")
        self.logger.info(f"[Callback Debug] Callback队列中的元素: {list(self._callbacks.keys())}")
        self.logger.info(f"[Callback Debug] 事件 '{event}' 的回调数量: {len(self._callbacks.get(event, []))}")

        for callback in self._callbacks.get(event, []):
            try:
                self.logger.info(f"[Callback Debug] 执行回调: {callback}")
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(*args, **kwargs))
                else:
                    callback(*args, **kwargs)
            except Exception as e:
                self.logger.warning(f"Callback {event} failed: {e}")

    # ==================== 包装方法（用于回调注入） ====================

    async def _train_round(self, round_num: int, client_ids: List[str]) -> RoundResult:
        """包装方法：负责回调触发，调用用户实现的 train_round

        Args:
            round_num: 轮次编号
            client_ids: 客户端ID列表

        Returns:
            RoundResult: 轮次训练结果
        """
        # 触发前置回调
        self._trigger_callbacks('before_round', round_num)

        # 调用用户实现的训练方法
        result = await self.train_round(round_num, client_ids)

        # 触发后置回调
        self._trigger_callbacks('after_round', round_num, result)

        return result

    async def _evaluate_global_model(self) -> EvaluationResult:
        """包装方法：负责回调触发，调用用户实现的 evaluate_global_model

        Returns:
            EvaluationResult: 评估结果
        """
        # 调用用户实现的评估方法
        result = await self.evaluate_global_model()

        # 触发回调
        self._trigger_callbacks('after_evaluation', result)

        return result

    # ==================== 训练循环方法 (框架提供) ====================

    async def run_training(self, max_rounds: int) -> FederationResult:
        """
        执行完整的联邦学习训练流程

        Args:
            max_rounds: 最大训练轮数

        Returns:
            FederationResult: 训练结果

        Raises:
            FederationError: 训练失败
        """
        result = FederationResult()
        result.total_rounds = max_rounds
        self.training_status.total_rounds = max_rounds
        self.training_status.is_training = True
        self.training_status.start_time = datetime.now()

        self.logger.info(f"Starting federated training for {max_rounds} rounds...")

        try:
            # 触发训练开始回调
            await self._trigger_training_callbacks("TRAINING_STARTED", {
                "start_time": self.training_status.start_time.isoformat(),
                "max_rounds": max_rounds
            })

            # 训练循环
            for round_num in range(1, max_rounds + 1):
                round_start_time = datetime.now()

                try:
                    # 选择客户端
                    selected_clients = self.select_clients_for_round(round_num)
                    self.logger.info(f"\nRound {round_num}/{max_rounds}: Selected {len(selected_clients)} clients")

                    # 检查客户端就绪状态
                    client_readiness = await self.check_client_readiness(selected_clients)
                    ready_clients = [cid for cid, ready in client_readiness.items() if ready]

                    if len(ready_clients) < self.training_config.min_clients:
                        self.logger.info(f"Warning: Insufficient ready clients ({len(ready_clients)} < {self.training_config.min_clients})")
                        continue

                    # 更新训练状态
                    self.training_status.current_round = round_num
                    self.training_status.selected_clients = selected_clients
                    self.training_status.active_clients = ready_clients

                    # 执行训练轮次（使用包装方法触发回调）
                    round_result = await self._train_round(round_num, ready_clients)

                    # 更新全局模型
                    if "aggregated_model" in round_result:
                        self.global_model = round_result["aggregated_model"]
                        if self._best_model is None:
                            self._best_model = self.global_model

                    # 计算轮次时间
                    round_time = (datetime.now() - round_start_time).total_seconds()
                    round_result["round_time"] = round_time
                    round_result["round_number"] = round_num

                    # 更新统计
                    await self._update_round_statistics(round_num, round_result, round_start_time)

                    # 更新结果
                    result.completed_rounds = round_num
                    result.training_history.append(round_result)

                    # 更新最佳指标
                    round_metrics = round_result.get("round_metrics", {})
                    if "avg_accuracy" in round_metrics:
                        accuracy = round_metrics["avg_accuracy"]
                        if accuracy > result.final_accuracy:
                            result.final_accuracy = accuracy
                            result.best_model = (
                                self.global_model.copy()
                                if isinstance(self.global_model, dict)
                                else self.global_model
                            )

                    if "avg_loss" in round_metrics:
                        loss = round_metrics["avg_loss"]
                        if loss < result.final_loss:
                            result.final_loss = loss

                    # 触发轮次回调
                    await self._trigger_round_callbacks(round_num, round_result)

                    self.logger.info(f"Round {round_num} completed in {round_time:.2f}s")
                    self.logger.info(f"  Metrics: accuracy={round_metrics.get('avg_accuracy', 0):.4f}, loss={round_metrics.get('avg_loss', 0):.4f}")

                    # 检查收敛条件
                    if self.should_stop_training(round_num, round_result):
                        result.termination_reason = "converged"
                        result.convergence_round = round_num
                        self.logger.info(f"Training converged at round {round_num}")
                        break

                except Exception as e:
                    self.logger.exception(f"Round {round_num} failed: {e}")
                    # 可以选择继续下一轮或终止训练
                    continue

            # 正常完成所有轮次
            if result.termination_reason == "unknown":
                result.termination_reason = "max_rounds_reached"

            # 最终评估
            self.logger.info("Performing final evaluation...")
            try:
                final_evaluation = await self.evaluate_global_model()
                if final_evaluation.get("accuracy") is not None:
                    result.final_accuracy = final_evaluation["accuracy"]
                if final_evaluation.get("loss") is not None:
                    result.final_loss = final_evaluation["loss"]
            except Exception as e:
                self.logger.exception(f"Final evaluation failed: {e}")

            # 保存最终模型
            if result.best_model is None:
                result.best_model = self.global_model

            result.success = True

        except Exception as e:
            result.error_message = str(e)
            result.success = False
            self.logger.exception(f"Training failed: {e}")
            raise FederationError(f"Training failed: {str(e)}")

        finally:
            # 更新训练状态
            self.training_status.is_training = False
            self.training_status.end_time = datetime.now()

            if self.training_status.start_time:
                result.total_time = (
                    self.training_status.end_time - self.training_status.start_time
                ).total_seconds()

            # 触发训练完成回调
            await self._trigger_training_callbacks("TRAINING_COMPLETED", {
                "end_time": self.training_status.end_time.isoformat(),
                "completed_rounds": result.completed_rounds,
                "final_accuracy": result.final_accuracy,
                "final_loss": result.final_loss,
                "total_time": result.total_time,
                "termination_reason": result.termination_reason
            })

            self.logger.info(f"Training completed: {result.completed_rounds}/{max_rounds} rounds")
            self.logger.info(f"  Final accuracy: {result.final_accuracy:.4f}")
            self.logger.info(f"  Final loss: {result.final_loss:.4f}")
            self.logger.info(f"  Total time: {result.total_time:.2f}s")
            self.logger.info(f"  Termination reason: {result.termination_reason}")

        return result
