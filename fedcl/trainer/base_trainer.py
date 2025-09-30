"""
MOE-FedCL 服务端训练器抽象基类
moe_fedcl/trainer/base_trainer.py
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timedelta

from ..learner.proxy import LearnerProxy
from ..types import ModelData, TrainingResult, EvaluationResult, RoundResult
from ..exceptions import TrainingError, ClientNotFoundError, FederationError
from ..communication.layer_event import ProxyManagerEventHandler


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
    
    def __init__(self, trainer: 'BaseTrainer'):
        self.trainer = trainer
        self.proxies: Dict[str, LearnerProxy] = {}
        self._lock = asyncio.Lock()
        
        # 导入日志记录器
        from ..utils.auto_logger import get_comm_logger
        self.logger = get_comm_logger("proxy_manager")
    
    async def on_proxy_ready(self, client_id: str, proxy: LearnerProxy):
        """接收业务通信层创建的代理"""
        self.logger.info(f"🎯 [代理管理器] 收到代理就绪通知: {client_id}")
        
        async with self._lock:
            self.proxies[client_id] = proxy
            
            # 更新trainer的客户端统计
            self.trainer.client_statistics[client_id] = ClientStatistics(client_id)
            
            self.logger.info(f"✅ [代理管理器] 学习器代理已注册: {client_id}, 当前总数: {len(self.proxies)}")
            self.logger.info(f"📊 [代理管理器] 可用客户端列表: {list(self.proxies.keys())}")
    
    async def on_proxy_disconnected(self, client_id: str):
        """处理代理断开"""
        self.logger.info(f"❌ [代理管理器] 收到代理断开通知: {client_id}")
        
        async with self._lock:
            if client_id in self.proxies:
                del self.proxies[client_id]
                
                # 清理trainer的客户端统计
                if client_id in self.trainer.client_statistics:
                    del self.trainer.client_statistics[client_id]
                
                self.logger.info(f"🗑️ [代理管理器] 学习器代理已移除: {client_id}, 剩余数量: {len(self.proxies)}")
    
    def get_proxy(self, client_id: str) -> Optional[LearnerProxy]:
        """获取指定客户端的代理"""
        return self.proxies.get(client_id)
    
    def get_all_proxies(self) -> Dict[str, LearnerProxy]:
        """获取所有代理"""
        return self.proxies.copy()
    
    def get_available_clients(self) -> List[str]:
        """获取可用客户端列表"""
        available_clients = []
        self.logger.debug(f"🔍 [代理管理器] 检查可用客户端，总代理数: {len(self.proxies)}")
        
        for client_id, proxy in self.proxies.items():
            if proxy.is_client_ready():
                available_clients.append(client_id)
                self.logger.debug(f"✅ [代理管理器] 客户端[{client_id}]可用")
            else:
                self.logger.debug(f"❌ [代理管理器] 客户端[{client_id}]不可用")
        
        self.logger.info(f"📊 [代理管理器] 可用客户端总数: {len(available_clients)}/{len(self.proxies)}")
        return available_clients


class BaseTrainer(ABC):
    """服务端训练器抽象基类 - 用户继承实现联邦学习算法"""
    
    def __init__(self,
                 global_model: ModelData,
                 training_config: Optional[TrainingConfig] = None,
                 logger: Any = None):
        """
        初始化训练器
        
        Args:
            global_model: 全局模型初始状态
            training_config: 训练配置
            logger: 日志记录器
        """
        # 🎯 自动实例化代理管理器（用户无感知）
        self._proxy_manager = ProxyManager(self)
        
        # 创建事件处理器，用于接收业务层的代理创建事件
        self._proxy_event_handler = ProxyManagerEventHandler(self._proxy_manager)
        
        # learner_proxies变成代理管理器的代理属性
        self.learner_proxies = self._proxy_manager.proxies
        
        self.global_model = global_model
        self.training_config = training_config or TrainingConfig()
        self.logger = logger
        
        # 训练状态
        self.training_status = TrainingStatus()
        self.training_status.total_rounds = self.training_config.max_rounds
        
        # 统计信息
        self.round_statistics: Dict[int, RoundStatistics] = {}
        self.client_statistics: Dict[str, ClientStatistics] = {}
        
        # 回调函数
        self.round_callbacks: List[Callable] = []
        self.training_callbacks: List[Callable] = []
        
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
            
            print(f"Checkpoint saved: {checkpoint_path}")
            return True
            
        except Exception as e:
            print(f"Failed to save checkpoint: {e}")
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
            
            print(f"Checkpoint loaded: {checkpoint_path}")
            return True
            
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
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
        """检查客户端就绪状态
        
        Args:
            client_ids: 要检查的客户端ID列表
            
        Returns:
            Dict[str, bool]: 客户端就绪状态映射
        """
        readiness = {}
        
        ping_tasks = []
        for client_id in client_ids:
            if client_id in self.learner_proxies:
                task = self.learner_proxies[client_id].ping()
                ping_tasks.append((client_id, task))
        
        # 并发ping所有客户端
        for client_id, task in ping_tasks:
            try:
                await asyncio.wait_for(task, timeout=5.0)
                readiness[client_id] = True
            except Exception:
                readiness[client_id] = False
        
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
            print("Checking client connections...")
            available_clients = self.get_available_clients()
            print(f"Found {len(available_clients)} available clients: {available_clients}")
            
            # 初始化全局模型
            if self.global_model is None:
                raise FederationError("Global model not provided")
            
            # 执行用户自定义初始化
            await self._perform_custom_initialization()
            
            print("BaseTrainer initialized successfully")
            return True
            
        except Exception as e:
            print(f"Trainer initialization failed: {e}")
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
        
        print("BaseTrainer cleaned up")
    
    async def handle_client_failure(self, client_id: str) -> None:
        """处理客户端故障
        
        Args:
            client_id: 故障的客户端ID
        """
        print(f"Handling client failure: {client_id}")
        
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
                print(f"Round callback {callback_id} error: {e}")
    
    async def _trigger_training_callbacks(self, event: str, data: Any):
        """触发训练回调"""
        for callback_id, callback in self.training_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event, data)
                else:
                    callback(event, data)
            except Exception as e:
                print(f"Training callback {callback_id} error: {e}")