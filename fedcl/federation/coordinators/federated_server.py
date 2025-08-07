# fedcl/federation/coordinators/improved_federated_server.py
"""
改进的联邦服务端协调器

基于现有FederatedServer的极简化改进：
- 增强错误处理和重试机制
- 添加超时控制
- 优化聚合流程
- 保持传统联邦学习架构
- 完全向后兼容
"""

import time
import asyncio
import threading
from typing import Dict, Any, List, Optional, Tuple, Union, Set
from dataclasses import dataclass
from collections import defaultdict, deque
import torch
import torch.nn as nn
from loguru import logger
from omegaconf import OmegaConf

from .base import (
    FederatedCommunicator, 
    CommunicationConfig, 
    CommunicatorRole, 
    MessageType
)
from ...core.execution_context import ExecutionContext
from ...core.base_aggregator import BaseAggregator
from ...engine.federation_engine import FederationEngine
from ..managers.model_manager import ModelManager
from ..managers.client_manager import ClientManager  
from ..state.state_enums import ServerState
from ..state.state_manager import StateManager, StateTransitionRecord
from ..exceptions import FederationError, ServerError
from ...utils.improved_logging_manager import get_component_logger, log_training_info, log_system_debug
from ...config.config_manager import DictConfig
from ...registry.component_composer import ComponentComposer


@dataclass
class TaskInfo:
    """任务信息"""
    task_id: str
    task_type: str  # 'classification', 'regression', 'detection', etc.
    dataset_info: Dict[str, Any]
    model_config: Dict[str, Any]
    learning_config: Dict[str, Any]
    sequence_position: int
    dependencies: List[str]  # 依赖的前置任务
    metadata: Dict[str, Any]


@dataclass
class RoundResult:
    """联邦学习轮次结果"""
    round_id: int
    task_id: str
    participating_clients: List[str]
    aggregated_model: Optional[torch.nn.Module]
    client_metrics: Dict[str, Dict[str, float]]
    aggregation_metrics: Dict[str, float]
    round_time: float
    convergence_info: Dict[str, Any]
    knowledge_transfer_info: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass
class ServerStatus:
    """服务端状态信息"""
    server_id: str
    current_task: Optional[str]
    current_round: int
    total_tasks: int
    completed_tasks: int
    registered_clients: int
    active_clients: int
    global_model_version: int
    uptime: float
    server_state: str


@dataclass
class RoundConfig:
    """轮次配置"""
    timeout: float = 300.0  # 轮次超时时间（秒）
    min_client_updates: int = 1  # 最小客户端更新数
    max_wait_time: float = 60.0  # 最大等待时间
    retry_attempts: int = 3  # 重试次数
    client_selection_timeout: float = 30.0  # 客户端选择超时


class ImprovedFederatedServer(FederatedCommunicator):
    """
    改进的联邦服务端协调器
    
    主要改进：
    1. 增强的错误处理和重试机制
    2. 超时控制和异常恢复
    3. 优化的聚合流程
    4. 改进的日志记录和监控
    5. 保持向后兼容性
    
    保持原有功能：
    - 任务序列管理和调度
    - 客户端注册和状态管理  
    - 模型聚合和分发协调
    - 知识蒸馏过程控制
    - 辅助模型管理
    - 持续学习策略执行
    """
    
    def __init__(self, server_id: str, config: DictConfig):
        """
        初始化改进的联邦服务端
        
        Args:
            server_id: 服务端唯一标识
            config: 服务端完整配置
        """
        # 构建通信配置
        comm_config = CommunicationConfig(
            role=CommunicatorRole.SERVER,
            component_id=server_id,
            host=config.get('communication', {}).get('host', 'localhost'),
            port=config.get('communication', {}).get('port', 8080),
            max_workers=config.get('communication', {}).get('max_workers', 10),
            heartbeat_interval=config.get('communication', {}).get('heartbeat_interval', 30.0),
            message_timeout=config.get('communication', {}).get('timeout', 60.0)
        )
        
        # 初始化通信基类
        super().__init__(comm_config)
        
        # 基本属性
        self.server_id = server_id
        self.server_config = config
        
        # 根据配置创建组件
        self.context = self._create_execution_context(config)
        self.aggregator = self._create_aggregator(config.get('aggregator', {}), self.context)
        self.model_manager = self._create_model_manager(config.get('model_manager', {}), self.context)
        self.client_manager = self._create_client_manager(config.get('client_manager', {}), self.context)
        self.federation_engine = self._create_federation_engine(config.get('federation', {}), self.context)
        
        # 添加评估器组件
        self.evaluators = self._create_evaluators(config.get('evaluators', {}), self.context)
        self.evaluation_config = config.get('evaluation', {})
        
        # 创建测试数据加载器
        self.test_data_loaders = self._create_test_data_loaders(config.get('test_datas', {}), self.context)
        
        # 任务管理
        self.task_queue: deque = deque()
        self.completed_tasks: List[TaskInfo] = []
        self.current_task: Optional[TaskInfo] = None
        self.task_history: Dict[str, RoundResult] = {}
        
        # 轮次管理
        self.current_round = 0
        self.global_model_version = 0
        self.round_history: List[RoundResult] = []
        
        # 改进：轮次配置
        round_config_dict = config.get('round_config', {})
        self.round_config = RoundConfig(
            timeout=round_config_dict.get('timeout', 300.0),
            min_client_updates=round_config_dict.get('min_client_updates', 1),
            max_wait_time=round_config_dict.get('max_wait_time', 60.0),
            retry_attempts=round_config_dict.get('retry_attempts', 3),
            client_selection_timeout=round_config_dict.get('client_selection_timeout', 30.0)
        )
        
        # 改进：并发控制
        self._round_lock = threading.RLock()
        self._aggregation_lock = threading.Lock()
        self._client_updates_lock = threading.Lock()
        
        # 改进：轮次状态追踪
        self._round_start_time: Optional[float] = None
        self._pending_client_updates: Dict[str, Dict[str, Any]] = {}
        self._expected_clients: Set[str] = set()
        self._round_timer: Optional[threading.Timer] = None
        
        # 持续学习特有组件
        self.auxiliary_models: Dict[str, torch.nn.Module] = {}
        self.knowledge_base: Dict[str, Any] = {}
        self.teacher_models: Dict[str, torch.nn.Module] = {}
        
        # 改进：性能监控
        self.performance_stats = {
            'successful_轮次': 0,
            'failed_轮次': 0,
            'average_round_time': 0.0,
            'start_time': time.time()
        }
        
        # 初始化状态管理器
        self.state_manager = self._initialize_state_management()
        
        # 注册Hook（如果配置了）
        self._register_hooks(config.get('hooks', {}))
        
        # 服务端启动状态
        self.startup_ready = False
        self.min_clients = config.get('federation', {}).get('min_客户端', 2)
        
        self.logger.debug(f"改进的联邦服务器初始化完成: {server_id}")
        self.logger.debug(f"轮次配置 - 超时时间: {self.round_config.timeout}s, min_updates: {self.round_config.min_client_updates}")
        self.logger.debug(f"聚合器: {type(self.aggregator).__name__}")
        self.logger.debug(f"模型管理器: {type(self.model_manager).__name__}")
        
        # 注册服务器特有的消息处理器
        self._register_server_handlers()
    
    def _register_server_handlers(self) -> None:
        """注册服务器特有的消息处理器"""
        # 客户端注册消息处理器
        self.register_message_handler(MessageType.REGISTRATION, self.handle_client_registration)
        self.logger.debug("注册服务器专用消息处理器")
    
    def handle_client_registration(self, message_data: Dict[str, Any]) -> Any:
        """
        处理客户端注册消息
        
        Args:
            message_data: 注册消息数据
            
        Returns:
            注册响应
        """
        try:
            client_info = message_data.get('data', {})
            self.logger.debug(f"处理客户端注册: {client_info.get('client_id', 'unknown')}")
            
            response = self.register_client(client_info)
            self.logger.debug(f"注册响应: {response}")
            
            return response
            
        except Exception as e:
            self.logger.error(f"处理客户端注册失败: {e}")
            return {"status": "error", "message": str(e)}
    
    # ===== FederatedCommunicator 抽象方法实现 =====
    
    def on_start(self) -> None:
        """服务端启动时的初始化"""
        try:
            self.logger.debug(f"Starting improved server {self.server_id}")
            
            # 状态转换：INITIALIZING -> LOADING_CONFIG
            if self.state_manager:
                self.state_manager.transition_to(ServerState.LOADING_CONFIG, {
                    "action": "loading_configuration",
                    "timestamp": time.time()
                })
            
            # 1. 初始化联邦引擎
            self.federation_engine.initialize_federation()
            
            # 状态转换：LOADING_CONFIG -> REGISTERING_COMPONENTS
            if self.state_manager:
                self.state_manager.transition_to(ServerState.REGISTERING_COMPONENTS, {
                    "action": "registering_components",
                    "timestamp": time.time()
                })
            
            # 2. 初始化模型管理器
            self._initialize_global_models()
            
            # 3. 加载任务序列
            self._load_task_sequence()
            
            # 状态转换：LOADING_CONFIG -> REGISTERING_COMPONENTS
            if self.state_manager:
                self.state_manager.transition_to(ServerState.REGISTERING_COMPONENTS, {
                    "action": "registering_components",
                    "timestamp": time.time()
                })
            
            # 3. 初始化模型管理器
            self._initialize_global_models()
            
            # 4. 加载任务序列
            self._load_task_sequence()
            
            # 状态转换：REGISTERING_COMPONENTS -> WAITING_FOR_CLIENTS
            if self.state_manager:
                self.state_manager.transition_to(ServerState.WAITING_FOR_CLIENTS, {
                    "action": "waiting_for_client_registration",
                    "min_客户端": self.min_clients,
                    "timestamp": time.time()
                })
            
            # 4. 发布服务端启动事件
            self.context.publish_event("server_started", {
                "server_id": self.server_id,
                "timestamp": time.time()
            })
            
            self.logger.debug(f"改进的服务器 {self.server_id} 启动成功")
            
        except Exception as e:
            self.logger.error(f"服务器启动失败 {self.server_id}: {e}")
            
            # 状态转换到错误状态
            if self.state_manager:
                self.state_manager.transition_to(ServerState.ERROR, {
                    "action": "startup_failed",
                    "error": str(e),
                    "timestamp": time.time()
                })
            
            raise FederationError(f"Server startup failed: {e}")
    
    def on_stop(self) -> None:
        """服务端停止时的清理"""
        try:
            self.logger.debug(f"停止改进的服务器 {self.server_id}")
            
            # 改进：取消正在进行的轮次
            self._cancel_current_round()
            
            # 1. 停止联邦引擎
            if hasattr(self.federation_engine, 'stop_federation'):
                self.federation_engine.stop_federation()
            
            # 2. 清理资源
            self.auxiliary_models.clear()
            self.knowledge_base.clear()
            self.teacher_models.clear()
            self._pending_client_updates.clear()
            self._expected_clients.clear()
            
            # 3. 发布服务端停止事件
            self.context.publish_event("server_已停止", {
                "server_id": self.server_id,
                "timestamp": time.time()
            })
            
            self.logger.debug(f"改进的服务器 {self.server_id} 已停止")
            
        except Exception as e:
            self.logger.error(f"停止服务器出错 {self.server_id}: {e}")
    
    def handle_model_distribution(self, message_data: Dict[str, Any]) -> Any:
        """处理模型分发请求（服务端不处理此消息）"""
        self.logger.warning("Server received model_distribution message - un期望")
        return {"status": "ignored"}
    
    def handle_model_update(self, message_data: Dict[str, Any]) -> Any:
        """处理客户端模型更新（改进版）"""
        try:
            client_id = message_data.get('metadata', {}).get('client_id')
            round_id = message_data.get('metadata', {}).get('round_id')
            
            if not client_id or round_id is None:
                self.logger.warning(f"Invalid model update message: missing client_id or round_id")
                return {"status": "error", "message": "Missing client_id or round_id"}
            
            self.logger.info(f"收到来自客户端的模型更新 {client_id} for round {round_id}")
            
            # 改进：验证轮次ID
            if round_id != self.current_round:
                self.logger.warning(f"轮次ID不匹配: expected {self.current_round}, got {round_id}")
                return {"status": "error", "message": "轮次ID不匹配"}
            
            # 改进：验证客户端是否被期待
            with self._client_updates_lock:
                if client_id not in self._expected_clients:
                    self.logger.warning(f"来自客户端的意外更新 {client_id}")
                    return {"status": "error", "message": "Unexpected client update"}
            
            # 改进：验证消息数据
            update_data = message_data.get('data', {})
            if not self._validate_client_update(update_data):
                self.logger.warning(f"客户端的更新数据无效 {client_id}")
                return {"status": "error", "message": "Invalid update data"}
            
            # 存储客户端更新
            self._store_client_update(client_id, round_id, update_data)
            
            # 检查是否收集到足够的更新
            self._check_aggregation_conditions()
            
            return {"status": "update_received", "client_id": client_id, "round_id": round_id}
            
        except Exception as e:
            self.logger.error(f"处理模型更新失败: {e}")
            return {"status": "error", "message": str(e)}
    
    def handle_training_trigger(self, message_data: Dict[str, Any]) -> Any:
        """处理训练触发请求（服务端主动触发，不处理此消息）"""
        self.logger.warning("Server received training_trigger message - un期望")
        return {"status": "ignored"}
    
    def handle_task_notification(self, message_data: Dict[str, Any]) -> Any:
        """处理任务通知（服务端主动分发任务）"""
        self.logger.warning("Server received task_notification message - un期望")
        return {"status": "ignored"}
    
    # ===== 公共接口方法 =====
    
    @classmethod
    def create_from_config(cls, config: DictConfig) -> 'ImprovedFederatedServer':
        """
        从配置创建服务端实例
        
        Args:
            config: 服务端配置
            
        Returns:
            ImprovedFederatedServer: 服务端实例
        """
        try:
            server_id = config.get('server', {}).get('id', 'main_server')
            logger.debug(f"从配置创建改进的服务器: {server_id}")
            
            # 直接创建服务端（内部会根据配置创建所有组件）
            server = cls(server_id, config)
            
            logger.debug(f"改进的服务器 created successfully: {server_id}")
            return server
            
        except Exception as e:
            logger.error(f"从配置创建改进的服务器失败: {e}")
            raise FederationError(f"Server creation failed: {e}")
    
    def register_client(self, client_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        注册客户端（改进版）
        
        Args:
            client_info: 客户端信息
            
        Returns:
            注册响应
        """
        try:
            client_id = client_info.get('client_id')
            if not client_id:
                return {"status": "error", "message": "Missing client_id"}
            
            # 改进：验证客户端信息
            if not self._validate_client_info(client_info):
                return {"status": "error", "message": "Invalid client information"}
            
            # 委托给客户端管理器
            success = self.client_manager.register_client(client_id, client_info)
            
            if success:
                self.logger.debug(f"Client registered: {client_id}")
                
                # 检查是否达到启动条件
                registered_count = len(self.client_manager.get_active_clients())
                if registered_count >= self.min_clients and not self.startup_ready:
                    self._on_startup_ready()
                
                return {
                    "status": "registered",
                    "client_id": client_id,
                    "server_ready": self.startup_ready
                }
            else:
                return {"status": "error", "message": "Registration failed"}
                
        except Exception as e:
            self.logger.error(f"Client registration failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def coordinate_federated_round(self, round_id: Optional[int] = None) -> Optional[RoundResult]:
        """
        协调联邦学习轮次（改进版）
        
        Args:
            round_id: 轮次ID，如果为None则自动递增
            
        Returns:
            轮次结果
        """
        with self._round_lock:
            try:
                if not self.current_task:
                    self.logger.warning("No current task, cannot coordinate round")
                    return None
                
                if round_id is None:
                    self.current_round += 1
                    round_id = self.current_round
                else:
                    self.current_round = round_id
                
                self.logger.info(f"Starting federated round {round_id} for task {self.current_task.task_id}")
                self._round_start_time = time.time()
                
                # 状态转换：确保从合法状态转换到COORDINATING
                if self.state_manager:
                    current_state = self.state_manager.current_state
                    
                    # 简化状态转换逻辑
                    if current_state == ServerState.ROUND_COMPLETED:
                        # 正常情况：上一轮完成，开始新轮
                        self.state_manager.transition_to(ServerState.TRAINING, {
                            "action": "start_new_round",
                            "round_id": round_id,
                            "timestamp": time.time()
                        })
                    elif current_state in [ServerState.WAITING_FOR_UPDATES, ServerState.AGGREGATING]:
                        # 异常情况：上一轮未正常完成，直接重置到TRAINING
                        self.logger.warning(f"Server in unexpected state {current_state} for new round, forcing reset to TRAINING")
                        # 强制重置状态
                        self.state_manager.current_state = ServerState.TRAINING
                        self.state_manager.state_history.append(StateTransitionRecord(
                            from_state=current_state,
                            to_state=ServerState.TRAINING,
                            timestamp=time.time(),
                            metadata={"reason": "force_reset_for_new_round", "round_id": round_id}
                        ))
                    elif current_state == ServerState.ERROR:
                        # 错误状态：需要先恢复
                        self.logger.debug("Recovering from ERROR state to TRAINING")
                        self.state_manager.current_state = ServerState.TRAINING
                        self.state_manager.state_history.append(StateTransitionRecord(
                            from_state=ServerState.ERROR,
                            to_state=ServerState.TRAINING,
                            timestamp=time.time(),
                            metadata={"reason": "error_recovery", "round_id": round_id}
                        ))
                    elif current_state != ServerState.TRAINING:
                        # 其他状态：尝试正常转换，失败则强制重置
                        try:
                            self.state_manager.transition_to(ServerState.TRAINING, {
                                "round_id": round_id,
                                "timestamp": time.time()
                            })
                        except Exception as e:
                            self.logger.warning(f"Cannot transition from {current_state} to TRAINING, forcing reset: {e}")
                            self.state_manager.current_state = ServerState.TRAINING
                            self.state_manager.state_history.append(StateTransitionRecord(
                                from_state=current_state,
                                to_state=ServerState.TRAINING,
                                timestamp=time.time(),
                                metadata={"reason": "force_reset_transition_failed", "round_id": round_id}
                            ))
                    
                    # 现在从TRAINING转换到COORDINATING
                    self.state_manager.transition_to(ServerState.COORDINATING, {
                        "round_id": round_id,
                        "task_id": self.current_task.task_id,
                        "timestamp": time.time()
                    })
                
                # 改进：重试机制
                for attempt in range(self.round_config.retry_attempts):
                    try:
                        result = self._execute_round_with_retry(round_id, attempt)
                        # 对于异步模式，成功启动轮次即认为成功
                        # 无论返回值是什么（None或其他），只要没有异常就认为成功
                        self.logger.info(f"轮次 {round_id} started successfully on attempt {attempt + 1}")
                        return result  # 可能是None，表示轮次已启动
                    except Exception as e:
                        self.logger.warning(f"轮次 {round_id} attempt {attempt + 1} failed: {e}")
                        if attempt == self.round_config.retry_attempts - 1:
                            raise
                        
                        # 重试前等待
                        time.sleep(min(2 ** attempt, 10))  # 指数退避
                
                return None
                
            except Exception as e:
                self.logger.error(f"Failed to coordinate round {round_id}: {e}")
                
                # 更新性能统计
                self.performance_stats['failed_轮次'] += 1
                
                # 状态转换到错误状态
                if self.state_manager:
                    self.state_manager.transition_to(ServerState.ERROR, {
                        "action": "round_coordination_failed",
                        "round_id": round_id,
                        "error": str(e),
                        "timestamp": time.time()
                    })
                
                return None
    
    def start_federation(self) -> Dict[str, Any]:
        """
        启动完整的联邦学习流程
        
        Returns:
            联邦学习结果
        """
        try:
            self.logger.info("开始联邦学习过程")
            
            # 检查服务端是否就绪
            if not self.startup_ready:
                raise ServerError("Server not ready for federation")
            
            # 获取联邦学习配置
            federation_config = self.server_config.get('federation', {})
            num_rounds = federation_config.get('num_轮次', 3)
            min_clients = federation_config.get('min_客户端', 1)
            
            self.logger.debug(f"联邦配置: {num_rounds} rounds, min {min_clients} 客户端")
            
            # 检查可用客户端数量
            available_clients = self.client_manager.get_active_clients()
            if len(available_clients) < min_clients:
                raise ServerError(f"Insufficient clients: {len(available_clients)} < {min_clients}")
            
            # 状态转换：IDLE -> TRAINING
            if self.state_manager:
                self.state_manager.transition_to(ServerState.TRAINING, {
                    "federation_start": True,
                    "num_轮次": num_rounds,
                    "available_客户端": len(available_clients)
                })
            
            # 创建联邦学习任务
            task_info = TaskInfo(
                task_id=f"federation_{int(time.time())}",
                task_type="federated_learning",
                dataset_info={"dataset_type": "mnist", "num_客户端": len(available_clients)},
                model_config={"model_type": "default", "architecture": "mlp"},
                learning_config={"num_轮次": num_rounds, "min_客户端": min_clients},
                sequence_position=1,
                dependencies=[],
                metadata={
                    "federation_type": "standard",
                    "start_time": time.time(),
                    "client_ids": available_clients
                }
            )
            self.current_task = task_info
            
            # 执行联邦学习轮次
            federation_results = {
                "task_id": task_info.task_id,
                "num_轮次": num_rounds,
                "round_results": [],
                "start_time": task_info.metadata["start_time"],
                "status": "running"
            }
            
            for round_num in range(1, num_rounds + 1):
                try:
                    self.logger.info(f"开始联邦训练轮次 {round_num}/{num_rounds}")
                    
                    # 协调单轮联邦学习
                    round_result = self.coordinate_federated_round(round_num)
                    
                    # 等待轮次完成（无论是同步还是异步）
                    if round_result is None:
                        # 异步模式：等待轮次真正完成
                        self.logger.info(f"轮次 {round_num} started asynchronously, waiting for completion...")
                        
                        # 等待轮次完成的逻辑
                        max_wait_time = self.round_config.timeout + 30  # 额外30秒缓冲
                        start_wait = time.time()
                        
                        while time.time() - start_wait < max_wait_time:
                            # 检查服务器状态
                            if self.state_manager:
                                current_state = self.state_manager.current_state
                                if current_state == ServerState.ROUND_COMPLETED:
                                    self.logger.info(f"轮次 {round_num} 成功完成")
                                    break
                                elif current_state == ServerState.ERROR:
                                    self.logger.warning(f"轮次 {round_num} 以错误状态结束")
                                    break
                            
                            # 短暂等待
                            time.sleep(2)
                        else:
                            # 超时
                            self.logger.warning(f"轮次 {round_num} completion timeout after {max_wait_time}s")
                        
                        # 记录异步完成的轮次
                        federation_results["round_results"].append({
                            "round_id": round_num,
                            "status": "async_完成",
                            "participants": available_clients,
                            "duration": time.time() - start_wait
                        })
                        self.performance_stats['successful_轮次'] += 1
                    else:
                        # 同步模式：轮次已经完成
                        federation_results["round_results"].append({
                            "round_id": round_num,
                            "status": "完成",
                            "participants": round_result.participating_clients,
                            "duration": getattr(round_result, 'round_duration', 0.0)
                        })
                        self.performance_stats['successful_轮次'] += 1
                        
                    # 轮次间间隔（确保状态稳定）
                    if round_num < num_rounds:
                        self.logger.info(f"准备轮次 {round_num + 1}...")
                        time.sleep(3)  # 增加间隔以确保状态清理完成
                        
                except Exception as e:
                    self.logger.error(f"轮次 {round_num} failed: {e}")
                    federation_results["round_results"].append({
                        "round_id": round_num,
                        "status": "失败",
                        "error": str(e),
                        "participants": [],
                        "duration": 0.0
                    })
                    self.performance_stats['failed_轮次'] += 1
                    
                    # 可选：失败后是否继续
                    continue_on_failure = federation_config.get('continue_on_failure', True)
                    if not continue_on_failure:
                        break
            
            # 完成联邦学习
            task_info.metadata["end_time"] = time.time()
            task_info.metadata["status"] = "完成"
            self.completed_tasks.append(task_info)
            
            federation_results.update({
                "end_time": task_info.metadata["end_time"],
                "total_duration": task_info.metadata["end_time"] - task_info.metadata["start_time"],
                "status": "完成",
                "successful_轮次": self.performance_stats['successful_轮次'],
                "failed_轮次": self.performance_stats['failed_轮次']
            })
            
            # 状态转换：TRAINING -> COMPLETED
            if self.state_manager:
                self.state_manager.transition_to(ServerState.COMPLETED, {
                    "federation_完成": True,
                    "total_轮次": len(federation_results["round_results"])
                })
            
            self.logger.info(f"联邦学习完成: {len(federation_results['round_results'])} 轮次")
            return federation_results
            
        except Exception as e:
            self.logger.error(f"联邦学习失败: {e}")
            
            # 状态转换到错误状态
            if self.state_manager:
                self.state_manager.transition_to(ServerState.ERROR, {
                    "federation_error": str(e),
                    "timestamp": time.time()
                })
            
            if hasattr(self, 'current_task') and self.current_task:
                self.current_task.metadata["status"] = "失败"
                self.current_task.metadata["end_time"] = time.time()
            
            raise ServerError(f"Federation execution failed: {e}")
    
    def get_server_status(self) -> ServerStatus:
        """获取服务端状态（改进版）"""
        uptime = time.time() - self.performance_stats['start_time']
        
        return ServerStatus(
            server_id=self.server_id,
            current_task=self.current_task.task_id if self.current_task else None,
            current_round=self.current_round,
            total_tasks=len(self.task_queue) + len(self.completed_tasks),
            completed_tasks=len(self.completed_tasks),
            registered_clients=len(self.client_manager.get_all_clients()) if self.client_manager else 0,
            active_clients=len(self.client_manager.get_active_clients()) if self.client_manager else 0,
            global_model_version=self.global_model_version,
            uptime=uptime,
            server_state=self.state_manager.get_current_state().name if self.state_manager else "UNKNOWN"
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        total_rounds = self.performance_stats['successful_轮次'] + self.performance_stats['failed_轮次']
        success_rate = (self.performance_stats['successful_轮次'] / total_rounds) if total_rounds > 0 else 0.0
        
        return {
            "total_轮次": total_rounds,
            "successful_轮次": self.performance_stats['successful_轮次'],
            "failed_轮次": self.performance_stats['failed_轮次'],
            "success_rate": success_rate,
            "average_round_time": self.performance_stats['average_round_time'],
            "uptime": time.time() - self.performance_stats['start_time']
        }
    
    # ===== 改进的内部实现方法 =====
    
    def _execute_round_with_retry(self, round_id: int, attempt: int) -> Optional[RoundResult]:
        """执行单轮联邦学习（带重试）"""
        try:
            self.logger.info(f"执行轮次 {round_id}, attempt {attempt + 1}")
            
            # 1. 选择参与客户端（带超时）
            participating_clients = self._select_participating_clients_with_timeout()
            if not participating_clients:
                raise ServerError("No clients available for round")
            self.logger.info(f"步骤 1 completed: Selected {len(participating_clients)} 客户端")
            
            # 2. 准备轮次状态
            self._prepare_round_state(participating_clients)
            self.logger.info(f"步骤 2 completed: 轮次状态准备完成")
            
            # 3. 分发全局模型（带确认）
            self.logger.info(f"步骤 3 starting: Model distribution to {len(participating_clients)} 客户端")
            distribution_success = self._distribute_global_model_with_confirmation(participating_clients)
            if not distribution_success:
                raise ServerError("Model distribution failed")
            self.logger.info(f"步骤 3 completed: Model distribution successful")
            
            # 4. 触发客户端训练（带超时监控）
            self.logger.info(f"步骤 4 starting: Triggering client training")
            self._trigger_client_training_with_monitoring(participating_clients, round_id)
            self.logger.info(f"步骤 4 completed: Client training triggered")
            
            # 状态转换：COORDINATING -> WAITING_FOR_UPDATES
            if self.state_manager:
                self.state_manager.transition_to(ServerState.WAITING_FOR_UPDATES, {
                    "participating_客户端": participating_clients,
                    "timeout": self.round_config.timeout,
                    "round_id": round_id
                })
            
            # 5. 设置轮次超时
            self._set_round_timeout(round_id)
            
            # 轮次执行成功启动，等待聚合完成
            # 返回None表示轮次已启动但尚未完成
            return None
            
        except Exception as e:
            self.logger.error(f"轮次 {round_id} attempt {attempt + 1} failed: {e}")
            self._cleanup_failed_round()
            raise
    
    def _select_participating_clients_with_timeout(self) -> List[str]:
        """选择参与客户端（带超时）"""
        try:
            start_time = time.time()
            
            # 获取可用客户端
            available_clients = self.client_manager.get_active_clients()
            
            if time.time() - start_time > self.round_config.client_selection_timeout:
                self.logger.warning("Client selection timeout")
                return []
            
            # 使用ClientManager选择客户端
            if len(available_clients) == 0:
                self.logger.warning("No active clients available")
                return []
            
            # 简单选择所有可用客户端，或根据配置选择子集
            client_fraction = self.server_config.get('federation', {}).get('client_fraction', 1.0)
            num_to_select = max(1, int(len(available_clients) * client_fraction))
            
            selected = available_clients[:num_to_select]  # 简单选择前N个
            
            self.logger.info(f"选择 {len(selected)} clients for round {self.current_round}: {selected}")
            return selected
            
        except Exception as e:
            self.logger.error(f"Failed to select participating clients: {e}")
            return []
    
    def _prepare_round_state(self, participating_clients: List[str]) -> None:
        """准备轮次状态"""
        self.logger.debug(f"Entering _prepare_round_state with {len(participating_clients)} 客户端")
        try:
            self.logger.debug(f"About to acquire _client_updates_lock")
            with self._client_updates_lock:
                self.logger.debug(f"Acquired _client_updates_lock successfully")
                self._expected_clients = set(participating_clients)
                self._pending_client_updates.clear()
                self.logger.debug(f" Set expected clients and cleared pending updates")
                
                # 清理上一轮的更新数据
                round_key_prefix = f"round_{self.current_round}_"
                self.logger.debug(f"About to get all federation states")
                old_keys = list(self.context.get_all_states(scope="federation").keys())
                self.logger.debug(f"Found {len(old_keys)} federation states to check for cleanup")
                
                cleanup_count = 0
                for key in old_keys:
                    if key.startswith(round_key_prefix):
                        self.context.remove_state(key, scope="federation")
                        cleanup_count += 1
                        
                self.logger.debug(f"DEBUG: Round state cleanup completed, removed {cleanup_count} old states")
        except Exception as e:
            self.logger.error(f"ERROR in _prepare_round_state: {e}")
            raise
    
    def _distribute_global_model_with_confirmation(self, client_ids: List[str]) -> bool:
        """分发全局模型到指定客户端（带确认）"""
        try:
            self.logger.info(f"Starting global model distribution to {len(client_ids)} 客户端")
            
            if not hasattr(self.model_manager, 'get_current_model'):
                self.logger.warning("模型管理器 doesn't support getting current model")
                return False
            
            self.logger.debug("Getting current model from model manager...")
            global_model = self.model_manager.get_current_model()
            if global_model is None:
                self.logger.warning("No global model available for distribution")
                return False
            
            self.logger.debug("Creating model data structure...")
            model_data = {
                "model_state": global_model.state_dict(),
                "model_version": self.global_model_version,
                "auxiliary_models": {
                    name: model.state_dict() 
                    for name, model in self.auxiliary_models.items()
                }
            }
            
            self.logger.debug(f"Model data created, starting distribution to {len(client_ids)} 客户端")
            
            # 向选中的客户端发送模型
            successful_distributions = 0
            for client_id in client_ids:
                try:
                    self.send_message(
                        target=client_id,
                        message_type=MessageType.MODEL_DISTRIBUTION,
                        data=model_data,
                        metadata={
                            "round_id": self.current_round,
                            "model_version": self.global_model_version,
                            "server_id": self.server_id
                        }
                    )
                    successful_distributions += 1
                except Exception as e:
                    self.logger.error(f"Failed to send model to client {client_id}: {e}")
            
            success_rate = successful_distributions / len(client_ids)
            self.logger.info(f"Global model distributed to {successful_distributions}/{len(client_ids)} 客户端")
            
            # 在模型分发后进行评估前置评估（如果配置了）
            if success_rate >= 0.5:
                self._log_model_distribution_with_evaluation()
            
            # 如果成功率太低，认为分发失败
            return success_rate >= 0.5  # 至少50%成功率
            
        except Exception as e:
            self.logger.error(f"Failed to distribute global model: {e}")
            return False
    
    def _trigger_client_training_with_monitoring(self, client_ids: List[str], round_id: int) -> None:
        """触发客户端训练（带监控）"""
        try:
            training_config = self.server_config.get('training', {})
            
            training_params = {
                "epochs": training_config.get('epochs', 5),
                "batch_size": training_config.get('batch_size', 32),
                "learning_rate": training_config.get('learning_rate', 0.01),
                "timeout": self.round_config.timeout
            }
            
            # 向选中的客户端发送训练触发消息
            successful_triggers = 0
            for client_id in client_ids:
                try:
                    self.send_message(
                        target=client_id,
                        message_type=MessageType.TRAINING_TRIGGER,
                        data=training_params,
                        metadata={
                            "round_id": round_id,
                            "task_id": self.current_task.task_id if self.current_task else None,
                            "server_id": self.server_id
                        }
                    )
                    successful_triggers += 1
                except Exception as e:
                    self.logger.error(f"Failed to trigger training for client {client_id}: {e}")
            
            self.logger.info(f"Training triggered for {successful_triggers}/{len(client_ids)} 客户端")
            
        except Exception as e:
            self.logger.error(f"Failed to trigger client training: {e}")
    
    def _set_round_timeout(self, round_id: int) -> None:
        """设置轮次超时"""
        def timeout_handler():
            self.logger.warning(f"轮次 {round_id} timeout after {self.round_config.timeout}s")
            self._handle_round_timeout(round_id)
        
        # 取消之前的计时器
        if self._round_timer:
            self._round_timer.cancel()
        
        # 设置新的计时器
        self._round_timer = threading.Timer(self.round_config.timeout, timeout_handler)
        self._round_timer.start()
    
    def _handle_round_timeout(self, round_id: int) -> None:
        """处理轮次超时"""
        try:
            with self._client_updates_lock:
                received_updates = len(self._pending_client_updates)
                expected_updates = len(self._expected_clients)
                
                self.logger.warning(f"轮次 {round_id} timeout: received {received_updates}/{expected_updates} updates")
                
                # 如果收到了最小数量的更新，尝试进行聚合
                if received_updates >= self.round_config.min_client_updates:
                    self.logger.info(f"Proceeding with aggregation using {received_updates} updates")
                    self._force_aggregation()
                else:
                    self.logger.error(f"Insufficient updates for aggregation: {received_updates} < {self.round_config.min_client_updates}")
                    self._handle_round_failure("Insufficient client updates")
                    
        except Exception as e:
            self.logger.error(f"Error handling round timeout: {e}")
            self._handle_round_failure(f"Timeout handling error: {e}")
    
    def _validate_client_info(self, client_info: Dict[str, Any]) -> bool:
        """验证客户端信息"""
        return 'client_id' in client_info
    
    def _validate_client_update(self, update_data: Dict[str, Any]) -> bool:
        """验证客户端更新数据"""
        return isinstance(update_data, dict) and 'client_id' in update_data
    
    def _store_client_update(self, client_id: str, round_id: int, update_data: Dict[str, Any]) -> None:
        """存储客户端更新（改进版）"""
        try:
            with self._client_updates_lock:
                # 存储到内存中
                self._pending_client_updates[client_id] = update_data
                
                # 同时存储到上下文中
                update_key = f"round_{round_id}_client_{client_id}_update"
                self.context.set_state(update_key, update_data, scope="federation")
                
                self.logger.debug(f"Stored update from client {client_id} for round {round_id}")
                
        except Exception as e:
            self.logger.error(f"Failed to store client update: {e}")
    
    def _check_aggregation_conditions(self) -> None:
        """检查是否可以开始聚合（改进版）"""
        try:
            # 在独立的代码块中获取聚合状态，避免锁嵌套
            aggregation_ready = False
            received_count = 0
            expected_count = 0
            
            with self._client_updates_lock:
                received_count = len(self._pending_client_updates)
                expected_count = len(self._expected_clients)
                
                self.logger.debug(f"Aggregation check: {received_count}/{expected_count} updates received")
                
                # 检查是否收集到足够的更新
                min_updates = max(self.round_config.min_client_updates, 1)
                
                if received_count >= min_updates:
                    # 检查是否收到所有期待的更新，或者是否已经等待足够长时间
                    all_received = received_count == expected_count
                    wait_time = time.time() - (self._round_start_time or time.time())
                    max_wait_exceeded = wait_time > self.round_config.max_wait_time
                    
                    aggregation_ready = all_received or max_wait_exceeded
                    
            # 在锁外进行聚合，避免死锁
            if aggregation_ready:
                self.logger.info(f"Starting aggregation with {received_count} updates")
                self._start_aggregation_improved()
                        
        except Exception as e:
            self.logger.error(f"Failed to check aggregation conditions: {e}")
    
    def _start_aggregation_improved(self) -> None:
        """开始聚合过程（改进版）"""
        # 预先标记客户端数据已被提取，避免重复清理
        client_data_extracted = False
        
        self.logger.debug(f"INFO: Entering _start_aggregation_improved, about to acquire aggregation lock")
        
        with self._aggregation_lock:
            self.logger.debug(f"INFO: Acquired aggregation lock, starting aggregation process")
            try:
                # 取消轮次超时计时器
                if self._round_timer:
                    self._round_timer.cancel()
                    self._round_timer = None
                
                # 快速获取客户端更新数据，然后立即释放锁
                client_updates = []
                participating_clients = []
                
                self.logger.debug(f"About to acquire client updates lock")
                
                with self._client_updates_lock:
                    self.logger.debug(f"Acquired client updates lock, extracting client data")
                    client_updates = list(self._pending_client_updates.values())
                    participating_clients = list(self._pending_client_updates.keys())
                    # 在锁内清理数据，避免后续长时间持有锁
                    self._pending_client_updates.clear()
                    self._expected_clients.clear()
                    client_data_extracted = True
                    self.logger.debug(f"Extracted {len(client_updates)} client updates, cleared pending data")
                
                if not client_updates:
                    raise ServerError("No client updates available for aggregation")
                
                self.logger.info(f"Starting aggregation for round {self.current_round} with {len(client_updates)} updates")
                
                # 聚合客户端传递的评估结果（而不是重复计算）
                client_evaluation_summary = None
                if any('evaluation_results' in update for update in client_updates):
                    client_evaluation_summary = self._aggregate_client_evaluation_results(client_updates)
                
                self.logger.info(f"About to transition to AGGREGATING state")
                # 状态转换：WAITING_FOR_UPDATES -> AGGREGATING
                if self.state_manager:
                    self.state_manager.transition_to(ServerState.AGGREGATING, {
                        "num_updates": len(client_updates),
                        "round_id": self.current_round,
                        "timestamp": time.time()
                    })
                    self.logger.debug(f"Successfully transitioned to AGGREGATING state")
                
                self.logger.debug(f"About to check model_manager and call update_global_model")
                # 委托给模型管理器执行聚合
                if hasattr(self.model_manager, 'update_global_model'):
                    try:
                        # 更新全局模型（通过配置的聚合器）
                        self.logger.info(f"Starting model aggregation for round {self.current_round}")
                        self.logger.debug(f"About to call model_manager.update_global_model with {len(client_updates)} updates")
                        
                        # 检查client_updates格式
                        for i, update in enumerate(client_updates[:2]):  # 只检查前2个
                            self.logger.info(f"client_update[{i}] keys: {list(update.keys())}")
                            if 'aggregated_model_update' in update:
                                self.logger.info(f"client_update[{i}] aggregated_model_update keys: {list(update['aggregated_model_update'].keys())}")
                        
                        aggregated_model = self.model_manager.update_global_model(client_updates)
                        self.logger.info(f"model_manager.update_global_model 成功完成")
                        self.global_model_version += 1
                        self.logger.info(f"Global model version incremented to {self.global_model_version}")
                        
                        # 保存最新聚合模型的引用，供checkpoint hook使用
                        self._last_aggregated_model = aggregated_model
                        self.logger.debug(f"Saved aggregated model reference for checkpoint hook: {type(aggregated_model)}")
                        
                        # 计算简单的聚合指标
                        self.logger.debug(f"Starting aggregation metrics calculation")
                        client_results = {
                            f"client_{i}": update.get('training_metrics', {}) 
                            for i, update in enumerate(client_updates)
                        }
                        enhanced_metrics = {}
                        if client_results:
                            # 简单平均计算
                            all_accuracies = [metrics.get('accuracy', 0.0) for metrics in client_results.values() if 'accuracy' in metrics]
                            all_losses = [metrics.get('loss', 0.0) for metrics in client_results.values() if 'loss' in metrics]
                            if all_accuracies:
                                enhanced_metrics['accuracy'] = sum(all_accuracies) / len(all_accuracies)
                            if all_losses:
                                enhanced_metrics['loss'] = sum(all_losses) / len(all_losses)
                        self.logger.debug(f"Aggregation metrics calculation 完成")
                        
                        # 创建轮次结果
                        self.logger.debug(f"Creating round result")
                        round_result = self._create_round_result_improved(client_updates, aggregated_model, participating_clients, enhanced_metrics)
                        self.round_history.append(round_result)
                        self.logger.debug(f"Round result created and added to history")
                        
                        # 服务端评估：在模型聚合完成后进行，同时传递客户端评估聚合结果
                        server_evaluation_results = self._perform_server_evaluation(aggregated_model, client_evaluation_summary)
                        
                        # 简单的收敛检查
                        convergence_info = {"is_converged": False, "improvement": 0.0}
                        if len(self.round_history) >= 2:
                            current_acc = round_result.aggregation_metrics.get('accuracy', 0.0)
                            prev_acc = self.round_history[-2].aggregation_metrics.get('accuracy', 0.0)
                            improvement = current_acc - prev_acc
                            convergence_info = {"is_converged": abs(improvement) < 0.01, "improvement": improvement}
                        
                        round_result.convergence_info = convergence_info
                        # 将服务端评估结果添加到轮次结果中
                        round_result.metadata['server_evaluation'] = server_evaluation_results
                        # 将客户端评估聚合结果添加到轮次结果中
                        if client_evaluation_summary:
                            round_result.metadata['client_evaluation_summary'] = client_evaluation_summary
                        
                        # 更新性能统计
                        self._update_performance_stats(round_result)
                        
                        # 状态转换：AGGREGATING -> VALIDATING_AGGREGATION -> DISTRIBUTING -> BROADCASTING -> ROUND_COMPLETED
                        if self.state_manager:
                            # 首先转换到验证聚合状态
                            self.state_manager.transition_to(ServerState.VALIDATING_AGGREGATION, {
                                "round_id": self.current_round,
                                "aggregation_success": True,
                                "timestamp": time.time()
                            })
                            # 然后转换到分发状态
                            self.state_manager.transition_to(ServerState.DISTRIBUTING, {
                                "round_id": self.current_round,
                                "timestamp": time.time()
                            })
                            # 接着转换到广播状态
                            self.state_manager.transition_to(ServerState.BROADCASTING, {
                                "round_id": self.current_round,
                                "timestamp": time.time()
                            })
                            # 最后转换到轮次完成状态
                            self.state_manager.transition_to(ServerState.ROUND_COMPLETED, {
                                "round_id": self.current_round,
                                "aggregation_success": True,
                                "is_converged": convergence_info.get("is_converged", False),
                                "timestamp": time.time()
                            })
                        
                        self.logger.debug(f"Aggregation completed for round {self.current_round}")
                        
                        # 发布轮次完成事件
                        self.context.publish_event("round_完成", {
                            "server_id": self.server_id,
                            "round_id": self.current_round,
                            "num_updates": len(client_updates),
                            "model_version": self.global_model_version,
                            "round_time": round_result.round_time,
                            "convergence_info": convergence_info
                        })
                        
                        # 执行轮次完成后的hooks（如checkpoint保存）
                        self._execute_after_round_hooks(round_result, convergence_info)
                        
                        # 清理轮次状态（不再需要，因为已经在上面清理了）
                        # self._cleanup_round_state()
                        self.logger.debug(f"轮次 state already cleaned up during aggregation start")
                        
                    except Exception as e:
                        self.logger.error(f"Model aggregation failed: {e}")
                        self._handle_aggregation_failure(e, client_data_extracted)
                else:
                    self.logger.error("模型管理器 doesn't support update_global_model")
                    self._handle_aggregation_failure(Exception("模型管理器 not compatible"), client_data_extracted)
                
            except Exception as e:
                self.logger.error(f"Aggregation failed for round {self.current_round}: {e}")
                self._handle_aggregation_failure(e, client_data_extracted)
    
    def _create_round_result_improved(self, client_updates: List[Dict[str, Any]], 
                                    aggregated_model: torch.nn.Module,
                                    participating_clients: List[str],
                                    enhanced_metrics: Dict[str, float] = None) -> RoundResult:
        """创建轮次结果（改进版）"""
        # 计算轮次时间
        round_time = 0.0
        if self._round_start_time:
            round_time = time.time() - self._round_start_time
        
        # 提取客户端指标
        client_metrics = {}
        for i, update in enumerate(client_updates):
            if i < len(participating_clients):
                client_id = participating_clients[i]
                metrics = update.get('training_metrics', {})
                client_metrics[client_id] = metrics
        
        # 使用增强指标或计算聚合指标
        aggregation_metrics = enhanced_metrics or self._calculate_aggregation_metrics_improved(client_updates)
        
        return RoundResult(
            round_id=self.current_round,
            task_id=self.current_task.task_id if self.current_task else "unknown",
            participating_clients=participating_clients,
            aggregated_model=aggregated_model,
            client_metrics=client_metrics,
            aggregation_metrics=aggregation_metrics,
            round_time=round_time,
            convergence_info={},  # 将在后续设置
            knowledge_transfer_info={},
            metadata={
                "server_id": self.server_id,
                "model_version": self.global_model_version,
                "timestamp": time.time()
            }
        )
    
    def _calculate_aggregation_metrics_improved(self, client_updates: List[Dict[str, Any]]) -> Dict[str, float]:
        """计算聚合指标（改进版）"""
        if not client_updates:
            return {}
        
        try:
            # 基础统计
            total_samples = sum(update.get('total_samples', 0) for update in client_updates)
            avg_training_time = sum(update.get('total_training_time', 0) for update in client_updates) / len(client_updates)
            
            # 计算指标统计
            all_losses = []
            all_accuracies = []
            
            for update in client_updates:
                metrics = update.get('training_metrics', {})
                if 'average_loss' in metrics:
                    all_losses.append(metrics['average_loss'])
                elif 'loss' in metrics:
                    all_losses.append(metrics['loss'])
                    
                if 'average_accuracy' in metrics:
                    all_accuracies.append(metrics['average_accuracy'])
                elif 'accuracy' in metrics:
                    all_accuracies.append(metrics['accuracy'])
            
            aggregated_metrics = {
                "total_samples": float(total_samples),
                "avg_training_time": avg_training_time,
                "num_participants": float(len(client_updates)),
                "aggregation_timestamp": time.time()
            }
            
            # 添加损失和准确率统计
            if all_losses:
                aggregated_metrics["avg_loss"] = sum(all_losses) / len(all_losses)
                aggregated_metrics["min_loss"] = min(all_losses)
                aggregated_metrics["max_loss"] = max(all_losses)
            
            if all_accuracies:
                aggregated_metrics["avg_accuracy"] = sum(all_accuracies) / len(all_accuracies)
                aggregated_metrics["min_accuracy"] = min(all_accuracies)
                aggregated_metrics["max_accuracy"] = max(all_accuracies)
            
            return aggregated_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating aggregation metrics: {e}")
            return {"error": str(e)}
    
    def _calculate_convergence_info(self) -> Dict[str, Any]:
        """计算收敛信息"""
        try:
            if len(self.round_history) < 2:
                return {"status": "insufficient_data"}
            
            # 简单的收敛检测
            recent_rounds = self.round_history[-3:]  # 最近3轮
            if len(recent_rounds) >= 2:
                accuracies = [r.aggregation_metrics.get("avg_accuracy", 0) for r in recent_rounds]
                if len(accuracies) >= 2:
                    improvement = accuracies[-1] - accuracies[0]
                    is_converging = abs(improvement) < 0.01  # 1%的改进阈值
                    
                    return {
                        "status": "converging" if is_converging else "improving",
                        "improvement": improvement,
                        "current_accuracy": accuracies[-1]
                    }
            
            return {"status": "unknown"}
            
        except Exception as e:
            self.logger.error(f"Error calculating convergence info: {e}")
            return {"status": "error", "error": str(e)}
    
    def _update_performance_stats(self, round_result: RoundResult) -> None:
        """更新性能统计"""
        try:
            self.performance_stats['successful_轮次'] += 1
            
            # 更新平均轮次时间
            total_time = self.performance_stats['average_round_time'] * (self.performance_stats['successful_轮次'] - 1)
            total_time += round_result.round_time
            self.performance_stats['average_round_time'] = total_time / self.performance_stats['successful_轮次']
            
        except Exception as e:
            self.logger.error(f"Error updating performance stats: {e}")
    
    def _handle_round_failure(self, error_msg: str) -> None:
        """处理轮次失败"""
        try:
            self.logger.error(f"轮次 {self.current_round} failed: {error_msg}")
            self.performance_stats['failed_轮次'] += 1
            
            # 状态转换到错误状态
            if self.state_manager:
                self.state_manager.transition_to(ServerState.ERROR, {
                    "action": "round_failed",
                    "round_id": self.current_round,
                    "error": error_msg,
                    "timestamp": time.time()
                })
            
            # 清理轮次状态
            self._cleanup_round_state()
            
        except Exception as e:
            self.logger.error(f"Error handling round failure: {e}")
    
    def _handle_aggregation_failure(self, error: Exception, client_data_extracted: bool = False) -> None:
        """处理聚合失败"""
        try:
            error_msg = str(error)
            self.logger.error(f"Aggregation failed for round {self.current_round}: {error_msg}")
            
            # 状态转换到错误状态
            if self.state_manager:
                self.state_manager.transition_to(ServerState.ERROR, {
                    "action": "aggregation_failed",
                    "round_id": self.current_round,
                    "error": error_msg,
                    "timestamp": time.time()
                })
            
            # 清理轮次状态 - 只有在客户端数据没有被提取时才清理
            if not client_data_extracted:
                self._cleanup_round_state()
            else:
                # 只清理计时器，客户端数据已经在聚合开始时清理了
                try:
                    if self._round_timer:
                        self._round_timer.cancel()
                        self._round_timer = None
                    self._round_start_time = None
                except Exception as cleanup_e:
                    self.logger.warning(f"Error during minimal cleanup: {cleanup_e}")
            
        except Exception as e:
            self.logger.error(f"Error handling aggregation failure: {e}")
    
    def _force_aggregation(self) -> None:
        """强制开始聚合（超时情况下）"""
        try:
            self.logger.info(f"Force starting aggregation for round {self.current_round}")
            self._start_aggregation_improved()
        except Exception as e:
            self.logger.error(f"Force aggregation failed: {e}")
            self._handle_round_failure(f"Force aggregation failed: {e}")
    
    def _cleanup_round_state(self) -> None:
        """清理轮次状态"""
        try:
            # 使用超时机制安全清理客户端更新数据
            lock_acquired = False
            try:
                # 尝试在1秒内获取锁
                lock_acquired = self._client_updates_lock.acquire(timeout=1.0)
                if lock_acquired:
                    # 安全清理：只有在有数据时才清理
                    if self._pending_client_updates:
                        self._pending_client_updates.clear()
                    if self._expected_clients:
                        self._expected_clients.clear()
                else:
                    self.logger.warning("Failed to acquire client updates lock for cleanup (timeout)")
            except Exception as e:
                self.logger.warning(f"Error acquiring client updates lock for cleanup: {e}")
            finally:
                if lock_acquired:
                    try:
                        self._client_updates_lock.release()
                    except Exception as e:
                        self.logger.warning(f"Error releasing client updates lock: {e}")
            
            # 清理计时器
            if self._round_timer:
                self._round_timer.cancel()
                self._round_timer = None
            
            self._round_start_time = None
            
        except Exception as e:
            self.logger.error(f"Error cleaning up round state: {e}")
    
    def _cleanup_failed_round(self) -> None:
        """清理失败的轮次"""
        self._cleanup_round_state()
    
    def _cancel_current_round(self) -> None:
        """取消当前轮次"""
        try:
            with self._round_lock:
                if self._round_timer:
                    self._round_timer.cancel()
                    self._round_timer = None
                
                self._cleanup_round_state()
                self.logger.info(f"Cancelled current round {self.current_round}")
                
        except Exception as e:
            self.logger.error(f"Error cancelling current round: {e}")
    
    # ===== 保持原有的其他方法（略去，与原代码相同） =====
    
    def add_task(self, task_info: TaskInfo) -> bool:
        """添加任务到任务队列"""
        try:
            self.task_queue.append(task_info)
            self.logger.info(f"Task added to queue: {task_info.task_id}")
            
            # 发布任务添加事件
            self.context.publish_event("task_added", {
                "task_id": task_info.task_id,
                "task_type": task_info.task_type,
                "sequence_position": task_info.sequence_position
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add task: {e}")
            return False
    
    def start_next_task(self) -> Optional[TaskInfo]:
        """启动下一个任务"""
        try:
            if not self.startup_ready:
                self.logger.warning("Server not ready, cannot start task")
                return None
            
            if not self.task_queue:
                self.logger.info("No tasks in queue")
                return None
            
            # 获取下一个任务
            next_task = self.task_queue.popleft()
            self.current_task = next_task
            
            # 状态转换：当前状态 -> TRAINING
            if self.state_manager:
                self.state_manager.transition_to(ServerState.TRAINING, {
                    "task_id": next_task.task_id,
                    "task_type": next_task.task_type,
                    "timestamp": time.time()
                })
            
            self.logger.info(f"Starting task: {next_task.task_id}")
            
            # 分发任务到客户端
            self._distribute_task(next_task)
            
            return next_task
            
        except Exception as e:
            self.logger.error(f"Failed to start next task: {e}")
            
            # 状态转换到错误状态
            if self.state_manager:
                self.state_manager.transition_to(ServerState.ERROR, {
                    "action": "task_start_failed",
                    "error": str(e),
                    "timestamp": time.time()
                })
            
            return None
    
    # ===== 其他原有方法保持不变 =====
    
    def _create_execution_context(self, config: DictConfig) -> ExecutionContext:
        """创建执行上下文"""
        from ...core.execution_context import ExecutionContext
        
        context_config = config.get('context', {})
        experiment_id = f"server_experiment_{self.server_id}"
        
        context = ExecutionContext(
            config=OmegaConf.create(context_config),
            experiment_id=experiment_id
        )
        
        # 存储完整配置
        context._server_config = config
        
        # 设置实验目录信息（优先从配置中获取，然后从实例属性）
        shared_experiment_dir = config.get('experiment.shared_experiment_dir')
        if shared_experiment_dir:
            context._base_experiment_dir = str(shared_experiment_dir)
            context._shared_experiment_dir = str(shared_experiment_dir)
            logger.debug(f"Server context: using shared_experiment_dir from config: {shared_experiment_dir}")
        elif hasattr(self, '_experiment_dir'):
            context._base_experiment_dir = str(self._experiment_dir)
            context._shared_experiment_dir = str(self._experiment_dir)
            logger.debug(f"Server context: using _experiment_dir from instance: {self._experiment_dir}")
        else:
            logger.warning("Server context: no experiment directory found in config or instance")
        
        return context
    
    def _create_aggregator(self, aggregator_config: Dict[str, Any], context: ExecutionContext) -> BaseAggregator:
        """创建聚合器"""
        from ...registry.component_composer import ComponentComposer
        from ...registry import registry
        
        composer = ComponentComposer(registry)
        
        # 包装聚合器配置
        config = OmegaConf.create({'aggregator': aggregator_config})
        
        aggregator = composer.create_aggregator(config, context)
        
        return aggregator
    
    def _create_model_manager(self, model_config: Dict[str, Any], context: ExecutionContext) -> ModelManager:
        """创建模型管理器"""
        from ..managers.model_manager import ModelManager
        
        # 使用默认聚合器创建模型管理器
        model_manager = ModelManager(
            config=OmegaConf.create(model_config),
            aggregator=self.aggregator
        )
        
        return model_manager
    
    def _create_client_manager(self, client_config: Dict[str, Any], context: ExecutionContext) -> ClientManager:
        """创建客户端管理器"""
        from ..managers.client_manager import ClientManager
        
        client_manager = ClientManager(
            config=OmegaConf.create(client_config)
        )
        
        return client_manager
    
    def _create_federation_engine(self, federation_config: Dict[str, Any], context: ExecutionContext) -> 'FederationEngine':
        """创建轻量化联邦引擎"""
        from ...engine.federation_engine import FederationEngine
        
        federation_engine = FederationEngine(context, federation_config)
        
        return federation_engine
    
    def _create_evaluators(self, evaluators_config: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        """创建服务端评估器"""
        evaluators = {}
        
        # 如果没有配置评估器，返回空字典
        if not evaluators_config:
            self.logger.debug("服务端未配置评估器")
            return evaluators
        
        # 直接创建评估器，不使用复杂的组件组合器
        from ...implementations.evaluators.accuracy_evaluator import AccuracyEvaluator
        
        try:
            for evaluator_name, evaluator_config in evaluators_config.items():
                try:
                    # 获取评估器配置中的类型
                    evaluator_type = evaluator_config.get('evaluator', {}).get('class', 'accuracy')
                    
                    if evaluator_type == 'accuracy':
                        # 直接创建AccuracyEvaluator实例
                        evaluator = AccuracyEvaluator(context, evaluator_config)
                        evaluators[evaluator_name] = evaluator
                        self.logger.debug(f"创建服务端评估器: {evaluator_name} (类型: {evaluator_type})")
                    else:
                        self.logger.warning(f"不支持的评估器类型: {evaluator_type}")
                        
                except Exception as e:
                    self.logger.error(f"创建评估器 {evaluator_name} 失败: {e}")
                    
        except Exception as e:
            self.logger.error(f"初始化服务端评估器失败: {e}")
                
        return evaluators
    
    def _create_test_data_loaders(self, test_datas_config: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        """创建服务端测试数据加载器"""
        test_data_loaders = {}
        
        # 如果没有配置测试数据，返回空字典
        if not test_datas_config:
            self.logger.debug("服务端未配置测试数据")
            return test_data_loaders
        
        try:
            from ...data.dataloader_factory import dataloader_factory
            
            for data_name, data_config in test_datas_config.items():
                try:
                    # 创建数据加载器
                    dataloader = dataloader_factory.create_dataloader(data_name, data_config)
                    test_data_loaders[data_name] = dataloader
                    self.logger.debug(f"创建服务端测试数据加载器: {data_name}")
                    
                except Exception as e:
                    self.logger.error(f"创建测试数据加载器 {data_name} 失败: {e}")
                    
        except Exception as e:
            self.logger.error(f"初始化服务端测试数据加载器失败: {e}")
                
        return test_data_loaders
    
    def _initialize_state_management(self) -> 'StateManager':
        """初始化服务端状态管理"""
        try:
            
            state_manager = StateManager(
                initial_state=ServerState.INITIALIZING,
                context=self.context,
                component_id=self.server_id,
                max_history=self.server_config.get('state_management', {}).get('max_history', 1000)
            )
            
            # 注册状态变化回调
            self._register_state_callbacks(state_manager)
            
            # 启用自动汇报重要状态
            important_states = [
                ServerState.READY,
                ServerState.TRAINING,
                ServerState.AGGREGATING,
                ServerState.ROUND_COMPLETED,
                ServerState.ERROR
            ]
            state_manager.enable_auto_reporting(important_states)
            
            self.logger.debug(f"Server {self.server_id} state management initialized")
            return state_manager
            
        except ImportError:
            self.logger.warning("State management module not available, skipping state management")
            return None
        except Exception as e:
            self.logger.error(f"Failed to initialize state management for server {self.server_id}: {e}")
            raise
    
    def _register_state_callbacks(self, state_manager: 'StateManager') -> None:
        """注册状态变化回调"""
        def state_change_callback(old_state, new_state, metadata):
            """状态变化回调"""
            self.logger.debug(f"Server {self.server_id} state: {old_state} -> {new_state}")
            
            # 更新上下文状态
            self.context.set_state(
                f"server_{self.server_id}_state", 
                new_state.name if hasattr(new_state, 'name') else str(new_state), 
                scope="federation"
            )
            
            # 发布状态变化事件
            self.context.publish_event("server_state_changed", {
                "server_id": self.server_id,
                "old_state": old_state.name if hasattr(old_state, 'name') else str(old_state),
                "new_state": new_state.name if hasattr(new_state, 'name') else str(new_state),
                "metadata": metadata,
                "timestamp": time.time()
            })
        
        # 注册全局状态回调
        state_manager.register_callback(state_change_callback)
    
    def _register_hooks(self, hooks_config: Dict[str, Any]) -> None:
        """注册Hook到联邦引擎"""
        try:
            if not hooks_config:
                self.logger.debug("No hooks configured")
                return
            
            # 简化的Hook注册
            if hasattr(self.federation_engine, 'hook_executor') and self.federation_engine.hook_executor:
                # 这里可以根据hooks_config创建具体的hook实例
                self.logger.debug("Hooks registered to federation engine")
            else:
                self.logger.debug("Federation engine has no hook executor")
            
        except Exception as e:
            self.logger.error(f"Failed to register hooks: {e}")
    
    def _execute_after_round_hooks(self, round_result, convergence_info: Dict[str, Any]) -> None:
        """执行轮次完成后的hooks（如checkpoint保存）"""
        try:
            # 检查是否配置了checkpoint_hook
            hooks_config = self.server_config.get('hooks', {})
            checkpoint_hook_config = hooks_config.get('checkpoint_hook', {})
            
            if not checkpoint_hook_config.get('enabled', False):
                self.logger.debug("checkpoint_hook not enabled for server")
                return
            
            if checkpoint_hook_config.get('phase') != 'after_round':
                self.logger.debug(f"checkpoint_hook phase is {checkpoint_hook_config.get('phase')}, not after_round")
                return
            
            # 创建CheckpointHook实例并执行
            from ...core.checkpoint_hook import CheckpointHook
            from omegaconf import DictConfig
            
            # 获取用户配置并与默认配置合并
            user_checkpoint_config = hooks_config.get('checkpoint', {})
            default_checkpoint_config = {
                'save_frequency': 1,
                'checkpoint_dir': './checkpoints/server',
                'save_model': True,
                'save_optimizer': False,
                'save_experiment_state': True
            }
            
            # 合并配置：用户配置优先，默认配置作为备选
            checkpoint_config = {**default_checkpoint_config, **user_checkpoint_config}
            self.logger.debug(f"Merged checkpoint config: {checkpoint_config}")
            
            hook = CheckpointHook(
                phase='after_round',
                checkpoint_config=DictConfig(checkpoint_config),
                priority=checkpoint_hook_config.get('priority', 0)
            )
            
            # 设置当前轮次到执行上下文
            self.context.set_state('current_round', self.current_round, scope='global')
            
            # 执行hook，传递服务端全局模型
            # 获取全局模型 - 优先从model_manager获取
            global_model = None
            self.logger.debug(f"Attempting to get global model for checkpoint hook")
            
            if hasattr(self, 'model_manager') and self.model_manager:
                try:
                    self.logger.debug(f"Trying to get model from model_manager: {type(self.model_manager)}")
                    global_model = self.model_manager.get_current_model()
                    if global_model is not None:
                        self.logger.debug(f"Successfully got model from model_manager: {type(global_model)}")
                    else:
                        self.logger.debug("model_manager.get_current_model() returned None")
                except Exception as e:
                    self.logger.warning(f"Failed to get model from model_manager: {e}")
                    import traceback
                    self.logger.debug(f"Model manager error traceback: {traceback.format_exc()}")
            else:
                self.logger.debug(f"model_manager not available: hasattr={hasattr(self, 'model_manager')}, manager={getattr(self, 'model_manager', None)}")
            
            # 如果model_manager没有模型，尝试从federation_engine获取
            if global_model is None and hasattr(self, 'federation_engine'):
                self.logger.debug("Trying to get model from federation_engine")
                global_model = getattr(self.federation_engine, 'global_model', None)
                if global_model is not None:
                    self.logger.debug(f"Successfully got model from federation_engine: {type(global_model)}")
                else:
                    self.logger.debug("federation_engine.global_model is None")
            
            # 如果还没有模型，尝试从最近的聚合结果获取
            if global_model is None and hasattr(self, '_last_aggregated_model'):
                self.logger.debug("Trying to get model from _last_aggregated_model")
                global_model = getattr(self, '_last_aggregated_model', None)
                if global_model is not None:
                    self.logger.debug(f"Successfully got model from _last_aggregated_model: {type(global_model)}")
            
            # 最终日志
            if global_model is None:
                self.logger.warning("No global model available for checkpoint hook - checkpoint will not contain model")
            else:
                self.logger.debug(f"Global model ready for checkpoint: {type(global_model)}")
            
            hook_kwargs = {
                'round': self.current_round,
                'model': global_model,
                'convergence_info': convergence_info,
                'round_result': round_result,
                'server_id': self.server_id
            }
            
            if hook.should_execute(self.context, **hook_kwargs):
                self.logger.debug(f"Executing server checkpoint hook for round {self.current_round}")
                hook.execute(self.context, **hook_kwargs)
                self.logger.info(f"Server checkpoint saved for round {self.current_round}")
            else:
                self.logger.debug(f"Server checkpoint hook should not execute for round {self.current_round}")
                
        except Exception as e:
            self.logger.error(f"Failed to execute after_round hooks: {e}")
            import traceback
            self.logger.debug(f"Hook execution traceback: {traceback.format_exc()}")
    
    def _initialize_global_models(self) -> None:
        """初始化全局模型"""
        try:
            # 创建默认模型
            default_model = self._create_default_model()
            
            # 委托给模型管理器初始化全局模型
            if hasattr(self.model_manager, 'initialize_global_model'):
                global_model = self.model_manager.initialize_global_model(default_model)
                self.global_model_version = 1
                
                self.logger.debug("Global model initialized")
            
            # 初始化辅助模型（持续学习特有）
            self._initialize_auxiliary_models()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize global models: {e}")
            raise ServerError(f"Global model initialization failed: {e}")
    
    def _create_default_model(self) -> torch.nn.Module:
        """创建默认模型（使用注册系统）"""
        try:
            # 从配置中获取全局模型参数 - 修复：读取正确的配置路径
            aggregator_config = self.server_config.get('aggregator', {})
            global_model_config = aggregator_config.get('global_model', {})
            
            # 如果没有找到global_model配置，尝试从model配置中读取（向后兼容）
            if not global_model_config:
                global_model_config = self.server_config.get('model', {})
            
            model_type = global_model_config.get('type', 'SimpleMLP')  # 默认使用SimpleMLP
            
            # 根据模型类型创建模型
            if model_type == 'mnist_cnn':
                # 使用mnist_cnn模型（与客户端保持一致）
                try:
                    from ...implementations.factory import ModelFactory
                    model = ModelFactory.create_model(global_model_config)
                    self.logger.debug(f"Created mnist_cnn model using ModelFactory")
                    return model
                except Exception as e:
                    self.logger.warning(f"Failed to create mnist_cnn via ModelFactory: {e}, falling back to SimpleMLP")
            
            # 回退到SimpleMLP或其他已知模型类型
            input_size = global_model_config.get('input_size', 784)  # MNIST
            hidden_sizes = global_model_config.get('hidden_sizes', [256, 128])
            num_classes = global_model_config.get('num_classes', 10)
            dropout_rate = global_model_config.get('dropout_rate', 0.2)

            # 尝试使用注册的模型
            try:
                from ...registry.component_registry import registry
                from ...implementations.models.mnist import SimpleMLP
                
                # 使用注册的SimpleMLP模型（已具备自动展平功能）
                model = SimpleMLP(
                    input_size=input_size,
                    hidden_sizes=hidden_sizes, 
                    num_classes=num_classes,
                    dropout_rate=dropout_rate
                )
                
                self.logger.debug(f"Created registered SimpleMLP model: input={input_size}, hidden={hidden_sizes}, output={num_classes}")
                return model
                
            except ImportError:
                self.logger.warning("Cannot import registered models, falling back to simple model")
                
                # 回退到简单的内联模型
                import torch.nn as nn
                
                layers = []
                prev_size = input_size
                for hidden_size in hidden_sizes:
                    layers.append(nn.Linear(prev_size, hidden_size))
                    layers.append(nn.ReLU())
                    if dropout_rate > 0:
                        layers.append(nn.Dropout(dropout_rate))
                    prev_size = hidden_size
                layers.append(nn.Linear(prev_size, num_classes))
                
                model = nn.Sequential(*layers)
                self.logger.debug(f"Created fallback Sequential model: input={input_size}, hidden={hidden_sizes}, output={num_classes}")
                return model
            
        except Exception as e:
            self.logger.error(f"Failed to create default model: {e}")
            # 最简单的回退模型
            import torch.nn as nn
            return nn.Sequential(
                nn.Linear(784, 128),
                nn.ReLU(),
                nn.Linear(128, 10)
            )
    
    def _initialize_auxiliary_models(self) -> None:
        """初始化辅助模型（持续学习特有）"""
        try:
            auxiliary_config = self.server_config.get('auxiliary_models', {})
            
            for model_name, model_config in auxiliary_config.items():
                # 这里可以根据配置创建不同类型的辅助模型
                # 例如：记忆网络、提示网络、知识蒸馏教师模型等
                self.auxiliary_models[model_name] = self._create_auxiliary_model(model_name, model_config)
                
                self.logger.debug(f"Initialized auxiliary model: {model_name}")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize auxiliary models: {e}")
    
    def _create_auxiliary_model(self, model_name: str, model_config: Dict[str, Any]) -> torch.nn.Module:
        """创建辅助模型"""
        # 简化实现，返回一个简单的线性模型
        # 实际实现应该根据配置创建不同类型的辅助模型
        input_dim = model_config.get('input_dim', 784)
        hidden_dim = model_config.get('hidden_dim', 256)
        output_dim = model_config.get('output_dim', 10)
        
        class SimpleAuxiliaryModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, output_dim)
                )
            
            def forward(self, x):
                return self.layers(x)
        
        return SimpleAuxiliaryModel()
    
    def _load_task_sequence(self) -> None:
        """加载任务序列"""
        try:
            task_sequence_config = self.server_config.get('task_sequence', [])
            
            for i, task_config in enumerate(task_sequence_config):
                task_info = TaskInfo(
                    task_id=task_config.get('task_id', f'task_{i}'),
                    task_type=task_config.get('task_type', 'classification'),
                    dataset_info=task_config.get('dataset', {}),
                    model_config=task_config.get('model', {}),
                    learning_config=task_config.get('learning', {}),
                    sequence_position=i,
                    dependencies=task_config.get('dependencies', []),
                    metadata=task_config.get('metadata', {})
                )
                
                self.task_queue.append(task_info)
            
            self.logger.debug(f"Loaded {len(self.task_queue)} tasks in sequence")
            
        except Exception as e:
            self.logger.warning(f"Failed to load task sequence: {e}")
    
    def _on_startup_ready(self) -> None:
        """启动条件满足时的回调"""
        self.startup_ready = True
        
        # 状态转换：WAITING_FOR_CLIENTS -> READY
        if self.state_manager:
            self.state_manager.transition_to(ServerState.READY, {
                "action": "startup_ready",
                "registered_客户端": len(self.client_manager.get_active_clients()),
                "timestamp": time.time()
            })
        
        # 发布启动就绪事件
        self.context.publish_event("server_startup_ready", {
            "server_id": self.server_id,
            "registered_客户端": len(self.client_manager.get_active_clients()),
            "task_queue_size": len(self.task_queue),
            "timestamp": time.time()
        })
        
        self.logger.info("Server startup ready - federation can begin")
    
    def _distribute_task(self, task_info: TaskInfo) -> None:
        """分发任务到客户端"""
        try:
            available_clients = self.client_manager.get_active_clients()
            
            task_message = {
                "task_id": task_info.task_id,
                "task_type": task_info.task_type,
                "dataset_info": task_info.dataset_info,
                "model_config": task_info.model_config,
                "learning_config": task_info.learning_config,
                "sequence_position": task_info.sequence_position
            }
            
            # 向所有可用客户端发送任务通知
            for client_id in available_clients:
                self.send_message(
                    target=client_id,
                    message_type=MessageType.TASK_NOTIFICATION,
                    data=task_message,
                    metadata={
                        "task_id": task_info.task_id,
                        "server_id": self.server_id
                    }
                )
            
            self.logger.info(f"Task {task_info.task_id} distributed to {len(available_clients)} 客户端")
            
        except Exception as e:
            self.logger.error(f"Failed to distribute task {task_info.task_id}: {e}")
    
    def cleanup_server(self) -> None:
        """清理服务端资源"""
        try:
            self.logger.debug(f"Cleaning up improved server {self.server_id}")
            
            # 取消当前轮次
            self._cancel_current_round()
            
            # 停止通信
            self.stop()
            
            # 清理组件
            if hasattr(self.federation_engine, 'cleanup'):
                self.federation_engine.cleanup()
            
            # 清理状态管理器
            if self.state_manager:
                try:
                    self.state_manager.transition_to(ServerState.COMPLETED, {
                        "action": "server_cleanup",
                        "timestamp": time.time()
                    })
                except Exception as e:
                    self.logger.warning(f"Failed to transition to completed state: {e}")
                finally:
                    self.state_manager = None
            
            # 清理持续学习组件
            self.auxiliary_models.clear()
            self.knowledge_base.clear()
            self.teacher_models.clear()
            
            # 清理轮次管理
            self._pending_client_updates.clear()
            self._expected_clients.clear()
            
            self.logger.debug(f"改进的服务器 {self.server_id} cleanup 完成")
            
        except Exception as e:
            self.logger.error(f"Server cleanup failed: {e}")
    
    def _aggregate_client_evaluation_results(self, client_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        聚合客户端传递的评估结果，避免重复计算
        
        Args:
            client_updates: 客户端更新列表
            
        Returns:
            聚合后的评估结果
        """
        aggregated_results = {
            "total_evaluation_tasks": 0,
            "successful_evaluations": 0,
            "evaluation_metrics": {},
            "timestamp": time.time(),
            "client_count": len(client_updates)
        }
        
        all_accuracies = []
        all_losses = []
        total_samples = 0
        
        try:
            for update in client_updates:
                evaluation_results = update.get('evaluation_results', {})
                if not evaluation_results:
                    continue
                    
                phase_evaluations = evaluation_results.get('phase_evaluations', {})
                for phase_name, phase_eval in phase_evaluations.items():
                    if isinstance(phase_eval, dict):
                        for task_name, task_result in phase_eval.items():
                            if isinstance(task_result, dict):
                                accuracy = task_result.get('accuracy')
                                loss = task_result.get('loss')
                                samples = task_result.get('samples', 0)
                                
                                if accuracy is not None:
                                    all_accuracies.append(accuracy)
                                if loss is not None:
                                    all_losses.append(loss)
                                total_samples += samples
            
            # 计算平均指标
            if all_accuracies:
                avg_accuracy = sum(all_accuracies) / len(all_accuracies)
                aggregated_results["evaluation_metrics"]["aggregated_accuracy"] = {
                    "accuracy": avg_accuracy,
                    "loss": sum(all_losses) / len(all_losses) if all_losses else 'N/A',
                    "total_samples": total_samples,
                    "client_count": len(client_updates)
                }
                aggregated_results["successful_evaluations"] = 1
                aggregated_results["total_evaluation_tasks"] = 1
                
                # 构建日志消息，只显示有效的指标
                log_parts = [f"acc:{avg_accuracy:.3f}"]
                if all_losses:
                    avg_loss = sum(all_losses) / len(all_losses)
                    log_parts.append(f"loss:{avg_loss:.3f}")
                
                self.logger.info(f"📊 [客户端评估聚合] Round {self.current_round} - "
                               f"聚合来自{len(client_updates)}个客户端的评估结果: "
                               f"{', '.join(log_parts)}")
            else:
                self.logger.debug("📊 [客户端评估聚合] 未找到有效的客户端评估结果")
                
        except Exception as e:
            self.logger.error(f"❌ [客户端评估聚合] 聚合客户端评估结果失败: {e}")
            
        return aggregated_results

    def _perform_server_evaluation(self, model: torch.nn.Module, client_evaluation_summary: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        执行服务端评估，结合客户端传递的评估结果
        
        Args:
            model: 聚合后的全局模型
            client_evaluation_summary: 客户端评估结果聚合
            
        Returns:
            评估结果字典
        """
        evaluation_results = {
            "total_evaluation_tasks": 0,
            "successful_evaluations": 0,
            "evaluation_metrics": {},
            "timestamp": time.time()
        }
        
        try:
            # 检查是否配置了评估
            if not self.evaluation_config or not self.evaluators:
                self.logger.debug("🔍 [服务端评估] 未配置评估器或评估任务")
                # 即使没有服务端评估器，也可以展示客户端聚合的结果
                if client_evaluation_summary and client_evaluation_summary.get("evaluation_metrics"):
                    client_metrics = client_evaluation_summary["evaluation_metrics"].get("aggregated_accuracy", {})
                    if client_metrics:
                        acc = client_metrics.get("accuracy", "N/A")
                        loss = client_metrics.get("loss", "N/A")
                        self.logger.info(f"📊 [客户端评估聚合] Round {self.current_round} - "
                                       f"来自客户端的评估结果: acc:{acc:.3f if isinstance(acc, (int, float)) else acc}, "
                                       f"loss:{loss:.3f if isinstance(loss, (int, float)) else loss}")
                return evaluation_results
                
            self.logger.info(f"🏛️ [服务端评估] Round {self.current_round} - 开始评估聚合后的全局模型")
            
            # 获取评估任务配置
            evaluation_tasks = self.evaluation_config.get('tasks', [])
            evaluation_results["total_evaluation_tasks"] = len(evaluation_tasks)
            
            for task in evaluation_tasks:
                try:
                    evaluator_name = task.get('evaluator')
                    test_data = task.get('test_data', 'server_test_data')
                    
                    if evaluator_name not in self.evaluators:
                        self.logger.warning(f"🏛️ [服务端评估] 未找到评估器: {evaluator_name}")
                        continue
                    
                    self.logger.info(f"🔍 [服务端评估] 执行评估任务: {evaluator_name}")
                    
                    evaluator = self.evaluators[evaluator_name]
                    
                    # 获取实际的测试数据
                    if test_data in self.test_data_loaders:
                        data_loader = self.test_data_loaders[test_data]
                    else:
                        self.logger.warning(f"🏛️ [服务端评估] 未找到测试数据: {test_data}")
                        continue
                    
                    # 执行评估
                    if hasattr(evaluator, 'evaluate'):
                        try:
                            result = evaluator.evaluate(model, data_loader)
                            evaluation_results["evaluation_metrics"][evaluator_name] = result
                            evaluation_results["successful_evaluations"] += 1
                            
                            # 提取关键指标用于日志
                            accuracy = result.get('accuracy', 'N/A')
                            loss = result.get('loss', 'N/A')
                            
                            # 构建日志消息，只包含有效的指标
                            log_parts = []
                            if isinstance(accuracy, (int, float)):
                                log_parts.append(f"acc:{accuracy:.3f}")
                            elif accuracy != 'N/A':
                                log_parts.append(f"acc:{accuracy}")
                            
                            if isinstance(loss, (int, float)):
                                log_parts.append(f"loss:{loss:.3f}")
                            elif loss != 'N/A':
                                log_parts.append(f"loss:{loss}")
                            
                            metrics_str = ", ".join(log_parts) if log_parts else "无有效指标"
                            self.logger.info(f"✅ [服务端评估] 评估任务完成: {evaluator_name} - {metrics_str}")
                        except RuntimeError as e:
                            if "mat1 and mat2 shapes cannot be multiplied" in str(e) or "size mismatch" in str(e):
                                self.logger.warning(f"⚠️ [服务端评估] 数据形状不匹配，跳过评估 {evaluator_name}: 请检查服务端测试数据预处理是否与客户端一致")
                            else:
                                self.logger.error(f"🏛️ [服务端评估] 评估任务运行时错误 {evaluator_name}: {e}")
                        except Exception as eval_e:
                            self.logger.error(f"🏛️ [服务端评估] 评估任务失败 {evaluator_name}: {eval_e}")
                    else:
                        self.logger.warning(f"🏛️ [服务端评估] 评估器 {evaluator_name} 没有evaluate方法")
                        
                except Exception as e:
                    self.logger.error(f"🏛️ [服务端评估] 评估任务失败 {evaluator_name}: {e}")
            
            # 创建评估摘要
            if evaluation_results["successful_evaluations"] > 0:
                summary = self._create_server_evaluation_summary(evaluation_results["evaluation_metrics"], client_evaluation_summary)
                self.logger.info(f"📊 [服务端评估] Round {self.current_round} - 全局模型评估完成: {summary}")
            elif client_evaluation_summary and client_evaluation_summary.get("evaluation_metrics"):
                # 即使没有服务端评估器成功，也展示客户端聚合结果
                client_metrics = client_evaluation_summary["evaluation_metrics"].get("aggregated_accuracy", {})
                if client_metrics:
                    acc = client_metrics.get("accuracy", "N/A")
                    loss = client_metrics.get("loss", "N/A")
                    self.logger.info(f"📊 [客户端评估聚合] Round {self.current_round} - "
                                   f"来自客户端的评估结果: acc:{acc:.3f if isinstance(acc, (int, float)) else acc}, "
                                   f"loss:{loss:.3f if isinstance(loss, (int, float)) else loss}")
            
        except Exception as e:
            self.logger.error(f"🏛️ [服务端评估] 服务端评估失败: {e}")
            
        return evaluation_results
    
    def _create_server_evaluation_summary(self, metrics: Dict[str, Any], client_evaluation_summary: Dict[str, Any] = None) -> str:
        """创建服务端评估结果摘要，结合客户端loss值"""
        if not metrics:
            return "无评估结果"
            
        summary_parts = []
        server_acc = None
        server_loss = None
        client_loss = None
        
        # 提取服务端评估的准确率和损失
        for evaluator_name, result in metrics.items():
            if isinstance(result, dict):
                acc = result.get('accuracy')
                loss = result.get('loss')
                if acc is not None:
                    server_acc = acc
                if loss is not None:
                    server_loss = loss
        
        # 获取客户端聚合的loss值
        if client_evaluation_summary and client_evaluation_summary.get("evaluation_metrics"):
            client_metrics = client_evaluation_summary["evaluation_metrics"].get("aggregated_accuracy", {})
            if client_metrics:
                client_loss = client_metrics.get("loss")
        
        # 构建摘要
        if server_acc is not None:
            summary_parts.append(f"acc:{server_acc:.3f}")
        
        # 优先使用客户端传递的loss值，如果没有则使用服务端计算的
        if client_loss is not None and client_loss != 'N/A':
            summary_parts.append(f"loss:{client_loss:.3f}(客户端)")
        elif server_loss is not None and server_loss != 'N/A':
            summary_parts.append(f"loss:{server_loss:.3f}(服务端)")
        # 移除了"loss:N/A"的情况，不显示无效的loss
        
        return f"{len(metrics)}个评估器({', '.join(summary_parts)})" if summary_parts else f"{len(metrics)}个评估器"
    
    def _log_model_distribution_with_evaluation(self):
        """在模型分发时记录评估相关日志"""
        if self.evaluation_config and self.evaluators:
            evaluation_tasks_count = len(self.evaluation_config.get('tasks', []))
            evaluators_count = len(self.evaluators)
            self.logger.info(f"📤 [服务端模型分发] Round {self.current_round} - 模型已分发，准备在聚合后进行服务端评估({evaluators_count}个评估器，{evaluation_tasks_count}个任务)")
        else:
            self.logger.info(f"📤 [服务端模型分发] Round {self.current_round} - 模型已分发到客户端")


# 向后兼容的别名
FederatedServer = ImprovedFederatedServer