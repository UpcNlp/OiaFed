"""
联邦服务端管理器 - 负责服务端组件的初始化和管理（重构版）
fedcl/federation/server.py
"""

from typing import Dict, Any, Optional, List

from ..config import CommunicationConfig, TrainingConfig
from ..exceptions import FederationError
from ..trainer.trainer import BaseTrainer
from ..types import CommunicationMode
from ..utils.auto_logger import get_sys_logger
from .business_initializer import BusinessInitializer
from .communication_initializer import CommunicationInitializer
from .components import CommunicationComponents, ServerBusinessComponents


class FederationServer:
    """
    联邦服务端管理器（薄协调层）

    职责：
        - 接收配置对象（CommunicationConfig + TrainingConfig）
        - 委托初始化器完成通信层和业务层的初始化
        - 建立层间关系
        - 启动/停止服务

    使用方式：
        >>> comm_config = CommunicationConfig(mode="network", role="server")
        >>> train_config = TrainingConfig(trainer={"name": "FedAvgTrainer"})
        >>> server = FederationServer(comm_config, train_config)
        >>> await server.initialize()
        >>> await server.start_server()
    """

    def __init__(
        self,
        communication_config: CommunicationConfig,
        training_config: TrainingConfig,
        server_id: Optional[str] = None,
        logging_config: Optional['LoggingConfig'] = None
    ):
        """
        初始化服务端管理器

        Args:
            communication_config: 通信配置对象
            training_config: 训练配置对象
            server_id: 服务端ID（如果为 None，从配置中读取或自动生成）
            logging_config: 日志配置对象（用于初始化实验跟踪器）
        """
        self.comm_config = communication_config
        self.train_config = training_config
        self.logging_config = logging_config

        # 先设置 mode（_generate_server_id 需要使用）
        self.mode = CommunicationMode(communication_config.mode)

        # 确定 server_id
        self.server_id = server_id or communication_config.node_id or self._generate_server_id()

        # 组件引用
        self.comm_components: Optional[CommunicationComponents] = None
        self.business_components: Optional[ServerBusinessComponents] = None

        # 实验跟踪器（在initialize中创建）
        self.tracker = None

        # 状态管理
        self.is_initialized = False
        self.is_running = False

        # 使用节点特定的运行日志
        from fedcl.utils.auto_logger import get_logger
        self.logger = get_logger("runtime", self.server_id)
        self.logger.info(
            f"FederationServer created: server_id={self.server_id}, mode={self.mode}"
        )

    async def initialize(self) -> bool:
        """
        统一初始化方法（通信层 + 业务层）

        流程：
            1. 初始化通信层（委托给 CommunicationInitializer）
            2. 初始化业务层（委托给 BusinessInitializer）
            3. 建立层间关系

        Returns:
            bool: 初始化是否成功

        Raises:
            FederationError: 如果初始化失败
        """
        if self.is_initialized:
            self.logger.warning("Server already initialized")
            return False

        self.logger.info("Starting FederationServer initialization...")

        try:
            # Phase 1: 初始化通信层（委托给 CommunicationInitializer）
            self.logger.info("1.Initializing communication layer...")
            comm_initializer = CommunicationInitializer(
                self.comm_config,
                self.server_id,
                node_role="server"
            )
            self.comm_components = await comm_initializer.initialize()
            self.logger.info("✓ Communication layer ready")

            # Phase 2: 初始化业务层（委托给 BusinessInitializer）
            self.logger.info("2.Initializing business layer...")
            business_initializer = BusinessInitializer(
                self.train_config,
                node_role="server"
            )
            self.business_components = await business_initializer.initialize_server_components(
                self.server_id
            )
            self.logger.info("✓Business layer ready")

            # Phase 3: 建立层间关系
            self.logger.info("3.Establishing layer relationships...")
            self._establish_layer_relationships()
            self.logger.info("✓Layer relationships established")

            # Phase 4: 初始化实验跟踪器并创建TrackerContext
            if self.logging_config and self.logging_config.tracker.enabled:
                self.logger.info("4.Initializing experiment tracker...")
                await self._initialize_tracker_and_set_context()
                self.logger.info("✓Experiment tracker ready, TrackerContext propagated")
            else:
                self.logger.info("4.Experiment tracker disabled, skipping")

            self.is_initialized = True
            self.logger.info("FederationServer initialized successfully")

            return True

        except Exception as e:
            self.logger.error(f"FederationServer initialization failed: {e}")
            raise FederationError(f"Server initialization failed: {str(e)}")

    def _establish_layer_relationships(self):
        """
        建立层间关系（事件传递链）

        连接链：
            ConnectionManager → BusinessCommunicationLayer → Trainer.ProxyEventHandler
        """
        if not self.comm_components or not self.business_components:
            raise FederationError("Components not initialized")

        if not self.comm_components.business_layer:
            self.logger.warning("No business layer to establish relationships")
            return

        # 连接层间事件传递
        self.comm_components.connection_manager.set_upper_layer(
            self.comm_components.business_layer
        )
        self.comm_components.business_layer.set_upper_layer(
            self.business_components.trainer._proxy_event_handler
        )

        # 🎯 关键修复：监听传输层的CLIENT_REGISTERED事件（内存模式）
        def handle_transport_client_registered(data):
            """处理传输层的客户端注册事件"""
            client_id = data.get("client_id")
            if client_id:
                self.logger.debug(f"[传输层事件桥接] 收到CLIENT_REGISTERED事件: {client_id}")

                # 直接调用ConnectionManager处理层间事件
                self.comm_components.connection_manager.handle_layer_event("CLIENT_REGISTERED", {
                    "client_id": client_id,
                    "event_data": data,
                    "timestamp": data.get("timestamp")
                })

        # 注册传输层事件监听器
        self.comm_components.transport.register_event_listener(
            "system", "CLIENT_REGISTERED", handle_transport_client_registered
        )
        self.logger.debug("[传输层事件桥接] 已注册CLIENT_REGISTERED事件监听器")

        # 🎯 关键修复：监听CommunicationManager的注册事件
        def handle_client_registration_event(event):
            """处理客户端注册事件并转换为层间事件"""
            if event.event_type == "CLIENT_REGISTERED":
                client_id = event.source_id
                self.logger.debug(f"[事件桥接] 转换CLIENT_REGISTERED为层间事件: {client_id}")
                self.logger.debug(f"{event.data}")

                # event.data 是 ClientInfo 对象，不是 dict
                timestamp = None
                if hasattr(event, 'data') and hasattr(event.data, 'registration_time'):
                    timestamp = event.data.registration_time.isoformat()

                self.comm_components.connection_manager.handle_layer_event("CLIENT_REGISTERED", {
                    "client_id": client_id,
                    "event_data": event.data,
                    "timestamp": timestamp
                })

        # 注册事件回调到CommunicationManager的RegistryService
        self.comm_components.communication_manager.registry_service.register_event_callback(
            handle_client_registration_event
        )


        self.logger.info("Layer relationships established, event bridges activated")

    async def _initialize_tracker_and_set_context(self):
        """
        初始化实验跟踪器并创建 TrackerContext 传递给客户端

        流程：
            1. 根据 logging_config 创建 MLflowTracker（自动创建run）
            2. 提取 run_id 和跟踪器配置
            3. 创建 TrackerContext 对象
            4. 调用 communication_manager.set_tracker_context()

        Raises:
            FederationError: 如果跟踪器初始化失败
        """
        try:
            from ..loggers.mlflow_tracker import MLflowTracker
            from ..types import TrackerContext

            tracker_cfg = self.logging_config.tracker

            # 1. 创建 MLflowTracker（自动创建run）
            self.tracker = MLflowTracker(
                experiment_name=self.logging_config.experiment_name,
                run_name=f"federated_{self.server_id}",
                role="aggregator",  # Server角色是聚合器
                tracking_uri=tracker_cfg.config.get('uri'),
                config=tracker_cfg.config
            )

            # 启动tracker（创建run）
            self.tracker.start()

            # 2. 获取run_id
            run_id = self.tracker._run_id
            self.logger.info(f"[TrackerContext] Server创建MLflow run: {run_id}")

            # 3. 创建TrackerContext
            tracker_context = TrackerContext(
                enabled=True,
                tracker_type=tracker_cfg.type,
                shared_run_id=run_id,  # 关键：这是共享的run_id
                config={
                    'tracking_uri': tracker_cfg.config.get('uri'),
                    'experiment_name': self.logging_config.experiment_name,
                    'experiment_id': self.tracker.experiment_id,
                },
                metadata={
                    'server_id': self.server_id,
                    'created_at': str(__import__('datetime').datetime.now())
                }
            )

            # 4. 设置到communication_manager（会在注册响应中发送给客户端）
            if hasattr(self.comm_components.communication_manager, 'set_tracker_context'):
                self.comm_components.communication_manager.set_tracker_context(tracker_context)
                self.logger.info(f"[TrackerContext] 已设置到communication_manager，将在客户端注册时传递")
            else:
                self.logger.warning(
                    f"Communication manager does not support set_tracker_context, "
                    f"clients will not receive TrackerContext"
                )

        except Exception as e:
            self.logger.error(f"Failed to initialize tracker and set context: {e}")
            raise FederationError(f"Tracker initialization failed: {str(e)}")

    async def start_server(self) -> bool:
        """
        启动服务端

        前提：
            必须已调用 initialize() 完成初始化

        Returns:
            bool: 启动是否成功

        Raises:
            FederationError: 如果服务端未初始化
        """
        if not self.is_initialized:
            raise FederationError("Server not initialized. Call initialize() first.")

        if self.is_running:
            self.logger.warning("Server already running")
            return True

        self.logger.info("Starting FederationServer...")

        try:
            # 启动通信层
            self.logger.debug("Starting communication layers...")

            if hasattr(self.comm_components.communication_manager, 'start'):
                await self.comm_components.communication_manager.start()
                self.logger.debug("✓ Communication manager started")

            if hasattr(self.comm_components.connection_manager, 'start'):
                await self.comm_components.connection_manager.start()
                self.logger.debug("✓ Connection manager started")

            # 初始化 trainer
            self.logger.debug("Initializing trainer...")
            trainer_ready = await self.business_components.trainer.initialize()
            if not trainer_ready:
                raise FederationError("Trainer initialization failed")
            self.logger.debug("✓ Trainer initialized")

            self.is_running = True
            self.logger.info("FederationServer started successfully")

            return True

        except Exception as e:
            self.logger.error(f"Failed to start server: {e}")
            return False

    async def stop_server(self) -> bool:
        """
        停止服务端（使用统一的 SHUTDOWN 消息协议）

        流程：
            1. 向所有活跃客户端广播 SHUTDOWN 消息
            2. 等待客户端响应（短暂延迟）
            3. 停止 Trainer
            4. 停止通信层（逐层向下）

        Returns:
            bool: 停止是否成功
        """
        if not self.is_running:
            self.logger.info("Server not running, nothing to stop")
            return True

        self.logger.info("Stopping FederationServer...")

        try:
            # 步骤1: 广播 SHUTDOWN 消息给所有活跃客户端（统一停止协议）
            if self.trainer:
                available_clients = self.trainer.get_available_clients()
                if available_clients:
                    self.logger.info(f"Broadcasting SHUTDOWN to {len(available_clients)} clients...")
                    await self._broadcast_shutdown(available_clients)

                    # 等待客户端处理 SHUTDOWN 消息（给予足够时间）
                    import asyncio
                    await asyncio.sleep(0.5)
                    self.logger.info("✓ SHUTDOWN broadcast completed")

            # 步骤2: 停止 trainer
            if self.business_components and self.business_components.trainer:
                await self.business_components.trainer.cleanup()
                self.logger.info("✓ Trainer stopped")

            # 停止通信层
            if self.comm_components:
                if hasattr(self.comm_components.connection_manager, 'stop'):
                    await self.comm_components.connection_manager.stop()
                    self.logger.debug("✓ Connection manager stopped")

                if hasattr(self.comm_components.communication_manager, 'stop'):
                    await self.comm_components.communication_manager.stop()
                    self.logger.debug("✓ Communication manager stopped")

                if hasattr(self.comm_components.transport, 'stop'):
                    await self.comm_components.transport.stop()
                    self.logger.debug("✓ Transport stopped")

            self.is_running = False
            self.logger.info("FederationServer stopped successfully")

            return True

        except Exception as e:
            self.logger.error(f"Failed to stop server: {e}")
            return False

    async def _broadcast_shutdown(self, client_ids: List[str]):
        """
        广播 SHUTDOWN 消息给指定的客户端列表

        Args:
            client_ids: 客户端ID列表
        """
        if not self.comm_components or not self.comm_components.communication_manager:
            self.logger.warning("Communication manager not available for SHUTDOWN broadcast")
            return

        shutdown_message = {
            "reason": "server_shutdown",
            "timestamp": str(__import__('datetime').datetime.now())
        }

        # 使用 RPC 消息机制广播 SHUTDOWN（与客户端的 register_message_handler 匹配）
        try:
            for client_id in client_ids:
                try:
                    # 使用 send_business_message 而不是 send_control_message
                    # 因为客户端使用 register_message_handler 注册处理器
                    await self.comm_components.communication_manager.send_business_message(
                        client_id,
                        "SHUTDOWN",
                        shutdown_message
                    )
                    self.logger.debug(f"  → Sent SHUTDOWN to {client_id}")
                except Exception as e:
                    self.logger.warning(f"  ✗ Failed to send SHUTDOWN to {client_id}: {e}")
        except Exception as e:
            self.logger.error(f"SHUTDOWN broadcast error: {e}")

    # ========== 便捷访问属性 ==========

    @property
    def trainer(self) -> Optional[BaseTrainer]:
        """获取训练器实例"""
        return self.business_components.trainer if self.business_components else None

    def get_server_status(self) -> Dict[str, Any]:
        """
        获取服务端状态

        Returns:
            服务端状态字典
        """
        return {
            "server_id": self.server_id,
            "mode": self.mode.value,
            "is_initialized": self.is_initialized,
            "is_running": self.is_running,
            "available_clients": len(self.trainer.get_available_clients()) if self.trainer else 0,
            "trainer_status": self.trainer.get_training_status() if self.trainer else None
        }

    def _generate_server_id(self) -> str:
        """生成服务端ID"""
        if self.mode == CommunicationMode.MEMORY:
            return "memory_server"
        elif self.mode == CommunicationMode.PROCESS:
            port = self.comm_config.transport.get("port", 8000) if self.comm_config.transport else 8000
            return f"process_server_{port}"
        elif self.mode == CommunicationMode.NETWORK:
            host = self.comm_config.transport.get("host", "localhost") if self.comm_config.transport else "localhost"
            port = self.comm_config.transport.get("port", 8000) if self.comm_config.transport else 8000
            return f"network_server_{host}_{port}"
        else:
            return "unknown_server"

    # ========== 工厂方法 ==========

    @classmethod
    def create_server(
        cls,
        communication_config: CommunicationConfig,
        training_config: TrainingConfig
    ) -> 'FederationServer':
        """
        工厂方法：创建服务端实例

        Args:
            communication_config: 通信配置对象
            training_config: 训练配置对象

        Returns:
            FederationServer 实例
        """
        return cls(communication_config, training_config)
