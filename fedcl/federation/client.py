"""
联邦客户端管理器 - 负责客户端组件的初始化和管理（重构版）
fedcl/federation/client.py
"""

from typing import Dict, Any, Optional

from ..config import CommunicationConfig, TrainingConfig
from ..exceptions import FederationError
from ..learner.base_learner import BaseLearner
from ..learner.stub import LearnerStub, StubConfig
from ..types import CommunicationMode, RegistrationStatus
from ..utils.auto_logger import get_sys_logger
from .business_initializer import BusinessInitializer
from .communication_initializer import CommunicationInitializer
from .components import CommunicationComponents, ClientBusinessComponents


class FederationClient:
    """
    联邦客户端管理器（薄协调层）

    职责：
        - 接收配置对象（CommunicationConfig + TrainingConfig）
        - 委托初始化器完成通信层和业务层的初始化
        - 创建 LearnerStub
        - 启动/停止客户端

    使用方式：
        >>> comm_config = CommunicationConfig(mode="network", role="client")
        >>> train_config = TrainingConfig(learner={"name": "StandardLearner"})
        >>> client = FederationClient(comm_config, train_config)
        >>> await client.initialize()
        >>> await client.start_client()
    """

    def __init__(
        self,
        communication_config: CommunicationConfig,
        training_config: TrainingConfig,
        client_id: Optional[str] = None,
        logging_config: Optional['LoggingConfig'] = None
    ):
        """
        初始化客户端管理器

        Args:
            communication_config: 通信配置对象
            training_config: 训练配置对象
            client_id: 客户端ID（如果为 None，从配置中读取或自动生成）
            logging_config: 日志配置对象（用于配置实验跟踪器）
        """
        self.comm_config = communication_config
        self.train_config = training_config
        self.logging_config = logging_config

        # 先设置 mode（_generate_client_id 需要使用）
        self.mode = CommunicationMode(communication_config.mode)

        # 确定 client_id
        self.client_id = client_id or communication_config.node_id or self._generate_client_id()

        # 组件引用
        self.comm_components: Optional[CommunicationComponents] = None
        self.business_components: Optional[ClientBusinessComponents] = None
        self.learner_stub: Optional[LearnerStub] = None

        # 实验跟踪器（根据logging_config动态创建）
        self.tracker = None

        # 状态管理
        self.is_initialized = False
        self.is_running = False
        self.is_registered = False

        # 使用节点特定的运行日志
        from fedcl.utils.auto_logger import get_logger
        self.system_logger = get_logger("runtime", self.client_id)
        self.system_logger.info(
            f"FederationClient created: client_id={self.client_id}, mode={self.mode}"
        )

    async def initialize(self) -> bool:
        """
        统一初始化方法（通信层 + 业务层）

        流程：
            1. 初始化通信层（委托给 CommunicationInitializer）
            2. 初始化业务层（委托给 BusinessInitializer）
            3. 创建 LearnerStub

        Returns:
            bool: 初始化是否成功

        Raises:
            FederationError: 如果初始化失败
        """
        if self.is_initialized:
            self.system_logger.warning("Client already initialized")
            return False

        self.system_logger.debug("Starting FederationClient initialization...")

        try:
            # Phase 1: 初始化通信层（委托给 CommunicationInitializer）
            self.system_logger.debug("Phase 1: Initializing communication layer...")
            comm_initializer = CommunicationInitializer(
                self.comm_config,
                self.client_id,
                node_role="client"
            )
            self.comm_components = await comm_initializer.initialize()
            self.system_logger.debug("✓ Phase 1 completed: Communication layer ready")

            # Phase 2: 初始化业务层（委托给 BusinessInitializer）
            self.system_logger.debug("Phase 2: Initializing business layer...")
            business_initializer = BusinessInitializer(
                self.train_config,
                node_role="client"
            )
            self.business_components = await business_initializer.initialize_client_components(
                self.client_id
            )
            self.system_logger.debug("✓ Phase 2 completed: Business layer ready")

            # Phase 3: 创建 LearnerStub
            self.system_logger.debug("Phase 3: Creating LearnerStub...")
            await self._initialize_learner_stub()
            self.system_logger.debug("✓ Phase 3 completed: LearnerStub created")

            self.is_initialized = True
            self.system_logger.debug("FederationClient initialized successfully")

            return True

        except Exception as e:
            self.system_logger.exception(f"FederationClient initialization failed: {e}")
            raise FederationError(f"Client initialization failed: {str(e)}")

    async def _initialize_learner_stub(self):
        """创建 LearnerStub"""
        if not self.business_components or not self.business_components.learner:
            raise FederationError("Learner not initialized")

        # 创建存根配置
        stub_config = StubConfig(
            auto_register=True,
            registration_retry_attempts=self.comm_config.get("registration_retry_attempts", 3),
            registration_retry_delay=self.comm_config.get("registration_retry_delay", 1.0),
            request_timeout=self.comm_config.get("timeout", 120.0),
            max_concurrent_requests=self.comm_config.get("max_concurrent_requests", 5)
        )

        # 创建 LearnerStub 实例
        self.learner_stub = LearnerStub(
            learner=self.business_components.learner,
            communication_manager=self.comm_components.communication_manager,
            connection_manager=self.comm_components.connection_manager,
            config=stub_config
        )

        self.system_logger.debug("LearnerStub created successfully")

    async def _setup_tracker_from_context(self):
        """
        根据 TrackerContext 配置实验跟踪器（使用共享 run）

        流程：
            1. 从 stub 获取 TrackerContext
            2. 根据 logging_config.tracker.type 创建对应的 tracker
            3. 使用 TrackerContext 中的 shared_run_id 配置 tracker
            4. 将 tracker 传递给 learner（用于训练时记录指标）
        """
        # 1. 获取 TrackerContext
        tracker_context = self.learner_stub.get_tracker_context()

        if not tracker_context or not tracker_context.enabled:
            self.system_logger.info("TrackerContext not available or disabled, skipping tracker setup")
            return

        # 2. 检查本地是否有 logging_config
        if not self.logging_config or not self.logging_config.tracker.enabled:
            self.system_logger.warning(
                "Local logging_config disabled, but server sent TrackerContext. "
                "Client will NOT log metrics."
            )
            return

        try:
            tracker_type = self.logging_config.tracker.type
            shared_run_id = tracker_context.shared_run_id

            self.system_logger.info(
                f"[TrackerContext] 配置 {tracker_type} tracker, 使用共享 run: {shared_run_id}"
            )

            # 3. 根据 tracker_type 创建对应的 tracker
            if tracker_type == "mlflow":
                from ..loggers.mlflow_tracker import MLflowTracker

                # 创建 MLflowTracker（使用共享 run_id）
                self.tracker = MLflowTracker(
                    experiment_name=tracker_context.config.get('experiment_name'),
                    run_name=f"client_{self.client_id}",
                    role="client",
                    tracking_uri=tracker_context.config.get('tracking_uri'),
                    config=self.logging_config.tracker.config,
                    shared_run_id=shared_run_id  # 关键：使用服务端传来的 run_id
                )

                # 不需要调用 start()，因为 shared_run_id 已经是激活的 run
                self.system_logger.info(
                    f"[TrackerContext] MLflowTracker 已配置, 共享 run_id: {shared_run_id}"
                )

            elif tracker_type == "wandb":
                # TODO: 实现 WandB shared run
                self.system_logger.warning(f"WandB shared run not implemented yet")
                return

            elif tracker_type == "tensorboard":
                # TODO: 实现 TensorBoard shared run
                self.system_logger.warning(f"TensorBoard shared run not implemented yet")
                return

            else:
                self.system_logger.warning(f"Unknown tracker type: {tracker_type}")
                return

            # 4. 将 tracker 传递给 learner（用于训练时记录指标）
            if self.business_components and self.business_components.learner:
                self.business_components.learner.tracker = self.tracker
                self.system_logger.info(
                    f"[TrackerContext] Tracker 已传递给 learner: {self.client_id}"
                )

        except Exception as e:
            self.system_logger.error(f"Failed to setup tracker from context: {e}")
            # 不抛出异常，允许客户端继续运行（只是不记录指标）

    async def start_client(self) -> bool:
        """
        启动客户端

        前提：
            必须已调用 initialize() 完成初始化

        Returns:
            bool: 启动是否成功

        Raises:
            FederationError: 如果客户端未初始化
        """
        if not self.is_initialized:
            raise FederationError("Client not initialized. Call initialize() first.")

        if self.is_running:
            self.system_logger.warning("Client already running")
            return True

        self.system_logger.info("Starting FederationClient...")

        try:
            # 启动通信层
            self.system_logger.debug("Starting communication layers...")

            if hasattr(self.comm_components.communication_manager, 'start'):
                await self.comm_components.communication_manager.start()
                self.system_logger.debug("✓ Communication manager started")

            if hasattr(self.comm_components.connection_manager, 'start'):
                await self.comm_components.connection_manager.start()
                self.system_logger.debug("✓ Connection manager started")

            # 注册 SHUTDOWN 消息处理器（统一停止协议）
            self._register_shutdown_handler()
            self.system_logger.debug("✓ SHUTDOWN handler registered")

            # 启动 LearnerStub（会自动注册到服务端）
            self.system_logger.debug("Starting LearnerStub...")
            await self.learner_stub.start_listening()
            self.system_logger.debug("✓ LearnerStub started")

            # 更新注册状态
            self.is_registered = (
                self.learner_stub.get_registration_status() == RegistrationStatus.REGISTERED
            )

            if self.is_registered:
                self.system_logger.info(f"✓ Client {self.client_id} registered successfully")

                # 配置实验跟踪器（使用从服务端接收的TrackerContext）
                await self._setup_tracker_from_context()
            else:
                self.system_logger.warning(
                    f"✗ Client {self.client_id} registration may have failed"
                )

            self.is_running = True
            self.system_logger.info("FederationClient started successfully")

            return True

        except Exception as e:
            self.system_logger.error(f"Failed to start client: {e}")
            return False

    async def stop_client(self) -> bool:
        """
        停止客户端

        Returns:
            bool: 停止是否成功
        """
        if not self.is_running:
            self.system_logger.debug("Client not running, nothing to stop")
            return True

        self.system_logger.debug("Stopping FederationClient...")

        try:
            # 从服务端注销
            if self.is_registered and self.learner_stub:
                await self.learner_stub.unregister_from_server()
                self.is_registered = False
                self.system_logger.debug("✓ Client unregistered from server")

            # 停止 LearnerStub
            if self.learner_stub:
                await self.learner_stub.stop_listening()
                self.system_logger.debug("✓ LearnerStub stopped")

            # 停止通信层
            if self.comm_components:
                if hasattr(self.comm_components.connection_manager, 'stop'):
                    await self.comm_components.connection_manager.stop()
                    self.system_logger.debug("✓ Connection manager stopped")

                if hasattr(self.comm_components.communication_manager, 'stop'):
                    await self.comm_components.communication_manager.stop()
                    self.system_logger.debug("✓ Communication manager stopped")

                if hasattr(self.comm_components.transport, 'stop'):
                    await self.comm_components.transport.stop()
                    self.system_logger.debug("✓ Transport stopped")

            self.is_running = False
            self.system_logger.debug("FederationClient stopped successfully")

            return True

        except Exception as e:
            self.system_logger.error(f"Failed to stop client: {e}")
            return False

    def _register_shutdown_handler(self):
        """注册 SHUTDOWN 消息处理器（统一停止协议）"""
        if not self.comm_components or not self.comm_components.communication_manager:
            return

        # 注册到 CommunicationManager 的消息处理器
        comm_manager = self.comm_components.communication_manager
        if hasattr(comm_manager, 'register_message_handler'):
            comm_manager.register_message_handler("SHUTDOWN", self._handle_shutdown_message)
            self.system_logger.debug(f"[{self.client_id}] Registered SHUTDOWN handler")

    def _handle_shutdown_message(self, source: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理来自服务端的 SHUTDOWN 消息

        Args:
            source: 消息来源（通常是服务端）
            data: SHUTDOWN 消息数据

        Returns:
            Dict[str, Any]: 响应数据
        """
        import asyncio

        self.system_logger.info(f"[{self.client_id}] Received SHUTDOWN from {source}")

        # 创建异步任务来停止客户端（不阻塞消息处理）
        async def shutdown_task():
            try:
                # 等待一小段时间，让响应先发出去
                await asyncio.sleep(0.1)
                await self.stop_client()
                self.system_logger.info(f"[{self.client_id}] Shutdown completed")
            except Exception as e:
                self.system_logger.error(f"[{self.client_id}] Shutdown error: {e}")

        # 在后台执行 shutdown
        asyncio.create_task(shutdown_task())

        # 立即返回确认响应
        return {
            "success": True,
            "message": f"Client {self.client_id} is shutting down",
            "timestamp": str(__import__('datetime').datetime.now())
        }

    # ========== 便捷访问属性 ==========

    @property
    def learner(self) -> Optional[BaseLearner]:
        """获取学习器实例"""
        return self.business_components.learner if self.business_components else None

    def get_client_status(self) -> Dict[str, Any]:
        """
        获取客户端状态

        Returns:
            客户端状态字典
        """
        return {
            "client_id": self.client_id,
            "mode": self.mode.value,
            "is_initialized": self.is_initialized,
            "is_running": self.is_running,
            "is_registered": self.is_registered,
            "learner_type": type(self.learner).__name__ if self.learner else None,
            "registration_status": (
                self.learner_stub.get_registration_status() if self.learner_stub else None
            )
        }

    def _generate_client_id(self) -> str:
        """生成客户端ID"""
        import uuid
        unique_id = str(uuid.uuid4())[:8]

        if self.mode == CommunicationMode.MEMORY:
            return f"memory_client_{unique_id}"
        elif self.mode == CommunicationMode.PROCESS:
            port = self.comm_config.transport.get("port", 0) if self.comm_config.transport else 0
            return f"process_client_{port}_{unique_id}"
        elif self.mode == CommunicationMode.NETWORK:
            host = self.comm_config.transport.get("host", "localhost") if self.comm_config.transport else "localhost"
            port = self.comm_config.transport.get("port", 8001) if self.comm_config.transport else 8001
            return f"network_client_{host}_{port}_{unique_id}"
        else:
            return f"unknown_client_{unique_id}"

    # ========== 工厂方法 ==========

    @classmethod
    def create_client(
        cls,
        communication_config: CommunicationConfig,
        training_config: TrainingConfig,
        client_id: Optional[str] = None
    ) -> 'FederationClient':
        """
        工厂方法：创建客户端实例

        Args:
            communication_config: 通信配置对象
            training_config: 训练配置对象
            client_id: 客户端ID（可选）

        Returns:
            FederationClient 实例
        """
        return cls(communication_config, training_config, client_id)
