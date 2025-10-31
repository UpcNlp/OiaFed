"""
联邦服务端管理器 - 负责服务端组件的初始化和管理（重构版）
fedcl/federation/server.py
"""

from typing import Dict, Any, Optional

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
        server_id: Optional[str] = None
    ):
        """
        初始化服务端管理器

        Args:
            communication_config: 通信配置对象
            training_config: 训练配置对象
            server_id: 服务端ID（如果为 None，从配置中读取或自动生成）
        """
        self.comm_config = communication_config
        self.train_config = training_config

        # 确定 server_id
        self.server_id = server_id or communication_config.node_id or self._generate_server_id()
        self.mode = CommunicationMode(communication_config.mode)

        # 组件引用
        self.comm_components: Optional[CommunicationComponents] = None
        self.business_components: Optional[ServerBusinessComponents] = None

        # 状态管理
        self.is_initialized = False
        self.is_running = False

        self.logger = get_sys_logger()
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

        self.logger.info("🔗 Layer relationships established, event bridges activated")

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
        停止服务端

        Returns:
            bool: 停止是否成功
        """
        if not self.is_running:
            self.logger.info("Server not running, nothing to stop")
            return True

        self.logger.info("Stopping FederationServer...")

        try:
            # 按相反顺序停止组件

            # 停止 trainer
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
