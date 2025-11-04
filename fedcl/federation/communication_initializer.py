"""
通信层初始化器 - 负责初始化5层通信栈
fedcl/federation/communication_initializer.py
"""

from typing import Optional

from ..communication.base import CommunicationManagerBase
from ..communication.business_layer import BusinessCommunicationLayer
from ..config import CommunicationConfig
from ..connection.manager import ConnectionManager
from ..factory.factory import ComponentFactory
from ..transport.base import TransportBase
from ..types import CommunicationMode
from ..utils.auto_logger import get_sys_logger
from .components import CommunicationComponents


class CommunicationInitializer:
    """
    通信层初始化器

    职责：
        - 接收 CommunicationConfig 配置对象
        - 初始化5层通信栈（Transport → CommunicationManager → ConnectionManager → BusinessCommunicationLayer）
        - 返回 CommunicationComponents 组件集合

    使用方式：
        >>> comm_config = CommunicationConfig(mode="network", role="server")
        >>> initializer = CommunicationInitializer(comm_config, "server_main", "server")
        >>> comm_components = await initializer.initialize()
        >>> # comm_components 包含所有通信组件
    """

    def __init__(self, comm_config: CommunicationConfig, node_id: str, node_role: str):
        """
        初始化通信层初始化器

        Args:
            comm_config: 通信配置对象（CommunicationConfig）
            node_id: 节点ID（server_id 或 client_id）
            node_role: 节点角色（"server" 或 "client"）
        """
        self.comm_config = comm_config
        self.node_id = node_id
        self.node_role = node_role
        self.mode = CommunicationMode(comm_config.mode)
        self.logger = get_sys_logger()

        self.logger.info(
            f"CommunicationInitializer created: node_id={node_id}, "
            f"role={node_role}, mode={self.mode}"
        )

    async def initialize(self) -> CommunicationComponents:
        """
        初始化通信栈

        流程：
            1. Layer 5: Transport（传输层）
            2. Layer 4: CommunicationManager（通信管理层）
            3. Layer 3: ConnectionManager（连接管理层）
            4. Layer 2: BusinessCommunicationLayer（业务通信层，仅服务端）

        Returns:
            CommunicationComponents: 包含所有通信组件的数据类

        Raises:
            Exception: 如果初始化失败
        """
        self.logger.info("Starting communication stack initialization...")

        try:
            # 将 CommunicationConfig 转换为字典格式供工厂使用
            config_dict = self._prepare_config_dict()

            # 创建组件工厂
            factory = ComponentFactory(config_dict)

            # Layer 5: Transport
            transport = self._create_transport(factory, config_dict)

            # Layer 4: CommunicationManager
            communication_manager = self._create_communication_manager(
                factory, config_dict, transport
            )

            # Layer 3: ConnectionManager
            connection_manager = self._create_connection_manager(
                factory, config_dict, communication_manager
            )

            # Layer 2: BusinessCommunicationLayer (仅服务端需要)
            business_layer = None
            if self.node_role == "server":
                business_layer = self._create_business_layer(
                    communication_manager, connection_manager
                )

            self.logger.debug("Communication stack initialized successfully")

            return CommunicationComponents(
                transport=transport,
                communication_manager=communication_manager,
                connection_manager=connection_manager,
                business_layer=business_layer
            )

        except Exception as e:
            self.logger.error(f"Communication stack initialization failed: {e}")
            raise

    def _prepare_config_dict(self) -> dict:
        """
        准备配置字典

        将 CommunicationConfig 转换为字典格式，供 ComponentFactory 使用

        Returns:
            配置字典
        """
        config_dict = self.comm_config.to_dict()

        # 确保必要字段存在
        config_dict["mode"] = self.comm_config.mode
        config_dict["node_id"] = self.node_id

        # ✅ 正确保留 transport 配置的嵌套结构
        # 不要使用 update()，这会把 transport 字典的内容展开到顶层
        if hasattr(self.comm_config, "transport") and self.comm_config.transport:
            # 确保 transport 字段正确嵌套在配置字典中
            if "transport" not in config_dict or not config_dict["transport"]:
                config_dict["transport"] = {}

            # 保留 transport 的完整结构
            config_dict["transport"].update(self.comm_config.transport)

            # 确保有 specific_config 字段
            if "specific_config" not in config_dict["transport"]:
                config_dict["transport"]["specific_config"] = {}

            # 将顶层的 host, port 等字段移到 specific_config 中（如果存在）
            transport_keys = ["host", "port", "websocket_port", "timeout", "type"]
            for key in transport_keys:
                if key in config_dict["transport"]:
                    # 如果在 transport 顶层，移到 specific_config
                    if key not in config_dict["transport"]["specific_config"]:
                        config_dict["transport"]["specific_config"][key] = config_dict["transport"][key]

        self.logger.debug(f"Prepared config dict: mode={config_dict.get('mode')}, transport={config_dict.get('transport')}")

        return config_dict

    def _create_transport(
        self,
        factory: ComponentFactory,
        config_dict: dict
    ) -> TransportBase:
        """
        创建传输层（Layer 5）

        Args:
            factory: 组件工厂
            config_dict: 配置字典

        Returns:
            TransportBase: 传输层实例
        """
        self.logger.debug("Creating Layer 5: Transport...")

        transport_config = factory._create_transport_config(
            config_dict,
            self.mode,
            node_role=self.node_role
        )

        transport = factory.create_transport(transport_config, self.mode)

        self.logger.debug(f"✓ Layer 5 created: {type(transport).__name__}")

        return transport

    def _create_communication_manager(
        self,
        factory: ComponentFactory,
        config_dict: dict,
        transport: TransportBase
    ) -> CommunicationManagerBase:
        """
        创建通信管理层（Layer 4）

        Args:
            factory: 组件工厂
            config_dict: 配置字典
            transport: 传输层实例

        Returns:
            CommunicationManagerBase: 通信管理层实例
        """
        self.logger.debug("Creating Layer 4: CommunicationManager...")

        communication_config = factory._create_communication_config(config_dict)

        communication_manager = factory.create_communication_manager(
            self.node_id,
            transport,
            communication_config,
            self.mode,
            node_role=self.node_role
        )

        self.logger.debug(
            f"✓ Layer 4 created: {type(communication_manager).__name__}"
        )

        return communication_manager

    def _create_connection_manager(
        self,
        factory: ComponentFactory,
        config_dict: dict,
        communication_manager: CommunicationManagerBase
    ) -> ConnectionManager:
        """
        创建连接管理层（Layer 3）

        Args:
            factory: 组件工厂
            config_dict: 配置字典
            communication_manager: 通信管理层实例

        Returns:
            ConnectionManager: 连接管理层实例
        """
        self.logger.debug("Creating Layer 3: ConnectionManager...")

        communication_config = factory._create_communication_config(config_dict)

        connection_manager = factory.create_connection_manager(
            communication_manager,
            communication_config
        )

        self.logger.debug(
            f"✓ Layer 3 created: {type(connection_manager).__name__}"
        )

        return connection_manager

    def _create_business_layer(
        self,
        communication_manager: CommunicationManagerBase,
        connection_manager: ConnectionManager
    ) -> BusinessCommunicationLayer:
        """
        创建业务通信层（Layer 2）- 仅服务端需要

        Args:
            communication_manager: 通信管理层实例
            connection_manager: 连接管理层实例

        Returns:
            BusinessCommunicationLayer: 业务通信层实例
        """
        self.logger.debug("Creating Layer 2: BusinessCommunicationLayer (server only)...")

        business_layer = BusinessCommunicationLayer()
        business_layer.set_dependencies(
            communication_manager,
            connection_manager
        )

        self.logger.debug("✓ Layer 2 created: BusinessCommunicationLayer")

        return business_layer
