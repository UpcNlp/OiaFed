"""
MOE-FedCL 统一组件工厂
moe_fedcl/factory/factory.py
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# 通信层
from ..communication.base import CommunicationManagerBase
from ..communication.memory_manager import MemoryCommunicationManager
from ..communication.network_manager import NetworkCommunicationManager
# 连接管理层
from ..connection.manager import ConnectionManager
from ..exceptions import ConfigurationError
# 业务层
from ..learner.base_learner import BaseLearner
from ..learner.proxy import LearnerProxy, ProxyConfig
from ..learner.stub import LearnerStub, StubConfig
from ..trainer.trainer import BaseTrainer
# 传输层
from ..transport.base import TransportBase
from ..transport.memory import MemoryTransport
from ..transport.network import NetworkTransport
# 配置和类型
from ..types import (
    CommunicationMode, TransportConfig, CommunicationConfig,
    FederationConfig
)

from ..utils.auto_logger import get_sys_logger


@dataclass
class TrainerComponents:
    """服务端训练器组件集合"""
    base_trainer: BaseTrainer
    learner_proxies: Dict[str, LearnerProxy]
    communication_manager: CommunicationManagerBase
    connection_manager: ConnectionManager
    transport: TransportBase


@dataclass
class LearnerComponents:
    """客户端学习器组件集合"""
    base_learner: BaseLearner
    learner_stub: LearnerStub
    communication_manager: CommunicationManagerBase
    connection_manager: ConnectionManager
    transport: TransportBase


@dataclass
class StandaloneComponents:
    """单机模式组件集合 (适用于Memory模式)"""
    server_components: TrainerComponents
    client_components: List[LearnerComponents]


class ComponentFactory:
    """统一组件工厂 - 根据配置创建相应模式的组件实例"""
    
    def __init__(self, config: Dict[str, Any] = None):
        # 如果没有提供配置，使用默认配置
        if config is None:
            config = {
                'transport': {'type': 'memory'},
                'communication': {},
                'federation': {}
            }
        
        # 保存配置
        self.config = config
        self.logger = get_sys_logger()
        # Transport类映射 - Process和Network都使用NetworkTransport
        self._transport_classes = {
            CommunicationMode.MEMORY: MemoryTransport,
            CommunicationMode.PROCESS: NetworkTransport,  # Process模式使用NetworkTransport
            CommunicationMode.NETWORK: NetworkTransport,
        }
        
        # 通信管理器类映射 (Process和Network模式都使用NetworkCommunicationManager)
        self._communication_classes = {
            CommunicationMode.MEMORY: MemoryCommunicationManager,
            CommunicationMode.PROCESS: NetworkCommunicationManager,  # 统一使用NetworkCommunicationManager
            CommunicationMode.NETWORK: NetworkCommunicationManager
        }
    
    def get_factory_type(self) -> str:
        """获取工厂类型"""
        return "MOEFedCLComponentFactory"
    
    # ==================== 核心创建方法 ====================
    
    def create_server_components(self, 
                                config: Dict[str, Any],
                                trainer: BaseTrainer,
                                clients_config: Dict[str, Dict[str, Any]]) -> TrainerComponents:
        """创建服务端组件
        
        Args:
            config: 完整配置字典
            trainer: 用户提供的训练器实例
            clients_config: 客户端配置
            
        Returns:
            TrainerComponents: 服务端组件集合
        """
        try:
            # 解析配置
            mode = CommunicationMode(config.get("mode", "memory"))

            # 先生成服务端节点ID
            server_id = self._generate_server_id(mode, config)

            # 创建配置（传入 node_id）
            transport_config = self._create_transport_config(config, mode, node_role="server", node_id=server_id)
            communication_config = self._create_communication_config(config)
            federation_config = self._create_federation_config(config)

            # 创建传输层
            transport = self.create_transport(transport_config, mode)

            # 创建通信管理器（指定为服务端角色）
            communication_manager = self.create_communication_manager(
                server_id, transport, communication_config, mode, node_role="server"
            )
            
            # 创建连接管理器
            connection_manager = self.create_connection_manager(
                communication_manager, communication_config
            )
            
            # 创建客户端代理集合
            learner_proxies = self._create_learner_proxies(
                clients_config, communication_manager, connection_manager, config
            )
            
            # 更新训练器的代理集合
            trainer.learner_proxies = learner_proxies
            
            return TrainerComponents(
                base_trainer=trainer,
                learner_proxies=learner_proxies,
                communication_manager=communication_manager,
                connection_manager=connection_manager,
                transport=transport
            )
            
        except Exception as e:
            raise ConfigurationError(f"Failed to create server components: {str(e)}")
    
    def create_client_components(self, 
                               config: Dict[str, Any],
                               learner: BaseLearner,
                               client_id: str) -> LearnerComponents:
        """创建客户端组件
        
        Args:
            config: 完整配置字典
            learner: 用户提供的学习器实例
            client_id: 客户端ID
            
        Returns:
            LearnerComponents: 客户端组件集合
        """
        try:
            # 解析配置
            mode = CommunicationMode(config.get("mode", "memory"))

            # 先生成客户端节点ID
            if not client_id:
                client_id = self._generate_client_id(mode, config)
            else:
                client_id = self._ensure_client_id_format(client_id, mode, config)

            # 更新学习器的client_id
            learner.client_id = client_id

            # 创建配置（传入 node_id）
            transport_config = self._create_transport_config(config, mode, node_role="client", node_id=client_id)
            communication_config = self._create_communication_config(config)

            # 创建传输层
            transport = self.create_transport(transport_config, mode)

            # 创建通信管理器（指定为客户端角色）
            communication_manager = self.create_communication_manager(
                client_id, transport, communication_config, mode, node_role="client"
            )
            
            # 创建连接管理器
            connection_manager = self.create_connection_manager(
                communication_manager, communication_config
            )
            
            # 创建学习器存根
            stub_config = self._create_stub_config(config)
            learner_stub = self.create_learner_stub(
                learner, communication_manager, connection_manager, stub_config
            )
            
            return LearnerComponents(
                base_learner=learner,
                learner_stub=learner_stub,
                communication_manager=communication_manager,
                connection_manager=connection_manager,
                transport=transport
            )
            
        except Exception as e:
            raise ConfigurationError(f"Failed to create client components: {str(e)}")
    
    def create_standalone_components(self, 
                                   config: Dict[str, Any],
                                   trainer: BaseTrainer,
                                   learners: List[BaseLearner],
                                   clients_config: Dict[str, Dict[str, Any]] = None) -> StandaloneComponents:
        """创建单机模式组件集合 (适用于Memory模式)
        
        Args:
            config: 配置字典
            trainer: 训练器实例
            learners: 学习器实例列表
            clients_config: 客户端配置，如果为None则自动生成
            
        Returns:
            StandaloneComponents: 单机组件集合
        """
        try:
            mode = CommunicationMode(config.get("mode", "memory"))
            
            if mode != CommunicationMode.MEMORY:
                raise ConfigurationError("Standalone components only support Memory mode")
            
            # 生成客户端配置
            if clients_config is None:
                clients_config = {}
                for i, learner in enumerate(learners):
                    client_id = f"memory_client_{i+1}"
                    learner.client_id = client_id
                    clients_config[client_id] = {"index": i+1}
            
            # 创建服务端组件
            server_components = self.create_server_components(
                config, trainer, clients_config
            )
            
            # 创建客户端组件
            client_components = []
            for i, learner in enumerate(learners):
                client_id = learner.client_id or f"memory_client_{i+1}"
                client_comp = self.create_client_components(config, learner, client_id)
                client_components.append(client_comp)
            
            return StandaloneComponents(
                server_components=server_components,
                client_components=client_components
            )
            
        except Exception as e:
            raise ConfigurationError(f"Failed to create standalone components: {str(e)}")
    
    # ==================== 基础组件创建方法 ====================
    
    def create_transport(self, config: TransportConfig, mode: CommunicationMode = None) -> TransportBase:
        """创建传输层实例
        
        Args:
            config: 传输配置
            mode: 通信模式，如果为None则从配置推断
            
        Returns:
            TransportBase: 传输层实例
        """
        if mode is None:
            mode = CommunicationMode(config.type)
        
        transport_class = self._transport_classes.get(mode)
        if transport_class is None:
            raise ConfigurationError(f"Unsupported transport mode: {mode}")
        
        return transport_class(config)
    
    def create_communication_manager(self,
                                   node_id: str,
                                   transport: TransportBase,
                                   config: CommunicationConfig,
                                   mode: CommunicationMode = None,
                                   node_role: str = None) -> CommunicationManagerBase:
        """创建通信管理器实例

        Args:
            node_id: 节点ID
            transport: 传输层实例
            config: 通信配置
            mode: 通信模式，如果为None则从transport推断
            node_role: 节点角色 ('server' 或 'client')

        Returns:
            CommunicationManagerBase: 通信管理器实例
        """
        if mode is None:
            # 从transport配置推断模式
            mode = CommunicationMode(transport.config.type)

        manager_class = self._communication_classes.get(mode)
        if manager_class is None:
            raise ConfigurationError(f"Unsupported communication mode: {mode}")

        # 所有 CommunicationManager 都支持 node_role 参数
        if node_role is not None:
            return manager_class(node_id, transport, config, node_role=node_role)
        else:
            # 向后兼容：如果没有提供 node_role，使用旧的调用方式
            return manager_class(node_id, transport, config)
    
    def create_connection_manager(self, 
                                communication_manager: CommunicationManagerBase,
                                config: CommunicationConfig) -> ConnectionManager:
        """创建连接管理器实例"""
        return ConnectionManager(communication_manager, config)
    
    def create_learner_proxy(self,
                           client_id: str,
                           communication_manager: CommunicationManagerBase,
                           connection_manager: ConnectionManager,
                           config: Optional[ProxyConfig] = None) -> LearnerProxy:
        """创建学习器代理实例"""
        return LearnerProxy(client_id, communication_manager, connection_manager, config)
    
    def create_learner_stub(self,
                          learner: BaseLearner,
                          communication_manager: CommunicationManagerBase,
                          connection_manager: ConnectionManager,
                          config: Optional[StubConfig] = None) -> LearnerStub:
        """创建学习器存根实例"""
        return LearnerStub(learner, communication_manager, connection_manager, config)
    
    # ==================== 配置创建和验证方法 ====================
    
    def validate_configuration(self, config: Dict[str, Any]) -> bool:
        """验证配置有效性
        
        Args:
            config: 配置字典
            
        Returns:
            bool: 配置是否有效
            
        Raises:
            ConfigurationError: 配置无效
        """
        try:
            # 验证必需字段
            required_fields = ["mode"]
            for field in required_fields:
                if field not in config:
                    raise ConfigurationError(f"Missing required configuration field: {field}")
            
            # 验证模式
            try:
                mode = CommunicationMode(config["mode"])
            except ValueError:
                raise ConfigurationError(f"Invalid communication mode: {config['mode']}")
            
            # 验证模式特定配置
            if mode == CommunicationMode.NETWORK:
                network_config = config.get("transport", {}).get("specific_config", {})
                if not network_config.get("host") or not network_config.get("port"):
                    raise ConfigurationError("Network mode requires host and port configuration")
            
            # 验证客户端数量
            federation_config = config.get("federation", {})
            min_clients = federation_config.get("min_clients", 2)
            if min_clients < 1:
                raise ConfigurationError("min_clients must be at least 1")
            
            return True
            
        except Exception as e:
            if isinstance(e, ConfigurationError):
                raise
            else:
                raise ConfigurationError(f"Configuration validation failed: {str(e)}")
    
    def get_available_modes(self) -> List[str]:
        """获取可用的通信模式列表
        
        Returns:
            List[str]: 可用模式列表
        """
        return [mode.value for mode in CommunicationMode]
    
    def create_default_config(self, mode: str) -> Dict[str, Any]:
        """创建默认配置
        
        Args:
            mode: 通信模式
            
        Returns:
            Dict[str, Any]: 默认配置字典
        """
        mode_enum = CommunicationMode(mode)
        
        base_config = {
            "mode": mode,
            "transport": {
                "type": mode,
                "timeout": 30.0,
                "retry_attempts": 3,
                "specific_config": {}
            },
            "communication": {
                "heartbeat_interval": 30.0,
                "heartbeat_timeout": 90.0,
                "registration_timeout": 60.0,
                "max_clients": 100,
                "rpc_timeout": 120.0
            },
            "federation": {
                "coordinator_id": "fed_coordinator",
                "max_rounds": 100,
                "min_clients": 2,
                "client_selection": "all",
                "training_config": {}
            }
        }
        
        # 模式特定配置
        if mode_enum == CommunicationMode.MEMORY:
            base_config["transport"]["specific_config"] = {
                "shared_memory_size": "1GB",
                "event_queue_size": 1000,
                "direct_call_timeout": 30.0
            }
        elif mode_enum == CommunicationMode.PROCESS:
            base_config["transport"]["specific_config"] = {
                "queue_backend": "multiprocessing",
                "max_queue_size": 10000,
                "serialization": "pickle",
                "process_timeout": 60.0
            }
        elif mode_enum == CommunicationMode.NETWORK:
            base_config["transport"]["specific_config"] = {
                "protocol": "http",
                "host": "127.0.0.1",  # 默认使用本地地址
                "port": 8000,
                "websocket_port": 8001,
                "ssl_enabled": False,
                "connection_timeout": 30.0
            }
        
        return base_config
    
    # ==================== 私有辅助方法 ====================
    
    def _create_transport_config(
        self,
        config: Dict[str, Any],
        mode: CommunicationMode,
        node_role: str = None,
        node_id: str = ""
    ) -> TransportConfig:
        """创建传输配置

        Args:
            config: 完整配置字典
            mode: 通信模式
            node_role: 节点角色 ('server' 或 'client')
            node_id: 节点ID

        Returns:
            TransportConfig: 传输配置
        """
        transport_config = config.get("transport", {})

        if not transport_config:
            transport_config = {
                "specific_config": {}
            }

        # 获取特定配置
        specific_config = transport_config.get("specific_config", {}).copy()

        # ✅ 关键修复：如果 transport 字典顶层有配置字段，也要复制到 specific_config
        # 这样可以支持两种配置方式：
        # 1. transport: {host: "127.0.0.1", port: 8000}
        # 2. transport: {specific_config: {host: "127.0.0.1", port: 8000}}
        for key in ["host", "port", "websocket_port", "timeout", "type", "server"]:
            if key in transport_config and key not in specific_config:
                specific_config[key] = transport_config[key]

        # 添加节点角色到配置中（用于 NetworkTransport）
        if node_role is not None:
            specific_config["node_role"] = node_role

        # Process模式和Network模式下，处理主机和端口配置
        if mode == CommunicationMode.PROCESS or mode == CommunicationMode.NETWORK:
            # ✅ 主机处理：根据模式使用不同的默认值
            if "host" not in specific_config or specific_config["host"] is None or specific_config["host"] == "":
                # process模式: 默认 127.0.0.1
                # network模式: 默认 0.0.0.0
                default_host = "127.0.0.1" if mode == CommunicationMode.PROCESS else "0.0.0.0"
                specific_config["host"] = default_host
                self.logger.debug(f"[Factory] 使用默认主机: {default_host} (mode={mode.value})")
            else:
                self.logger.debug(f"[Factory] 使用配置主机: {specific_config['host']}")

            # ✅ 端口处理：只在未配置或配置为None时才使用默认值
            if "port" not in specific_config or specific_config["port"] is None:
                # 服务器默认8000，客户端默认0（自动分配）
                specific_config["port"] = 8000 if (node_role and node_role.lower() == "server") else 0
                self.logger.debug(f"[Factory] 使用默认端口: {specific_config['port']} (role={node_role})")
            else:
                # 如果明确配置了端口，即使是0也要使用（用户可能就是想要随机端口）
                self.logger.info(f"[Factory] 使用配置端口: {specific_config['port']} (role={node_role})")

        self.logger.debug(f"[Factory] transport最终specific_config : {specific_config}")

        return TransportConfig(
            type=str(mode.value),
            node_id=node_id,
            timeout=transport_config.get("timeout", specific_config.get("timeout", 30.0)),
            retry_attempts=transport_config.get("retry_attempts", 3),
            specific_config=specific_config
        )
    
    def _create_communication_config(self, config: Dict[str, Any]) -> CommunicationConfig:
        """创建通信配置"""
        comm_config = config.get("communication", {})
        
        return CommunicationConfig(
            heartbeat_interval=comm_config.get("heartbeat_interval", 30.0),
            heartbeat_timeout=comm_config.get("heartbeat_timeout", 90.0),
            registration_timeout=comm_config.get("registration_timeout", 60.0),
            max_clients=comm_config.get("max_clients", 100),
            rpc_timeout=comm_config.get("rpc_timeout", 120.0)
        )
    
    def _create_federation_config(self, config: Dict[str, Any]) -> FederationConfig:
        """创建联邦配置"""
        fed_config = config.get("federation", {})
        
        return FederationConfig(
            coordinator_id=fed_config.get("coordinator_id", "fed_coordinator"),
            max_rounds=fed_config.get("max_rounds", 100),
            min_clients=fed_config.get("min_clients", 2),
            client_selection=fed_config.get("client_selection", "all"),
            training_config=fed_config.get("training_config", {})
        )
    
    def _create_stub_config(self, config: Dict[str, Any]) -> StubConfig:
        """创建存根配置"""
        stub_config = config.get("stub", {})
        
        return StubConfig(
            auto_register=stub_config.get("auto_register", True),
            registration_retry_attempts=stub_config.get("registration_retry_attempts", 3),
            registration_retry_delay=stub_config.get("registration_retry_delay", 5.0),
            request_timeout=stub_config.get("request_timeout", 120.0),
            max_concurrent_requests=stub_config.get("max_concurrent_requests", 5)
        )
    
    def _create_learner_proxies(self, 
                              clients_config: Dict[str, Dict[str, Any]],
                              communication_manager: CommunicationManagerBase,
                              connection_manager: ConnectionManager,
                              config: Dict[str, Any]) -> Dict[str, LearnerProxy]:
        """创建客户端代理集合"""
        proxies = {}
        proxy_config = ProxyConfig()  # 使用默认配置
        
        for client_id in clients_config.keys():
            proxy = self.create_learner_proxy(
                client_id, communication_manager, connection_manager, proxy_config
            )
            proxies[client_id] = proxy
        
        return proxies
    
    def _generate_server_id(self, mode: CommunicationMode, config: Dict[str, Any]) -> str:
        """生成服务端节点ID"""
        if mode == CommunicationMode.MEMORY:
            return "memory_server"
        elif mode == CommunicationMode.PROCESS:
            port = config.get("transport", {}).get("specific_config", {}).get("port", 8000)
            return f"process_server_{port}"
        elif mode == CommunicationMode.NETWORK:
            transport_config = config.get("transport", {}).get("specific_config", {})
            host = transport_config.get("host", "localhost")
            port = transport_config.get("port", 8000)
            return f"network_server_{host}_{port}"
        else:
            return "unknown_server"
    
    def _generate_client_id(self, mode: CommunicationMode, config: Dict[str, Any]) -> str:
        """生成客户端节点ID"""
        import uuid
        
        if mode == CommunicationMode.MEMORY:
            return f"memory_client_{uuid.uuid4().hex[:8]}"
        elif mode == CommunicationMode.PROCESS:
            import os
            pid = os.getpid()
            port = config.get("transport", {}).get("specific_config", {}).get("port", 8001)
            return f"process_client_{pid}_{port}"
        elif mode == CommunicationMode.NETWORK:
            transport_config = config.get("transport", {}).get("specific_config", {})
            host = transport_config.get("host", "localhost")
            port = transport_config.get("port", 8001)
            client_uuid = uuid.uuid4().hex[:8]
            return f"network_client_{host}_{port}_{client_uuid}"
        else:
            return f"unknown_client_{uuid.uuid4().hex[:8]}"
    
    def _ensure_client_id_format(self, client_id: str, mode: CommunicationMode, config: Dict[str, Any]) -> str:
        """确保客户端ID格式正确"""
        mode_prefix = f"{mode.value}_"
        
        if not client_id.startswith(mode_prefix):
            # 补全前缀
            if mode == CommunicationMode.MEMORY:
                return f"memory_client_{client_id}"
            elif mode == CommunicationMode.PROCESS:
                import os
                pid = os.getpid()
                port = config.get("transport", {}).get("specific_config", {}).get("port", 8001)
                return f"process_client_{pid}_{port}_{client_id}"
            elif mode == CommunicationMode.NETWORK:
                transport_config = config.get("transport", {}).get("specific_config", {})
                host = transport_config.get("host", "localhost")
                port = transport_config.get("port", 8001)
                return f"network_client_{host}_{port}_{client_id}"
        
        return client_id


# ==================== 便捷创建函数 ====================

def create_memory_system(trainer: BaseTrainer, 
                        learners: List[BaseLearner],
                        max_rounds: int = 100) -> StandaloneComponents:
    """便捷创建Memory模式完整系统
    
    Args:
        trainer: 训练器实例
        learners: 学习器实例列表
        max_rounds: 最大训练轮数
        
    Returns:
        StandaloneComponents: 完整系统组件
    """
    factory = ComponentFactory()
    
    config = factory.create_default_config("memory")
    config["federation"]["max_rounds"] = max_rounds
    config["federation"]["min_clients"] = len(learners)
    
    return factory.create_standalone_components(config, trainer, learners)


def create_process_server(trainer: BaseTrainer,
                         clients_config: Dict[str, Dict[str, Any]],
                         port: int = 8000) -> TrainerComponents:
    """便捷创建Process模式服务端
    
    Args:
        trainer: 训练器实例
        clients_config: 客户端配置
        port: 服务端端口
        
    Returns:
        TrainerComponents: 服务端组件
    """
    factory = ComponentFactory()
    
    config = factory.create_default_config("process")
    config["transport"]["specific_config"]["port"] = port
    
    return factory.create_server_components(config, trainer, clients_config)


def create_process_client(learner: BaseLearner, 
                         client_id: str,
                         server_port: int = 8000) -> LearnerComponents:
    """便捷创建Process模式客户端
    
    Args:
        learner: 学习器实例
        client_id: 客户端ID
        server_port: 服务端端口
        
    Returns:
        LearnerComponents: 客户端组件
    """
    factory = ComponentFactory()
    
    config = factory.create_default_config("process")
    config["transport"]["specific_config"]["server_port"] = server_port
    
    return factory.create_client_components(config, learner, client_id)


def create_network_server(trainer: BaseTrainer,
                         clients_config: Dict[str, Dict[str, Any]],
                         host: str = "127.0.0.1",  # 默认使用本地地址
                         port: int = 8000) -> TrainerComponents:
    """便捷创建Network模式服务端
    
    Args:
        trainer: 训练器实例
        clients_config: 客户端配置
        host: 服务端主机
        port: 服务端端口
        
    Returns:
        TrainerComponents: 服务端组件
    """
    factory = ComponentFactory()
    
    config = factory.create_default_config("network")
    config["transport"]["specific_config"]["host"] = host
    config["transport"]["specific_config"]["port"] = port
    
    return factory.create_server_components(config, trainer, clients_config)


def create_network_client(learner: BaseLearner,
                         client_id: str,
                         server_host: str = "localhost",
                         server_port: int = 8000) -> LearnerComponents:
    """便捷创建Network模式客户端
    
    Args:
        learner: 学习器实例
        client_id: 客户端ID
        server_host: 服务端主机
        server_port: 服务端端口
        
    Returns:
        LearnerComponents: 客户端组件
    """
    factory = ComponentFactory()
    
    config = factory.create_default_config("network")
    config["transport"]["specific_config"]["server_host"] = server_host
    config["transport"]["specific_config"]["server_port"] = server_port
    
    return factory.create_client_components(config, learner, client_id)