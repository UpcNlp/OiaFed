"""
MOE-FedCL 联邦通信系统主包
moe_fedcl/__init__.py

提供统一的API接口，用户只需要导入这个包就能使用所有功能。
"""

# 版本信息
__version__ = "0.1.0"
__author__ = "MOE-FedCL Team"
__description__ = "A unified federated learning communication system supporting Memory/Process/Network modes"

# ==================== 核心抽象类 ====================
# 用户需要继承的基类
from .learner.base_learner import BaseLearner
from .trainer.base_trainer import BaseTrainer

# ==================== 配置类型 ====================
from .types import (
    # 核心数据类型
    ModelData, TrainingResult, EvaluationResult, RoundResult, MetricsData,
    
    # 配置类型
    TransportConfig, CommunicationConfig, FederationConfig,
    
    # 枚举类型
    CommunicationMode, ConnectionStatus, RegistrationStatus, 
    FederationStatus, TrainingStatus, HealthStatus,
    
    # 请求响应类型
    TrainingRequest, TrainingResponse, RegistrationRequest, RegistrationResponse,
    HeartbeatMessage, EventMessage, ClientInfo
)

# ==================== 核心组件 ====================
# 联邦学习统一入口 - 延迟导入避免循环依赖
# from .federation.coordinator import FederationCoordinator, FederationResult

# 代理和存根
from .learner.proxy import LearnerProxy, ProxyConfig
from .learner.stub import LearnerStub, StubConfig

# 训练器配置
from .trainer.base_trainer import TrainingConfig

# ==================== 组件工厂 ====================
from .factory.factory import (
    ComponentFactory,
    TrainerComponents, LearnerComponents, StandaloneComponents,
    
    # 便捷创建函数
    create_memory_system,
    create_process_server, create_process_client,
    create_network_server, create_network_client
)

# ==================== 异常类 ====================
from .exceptions import (
    MOEFedCLError, TransportError, ConnectionError, RegistrationError,
    CommunicationError, TrainingError, TimeoutError, ValidationError,
    ConfigurationError, ClientNotFoundError, FederationError, SerializationError
)

# ==================== 工具函数 ====================
from .utils.serialization import serialize_data, deserialize_data
from .utils.retry import retry_async, RetryConfig


# ==================== 统一API类 ====================

class MOEFedCL:
    """MOE-FedCL联邦学习系统主类 - 提供统一的高级API"""
    
    def __init__(self, mode: str = "memory", config: dict = None):
        """
        初始化MOE-FedCL系统
        
        Args:
            mode: 通信模式 ("memory", "process", "network")
            config: 系统配置字典
        """
        self.mode = CommunicationMode(mode)
        self.factory = ComponentFactory()
        
        # 创建或使用默认配置
        if config is None:
            self.config = self.factory.create_default_config(mode)
        else:
            self.config = config
        
        # 验证配置
        self.factory.validate_configuration(self.config)
        
        # 组件存储
        self.server_components: Optional[TrainerComponents] = None
        self.client_components: List[LearnerComponents] = []
        self.standalone_components: Optional[StandaloneComponents] = None
    
    def create_server(self, trainer: BaseTrainer, clients_config: dict) -> TrainerComponents:
        """创建服务端组件
        
        Args:
            trainer: 用户定义的训练器
            clients_config: 客户端配置
            
        Returns:
            TrainerComponents: 服务端组件集合
        """
        self.server_components = self.factory.create_server_components(
            self.config, trainer, clients_config
        )
        return self.server_components
    
    def create_client(self, learner: BaseLearner, client_id: str = None) -> LearnerComponents:
        """创建客户端组件
        
        Args:
            learner: 用户定义的学习器
            client_id: 客户端ID
            
        Returns:
            LearnerComponents: 客户端组件集合
        """
        client_components = self.factory.create_client_components(
            self.config, learner, client_id
        )
        self.client_components.append(client_components)
        return client_components
    
    def create_standalone_system(self, trainer: BaseTrainer, learners: list) -> StandaloneComponents:
        """创建单机完整系统 (仅Memory模式)
        
        Args:
            trainer: 训练器实例
            learners: 学习器实例列表
            
        Returns:
            StandaloneComponents: 完整系统组件
        """
        if self.mode != CommunicationMode.MEMORY:
            raise ConfigurationError("Standalone system only supports Memory mode")
        
        self.standalone_components = self.factory.create_standalone_components(
            self.config, trainer, learners
        )
        return self.standalone_components
    
    async def start_server(self) -> bool:
        """启动服务端"""
        if not self.server_components:
            raise MOEFedCLError("Server components not created")
        
        try:
            # 启动通信管理器
            await self.server_components.communication_manager.start()
            
            # 启动连接管理器
            await self.server_components.connection_manager.start()
            
            print(f"Server started in {self.mode.value} mode")
            return True
            
        except Exception as e:
            print(f"Failed to start server: {e}")
            return False
    
    async def start_client(self, client_components: LearnerComponents) -> bool:
        """启动指定客户端"""
        try:
            # 启动通信管理器
            await client_components.communication_manager.start()
            
            # 启动连接管理器
            await client_components.connection_manager.start()
            
            # 启动学习器存根监听
            await client_components.learner_stub.start_listening()
            
            print(f"Client {client_components.base_learner.client_id} started in {self.mode.value} mode")
            return True
            
        except Exception as e:
            print(f"Failed to start client: {e}")
            return False
    
    async def start_standalone_system(self) -> bool:
        """启动单机完整系统"""
        if not self.standalone_components:
            raise MOEFedCLError("Standalone components not created")
        
        try:
            # 启动服务端组件
            await self.start_server()
            
            # 启动所有客户端组件
            for client_comp in self.standalone_components.client_components:
                await self.start_client(client_comp)
            
            print(f"Standalone system started with {len(self.standalone_components.client_components)} clients")
            return True
            
        except Exception as e:
            print(f"Failed to start standalone system: {e}")
            return False
    
    async def run_federation(self):
        """运行联邦学习训练"""
        # 新架构中，联邦训练由FederationServer和用户的trainer直接管理
        # 这个方法暂时保留为兼容性接口
        raise NotImplementedError(
            "在新架构中，请直接使用 FederationServer 和自定义的 BaseTrainer 来运行联邦训练。"
            "参考 examples/complete_new_architecture_demo.py 获取详细示例。"
        )
    
    async def stop_all(self):
        """停止所有组件"""
        try:
            # 停止服务端
            if self.server_components:
                await self.server_components.communication_manager.stop()
                await self.server_components.connection_manager.stop()
            
            # 停止所有客户端
            for client_comp in self.client_components:
                await client_comp.learner_stub.stop_listening()
                await client_comp.communication_manager.stop()
                await client_comp.connection_manager.stop()
            
            # 停止单机系统
            if self.standalone_components:
                await self.standalone_components.server_components.communication_manager.stop()
                for client_comp in self.standalone_components.client_components:
                    await client_comp.learner_stub.stop_listening()
                    await client_comp.communication_manager.stop()
            
            print("All components stopped")
            
        except Exception as e:
            print(f"Error stopping components: {e}")
    
    def get_system_status(self) -> dict:
        """获取系统状态"""
        status = {
            "mode": self.mode.value,
            "server_status": None,
            "clients_status": [],
            "federation_status": None
        }
        
        # 服务端状态
        if self.server_components:
            status["server_status"] = self.server_components.federation_coordinator.get_federation_status()
        elif self.standalone_components:
            status["server_status"] = self.standalone_components.server_components.federation_coordinator.get_federation_status()
        
        # 客户端状态
        clients_to_check = self.client_components
        if self.standalone_components:
            clients_to_check.extend(self.standalone_components.client_components)
        
        for client_comp in clients_to_check:
            client_status = {
                "client_id": client_comp.base_learner.client_id,
                "registration_status": client_comp.learner_stub.get_registration_status().value,
                "learner_info": client_comp.base_learner.get_learner_info()
            }
            status["clients_status"].append(client_status)
        
        return status


# ==================== 便捷函数 ====================

def quick_start_memory(trainer: BaseTrainer, learners: list, max_rounds: int = 100) -> MOEFedCL:
    """快速启动Memory模式联邦学习
    
    Args:
        trainer: 训练器实例
        learners: 学习器实例列表  
        max_rounds: 最大训练轮数
        
    Returns:
        MOEFedCL: 配置好的系统实例
        
    Usage:
        system = quick_start_memory(my_trainer, my_learners, 50)
        await system.start_standalone_system()
        result = await system.run_federation()
    """
    system = MOEFedCL("memory")
    system.config["federation"]["max_rounds"] = max_rounds
    system.config["federation"]["min_clients"] = len(learners)
    
    system.create_standalone_system(trainer, learners)
    return system


def quick_start_process(trainer: BaseTrainer, clients_config: dict, port: int = 8000) -> MOEFedCL:
    """快速启动Process模式服务端
    
    Args:
        trainer: 训练器实例
        clients_config: 客户端配置
        port: 服务端端口
        
    Returns:
        MOEFedCL: 配置好的服务端系统
    """
    system = MOEFedCL("process")
    system.config["transport"]["specific_config"]["port"] = port
    
    system.create_server(trainer, clients_config)
    return system


def quick_start_network(trainer: BaseTrainer, clients_config: dict, host: str = "0.0.0.0", port: int = 8000) -> MOEFedCL:
    """快速启动Network模式服务端
    
    Args:
        trainer: 训练器实例
        clients_config: 客户端配置  
        host: 服务端主机
        port: 服务端端口
        
    Returns:
        MOEFedCL: 配置好的服务端系统
    """
    system = MOEFedCL("network")
    system.config["transport"]["specific_config"]["host"] = host
    system.config["transport"]["specific_config"]["port"] = port
    
    system.create_server(trainer, clients_config)
    return system


def create_process_client_system(learner: BaseLearner, client_id: str, server_port: int = 8000) -> MOEFedCL:
    """创建Process模式客户端系统
    
    Args:
        learner: 学习器实例
        client_id: 客户端ID
        server_port: 服务端端口
        
    Returns:
        MOEFedCL: 配置好的客户端系统
    """
    system = MOEFedCL("process")
    system.config["transport"]["specific_config"]["server_port"] = server_port
    
    system.create_client(learner, client_id)
    return system


def create_network_client_system(learner: BaseLearner, client_id: str, server_host: str = "localhost", server_port: int = 8000) -> MOEFedCL:
    """创建Network模式客户端系统
    
    Args:
        learner: 学习器实例
        client_id: 客户端ID
        server_host: 服务端主机
        server_port: 服务端端口
        
    Returns:
        MOEFedCL: 配置好的客户端系统
    """
    system = MOEFedCL("network")
    system.config["transport"]["specific_config"]["server_host"] = server_host
    system.config["transport"]["specific_config"]["server_port"] = server_port
    
    system.create_client(learner, client_id)
    return system


# ==================== 导出API ====================

__all__ = [
    # 版本信息
    "__version__", "__author__", "__description__",
    
    # 核心抽象类 (用户继承)
    "BaseLearner", "BaseTrainer",
    
    # 主系统类
    "MOEFedCL",
    
    # 配置类型
    "TransportConfig", "CommunicationConfig", "FederationConfig", "TrainingConfig",
    "ProxyConfig", "StubConfig",
    
    # 枚举类型
    "CommunicationMode", "ConnectionStatus", "RegistrationStatus", 
    "FederationStatus", "TrainingStatus", "HealthStatus",
    
    # 数据类型
    "ModelData", "TrainingResult", "EvaluationResult", "RoundResult", "MetricsData",
    
    # 请求响应类型
    "TrainingRequest", "TrainingResponse", "RegistrationRequest", "RegistrationResponse",
    "HeartbeatMessage", "EventMessage", "ClientInfo",
    
    # 核心组件
    "LearnerProxy", "LearnerStub",
    
    # 组件工厂
    "ComponentFactory", "TrainerComponents", "LearnerComponents", "StandaloneComponents",
    
    # 便捷创建函数
    "create_memory_system", "create_process_server", "create_process_client",
    "create_network_server", "create_network_client",
    
    # 快速启动函数
    "quick_start_memory", "quick_start_process", "quick_start_network",
    "create_process_client_system", "create_network_client_system",
    
    # 异常类
    "MOEFedCLError", "TransportError", "ConnectionError", "RegistrationError",
    "CommunicationError", "TrainingError", "TimeoutError", "ValidationError",
    "ConfigurationError", "ClientNotFoundError", "FederationError", "SerializationError",
    
    # 工具函数
    "serialize_data", "deserialize_data", "retry_async", "RetryConfig"
]