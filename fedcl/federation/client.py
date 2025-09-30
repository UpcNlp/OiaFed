"""
联邦客户端管理器 - 负责客户端组件的初始化和管理
fedcl/federation/client.py
"""

import asyncio
from typing import Dict, Any, Type, Optional
from ..learner.base_learner import BaseLearner
from ..learner.stub import LearnerStub, StubConfig
from ..connection.manager import ConnectionManager
from ..communication.base import CommunicationManagerBase
from ..transport.base import TransportBase
from ..factory.factory import ComponentFactory
from ..types import CommunicationMode
from ..exceptions import FederationError


class FederationClient:
    """联邦客户端管理器 - 专门负责客户端组件的初始化、装配和管理"""
    
    def __init__(self, config: Dict[str, Any], client_id: str = None):
        self.config = config
        self.mode = CommunicationMode(config.get("mode", "memory"))
        self.client_id = client_id or self._generate_client_id()
        
        # 组件引用
        self.transport: Optional[TransportBase] = None
        self.communication_manager: Optional[CommunicationManagerBase] = None
        self.connection_manager: Optional[ConnectionManager] = None
        self.learner_stub: Optional[LearnerStub] = None
        self.learner: Optional[BaseLearner] = None
        
        # 状态管理
        self.is_initialized = False
        self.is_running = False
        self.is_registered = False
        
        print(f"FederationClient created with mode: {self.mode}, client_id: {self.client_id}")
    
    async def initialize_with_learner(self, 
                                    learner_class: Type[BaseLearner],
                                    learner_config: Dict[str, Any] = None) -> BaseLearner:
        """初始化客户端并创建learner
        
        Args:
            learner_class: 用户的学习器类
            learner_config: 学习器配置
            
        Returns:
            BaseLearner: 初始化好的学习器实例
        """
        if self.is_initialized:
            raise FederationError("Client already initialized")
        
        try:
            # 1. 创建learner实例
            self.learner = learner_class(
                client_id=self.client_id,
                config=learner_config or {},
                logger=None  # TODO: 添加日志配置
            )
            
            # 2. 初始化通信组件栈（严格按层次顺序）
            await self._initialize_communication_stack()
            
            # 3. 创建LearnerStub
            await self._initialize_learner_stub()
            
            # 4. 标记初始化完成
            self.is_initialized = True
            
            print("FederationClient initialized successfully")
            return self.learner
            
        except Exception as e:
            print(f"FederationClient initialization failed: {e}")
            raise FederationError(f"Client initialization failed: {str(e)}")
    
    async def _initialize_communication_stack(self):
        """初始化通信组件栈 - 严格按照层次顺序"""
        factory = ComponentFactory(self.config)
        
        # 第5层：创建传输层（最底层，无依赖）
        transport_config = factory._create_transport_config(self.config, self.mode)
        self.transport = factory.create_transport(transport_config, self.mode)
        print(f"Layer 5: Transport layer created - {type(self.transport).__name__}")
        
        # 第4层：创建通用通信层（依赖传输层）
        communication_config = factory._create_communication_config(self.config)
        self.communication_manager = factory.create_communication_manager(
            self.client_id, self.transport, communication_config, self.mode
        )
        print(f"Layer 4: Communication manager created - {type(self.communication_manager).__name__}")
        
        # 第3层：创建连接管理层（依赖通信层）
        self.connection_manager = factory.create_connection_manager(
            self.communication_manager, communication_config
        )
        print(f"Layer 3: Connection manager created - {type(self.connection_manager).__name__}")
    
    async def _initialize_learner_stub(self):
        """初始化LearnerStub"""
        # 创建存根配置
        stub_config = StubConfig(
            auto_register=True,
            registration_retry_attempts=self.config.get("registration_retry_attempts", 3),
            registration_retry_delay=self.config.get("registration_retry_delay", 1.0),
            request_timeout=self.config.get("timeout", 120.0),
            max_concurrent_requests=self.config.get("max_concurrent_requests", 5)
        )
        
        # 创建LearnerStub实例
        self.learner_stub = LearnerStub(
            learner=self.learner,
            communication_manager=self.communication_manager,
            connection_manager=self.connection_manager,
            config=stub_config
        )
        
        print("LearnerStub created")
    
    async def start_client(self) -> bool:
        """启动客户端"""
        if not self.is_initialized:
            raise FederationError("Client not initialized")
        
        if self.is_running:
            return True
        
        try:
            # 启动各层组件
            if hasattr(self.transport, 'start'):
                await self.transport.start()
            
            if hasattr(self.communication_manager, 'start'):
                await self.communication_manager.start()
            
            if hasattr(self.connection_manager, 'start'):
                await self.connection_manager.start()
            
            # 启动LearnerStub并向服务端注册
            await self.learner_stub.start_listening()
            
            # 执行客户端注册
            await self._register_to_server()
            
            self.is_running = True
            print("FederationClient started successfully")
            return True
            
        except Exception as e:
            print(f"Failed to start client: {e}")
            return False
    
    async def _register_to_server(self) -> bool:
        """向服务端注册"""
        try:
            # 通过LearnerStub向服务端注册
            registration_result = await self.learner_stub.register_to_server()
            
            if registration_result.success:
                self.is_registered = True
                print(f"Client {self.client_id} registered to server successfully")
                return True
            else:
                error_msg = getattr(registration_result, 'error_message', 'Unknown error')
                print(f"Client {self.client_id} registration failed: {error_msg}")
                return False
                
        except Exception as e:
            print(f"Client registration error: {e}")
            return False
    
    async def stop_client(self) -> bool:
        """停止客户端"""
        if not self.is_running:
            return True
        
        try:
            # 从服务端注销
            if self.is_registered:
                await self.learner_stub.unregister_from_server()
                self.is_registered = False
            
            # 停止LearnerStub
            if self.learner_stub:
                await self.learner_stub.stop_listening()
            
            # 按相反顺序停止组件
            if hasattr(self.connection_manager, 'stop'):
                await self.connection_manager.stop()
            
            if hasattr(self.communication_manager, 'stop'):
                await self.communication_manager.stop()
            
            if hasattr(self.transport, 'stop'):
                await self.transport.stop()
            
            self.is_running = False
            print("FederationClient stopped successfully")
            return True
            
        except Exception as e:
            print(f"Failed to stop client: {e}")
            return False
    
    def get_learner(self) -> Optional[BaseLearner]:
        """获取学习器实例"""
        return self.learner
    
    def get_client_status(self) -> Dict[str, Any]:
        """获取客户端状态"""
        return {
            "client_id": self.client_id,
            "mode": self.mode.value,
            "is_initialized": self.is_initialized,
            "is_running": self.is_running,
            "is_registered": self.is_registered,
            "learner_type": type(self.learner).__name__ if self.learner else None,
            "registration_status": self.learner_stub.get_registration_status() if self.learner_stub else None
        }
    
    def _generate_client_id(self) -> str:
        """生成客户端ID"""
        import uuid
        unique_id = str(uuid.uuid4())[:8]
        
        if self.mode == CommunicationMode.MEMORY:
            return f"memory_client_{unique_id}"
        elif self.mode == CommunicationMode.PROCESS:
            import os
            pid = os.getpid()
            port = self.config.get("port", 8001)
            return f"process_client_{pid}_{port}_{unique_id}"
        elif self.mode == CommunicationMode.NETWORK:
            host = self.config.get("host", "localhost")
            port = self.config.get("port", 8001)
            return f"network_client_{host}_{port}_{unique_id}"
        else:
            return f"unknown_client_{unique_id}"
    
    @classmethod
    def create_client(cls, config: Dict[str, Any], client_id: str = None) -> 'FederationClient':
        """工厂方法：创建客户端实例"""
        return cls(config, client_id)
