"""
联邦服务端管理器 - 负责服务端组件的初始化和管理
fedcl/federation/server.py
"""

import asyncio
from typing import Dict, Any, Type, Optional
from ..trainer.base_trainer import BaseTrainer
from ..communication.business_layer import BusinessCommunicationLayer
from ..connection.manager import ConnectionManager
from ..communication.base import CommunicationManagerBase
from ..transport.base import TransportBase
from ..factory.factory import ComponentFactory
from ..types import CommunicationMode, ModelData
from ..exceptions import FederationError
from ..utils.auto_logger import get_sys_logger


class FederationServer:
    """联邦服务端管理器 - 专门负责服务端组件的初始化、装配和管理"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_sys_logger()
        self.mode = CommunicationMode(config.get("mode", "memory"))
        self.server_id = self._generate_server_id()
        
        # 组件引用
        self.transport: Optional[TransportBase] = None
        self.communication_manager: Optional[CommunicationManagerBase] = None
        self.connection_manager: Optional[ConnectionManager] = None
        self.business_layer: Optional[BusinessCommunicationLayer] = None
        self.trainer: Optional[BaseTrainer] = None
        
        # 状态管理
        self.is_initialized = False
        self.is_running = False
        
        self.logger.info(f"使用模式：{self.mode}创建联邦服务器, server_id: {self.server_id}")
    
    async def initialize_with_trainer(self, 
                                    trainer_class: Type[BaseTrainer],
                                    global_model: ModelData,
                                    trainer_config: Dict[str, Any] = None) -> BaseTrainer:
        """初始化服务端并创建trainer
        
        Args:
            trainer_class: 用户的训练器类
            global_model: 全局模型
            trainer_config: 训练器配置
            
        Returns:
            BaseTrainer: 初始化好的训练器实例
        """
        if self.is_initialized:
            raise FederationError("Server already initialized")
        
        try:
            # 1. 创建trainer实例
            self.trainer = trainer_class(
                global_model=global_model,
                training_config=trainer_config,
            )
            
            # 2. 初始化通信组件栈（严格按层次顺序）
            await self._initialize_communication_stack()
            
            # 3. 建立层间关系链
            self._establish_layer_relationships()
            
            # 4. 标记初始化完成
            self.is_initialized = True
            
            self.logger.info("联邦服务器初始化成功")
            return self.trainer
            
        except Exception as e:
            self.logger.error(f"FederationServer initialization failed: {e}")
            raise FederationError(f"Server initialization failed: {str(e)}")
    
    async def _initialize_communication_stack(self):
        """初始化通信组件栈 - 严格按照层次顺序"""
        factory = ComponentFactory(self.config)
        
        # 第5层：创建传输层（最底层，无依赖）
        transport_config = factory._create_transport_config(self.config, self.mode)
        self.transport = factory.create_transport(transport_config, self.mode)
        self.logger.info(f"Layer 5: Transport layer created - {type(self.transport).__name__}")
        
        # 第4层：创建通用通信层（依赖传输层）
        communication_config = factory._create_communication_config(self.config)
        self.communication_manager = factory.create_communication_manager(
            self.server_id, self.transport, communication_config, self.mode
        )
        self.logger.info(f"Layer 4: Communication manager created - {type(self.communication_manager).__name__}")
        
        # 第3层：创建连接管理层（依赖通信层）
        self.connection_manager = factory.create_connection_manager(
            self.communication_manager, communication_config
        )
        self.logger.info(f"Layer 3: Connection manager created - {type(self.connection_manager).__name__}")
        
        # 第2层：创建业务通信层（依赖连接层）
        self.business_layer = BusinessCommunicationLayer()
        self.business_layer.set_dependencies(
            self.communication_manager, 
            self.connection_manager
        )
        self.logger.info("Layer 2: Business communication layer created")
    
    def _establish_layer_relationships(self):
        """建立层间关系链 - 确保事件能够正确向上传递"""
        # 建立向上传递链：
        # ConnectionManager → BusinessCommunicationLayer → ProxyManager
        
        # 连接管理层的上层是业务通信层
        self.connection_manager.set_upper_layer(self.business_layer)
        
        # 业务通信层的上层是trainer的代理事件处理器
        self.business_layer.set_upper_layer(self.trainer._proxy_event_handler)
        
        # 🎯 关键修复：监听传输层的CLIENT_REGISTERED事件（内存模式）
        def handle_transport_client_registered(data):
            """处理传输层的客户端注册事件"""
            print(f"[传输层事件桥接] *** 收到事件调用 *** 数据: {data}")
            client_id = data.get("client_id")
            if client_id:
                self.logger.info(f"[传输层事件桥接] 收到CLIENT_REGISTERED事件: {client_id}")
                
                # 直接调用ConnectionManager处理层间事件
                self.connection_manager.handle_layer_event("CLIENT_REGISTERED", {
                    "client_id": client_id,
                    "event_data": data,
                    "timestamp": data.get("timestamp")
                })
            else:
                print(f"[传输层事件桥接] *** 事件数据中没有client_id: {data} ***")
        
        # 注册传输层事件监听器 - 监听"system"目标的事件
        self.transport.register_event_listener("system", "CLIENT_REGISTERED", handle_transport_client_registered)
        self.logger.info("[传输层事件桥接] 已注册CLIENT_REGISTERED事件监听器（目标：system）")
        
        # 🎯 关键修复：监听CommunicationManager的注册事件，并转换为层间事件
        def handle_client_registration_event(event):
            """处理客户端注册事件并转换为层间事件"""
            self.logger.info(f"[事件桥接] 收到注册服务事件: {event.event_type}, 源: {event.source_id}")
            
            if event.event_type == "CLIENT_REGISTERED":
                client_id = event.source_id
                self.logger.info(f"[事件桥接] 转换CLIENT_REGISTERED为层间事件: {client_id}")
                
                # 通过第3层（ConnectionManager）处理第4层事件
                self.logger.info(f"[事件桥接] 向第3层传递CLIENT_REGISTERED事件: {client_id}")
                self.connection_manager.handle_layer_event("CLIENT_REGISTERED", {
                    "client_id": client_id,
                    "event_data": event.data,
                    "timestamp": event.data.get("timestamp") if hasattr(event, 'data') else None
                })
        
        # 注册事件回调到CommunicationManager的RegistryService
        self.logger.debug(f"[事件桥接] 正在注册事件回调到RegistryService...")
        self.logger.debug(f"[事件桥接] RegistryService实例: {id(self.communication_manager.registry_service)}")
        try:
            self.communication_manager.registry_service.register_event_callback(handle_client_registration_event)
            self.logger.debug(f"[事件桥接] 事件回调注册成功")
        except Exception as e:
            self.logger.error(f"[事件桥接] 事件回调注册失败: {e}")
            import traceback
            self.logger.error(f"[事件桥接] 错误详情: {traceback.format_exc()}")
        
        callback_count = len(self.communication_manager.registry_service.event_callbacks)
        self.logger.debug(f"[事件桥接] 当前注册的回调数量: {callback_count}")
        
        # 列出所有回调的详细信息
        for i, cb in enumerate(self.communication_manager.registry_service.event_callbacks):
            cb_name = cb.__name__ if hasattr(cb, '__name__') else str(cb)
            self.logger.debug(f"[事件桥接] 回调 #{i+1}: {cb_name}")
        
        if callback_count >= 2:
            self.logger.info("[事件桥接] 桥接回调注册成功")
        else:
            self.logger.warning("[事件桥接] 桥接回调可能注册失败，回调数量不正确")
        
        self.logger.info("🔗 层级关系建立完成，事件桥接已激活")
        
        self.logger.info("层级关系建立完成，事件桥接已激活")
    
    async def start_server(self) -> bool:
        """启动服务端"""
        print(f"🟢 [Server] 开始启动服务器: {self.server_id}")
        
        if not self.is_initialized:
            raise FederationError("Server not initialized")
        
        if self.is_running:
            print(f"🟡 [Server] 服务器已经在运行")
            return True
        
        try:
            # 启动各层组件
            print(f"🚀 [Server] 启动传输层...")
            if hasattr(self.transport, 'start'):
                await self.transport.start()
            
            print(f"🌐 [Server] 启动通信管理器...")
            if hasattr(self.communication_manager, 'start'):
                await self.communication_manager.start()
            
            print(f"🔗 [Server] 启动连接管理器...")
            if hasattr(self.connection_manager, 'start'):
                await self.connection_manager.start()
            
            # 初始化trainer
            trainer_ready = await self.trainer.initialize()
            if not trainer_ready:
                raise FederationError("Trainer initialization failed")
            
            self.is_running = True
            print("FederationServer started successfully")
            return True
            
        except Exception as e:
            print(f"Failed to start server: {e}")
            return False
    
    async def stop_server(self) -> bool:
        """停止服务端"""
        if not self.is_running:
            return True
        
        try:
            # 按相反顺序停止组件
            if self.trainer:
                await self.trainer.cleanup()
            
            if hasattr(self.connection_manager, 'stop'):
                await self.connection_manager.stop()
            
            if hasattr(self.communication_manager, 'stop'):
                await self.communication_manager.stop()
            
            if hasattr(self.transport, 'stop'):
                await self.transport.stop()
            
            self.is_running = False
            print("FederationServer stopped successfully")
            return True
            
        except Exception as e:
            print(f"Failed to stop server: {e}")
            return False
    
    def get_trainer(self) -> Optional[BaseTrainer]:
        """获取训练器实例"""
        return self.trainer
    
    def get_server_status(self) -> Dict[str, Any]:
        """获取服务端状态"""
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
            port = self.config.get("port", 8000)
            return f"process_server_{port}"
        elif self.mode == CommunicationMode.NETWORK:
            host = self.config.get("host", "localhost")
            port = self.config.get("port", 8000)
            return f"network_server_{host}_{port}"
        else:
            return "unknown_server"
    
    @classmethod
    def create_server(cls, config: Dict[str, Any]) -> 'FederationServer':
        """工厂方法：创建服务端实例"""
        return cls(config)
