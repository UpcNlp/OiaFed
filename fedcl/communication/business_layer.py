"""
业务通信层 - 负责创建和管理LearnerProxy
fedcl/communication/business_layer.py
"""

from typing import Dict, Any, Optional
from ..learner.proxy import LearnerProxy, ProxyConfig
from ..communication.layer_event import LayerEventHandler
from ..exceptions import CommunicationError
from ..utils.auto_logger import get_comm_logger


class BusinessCommunicationLayer(LayerEventHandler):
    """第2层：业务通信层 - 负责创建LearnerProxy"""
    
    def __init__(self, upper_layer: Optional[LayerEventHandler] = None):
        super().__init__(upper_layer)
        self.created_proxies: Dict[str, LearnerProxy] = {}
        self.communication_manager = None
        self.connection_manager = None
        self.logger = get_comm_logger("business_layer")
    
    def set_dependencies(self, communication_manager, connection_manager):
        """设置依赖的下层组件"""
        self.communication_manager = communication_manager
        self.connection_manager = connection_manager
    
    def handle_layer_event(self, event_type: str, event_data: Dict[str, Any]):
        """处理层间事件 - 严格按照层次分离原则"""
        self.logger.info(f"[第2层-业务通信层] 收到事件: {event_type}, 数据: {event_data}")
        
        if event_type == "CONNECTION_ESTABLISHED":
            # 统一处理连接建立事件（包括来自下层转换的客户端注册事件）
            self.logger.info(f"[第2层-业务通信层] 处理连接建立事件")
            self._handle_connection_established(event_data)
        
        elif event_type == "CONNECTION_LOST":
            client_id = event_data["client_id"]
            self.logger.info(f"[第2层-业务通信层] 处理连接丢失事件: {client_id}")
            
            # 清理断开的代理
            if client_id in self.created_proxies:
                del self.created_proxies[client_id]
                
                # 向上传递代理断开信息
                self.logger.info(f"[第2层-业务通信层] 向上传递代理断开事件: {client_id}")
                self.propagate_to_upper("LEARNER_PROXY_DISCONNECTED", {
                    "client_id": client_id
                })
                
                self.logger.info(f"[第2层-业务通信层] 客户端[{client_id}] 断开连接，学习器代理已被移除")
        
        else:
            self.logger.warning(f"[第2层-业务通信层] 未知事件类型：'{event_type}'，忽略")
    
    def _handle_connection_established(self, event_data: Dict[str, Any]):
        """统一处理连接建立事件 - 包括客户端注册和连接建立"""
        client_id = event_data.get("client_id")
        if not client_id:
            self.logger.error("[第2层-业务通信层] 建立连接时缺少客户端ID")
            return
        
        connection = event_data.get("connection")
        connection_config = event_data.get("connection_config", {})
        
        self.logger.info(f"[第2层-业务通信层] 开始处理客户端[{client_id}]的连接建立")
        self.logger.debug(f"[第2层-业务通信层] 连接配置: {connection_config}")
        
        # 如果没有提供连接配置，使用默认配置
        if not connection_config:
            connection_config = {
                "timeout": 120.0,
                "retry_attempts": 3
            }
            self.logger.info(f"[第2层-业务通信层] 使用默认连接配置: {connection_config}")
        
        # 🎯 核心：创建LearnerProxy（本层职责）
        try:
            self.logger.info(f"[第2层-业务通信层] 正在为客户端[{client_id}]创建学习器代理...")
            proxy = self._create_learner_proxy(client_id, connection, connection_config)
            
            # 设置代理为连接状态 (Memory模式下代理立即可用)
            from fedcl.types import ConnectionStatus
            proxy._connection_status = ConnectionStatus.CONNECTED
            
            self.created_proxies[client_id] = proxy
            self.logger.info(f"[第2层-业务通信层] 学习器代理创建成功: {client_id}")
            
            # 获取代理能力
            proxy_capabilities = self._get_proxy_capabilities(proxy)
            self.logger.debug(f"[第2层-业务通信层] 代理能力: {proxy_capabilities}")
            
            # 向上传递代理就绪信息
            self.logger.info(f"[第2层-业务通信层] 向上传递LEARNER_PROXY_READY事件: {client_id}")
            self.propagate_to_upper("LEARNER_PROXY_READY", {
                "client_id": client_id,
                "proxy": proxy,
                "proxy_capabilities": proxy_capabilities
            })
            
        except Exception as e:
            self.logger.error(f"[第2层-业务通信层] 为客户端[{client_id}]创建学习器代理失败: {e}")
            import traceback
            self.logger.error(f"[第2层-业务通信层] 错误详情: {traceback.format_exc()}")
            raise CommunicationError(f"Failed to create learner proxy: {str(e)}")

    def _create_learner_proxy(self, client_id: str, connection, connection_config: Dict[str, Any]) -> LearnerProxy:
        """创建学习器代理"""
        # 创建代理配置
        proxy_config = ProxyConfig(
            default_timeout=connection_config.get("timeout", 120.0),
            max_retries=connection_config.get("retry_attempts", 3),
            **connection_config.get("proxy_config", {})
        )
        
        # 创建LearnerProxy实例
        proxy = LearnerProxy(
            client_id=client_id,
            communication_manager=self.communication_manager,
            connection_manager=self.connection_manager,
            config=proxy_config
        )
        
        return proxy
    
    def _get_proxy_capabilities(self, proxy: LearnerProxy) -> Dict[str, Any]:
        """获取代理能力信息"""
        try:
            # 可以调用代理的能力检测方法
            return {
                "methods": ["train", "evaluate", "get_model", "set_model"],
                "ready": proxy.is_client_ready() if hasattr(proxy, 'is_client_ready') else False
            }
        except Exception:
            return {"methods": [], "ready": False}
    
    def get_proxy(self, client_id: str) -> Optional[LearnerProxy]:
        """供外部获取代理（如果需要）"""
        return self.created_proxies.get(client_id)
    
    def get_all_proxies(self) -> Dict[str, LearnerProxy]:
        """获取所有已创建的代理"""
        return self.created_proxies.copy()
