"""
层间事件处理接口
fedcl/communication/layer_event.py
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class LayerEventHandler(ABC):
    """层间事件处理接口 - 确保严格的向上传递"""
    
    def __init__(self, upper_layer: Optional['LayerEventHandler'] = None):
        self.upper_layer = upper_layer
    
    @abstractmethod
    def handle_layer_event(self, event_type: str, event_data: Dict[str, Any]):
        """处理本层事件
        
        Args:
            event_type: 事件类型
            event_data: 事件数据
        """
        pass
    
    def propagate_to_upper(self, event_type: str, event_data: Dict[str, Any]):
        """向上层传递事件
        
        Args:
            event_type: 事件类型  
            event_data: 事件数据
        """
        if self.upper_layer:
            self.upper_layer.handle_layer_event(event_type, event_data)
    
    def set_upper_layer(self, upper_layer: 'LayerEventHandler'):
        """设置上层处理器"""
        self.upper_layer = upper_layer


class ProxyManagerEventHandler(LayerEventHandler):
    """代理管理器事件处理器 - BaseTrainer的代理管理组件"""
    
    def __init__(self, proxy_manager):
        super().__init__()
        self.proxy_manager = proxy_manager
        # 导入日志记录器
        from ..utils.auto_logger import get_comm_logger
        self.logger = get_comm_logger("proxy_manager_handler")
    
    def handle_layer_event(self, event_type: str, event_data: Dict[str, Any]):
        """处理代理相关事件"""
        self.logger.info(f"[第1层-应用层] 代理管理器收到事件: {event_type}, 数据: {event_data}")
        
        if event_type == "LEARNER_PROXY_READY":
            client_id = event_data["client_id"]
            proxy = event_data["proxy"]
            
            self.logger.info(f"[第1层-应用层] 学习器代理就绪: {client_id}")
            
            # 异步处理代理就绪事件
            import asyncio
            asyncio.create_task(self.proxy_manager.on_proxy_ready(client_id, proxy))
            
        elif event_type == "LEARNER_PROXY_DISCONNECTED":
            client_id = event_data["client_id"]
            
            self.logger.info(f"[第1层-应用层] 学习器代理断开: {client_id}")
            
            # 异步处理代理断开事件
            import asyncio
            asyncio.create_task(self.proxy_manager.on_proxy_disconnected(client_id))
        
        else:
            self.logger.warning(f"[第1层-应用层] 未知事件类型: {event_type}")
