# fedcl/communication/message_handler.py
"""
消息处理器

负责处理联邦学习中的消息路由和处理，包括模型分发、更新收集等。
"""

from typing import Dict, Any, Optional, Callable
import asyncio
from enum import Enum
from dataclasses import dataclass
from loguru import logger


class MessageType(Enum):
    """消息类型枚举"""
    MODEL_DISTRIBUTION = "model_distribution"
    MODEL_UPDATE = "model_update"
    CLIENT_REGISTRATION = "client_registration"
    TRAINING_START = "training_start"
    TRAINING_COMPLETE = "training_complete"
    HEARTBEAT = "heartbeat"


@dataclass
class Message:
    """消息数据结构"""
    message_type: MessageType
    sender_id: str
    receiver_id: str
    payload: Dict[str, Any]
    timestamp: float
    message_id: str


class MessageHandler:
    """
    消息处理器基类
    
    定义消息处理的基本接口和路由逻辑
    """
    
    def __init__(self, component_id: str):
        self.component_id = component_id
        self.message_handlers: Dict[MessageType, Callable] = {}
        self.logger = logger.bind(component=f"MessageHandler[{component_id}]")
        
    def register_handler(self, message_type: MessageType, handler: Callable):
        """注册消息处理器"""
        self.message_handlers[message_type] = handler
        self.logger.debug(f"Registered handler for {message_type}")
        
    async def handle_message(self, message: Message) -> Optional[Dict[str, Any]]:
        """处理消息"""
        try:
            if message.message_type not in self.message_handlers:
                self.logger.warning(f"No handler for message type: {message.message_type}")
                return {"status": "error", "message": "No handler found"}
            
            handler = self.message_handlers[message.message_type]
            result = await handler(message)
            
            self.logger.debug(f"Processed message {message.message_id}: {message.message_type}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error handling message {message.message_id}: {e}")
            return {"status": "error", "message": str(e)}


class ClientMessageHandler(MessageHandler):
    """客户端消息处理器"""
    
    def __init__(self, client_id: str, federated_client):
        super().__init__(client_id)
        self.federated_client = federated_client
        self._register_default_handlers()
        
    def _register_default_handlers(self):
        """注册默认的消息处理器"""
        self.register_handler(MessageType.MODEL_DISTRIBUTION, self._handle_model_distribution)
        self.register_handler(MessageType.TRAINING_START, self._handle_training_start)
        self.register_handler(MessageType.HEARTBEAT, self._handle_heartbeat)
        
    async def _handle_model_distribution(self, message: Message) -> Dict[str, Any]:
        """处理模型分发消息"""
        try:
            payload = message.payload
            round_id = payload.get('round_id')
            model_state_dict = payload.get('model_state_dict')
            
            if not model_state_dict:
                return {"status": "error", "message": "No model state dict in payload"}
            
            # 重构：不再直接调用 receive_global_model，而是保存模型并触发训练
            self.federated_client._receive_and_store_global_model(model_state_dict, round_id)
            
            self.logger.debug(f"Received and stored global model for round {round_id}")
            return {"status": "success", "round_id": round_id}
            
        except Exception as e:
            self.logger.error(f"Failed to handle model distribution: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _handle_training_start(self, message: Message) -> Dict[str, Any]:
        """处理训练开始消息"""
        try:
            payload = message.payload
            round_id = payload.get('round_id')
            
            # 触发本地训练
            result = await self.federated_client._start_local_training(round_id)
            
            return {"status": "success", "training_result": result}
            
        except Exception as e:
            self.logger.error(f"Failed to start training: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _handle_heartbeat(self, message: Message) -> Dict[str, Any]:
        """处理心跳消息"""
        return {
            "status": "alive",
            "client_id": self.component_id,
            "timestamp": message.timestamp
        }


class ServerMessageHandler(MessageHandler):
    """服务器消息处理器"""
    
    def __init__(self, server_id: str, federated_server):
        super().__init__(server_id)
        self.federated_server = federated_server
        self._register_default_handlers()
        
    def _register_default_handlers(self):
        """注册默认的消息处理器"""
        self.register_handler(MessageType.MODEL_UPDATE, self._handle_model_update)
        self.register_handler(MessageType.CLIENT_REGISTRATION, self._handle_client_registration)
        self.register_handler(MessageType.TRAINING_COMPLETE, self._handle_training_complete)
        
    async def _handle_model_update(self, message: Message) -> Dict[str, Any]:
        """处理客户端模型更新"""
        try:
            payload = message.payload
            client_id = message.sender_id
            
            # 将模型更新传递给服务器
            self.federated_server._receive_client_update(client_id, payload)
            
            self.logger.debug(f"收到来自客户端的模型更新 {client_id}")
            return {"status": "success"}
            
        except Exception as e:
            self.logger.error(f"处理模型更新失败: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _handle_client_registration(self, message: Message) -> Dict[str, Any]:
        """处理客户端注册"""
        try:
            payload = message.payload
            client_id = message.sender_id
            
            # 注册客户端
            success = self.federated_server._register_client(client_id, payload)
            
            return {"status": "success" if success else "失败"}
            
        except Exception as e:
            self.logger.error(f"处理客户端注册失败: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _handle_training_complete(self, message: Message) -> Dict[str, Any]:
        """处理训练完成通知"""
        try:
            payload = message.payload
            client_id = message.sender_id
            round_id = payload.get('round_id')
            
            self.federated_server._mark_client_training_complete(client_id, round_id)
            
            return {"status": "success"}
            
        except Exception as e:
            self.logger.error(f"Failed to handle training complete: {e}")
            return {"status": "error", "message": str(e)}
