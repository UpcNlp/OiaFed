# fedcl/communication/message_protocol.py
"""
消息协议模块

实现FedCL框架的消息协议，包括消息的序列化、反序列化、验证和压缩功能。
支持多种消息类型，确保分布式通信的可靠性和安全性。
"""

import json
import gzip
import hashlib
import uuid
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Union
from loguru import logger

from .exceptions import ProtocolError, SerializationError


@dataclass
class Message:
    """
    消息数据结构
    
    定义了联邦学习通信中使用的标准消息格式。
    """
    message_id: str
    message_type: str
    sender: str
    receiver: str
    timestamp: datetime
    data: Any
    metadata: Dict[str, Any]
    checksum: str
    
    def validate_checksum(self) -> bool:
        """
        验证消息校验和
        
        Returns:
            bool: 校验和是否有效
        """
        try:
            calculated_checksum = self.calculate_checksum()
            return calculated_checksum == self.checksum
        except Exception as e:
            logger.error(f"Failed to validate checksum: {e}")
            return False
    
    def calculate_checksum(self) -> str:
        """
        计算消息校验和
        
        Returns:
            str: 消息的MD5校验和
        """
        try:
            # 创建消息内容的字符串表示（不包括checksum字段）
            content_dict = {
                'message_id': self.message_id,
                'message_type': self.message_type,
                'sender': self.sender,
                'receiver': self.receiver,
                'timestamp': self.timestamp.isoformat(),
                'metadata': self.metadata
            }
            
            # 处理数据字段，如果是bytes则使用特殊表示
            if isinstance(self.data, bytes):
                content_dict['data'] = f"<bytes:{len(self.data)}:{hashlib.md5(self.data).hexdigest()}>"
            else:
                content_dict['data'] = str(self.data)
            
            content_str = json.dumps(content_dict, sort_keys=True)
            return hashlib.md5(content_str.encode('utf-8')).hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate checksum: {e}")
            raise SerializationError(f"Failed to calculate checksum: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将消息转换为字典格式
        
        Returns:
            Dict[str, Any]: 消息的字典表示
        """
        try:
            result = asdict(self)
            result['timestamp'] = self.timestamp.isoformat()
            return result
        except Exception as e:
            logger.error(f"Failed to convert message to dict: {e}")
            raise SerializationError(f"Failed to convert message to dict: {e}")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """
        从字典创建消息对象
        
        Args:
            data: 消息字典数据
            
        Returns:
            Message: 消息对象
        """
        try:
            # 处理时间戳
            if isinstance(data['timestamp'], str):
                data['timestamp'] = datetime.fromisoformat(data['timestamp'])
            
            return cls(**data)
        except Exception as e:
            logger.error(f"Failed to create message from dict: {e}")
            raise SerializationError(f"Failed to create message from dict: {e}")


class MessageProtocol:
    """
    消息协议类
    
    负责消息的序列化、反序列化、验证和压缩等功能。
    支持联邦学习中的各种消息类型。
    """
    
    # 消息类型常量
    MODEL_UPDATE = "model_update"
    GLOBAL_MODEL = "global_model"
    CLIENT_READY = "client_ready"
    TRAINING_COMPLETE = "training_complete"
    HEARTBEAT = "heartbeat"
    ERROR_REPORT = "error_report"
    SHUTDOWN = "shutdown"
    CLIENT_REGISTRATION = "client_registration"
    REGISTRATION_ACK = "registration_ack"
    ROUND_COMPLETE = "round_complete"
    EXPERIMENT_END = "experiment_end"
    CLIENT_DISCONNECT = "client_disconnect"
    FEATURE_TRANSFER = "feature_transfer"
    GRADIENT_RETURN = "gradient_return"
    
    # 支持的消息类型集合
    SUPPORTED_MESSAGE_TYPES = {
        MODEL_UPDATE, GLOBAL_MODEL, CLIENT_READY, TRAINING_COMPLETE,
        HEARTBEAT, ERROR_REPORT, SHUTDOWN, CLIENT_REGISTRATION,
        REGISTRATION_ACK, ROUND_COMPLETE, EXPERIMENT_END,
        CLIENT_DISCONNECT, FEATURE_TRANSFER, GRADIENT_RETURN
    }
    
    def __init__(self, version: str = "1.0") -> None:
        """
        初始化消息协议
        
        Args:
            version: 协议版本号
        """
        self.version = version
        self.encoding = 'utf-8'
        logger.debug(f"MessageProtocol initialized with version {version}")
    
    def serialize_message(self, message_type: str, data: Any,
                         metadata: Optional[Dict] = None) -> bytes:
        """
        序列化消息
        
        Args:
            message_type: 消息类型
            data: 消息数据
            metadata: 元数据
            
        Returns:
            bytes: 序列化后的消息字节流
        """
        try:
            # 验证消息类型
            if message_type not in self.SUPPORTED_MESSAGE_TYPES:
                raise ProtocolError(f"Unsupported message type: {message_type}")
            
            # 创建消息对象
            message = self.create_message(
                message_type=message_type,
                data=data,
                sender="unknown",  # 这里应该由调用者提供
                receiver="unknown",  # 这里应该由调用者提供
                metadata=metadata
            )
            
            # 使用pickle序列化整个消息对象
            import pickle
            return pickle.dumps(message)
            
        except ProtocolError:
            # 重新抛出ProtocolError，不要包装成SerializationError
            raise
        except Exception as e:
            logger.error(f"Failed to serialize message: {e}")
            raise SerializationError(f"Failed to serialize message: {e}")
    
    def deserialize_message(self, raw_data: bytes) -> Message:
        """
        反序列化消息
        
        Args:
            raw_data: 原始字节数据
            
        Returns:
            Message: 反序列化后的消息对象
        """
        try:
            # 尝试使用pickle反序列化
            import pickle
            try:
                message = pickle.loads(raw_data)
                # 验证消息
                if not self.validate_message(message):
                    raise ProtocolError("Message validation failed")
                return message
            except (pickle.PickleError, UnicodeDecodeError):
                # 如果pickle失败，尝试JSON格式
                import json
                json_str = raw_data.decode('utf-8')
                message_dict = json.loads(json_str)
                message = Message.from_dict(message_dict)
                
                # 验证消息
                if not self.validate_message(message):
                    raise ProtocolError("Message validation failed")
                return message
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            raise SerializationError(f"JSON decode error: {e}")
        except pickle.PickleError as e:
            logger.error(f"Pickle error: {e}")
            raise SerializationError(f"Pickle error: {e}")
        except Exception as e:
            logger.error(f"Failed to deserialize message: {e}")
            raise SerializationError(f"Failed to deserialize message: {e}")
    
    def validate_message(self, message: Message) -> bool:
        """
        验证消息格式
        
        Args:
            message: 待验证的消息
            
        Returns:
            bool: 消息是否有效
        """
        try:
            # 检查必需字段
            required_fields = ['message_id', 'message_type', 'sender', 'receiver', 'timestamp']
            for field in required_fields:
                if not hasattr(message, field):
                    logger.warning(f"Missing required field: {field}")
                    return False
                # 允许空字符串，但不允许None
                value = getattr(message, field)
                if value is None:
                    logger.warning(f"Required field {field} is None")
                    return False
                # message_id不能为空
                if field == 'message_id' and (not value or not str(value).strip()):
                    logger.warning(f"Required field {field} is empty")
                    return False
            
            # sender和receiver不能都为空
            if (not message.sender or not message.sender.strip()) and (not message.receiver or not message.receiver.strip()):
                logger.warning("Both sender and receiver cannot be empty")
                return False
            
            # 检查消息类型
            if message.message_type not in self.SUPPORTED_MESSAGE_TYPES:
                logger.warning(f"Unsupported message type: {message.message_type}")
                return False
            
            # 验证校验和
            if not message.validate_checksum():
                logger.warning("Checksum validation failed")
                return False
            
            # 检查时间戳格式
            if not isinstance(message.timestamp, datetime):
                logger.warning("Invalid timestamp format")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Message validation error: {e}")
            return False
    
    def create_message(self, message_type: str, data: Any,
                      sender: str, receiver: str,
                      metadata: Optional[Dict] = None) -> Message:
        """
        创建消息对象
        
        Args:
            message_type: 消息类型
            data: 消息数据
            sender: 发送方ID
            receiver: 接收方ID
            metadata: 元数据
            
        Returns:
            Message: 创建的消息对象
        """
        try:
            # 验证消息类型
            if message_type not in self.SUPPORTED_MESSAGE_TYPES:
                raise ProtocolError(f"Unsupported message type: {message_type}")
            
            # 生成消息ID
            message_id = str(uuid.uuid4())
            
            # 创建时间戳
            timestamp = datetime.now()
            
            # 处理元数据
            if metadata is None:
                metadata = {}
            
            # 添加协议版本到元数据
            metadata['protocol_version'] = self.version
            
            # 创建消息对象（不包含checksum）
            message = Message(
                message_id=message_id,
                message_type=message_type,
                sender=sender,
                receiver=receiver,
                timestamp=timestamp,
                data=data,
                metadata=metadata,
                checksum=""  # 临时设置为空
            )
            
            # 计算并设置校验和
            message.checksum = message.calculate_checksum()
            
            return message
            
        except Exception as e:
            logger.error(f"Failed to create message: {e}")
            raise ProtocolError(f"Failed to create message: {e}")
    
    def get_message_size(self, message: Message) -> int:
        """
        获取消息大小
        
        Args:
            message: 消息对象
            
        Returns:
            int: 消息大小（字节）
        """
        try:
            message_dict = message.to_dict()
            
            # 递归处理嵌套的bytes数据
            def handle_bytes_in_dict(obj):
                if isinstance(obj, bytes):
                    return f"<bytes:{len(obj)}>"
                elif isinstance(obj, dict):
                    return {k: handle_bytes_in_dict(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [handle_bytes_in_dict(item) for item in obj]
                else:
                    return obj
            
            # 处理可能包含bytes的数据
            safe_dict = handle_bytes_in_dict(message_dict)
            json_str = json.dumps(safe_dict, ensure_ascii=False)
            
            # 计算实际大小：JSON字符串大小 + 所有bytes数据的实际大小
            def calculate_bytes_size(obj):
                total_size = 0
                if isinstance(obj, bytes):
                    total_size += len(obj)
                elif isinstance(obj, dict):
                    for v in obj.values():
                        total_size += calculate_bytes_size(v)
                elif isinstance(obj, list):
                    for item in obj:
                        total_size += calculate_bytes_size(item)
                return total_size
            
            bytes_size = calculate_bytes_size(message_dict)
            json_size = len(json_str.encode(self.encoding))
            
            return json_size + bytes_size
            
        except Exception as e:
            logger.error(f"Failed to get message size: {e}")
            raise SerializationError(f"Failed to get message size: {e}")
    
    def compress_message(self, message: Message) -> Message:
        """
        压缩消息
        
        Args:
            message: 原始消息
            
        Returns:
            Message: 压缩后的消息
        """
        try:
            # 将整个消息数据使用pickle序列化，然后压缩
            import pickle
            data_bytes = pickle.dumps(message.data)
            compressed_data = gzip.compress(data_bytes)
            original_size = len(data_bytes)
            
            # 创建新的消息，包含压缩后的数据
            compressed_message = Message(
                message_id=message.message_id,
                message_type=message.message_type,
                sender=message.sender,
                receiver=message.receiver,
                timestamp=message.timestamp,
                data=compressed_data,
                metadata={**message.metadata, 'compressed': True, 'original_size': original_size},
                checksum=""
            )
            
            # 重新计算校验和
            compressed_message.checksum = compressed_message.calculate_checksum()
            
            logger.debug(f"Message compressed: {original_size} -> {len(compressed_data)} bytes")
            return compressed_message
            
        except Exception as e:
            logger.error(f"Failed to compress message: {e}")
            raise SerializationError(f"Failed to compress message: {e}")
    
    def decompress_message(self, message: Message) -> Message:
        """
        解压消息
        
        Args:
            message: 压缩的消息
            
        Returns:
            Message: 解压后的消息
        """
        try:
            # 检查是否为压缩消息
            if not message.metadata.get('compressed', False):
                logger.warning("Message is not compressed")
                return message
            
            # 解压数据
            compressed_data = message.data
            if isinstance(compressed_data, bytes):
                decompressed_bytes = gzip.decompress(compressed_data)
                
                # 使用pickle反序列化
                import pickle
                decompressed_data = pickle.loads(decompressed_bytes)
            else:
                raise SerializationError("Invalid compressed data format")
            
            # 创建解压后的消息
            decompressed_metadata = {k: v for k, v in message.metadata.items() 
                                   if k not in ['compressed', 'original_size']}
            
            decompressed_message = Message(
                message_id=message.message_id,
                message_type=message.message_type,
                sender=message.sender,
                receiver=message.receiver,
                timestamp=message.timestamp,
                data=decompressed_data,
                metadata=decompressed_metadata,
                checksum=""
            )
            
            # 重新计算校验和
            decompressed_message.checksum = decompressed_message.calculate_checksum()
            
            logger.debug("Message decompressed successfully")
            return decompressed_message
            
        except Exception as e:
            logger.error(f"Failed to decompress message: {e}")
            raise SerializationError(f"Failed to decompress message: {e}")
