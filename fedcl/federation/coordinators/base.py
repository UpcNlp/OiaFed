# fedcl/communication/federated_communicator.py
"""
联邦学习通信中间层基类

提供统一的通信接口，客户端和服务端都通过继承这个类进行交互。
使用线程化处理通信任务，不阻塞主业务逻辑。
"""

import threading
import time
import queue
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor, Future
from loguru import logger

from ...communication.communication_manager import CommunicationManager
from ...communication.communication_handler import CommunicationHandler
from ...communication.message_protocol import Message
from ...communication.exceptions import CommunicationError, ConnectionTimeoutError
from ...communication.adaptive_communication_manager import create_adaptive_communication_manager
from ...utils.improved_logging_manager import get_component_logger, log_training_info, log_system_debug


class MessageType(Enum):
    """消息类型枚举"""
    # 模型相关
    MODEL_DISTRIBUTION = "model_distribution"
    MODEL_UPDATE = "model_update"
    MODEL_REQUEST = "model_request"
    
    # 训练控制
    TRAINING_TRIGGER = "training_trigger"
    TRAINING_STATUS = "training_status"
    TRAINING_COMPLETE = "training_complete"
    
    # 持续学习特有
    TASK_NOTIFICATION = "task_notification"
    KNOWLEDGE_DISTILLATION = "knowledge_distillation"
    
    # 系统管理
    HEARTBEAT = "heartbeat"
    REGISTRATION = "registration"
    STATUS_QUERY = "status_query"
    SHUTDOWN = "shutdown"
    
    # 响应消息
    ACK = "ack"
    ERROR = "error"


class CommunicatorRole(Enum):
    """通信器角色"""
    SERVER = "server"
    CLIENT = "client"


@dataclass
class MessageHandler:
    """消息处理器配置"""
    message_type: MessageType
    handler_func: Callable[[Dict[str, Any]], Any]
    is_async: bool = False
    priority: int = 0  # 优先级，数字越大优先级越高


@dataclass
class CommunicationConfig:
    """通信配置"""
    role: CommunicatorRole
    component_id: str
    host: str = "localhost"
    port: int = 8080
    max_workers: int = 10
    heartbeat_interval: float = 30.0
    message_timeout: float = 60.0
    retry_attempts: int = 3
    enable_heartbeat: bool = True
    enable_stats: bool = True


class FederatedCommunicator(ABC):
    """
    联邦学习通信中间层基类
    
    提供统一的通信接口和线程化处理能力，支持客户端和服务端继承。
    
    主要功能：
    1. 统一的消息发送接收接口
    2. 线程化消息处理，不阻塞主业务逻辑
    3. 基于消息类型的业务逻辑分发
    4. 心跳监控和连接管理
    5. 错误处理和重试机制
    """
    
    def __init__(self, config: CommunicationConfig, 
                 communication_manager: Optional[CommunicationManager] = None):
        """
        初始化联邦通信器
        
        Args:
            config: 通信配置
            communication_manager: 通信管理器（可选，用于真实联邦环境）
        """
        self.config = config
        self.role = config.role
        self.component_id = config.component_id
        
        # 获取专用日志器
        self.logger = get_component_logger(
            self.role.value, 
            self.component_id
        )
        
        # 通信组件
        self.communication_manager = communication_manager
        if not self.communication_manager:
            # 创建默认的通信管理器
            self.communication_manager = self._create_default_communication_manager()
        
        # 如果是SimpleCommunicationManager，注册消息处理器
        if hasattr(self.communication_manager, 'register_message_handler'):
            # 为所有消息类型注册处理器
            for msg_type in MessageType:
                self.communication_manager.register_message_handler(
                    msg_type.value, 
                    self._handle_communication_message
                )
        
        # 线程管理
        self._executor = ThreadPoolExecutor(
            max_workers=config.max_workers,
            thread_name_prefix=f"{self.role.value}_{self.component_id}"
        )
        
        # 消息处理
        self._message_handlers: Dict[MessageType, MessageHandler] = {}
        self._message_queue = queue.PriorityQueue()
        self._response_futures: Dict[str, Future] = {}
        
        # 状态管理
        self._is_running = False
        self._is_connected = False
        self._connection_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        
        # 心跳管理
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._last_heartbeat = time.time()
        
        # 处理线程
        self._message_processor_thread: Optional[threading.Thread] = None
        
        # 统计信息
        self._stats = {
            'messages_sent': 0,
            'messages_received': 0,
            'messages_processed': 0,
            'errors': 0,
            'connections': 0,
            'last_activity': time.time()
        }
        
        # 注册默认消息处理器
        self._register_default_handlers()
        
        self.logger.debug(f"FederatedCommunicator initialized: {self.role.value}({self.component_id})")
    
    def start(self) -> None:
        """启动通信器"""
        try:
            with self._connection_lock:
                if self._is_running:
                    self.logger.warning("Communicator already running")
                    return
                
                self._is_running = True
            
            # 启动消息处理线程
            self._message_processor_thread = threading.Thread(
                target=self._message_processor_loop,
                name=f"{self.role.value}_{self.component_id}_processor",
                daemon=True
            )
            self._message_processor_thread.start()
            
            # 启动心跳线程（如果启用）
            if self.config.enable_heartbeat:
                self._heartbeat_thread = threading.Thread(
                    target=self._heartbeat_loop,
                    name=f"{self.role.value}_{self.component_id}_heartbeat",
                    daemon=True
                )
                self._heartbeat_thread.start()
            
            # 执行角色特定的启动逻辑
            self.on_start()
            
            self.logger.debug(f"Communicator started: {self.role.value}({self.component_id})")
            
        except Exception as e:
            self._is_running = False
            self.logger.error(f"Failed to start communicator: {e}")
            raise CommunicationError(f"Failed to start communicator: {e}")
    
    def stop(self) -> None:
        """停止通信器"""
        try:
            with self._connection_lock:
                if not self._is_running:
                    return
                
                self._is_running = False
            
            self.logger.debug("Stopping communicator...")
            
            # 执行角色特定的停止逻辑
            self.on_stop()
            
            # 停止心跳线程
            if self._heartbeat_thread and self._heartbeat_thread.is_alive():
                self._heartbeat_thread.join(timeout=2.0)
            
            # 停止消息处理线程
            if self._message_processor_thread and self._message_processor_thread.is_alive():
                # 发送停止信号
                self._message_queue.put((0, time.time(), None))  # 特殊停止消息
                self._message_processor_thread.join(timeout=5.0)
            
            # 关闭线程池
            self._executor.shutdown(wait=True)
            
            # 关闭通信管理器
            if hasattr(self.communication_manager, 'shutdown'):
                self.communication_manager.shutdown()
            
            self.logger.debug("Communicator 已停止")
            
        except Exception as e:
            self.logger.error(f"Error stopping communicator: {e}")
    
    def send_message(self, target: str, message_type: MessageType, 
                    data: Any, metadata: Optional[Dict[str, Any]] = None,
                    timeout: float = None, expect_response: bool = False) -> Optional[Any]:
        """
        发送消息
        
        Args:
            target: 目标地址
            message_type: 消息类型
            data: 消息数据
            metadata: 消息元数据
            timeout: 超时时间
            expect_response: 是否期待响应
            
        Returns:
            如果expect_response为True，返回响应数据；否则返回None
        """
        try:
            if not self._is_running:
                raise CommunicationError("Communicator not running")
            
            # 创建消息
            message = Message(
                message_id=f"{self.component_id}_{int(time.time() * 1000)}",
                message_type=message_type.value,
                sender=self.component_id,
                receiver=target,
                timestamp=time.time(),
                data=data,
                metadata=metadata or {},
                checksum=""
            )
            message.checksum = message.calculate_checksum()
            
            # 如果期待响应，创建Future
            response_future = None
            if expect_response:
                response_future = Future()
                self._response_futures[message.message_id] = response_future
            
            # 发送消息
            success = self._send_message_impl(target, message)
            
            if not success:
                if expect_response and message.message_id in self._response_futures:
                    del self._response_futures[message.message_id]
                raise CommunicationError(f"Failed to send message to {target}")
            
            # 更新统计
            with self._stats_lock:
                self._stats['messages_sent'] += 1
                self._stats['last_activity'] = time.time()
            
            # 等待响应
            if expect_response:
                timeout = timeout or self.config.message_timeout
                try:
                    response = response_future.result(timeout=timeout)
                    return response
                except TimeoutError:
                    self.logger.warning(f"Message response timeout: {message.message_id}")
                    if message.message_id in self._response_futures:
                        del self._response_futures[message.message_id]
                    return None
            
            return None
            
        except Exception as e:
            with self._stats_lock:
                self._stats['errors'] += 1
            self.logger.error(f"Error sending message: {e}")
            raise CommunicationError(f"Failed to send message: {e}")
    
    def broadcast_message(self, targets: List[str], message_type: MessageType,
                         data: Any, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, bool]:
        """
        广播消息
        
        Args:
            targets: 目标地址列表
            message_type: 消息类型
            data: 消息数据
            metadata: 消息元数据
            
        Returns:
            Dict[str, bool]: 每个目标的发送结果
        """
        try:
            if not self._is_running:
                raise CommunicationError("Communicator not running")
            
            # 并行发送
            futures = {}
            for target in targets:
                future = self._executor.submit(
                    self.send_message, target, message_type, data, metadata
                )
                futures[target] = future
            
            # 收集结果
            results = {}
            for target, future in futures.items():
                try:
                    future.result(timeout=self.config.message_timeout)
                    results[target] = True
                except Exception as e:
                    self.logger.error(f"Broadcast failed for {target}: {e}")
                    results[target] = False
            
            successful = sum(1 for success in results.values() if success)
            self.logger.debug(f"Broadcast completed: {successful}/{len(targets)} successful")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in broadcast: {e}")
            raise CommunicationError(f"Broadcast failed: {e}")
    
    def register_message_handler(self, message_type: MessageType, 
                                handler_func: Callable[[Dict[str, Any]], Any],
                                is_async: bool = False, priority: int = 0) -> None:
        """
        注册消息处理器
        
        Args:
            message_type: 消息类型
            handler_func: 处理函数
            is_async: 是否异步处理
            priority: 优先级
        """
        handler = MessageHandler(
            message_type=message_type,
            handler_func=handler_func,
            is_async=is_async,
            priority=priority
        )
        self._message_handlers[message_type] = handler
        self.logger.debug(f"Registered handler for {message_type.value}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._stats_lock:
            stats = self._stats.copy()
            stats['uptime'] = time.time() - self._stats.get('start_time', time.time())
            stats['is_running'] = self._is_running
            stats['is_connected'] = self._is_connected
            return stats
    
    # ===== 抽象方法 - 子类必须实现 =====
    
    @abstractmethod
    def on_start(self) -> None:
        """启动时的角色特定逻辑"""
        pass
    
    @abstractmethod
    def on_stop(self) -> None:
        """停止时的角色特定逻辑"""
        pass
    
    @abstractmethod
    def handle_model_distribution(self, message_data: Dict[str, Any]) -> Any:
        """处理模型分发消息"""
        pass
    
    @abstractmethod
    def handle_model_update(self, message_data: Dict[str, Any]) -> Any:
        """处理模型更新消息"""
        pass
    
    @abstractmethod
    def handle_training_trigger(self, message_data: Dict[str, Any]) -> Any:
        """处理训练触发消息"""
        pass
    
    @abstractmethod
    def handle_task_notification(self, message_data: Dict[str, Any]) -> Any:
        """处理任务通知消息（持续学习特有）"""
        pass
    
    # ===== 内部实现方法 =====
    
    def _handle_communication_message(self, message: Dict[str, Any]) -> None:
        """处理来自通信管理器的消息"""
        try:
            # 将字典消息转换为Message对象
            from ...communication.message_protocol import Message
            
            message_obj = Message(
                message_id=message.get('message_id', ''),
                message_type=message.get('type', 'unknown'),
                sender=message.get('sender', ''),
                receiver=self.component_id,
                timestamp=message.get('timestamp', time.time()),
                data=message.get('data', {}),
                metadata=message.get('metadata', {}),
                checksum=message.get('checksum', '')
            )
            
            # 将消息放入处理队列
            self._message_queue.put((1, time.time(), message_obj))
            
            # 更新统计
            with self._stats_lock:
                self._stats['messages_received'] += 1
                self._stats['last_activity'] = time.time()
                
            self.logger.debug(f"Message received from communication manager: {message.get('type', 'unknown')}")
            
        except Exception as e:
            self.logger.error(f"Error handling communication message: {e}")
    
    def _create_default_communication_manager(self) -> CommunicationManager:
        """创建默认通信管理器"""
        # 检查配置是否启用自适应模式
        config = getattr(self, 'config', {})
        
        # 处理不同类型的配置对象
        if hasattr(config, 'get'):
            # 如果是字典或DictConfig
            communication_config = config.get('communication', {})
        elif hasattr(config, 'communication'):
            # 如果是配置对象
            communication_config = getattr(config, 'communication', {})
        else:
            # 直接使用config作为通信配置
            communication_config = config
        
        # 检查是否有adaptive_mode属性
        adaptive_mode = True  # 默认启用自适应模式
        if hasattr(communication_config, 'get'):
            adaptive_mode = communication_config.get('adaptive_mode', True)
        elif hasattr(communication_config, 'adaptive_mode'):
            adaptive_mode = getattr(communication_config, 'adaptive_mode', True)
        
        if adaptive_mode:
            # 使用自适应通信管理器
            role = "server" if self.role == CommunicatorRole.SERVER else "client"
            return create_adaptive_communication_manager(
                config=communication_config,
                role=role,
                component_id=self.component_id
            )
        else:
            # 使用传统的SimpleCommunicationManager
            from ...communication.communication_manager import SimpleCommunicationManager
            role = "server" if self.role == CommunicatorRole.SERVER else "client"
            return SimpleCommunicationManager(
                config=communication_config,
                role=role,
                component_id=self.component_id
            )
    
    def _register_default_handlers(self) -> None:
        """注册默认消息处理器"""
        # 核心业务消息处理器
        self.register_message_handler(MessageType.MODEL_DISTRIBUTION, self.handle_model_distribution)
        self.register_message_handler(MessageType.MODEL_UPDATE, self.handle_model_update)
        self.register_message_handler(MessageType.TRAINING_TRIGGER, self.handle_training_trigger)
        self.register_message_handler(MessageType.TASK_NOTIFICATION, self.handle_task_notification)
        
        # 系统消息处理器
        self.register_message_handler(MessageType.HEARTBEAT, self._handle_heartbeat)
        self.register_message_handler(MessageType.STATUS_QUERY, self._handle_status_query)
        self.register_message_handler(MessageType.ACK, self._handle_ack)
        self.register_message_handler(MessageType.ERROR, self._handle_error)
    
    def _send_message_impl(self, target: str, message: Message) -> bool:
        """实际发送消息的实现"""
        try:
            # 新的SimpleCommunicationManager支持直接发送Message对象
            if hasattr(self.communication_manager, 'send_message'):
                return self.communication_manager.send_message(target, message)
            elif hasattr(self.communication_manager, 'send_message_to_server') and self.role == CommunicatorRole.CLIENT:
                # 客户端发送到服务端
                self.logger.debug(f"Sending message type: '{message.message_type}' from client to server")
                message_dict = {
                    'type': message.message_type,  # message.message_type 已经是字符串
                    'data': message.data,
                    'metadata': message.metadata,
                    'sender': message.sender,
                    'message_id': message.message_id,
                    'timestamp': message.timestamp,
                    'checksum': message.checksum
                }
                self.communication_manager.send_message_to_server(message_dict)
                return True
            elif hasattr(self.communication_manager, 'send_message_to_client') and self.role == CommunicatorRole.SERVER:
                # 服务端发送到客户端
                self.logger.debug(f"Sending message type: '{message.message_type}' from server to client")
                message_dict = {
                    'type': message.message_type,  # message.message_type 已经是字符串
                    'data': message.data,
                    'metadata': message.metadata,
                    'sender': message.sender,
                    'message_id': message.message_id,
                    'timestamp': message.timestamp,
                    'checksum': message.checksum
                }
                self.communication_manager.send_message_to_client(target, message_dict)
                return True
            else:
                self.logger.warning("No suitable send method found in communication manager")
                return False
                
        except Exception as e:
            self.logger.error(f"Error in send implementation: {e}")
            return False
    
    def _message_processor_loop(self) -> None:
        """消息处理循环"""
        self.logger.debug("Message processor started")
        
        while self._is_running:
            try:
                # 获取消息（带超时）
                try:
                    priority, timestamp, message = self._message_queue.get(timeout=1.0)
                    
                    # 检查停止信号
                    if message is None:
                        break
                    
                    # 处理消息
                    self._process_message(message)
                    
                except queue.Empty:
                    continue
                    
            except Exception as e:
                self.logger.error(f"Error in message processor: {e}")
                with self._stats_lock:
                    self._stats['errors'] += 1
        
        self.logger.debug("Message processor 已停止")
    
    def _process_message(self, message: Union[Dict[str, Any], Any]) -> None:
        """处理单个消息"""
        try:
            # 如果message是Message对象，先转换为字典
            if hasattr(message, 'message_type') and hasattr(message, 'data'):
                # 这是一个Message对象，转换为字典格式
                message_dict = {
                    'type': message.message_type,
                    'data': message.data,
                    'metadata': message.metadata,
                    'sender': message.sender,
                    'message_id': message.message_id,
                    'timestamp': message.timestamp,
                    'checksum': message.checksum
                }
            else:
                # 已经是字典格式
                message_dict = message
            
            message_type_str = message_dict.get('type', '')
            
            # 查找消息类型
            message_type = None
            for mt in MessageType:
                if mt.value == message_type_str:
                    message_type = mt
                    break
            
            if not message_type:
                self.logger.warning(f"Unknown message type: {message_type_str}")
                return
            
            # 查找处理器
            if message_type not in self._message_handlers:
                self.logger.warning(f"No handler for message type: {message_type.value}")
                return
            
            handler = self._message_handlers[message_type]
            
            # 执行处理器
            try:
                if handler.is_async:
                    # 异步处理
                    future = self._executor.submit(handler.handler_func, message_dict)
                    # 可以选择等待结果或继续
                else:
                    # 同步处理
                    result = handler.handler_func(message_dict)
                    
                    # 如果是响应消息，设置Future结果
                    if message_type == MessageType.ACK and 'response_to' in message_dict.get('metadata', {}):
                        response_to = message_dict['metadata']['response_to']
                        if response_to in self._response_futures:
                            self._response_futures[response_to].set_result(message_dict.get('data'))
                    
                    # 如果有返回值且不是ACK消息，发送响应给发送方
                    elif result is not None and message_type != MessageType.ACK:
                        sender = message_dict.get('sender')
                        original_message_id = message_dict.get('message_id')
                        
                        if sender and original_message_id:
                            try:
                                # 发送ACK响应
                                from datetime import datetime
                                response_message = Message(
                                    message_id=f"ack_{original_message_id}_{int(time.time() * 1000)}",
                                    sender=self.component_id,
                                    receiver=sender,
                                    message_type=MessageType.ACK.value,  # 使用.value转换为字符串
                                    data=result,
                                    metadata={'response_to': original_message_id},
                                    timestamp=datetime.now(),
                                    checksum=""
                                )
                                response_message.checksum = response_message.calculate_checksum()
                                
                                # 使用通信管理器发送响应
                                self._send_message_impl(sender, response_message)
                                self.logger.debug(f"Sent ACK response for message {original_message_id}")
                                
                            except Exception as e:
                                self.logger.error(f"Failed to send ACK response: {e}")
                
                with self._stats_lock:
                    self._stats['messages_processed'] += 1
                    
            except Exception as e:
                self.logger.error(f"Error processing message {message_type.value}: {e}")
                with self._stats_lock:
                    self._stats['errors'] += 1
            
        except Exception as e:
            self.logger.error(f"Error in message processing: {e}")
    
    def _heartbeat_loop(self) -> None:
        """心跳循环"""
        self.logger.debug("Heartbeat started")
        
        while self._is_running:
            try:
                time.sleep(self.config.heartbeat_interval)
                
                if not self._is_running:
                    break
                
                # 发送心跳
                self._send_heartbeat()
                
            except Exception as e:
                self.logger.error(f"Error in heartbeat: {e}")
        
        self.logger.debug("Heartbeat 已停止")
    
    def _send_heartbeat(self) -> None:
        """发送心跳"""
        try:
            heartbeat_data = {
                'timestamp': time.time(),
                'role': self.role.value,
                'component_id': self.component_id,
                'stats': self.get_stats()
            }
            
            # 确定心跳目标
            if self.role == CommunicatorRole.CLIENT:
                # 客户端向服务端发送心跳
                target = "server"
            else:
                # 服务端可能需要向所有客户端发送心跳
                # 这里简化处理，不发送心跳
                return
            
            self.send_message(target, MessageType.HEARTBEAT, heartbeat_data)
            self._last_heartbeat = time.time()
            
        except Exception as e:
            self.logger.debug(f"Heartbeat send failed: {e}")
    
    # ===== 默认消息处理器 =====
    
    def _handle_heartbeat(self, message_data: Dict[str, Any]) -> Any:
        """处理心跳消息"""
        sender = message_data.get('sender', 'unknown')
        self.logger.debug(f"Received heartbeat from {sender}")
        
        # 更新连接状态
        self._is_connected = True
        
        # 发送心跳响应
        response_data = {
            'timestamp': time.time(),
            'role': self.role.value,
            'component_id': self.component_id
        }
        
        # 这里可以选择是否发送ACK响应
        return response_data
    
    def _handle_status_query(self, message_data: Dict[str, Any]) -> Any:
        """处理状态查询消息"""
        return self.get_stats()
    
    def _handle_ack(self, message_data: Dict[str, Any]) -> Any:
        """处理ACK消息"""
        self.logger.debug("Received ACK message")
        return None
    
    def _handle_error(self, message_data: Dict[str, Any]) -> Any:
        """处理错误消息"""
        error_msg = message_data.get('data', {}).get('error', 'Unknown error')
        self.logger.error(f"Received error message: {error_msg}")
        return None
    
    # ===== 公共接口方法 =====
    
    def receive_message(self, message: Dict[str, Any]) -> None:
        """
        接收消息（由通信管理器调用）
        
        Args:
            message: 接收到的消息，可能是Message对象或字典
        """
        try:
            # 如果是Message对象，转换为字典
            if hasattr(message, 'to_dict'):
                message_dict = message.to_dict()
            elif hasattr(message, 'message_type'):
                # 可能是Message对象但没有to_dict方法
                message_dict = {
                    'message_id': getattr(message, 'message_id', ''),
                    'message_type': getattr(message, 'message_type', ''),
                    'sender': getattr(message, 'sender', ''),
                    'receiver': getattr(message, 'receiver', ''),
                    'timestamp': getattr(message, 'timestamp', time.time()),
                    'data': getattr(message, 'data', {}),
                    'metadata': getattr(message, 'metadata', {}),
                    'priority': getattr(message, 'priority', 0)
                }
            else:
                # 已经是字典
                message_dict = message
            
            # 将消息放入处理队列
            priority = message_dict.get('priority', 0)
            timestamp = time.time()
            
            self._message_queue.put((priority, timestamp, message_dict))
            
            with self._stats_lock:
                self._stats['messages_received'] += 1
                self._stats['last_activity'] = time.time()
            
        except Exception as e:
            self.logger.error(f"Error receiving message: {e}")
            # 打印更详细的错误信息以便调试
            self.logger.error(f"Message type: {type(message)}")
            self.logger.error(f"Message attributes: {dir(message) if hasattr(message, '__dict__') else 'No attributes'}")
    
    def is_running(self) -> bool:
        """检查通信器是否运行中"""
        return self._is_running
    
    def is_connected(self) -> bool:
        """检查是否已连接"""
        return self._is_connected