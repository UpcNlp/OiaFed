# fedcl/communication/communication_handler.py
"""
通信处理器模块

实现FedCL框架的通信处理功能，包括消息发送、接收、广播、错误处理等。
提供异步通信支持和自动重试机制。
"""

import asyncio
import time
import threading
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass
from loguru import logger

from .message_protocol import MessageProtocol, Message
from .data_serializer import DataSerializer
from .network_interface import NetworkInterface, Connection, ConnectionStatus
from .security_module import SecurityModule
from .exceptions import CommunicationError, NetworkError, ConnectionTimeoutError


@dataclass
class RetryPolicy:
    """重试策略配置"""
    max_retries: int = 3
    backoff_factor: float = 2.0
    initial_delay: float = 1.0


@dataclass
class CommunicationStats:
    """通信统计信息"""
    messages_sent: int = 0
    messages_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    failed_sends: int = 0
    failed_receives: int = 0
    total_connections: int = 0
    active_connections: int = 0


class CommunicationHandler:
    """
    通信处理器
    
    负责处理联邦学习通信的核心逻辑，包括消息发送、接收、广播、
    连接管理和错误处理。支持异步操作和自动重试。
    """
    
    def __init__(self, protocol: MessageProtocol, serializer: DataSerializer, 
                 security: SecurityModule, network: NetworkInterface) -> None:
        """
        初始化通信处理器
        
        Args:
            protocol: 消息协议
            serializer: 数据序列化器
            security: 安全模块
            network: 网络接口
        """
        self.protocol = protocol
        self.serializer = serializer
        self.security = security
        self.network = network
        
        # 重试策略
        self.retry_policy = RetryPolicy()
        
        # 统计信息
        self.stats = CommunicationStats()
        
        # 连接管理
        self._connections: Dict[str, Connection] = {}
        self._connection_lock = threading.Lock()
        
        # 异步支持
        self._executor = ThreadPoolExecutor(max_workers=10)
        
        # 消息队列
        self._pending_messages: Dict[str, List[Message]] = {}
        self._message_lock = threading.Lock()
        
        logger.debug("CommunicationHandler initialized")
    
    def send_message(self, target: str, message: Message) -> bool:
        """
        发送消息
        
        Args:
            target: 目标地址
            message: 要发送的消息
            
        Returns:
            bool: 发送是否成功
        """
        try:
            # 获取或建立连接
            connection = self._get_connection(target)
            if not connection:
                logger.error(f"Failed to get connection to {target}")
                self.stats.failed_sends += 1
                return False
            
            # 序列化消息
            # 使用Message对象的内置序列化方法
            import pickle
            serialized_data = pickle.dumps(message)
            
            # 加密数据（如果需要）
            if hasattr(self.security, 'encrypt_data'):
                try:
                    encrypted_data = self.security.encrypt_data(serialized_data)
                except Exception as e:
                    logger.warning(f"Failed to encrypt message: {e}, sending unencrypted")
                    encrypted_data = serialized_data
            else:
                encrypted_data = serialized_data
            
            # 发送数据
            success = self._send_with_retry(connection, encrypted_data)
            
            if success:
                self.stats.messages_sent += 1
                self.stats.bytes_sent += len(encrypted_data)
                logger.debug(f"Message sent to {target}, size: {len(encrypted_data)} bytes")
            else:
                self.stats.failed_sends += 1
                logger.error(f"Failed to send message to {target}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending message to {target}: {e}")
            self.stats.failed_sends += 1
            return False
    
    def receive_message(self, source: str, timeout: float = 30.0) -> Message:
        """
        接收消息
        
        Args:
            source: 消息源地址
            timeout: 超时时间（秒）
            
        Returns:
            Message: 接收到的消息
            
        Raises:
            ConnectionTimeoutError: 超时
            CommunicationError: 通信错误
        """
        try:
            start_time = time.time()
            
            # 检查是否有待处理的消息
            with self._message_lock:
                if source in self._pending_messages and self._pending_messages[source]:
                    message = self._pending_messages[source].pop(0)
                    if not self._pending_messages[source]:
                        del self._pending_messages[source]
                    self.stats.messages_received += 1
                    return message
            
            # 获取连接
            connection = self._get_connection(source)
            if not connection:
                raise CommunicationError(f"No connection to {source}")
            
            # 接收数据
            while time.time() - start_time < timeout:
                try:
                    data = connection.receive(timeout=min(5.0, timeout - (time.time() - start_time)))
                    if data:
                        # 解密数据（如果需要）
                        if hasattr(self.security, 'decrypt_data'):
                            try:
                                decrypted_data = self.security.decrypt_data(data)
                            except Exception as e:
                                logger.warning(f"Failed to decrypt message: {e}, using raw data")
                                decrypted_data = data
                        else:
                            decrypted_data = data
                        
                        # 反序列化消息
                        import pickle
                        message = pickle.loads(decrypted_data)
                        
                        self.stats.messages_received += 1
                        self.stats.bytes_received += len(data)
                        
                        logger.debug(f"Message received from {source}, size: {len(data)} bytes")
                        return message
                
                except ConnectionTimeoutError:
                    continue  # 继续等待，不增加失败计数
            
            # 超时
            self.stats.failed_receives += 1
            raise ConnectionTimeoutError(f"Timeout receiving message from {source}")
            
        except ConnectionTimeoutError:
            # 如果是从超时路径来的，不要重复增加计数
            raise
        except Exception as e:
            self.stats.failed_receives += 1
            logger.error(f"Error receiving message from {source}: {e}")
            raise CommunicationError(f"Failed to receive message from {source}: {e}")
    
    def broadcast_message(self, message: Message, targets: List[str]) -> Dict[str, bool]:
        """
        广播消息
        
        Args:
            message: 要广播的消息
            targets: 目标地址列表
            
        Returns:
            Dict[str, bool]: 每个目标的发送结果
        """
        results = {}
        
        # 并行发送
        futures = {}
        for target in targets:
            future = self._executor.submit(self.send_message, target, message)
            futures[target] = future
        
        # 收集结果
        for target, future in futures.items():
            try:
                results[target] = future.result(timeout=30.0)
            except Exception as e:
                logger.error(f"Error broadcasting to {target}: {e}")
                results[target] = False
        
        successful_sends = sum(1 for success in results.values() if success)
        logger.debug(f"Broadcast completed: {successful_sends}/{len(targets)} successful")
        
        return results
    
    def establish_connection(self, address: Tuple[str, int]) -> Connection:
        """
        建立连接
        
        Args:
            address: 目标地址 (host, port)
            
        Returns:
            Connection: 建立的连接
            
        Raises:
            NetworkError: 连接失败
        """
        try:
            # 使用网络接口创建客户端套接字
            client_socket = self.network.create_client_socket(address[0], address[1])
            
            # 创建连接对象
            connection_id = f"{address[0]}:{address[1]}"
            connection = Connection(client_socket, connection_id, address)
            
            with self._connection_lock:
                self._connections[connection_id] = connection
                self.stats.total_connections += 1
                self.stats.active_connections += 1
            
            logger.debug(f"Connection established to {address}")
            return connection
            
        except Exception as e:
            logger.error(f"Failed to establish connection to {address}: {e}")
            raise NetworkError(f"Failed to establish connection to {address}: {e}")
    
    def close_connection(self, connection_id: str) -> None:
        """
        关闭连接
        
        Args:
            connection_id: 连接ID
        """
        try:
            with self._connection_lock:
                if connection_id in self._connections:
                    connection = self._connections[connection_id]
                    connection.close()
                    del self._connections[connection_id]
                    self.stats.active_connections -= 1
                    logger.debug(f"Connection {connection_id} closed")
                
        except Exception as e:
            logger.error(f"Error closing connection {connection_id}: {e}")
    
    def handle_network_error(self, error: Exception, connection_id: str) -> None:
        """
        处理网络错误
        
        Args:
            error: 错误信息
            connection_id: 连接ID
        """
        logger.error(f"Network error on connection {connection_id}: {error}")
        
        # 关闭有问题的连接
        self.close_connection(connection_id)
        
        # 根据错误类型决定处理策略
        if isinstance(error, ConnectionTimeoutError):
            logger.debug(f"Connection {connection_id} timed out, will retry later")
        elif isinstance(error, NetworkError):
            logger.warning(f"Network error on {connection_id}, connection removed")
        else:
            logger.error(f"Unexpected error on {connection_id}: {error}")
    
    def set_retry_policy(self, max_retries: int, backoff_factor: float) -> None:
        """
        设置重试策略
        
        Args:
            max_retries: 最大重试次数
            backoff_factor: 退避因子
        """
        self.retry_policy.max_retries = max_retries
        self.retry_policy.backoff_factor = backoff_factor
        logger.debug(f"Retry policy updated: max_retries={max_retries}, backoff_factor={backoff_factor}")
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """
        获取通信统计
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        return {
            'messages_sent': self.stats.messages_sent,
            'messages_received': self.stats.messages_received,
            'bytes_sent': self.stats.bytes_sent,
            'bytes_received': self.stats.bytes_received,
            'failed_sends': self.stats.failed_sends,
            'failed_receives': self.stats.failed_receives,
            'total_connections': self.stats.total_connections,
            'active_connections': self.stats.active_connections,
            'success_rate_send': (self.stats.messages_sent / max(1, self.stats.messages_sent + self.stats.failed_sends)) * 100,
            'success_rate_receive': (self.stats.messages_received / max(1, self.stats.messages_received + self.stats.failed_receives)) * 100
        }
    
    async def async_send_message(self, target: str, message: Message) -> bool:
        """
        异步发送消息
        
        Args:
            target: 目标地址
            message: 要发送的消息
            
        Returns:
            bool: 发送是否成功
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.send_message, target, message)
    
    async def async_receive_message(self, source: str, timeout: float = 30.0) -> Message:
        """
        异步接收消息
        
        Args:
            source: 消息源地址
            timeout: 超时时间（秒）
            
        Returns:
            Message: 接收到的消息
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.receive_message, source, timeout)
    
    def _get_connection(self, target: str) -> Optional[Connection]:
        """获取到目标的连接"""
        with self._connection_lock:
            # 直接查找连接
            if target in self._connections:
                connection = self._connections[target]
                if connection.status == ConnectionStatus.CONNECTED:
                    return connection
                else:
                    # 连接已断开，移除
                    del self._connections[target]
                    self.stats.active_connections -= 1
            
            # 尝试解析地址并建立新连接
            try:
                if ':' in target:
                    host, port_str = target.split(':', 1)
                    port = int(port_str)
                    return self.establish_connection((host, port))
                else:
                    logger.error(f"Invalid target format: {target}")
                    return None
            except Exception as e:
                logger.error(f"Failed to parse or connect to {target}: {e}")
                return None
    
    def _send_with_retry(self, connection: Connection, data: bytes) -> bool:
        """带重试的发送"""
        for attempt in range(self.retry_policy.max_retries + 1):
            try:
                success = connection.send(data)
                if success:
                    return True
                
                if attempt < self.retry_policy.max_retries:
                    delay = self.retry_policy.initial_delay * (self.retry_policy.backoff_factor ** attempt)
                    logger.debug(f"Send failed, retrying in {delay:.2f}s (attempt {attempt + 1})")
                    time.sleep(delay)
                
            except Exception as e:
                if attempt < self.retry_policy.max_retries:
                    delay = self.retry_policy.initial_delay * (self.retry_policy.backoff_factor ** attempt)
                    logger.debug(f"Send error: {e}, retrying in {delay:.2f}s (attempt {attempt + 1})")
                    time.sleep(delay)
                else:
                    logger.error(f"Send failed after {self.retry_policy.max_retries} retries: {e}")
        
        return False
    
    def shutdown(self) -> None:
        """关闭通信处理器"""
        logger.info("Shutting down CommunicationHandler")
        
        # 关闭所有连接
        with self._connection_lock:
            for connection_id, connection in list(self._connections.items()):
                try:
                    connection.close()
                except Exception as e:
                    logger.error(f"Error closing connection {connection_id}: {e}")
            self._connections.clear()
            self.stats.active_connections = 0
        
        # 关闭线程池
        self._executor.shutdown(wait=True)
        
        logger.debug("CommunicationHandler shutdown 完成")
