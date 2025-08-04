# fedcl/communication/network_interface.py
"""
网络接口模块

实现网络连接、数据传输和连接管理功能。
支持TCP和WebSocket协议，提供SSL/TLS加密和连接池管理。
"""

import socket
import ssl
import asyncio
import threading
import time
import json
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from loguru import logger
from omegaconf import DictConfig

from .exceptions import NetworkError, ConnectionTimeoutError, SecurityError


class ConnectionStatus(Enum):
    """连接状态枚举"""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    ERROR = "error"
    TIMEOUT = "timeout"


class ProtocolType(Enum):
    """协议类型枚举"""
    TCP = "tcp"
    WEBSOCKET = "websocket"
    SSL_TCP = "ssl_tcp"


@dataclass
class ConnectionInfo:
    """连接信息数据结构"""
    connection_id: str
    peer_address: Tuple[str, int]
    local_address: Tuple[str, int]
    protocol: ProtocolType
    status: ConnectionStatus
    created_time: float
    last_activity: float
    bytes_sent: int = 0
    bytes_received: int = 0


class Connection:
    """
    网络连接类
    
    封装单个网络连接的操作，提供发送、接收、状态检查等功能。
    """
    
    def __init__(self, socket_obj: socket.socket, connection_id: str, 
                 peer_address: Tuple[str, int], protocol: ProtocolType = ProtocolType.TCP) -> None:
        """
        初始化连接
        
        Args:
            socket_obj: 套接字对象
            connection_id: 连接ID
            peer_address: 对端地址
            protocol: 协议类型
        """
        self.socket = socket_obj
        self.connection_id = connection_id
        self.peer_address = peer_address
        self.protocol = protocol
        self.status = ConnectionStatus.CONNECTED
        self.created_time = time.time()
        self.last_activity = time.time()
        self.bytes_sent = 0
        self.bytes_received = 0
        self.timeout = 30.0
        self.lock = threading.Lock()
        
        # 获取本地地址
        try:
            self.local_address = self.socket.getsockname()
        except:
            self.local_address = ("unknown", 0)
        
        logger.debug(f"Connection {connection_id} initialized: {peer_address}")
    
    def send(self, data: bytes) -> bool:
        """
        发送数据
        
        Args:
            data: 要发送的数据
            
        Returns:
            bool: 发送是否成功
        """
        try:
            with self.lock:
                if self.status != ConnectionStatus.CONNECTED:
                    logger.warning(f"Connection {self.connection_id} is not connected")
                    return False
                
                # 添加数据长度前缀
                data_length = len(data)
                length_prefix = data_length.to_bytes(4, byteorder='big')
                full_data = length_prefix + data
                
                # 发送数据
                total_sent = 0
                while total_sent < len(full_data):
                    try:
                        sent = self.socket.send(full_data[total_sent:])
                        if sent == 0:
                            raise NetworkError("Socket connection broken")
                        total_sent += sent
                    except socket.timeout:
                        raise ConnectionTimeoutError(f"Send timeout on connection {self.connection_id}")
                
                self.bytes_sent += len(data)
                self.last_activity = time.time()
                
                logger.debug(f"Sent {len(data)} bytes on connection {self.connection_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to send data on connection {self.connection_id}: {e}")
            self.status = ConnectionStatus.ERROR
            return False
    
    def receive(self, timeout: float = None) -> bytes:
        """
        接收数据
        
        Args:
            timeout: 超时时间
            
        Returns:
            bytes: 接收到的数据
        """
        try:
            with self.lock:
                if self.status != ConnectionStatus.CONNECTED:
                    raise NetworkError(f"Connection {self.connection_id} is not connected")
                
                # 设置超时
                original_timeout = self.socket.gettimeout()
                if timeout is not None:
                    self.socket.settimeout(timeout)
                else:
                    self.socket.settimeout(self.timeout)
                
                try:
                    # 首先接收长度前缀（4字节）
                    length_data = self._receive_exact(4)
                    data_length = int.from_bytes(length_data, byteorder='big')
                    
                    # 接收实际数据
                    data = self._receive_exact(data_length)
                    
                    self.bytes_received += len(data)
                    self.last_activity = time.time()
                    
                    logger.debug(f"Received {len(data)} bytes on connection {self.connection_id}")
                    return data
                    
                finally:
                    # 恢复原始超时设置
                    self.socket.settimeout(original_timeout)
                    
        except socket.timeout:
            logger.warning(f"Receive timeout on connection {self.connection_id}")
            raise ConnectionTimeoutError(f"Receive timeout on connection {self.connection_id}")
        except Exception as e:
            logger.error(f"Failed to receive data on connection {self.connection_id}: {e}")
            self.status = ConnectionStatus.ERROR
            raise NetworkError(f"Failed to receive data: {e}")
    
    def _receive_exact(self, size: int) -> bytes:
        """
        接收指定长度的数据
        
        Args:
            size: 要接收的字节数
            
        Returns:
            bytes: 接收到的数据
        """
        data = b''
        while len(data) < size:
            chunk = self.socket.recv(size - len(data))
            if not chunk:
                raise NetworkError("Socket connection broken")
            data += chunk
        return data
    
    def is_alive(self) -> bool:
        """
        检查连接是否活跃
        
        Returns:
            bool: 连接是否活跃
        """
        try:
            if self.status != ConnectionStatus.CONNECTED:
                return False
            
            # 尝试发送心跳包
            self.socket.send(b'')
            return True
            
        except:
            self.status = ConnectionStatus.DISCONNECTED
            return False
    
    def close(self) -> None:
        """关闭连接"""
        try:
            with self.lock:
                if self.socket:
                    self.socket.close()
                self.status = ConnectionStatus.DISCONNECTED
                logger.debug(f"Connection {self.connection_id} closed")
        except Exception as e:
            logger.error(f"Error closing connection {self.connection_id}: {e}")
    
    def get_peer_address(self) -> Tuple[str, int]:
        """
        获取对端地址
        
        Returns:
            Tuple[str, int]: 对端地址 (host, port)
        """
        return self.peer_address
    
    def get_connection_info(self) -> ConnectionInfo:
        """
        获取连接信息
        
        Returns:
            ConnectionInfo: 连接详细信息
        """
        return ConnectionInfo(
            connection_id=self.connection_id,
            peer_address=self.peer_address,
            local_address=self.local_address,
            protocol=self.protocol,
            status=self.status,
            created_time=self.created_time,
            last_activity=self.last_activity,
            bytes_sent=self.bytes_sent,
            bytes_received=self.bytes_received
        )
    
    def set_timeout(self, timeout: float) -> None:
        """
        设置超时时间
        
        Args:
            timeout: 超时时间（秒）
        """
        self.timeout = timeout
        logger.debug(f"Connection {self.connection_id} timeout set to {timeout}s")


class ConnectionPool:
    """
    连接池类
    
    管理多个网络连接，提供连接获取、释放和清理功能。
    """
    
    def __init__(self, max_connections: int = 100) -> None:
        """
        初始化连接池
        
        Args:
            max_connections: 最大连接数
        """
        self.max_connections = max_connections
        self.connections: Dict[str, Connection] = {}
        self.lock = threading.RLock()
        self.cleanup_interval = 60.0  # 清理间隔（秒）
        self.connection_timeout = 300.0  # 连接超时时间（秒）
        
        # 启动清理线程
        self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self.cleanup_thread.start()
        
        logger.debug(f"ConnectionPool initialized with max_connections={max_connections}")
    
    def get_connection(self, client_id: str) -> Optional[Connection]:
        """
        获取连接
        
        Args:
            client_id: 客户端ID
            
        Returns:
            Optional[Connection]: 连接对象，如果不存在则返回None
        """
        with self.lock:
            connection = self.connections.get(client_id)
            if connection and connection.is_alive():
                return connection
            elif connection:
                # 连接已失效，移除它
                self.remove_connection(client_id)
            return None
    
    def add_connection(self, client_id: str, connection: Connection) -> None:
        """
        添加连接
        
        Args:
            client_id: 客户端ID
            connection: 连接对象
        """
        with self.lock:
            if len(self.connections) >= self.max_connections:
                # 连接池已满，移除最旧的连接
                oldest_id = min(self.connections.keys(), 
                              key=lambda x: self.connections[x].last_activity)
                self.remove_connection(oldest_id)
                logger.warning(f"Connection pool full, removed oldest connection: {oldest_id}")
            
            self.connections[client_id] = connection
            logger.debug(f"Added connection for client {client_id}")
    
    def remove_connection(self, client_id: str) -> None:
        """
        移除连接
        
        Args:
            client_id: 客户端ID
        """
        with self.lock:
            connection = self.connections.pop(client_id, None)
            if connection:
                connection.close()
                logger.debug(f"Removed connection for client {client_id}")
    
    def get_active_connections(self) -> List[str]:
        """
        获取活跃连接列表
        
        Returns:
            List[str]: 活跃的客户端ID列表
        """
        with self.lock:
            active_connections = []
            for client_id, connection in self.connections.items():
                if connection.is_alive():
                    active_connections.append(client_id)
            return active_connections
    
    def cleanup_stale_connections(self) -> None:
        """清理失效连接"""
        current_time = time.time()
        stale_connections = []
        
        with self.lock:
            for client_id, connection in self.connections.items():
                # 检查连接是否超时或失效
                if (current_time - connection.last_activity > self.connection_timeout or
                    not connection.is_alive()):
                    stale_connections.append(client_id)
            
            # 移除失效连接
            for client_id in stale_connections:
                self.remove_connection(client_id)
                
        if stale_connections:
            logger.debug(f"Cleaned up {len(stale_connections)} stale connections")
    
    def _cleanup_worker(self) -> None:
        """清理工作线程"""
        while True:
            try:
                time.sleep(self.cleanup_interval)
                self.cleanup_stale_connections()
            except Exception as e:
                logger.error(f"Error in cleanup worker: {e}")
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """
        获取连接池统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        with self.lock:
            active_count = len(self.get_active_connections())
            total_count = len(self.connections)
            
            return {
                'total_connections': total_count,
                'active_connections': active_count,
                'max_connections': self.max_connections,
                'utilization': total_count / self.max_connections if self.max_connections > 0 else 0,
                'cleanup_interval': self.cleanup_interval,
                'connection_timeout': self.connection_timeout
            }
    
    def set_max_connections(self, max_connections: int) -> None:
        """
        设置最大连接数
        
        Args:
            max_connections: 新的最大连接数
        """
        with self.lock:
            old_max = self.max_connections
            self.max_connections = max_connections
            
            # 如果新的最大连接数小于当前连接数，移除多余连接
            if len(self.connections) > max_connections:
                excess_count = len(self.connections) - max_connections
                oldest_connections = sorted(
                    self.connections.items(),
                    key=lambda x: x[1].last_activity
                )[:excess_count]
                
                for client_id, _ in oldest_connections:
                    self.remove_connection(client_id)
                
                logger.warning(f"Reduced connections from {len(self.connections) + excess_count} "
                             f"to {len(self.connections)} due to max_connections change")
            
            logger.debug(f"Max connections changed from {old_max} to {max_connections}")


class NetworkInterface:
    """
    网络接口类
    
    提供网络通信的底层接口，支持服务器和客户端套接字创建，
    数据发送接收等功能。
    """
    
    def __init__(self, config: DictConfig) -> None:
        """
        初始化网络接口
        
        Args:
            config: 网络配置
        """
        self.config = config
        self.connection_pool = ConnectionPool(
            max_connections=config.get('max_connections', 100)
        )
        self.ssl_context = None
        self.executor = ThreadPoolExecutor(max_workers=config.get('max_workers', 10))
        
        # 初始化SSL上下文（如果需要）
        if config.get('use_ssl', False):
            self._init_ssl_context()
        
        logger.debug("NetworkInterface initialized")
    
    def _init_ssl_context(self) -> None:
        """初始化SSL上下文"""
        try:
            self.ssl_context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
            
            # 配置SSL选项
            if self.config.get('ssl_verify_mode') == 'none':
                self.ssl_context.check_hostname = False
                self.ssl_context.verify_mode = ssl.CERT_NONE
            
            # 加载证书文件
            cert_file = self.config.get('ssl_cert_file')
            key_file = self.config.get('ssl_key_file')
            if cert_file and key_file:
                self.ssl_context.load_cert_chain(cert_file, key_file)
            
            logger.debug("SSL context initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize SSL context: {e}")
            raise SecurityError(f"SSL initialization failed: {e}")
    
    def create_server_socket(self, host: str, port: int) -> socket.socket:
        """
        创建服务器套接字
        
        Args:
            host: 主机地址
            port: 端口号
            
        Returns:
            socket.socket: 服务器套接字
        """
        try:
            # 创建套接字
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # 绑定地址
            server_socket.bind((host, port))
            
            # 开始监听
            backlog = self.config.get('socket_backlog', 5)
            server_socket.listen(backlog)
            
            # 如果使用SSL，包装套接字
            if self.ssl_context:
                server_socket = self.ssl_context.wrap_socket(
                    server_socket, server_side=True
                )
            
            logger.debug(f"Server socket created and listening on {host}:{port}")
            return server_socket
            
        except Exception as e:
            logger.error(f"Failed to create server socket: {e}")
            raise NetworkError(f"Failed to create server socket: {e}")
    
    def create_client_socket(self, host: str, port: int) -> socket.socket:
        """
        创建客户端套接字
        
        Args:
            host: 服务器主机地址
            port: 服务器端口号
            
        Returns:
            socket.socket: 客户端套接字
        """
        try:
            # 创建套接字
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            
            # 设置超时
            timeout = self.config.get('connection_timeout', 30.0)
            client_socket.settimeout(timeout)
            
            # 连接到服务器
            client_socket.connect((host, port))
            
            # 如果使用SSL，包装套接字
            if self.ssl_context:
                client_socket = self.ssl_context.wrap_socket(
                    client_socket, server_hostname=host
                )
            
            logger.debug(f"Client socket connected to {host}:{port}")
            return client_socket
            
        except Exception as e:
            logger.error(f"Failed to create client socket: {e}")
            raise NetworkError(f"Failed to create client socket: {e}")
    
    def send_data(self, connection: Connection, data: bytes) -> bool:
        """
        发送数据
        
        Args:
            connection: 连接对象
            data: 要发送的数据
            
        Returns:
            bool: 发送是否成功
        """
        return connection.send(data)
    
    def receive_data(self, connection: Connection, timeout: float = 30.0) -> bytes:
        """
        接收数据
        
        Args:
            connection: 连接对象
            timeout: 超时时间
            
        Returns:
            bytes: 接收到的数据
        """
        return connection.receive(timeout)
    
    def close_connection(self, connection_id: str) -> None:
        """
        关闭连接
        
        Args:
            connection_id: 连接ID
        """
        self.connection_pool.remove_connection(connection_id)
    
    def get_connection_status(self, connection_id: str) -> ConnectionStatus:
        """
        获取连接状态
        
        Args:
            connection_id: 连接ID
            
        Returns:
            ConnectionStatus: 连接状态
        """
        connection = self.connection_pool.get_connection(connection_id)
        return connection.status if connection else ConnectionStatus.DISCONNECTED
    
    def set_socket_options(self, socket_obj: socket.socket, options: Dict[str, Any]) -> None:
        """
        设置套接字选项
        
        Args:
            socket_obj: 套接字对象
            options: 选项字典
        """
        try:
            for option, value in options.items():
                if option == 'SO_REUSEADDR':
                    socket_obj.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, value)
                elif option == 'SO_KEEPALIVE':
                    socket_obj.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, value)
                elif option == 'TCP_NODELAY':
                    socket_obj.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, value)
                # 可以添加更多选项
                
            logger.debug(f"Socket options set: {options}")
            
        except Exception as e:
            logger.error(f"Failed to set socket options: {e}")
            raise NetworkError(f"Failed to set socket options: {e}")
    
    async def async_send_data(self, connection: Connection, data: bytes) -> bool:
        """
        异步发送数据
        
        Args:
            connection: 连接对象
            data: 要发送的数据
            
        Returns:
            bool: 发送是否成功
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, connection.send, data
        )
    
    async def async_receive_data(self, connection: Connection, timeout: float = 30.0) -> bytes:
        """
        异步接收数据
        
        Args:
            connection: 连接对象
            timeout: 超时时间
            
        Returns:
            bytes: 接收到的数据
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, connection.receive, timeout
        )
    
    def get_network_stats(self) -> Dict[str, Any]:
        """
        获取网络统计信息
        
        Returns:
            Dict[str, Any]: 网络统计信息
        """
        pool_stats = self.connection_pool.get_pool_stats()
        
        return {
            'connection_pool': pool_stats,
            'ssl_enabled': self.ssl_context is not None,
            'max_workers': self.executor._max_workers,
            'config': dict(self.config)
        }
    
    def shutdown(self) -> None:
        """关闭网络接口"""
        try:
            # 清理所有连接
            active_connections = self.connection_pool.get_active_connections()
            for connection_id in active_connections:
                self.connection_pool.remove_connection(connection_id)
            
            # 关闭线程池
            self.executor.shutdown(wait=True)
            
            logger.debug("NetworkInterface shutdown 完成")
            
        except Exception as e:
            logger.error(f"Error during NetworkInterface shutdown: {e}")
