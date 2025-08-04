# fedcl/communication/adaptive_communication_manager.py
"""
自适应通信管理器

根据配置自动选择通信模式：
- 当服务端和客户端配置相同IP:端口时，使用同进程模拟通信
- 当配置不同IP:端口时，使用真实网络通信
"""

import socket
import threading
import time
import json
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from loguru import logger

from .communication_manager import SimpleCommunicationManager
from .message_protocol import Message
from .exceptions import CommunicationError, ConnectionTimeoutError
from ..config.config_manager import DictConfig


@dataclass
class EndpointInfo:
    """端点信息"""
    host: str
    port: int
    component_id: str
    role: str  # "server" or "client"
    
    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"


class NetworkCommunicationManager:
    """真实网络通信管理器"""
    
    def __init__(self, config: DictConfig, role: str, component_id: str):
        self.config = config
        self.role = role
        self.component_id = component_id
        self.logger = logger.bind(component=f"NetworkComm[{role}:{component_id}]")
        
        # 网络配置
        self.host = self._get_config_value('host', 'localhost')
        self.port = self._get_config_value('port', 8080)
        self.timeout = self._get_config_value('timeout', 60.0)
        self.max_workers = self._get_config_value('max_workers', 5)
        
        # 连接管理
        self.connections: Dict[str, socket.socket] = {}
        self.server_socket: Optional[socket.socket] = None
        self.message_handlers: Dict[str, callable] = {}
        self.is_running = False
        
        # 线程管理
        self._server_thread: Optional[threading.Thread] = None
        self._connection_lock = threading.Lock()
        
        self.logger.debug(f"NetworkCommunicationManager initialized: {self.host}:{self.port}")
    
    def _get_config_value(self, key: str, default=None):
        """获取配置值，从communication子配置中查找"""
        # 首先尝试从communication子配置中获取
        comm_config = None
        if hasattr(self.config, 'get'):
            comm_config = self.config.get('communication', {})
        elif hasattr(self.config, 'communication'):
            comm_config = self.config.communication
        
        if comm_config:
            if hasattr(comm_config, 'get'):
                return comm_config.get(key, default)
            elif hasattr(comm_config, key):
                return getattr(comm_config, key, default)
        
        # 如果communication中没有，从顶级配置中获取
        if hasattr(self.config, 'get'):
            return self.config.get(key, default)
        elif hasattr(self.config, key):
            return getattr(self.config, key, default)
        else:
            return default
    
    def start(self):
        """启动网络通信"""
        # 先设置运行状态，再启动服务端
        self.is_running = True
        if self.role == "server":
            self._start_server()
        self.logger.debug(f"Network communication started for {self.role}")
    
    def stop(self):
        """停止网络通信"""
        self.is_running = False
        
        if self.server_socket:
            try:
                self.server_socket.close()
            except Exception as e:
                self.logger.error(f"Error closing server socket: {e}")
        
        with self._connection_lock:
            for conn in self.connections.values():
                try:
                    conn.close()
                except Exception as e:
                    self.logger.error(f"Error closing connection: {e}")
            self.connections.clear()
        
        if self._server_thread and self._server_thread.is_alive():
            self._server_thread.join(timeout=2.0)
        
        self.logger.info("Network communication 已停止")
    
    def _start_server(self):
        """启动服务端监听"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.settimeout(1.0)  # 设置超时，避免无限阻塞
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(self.max_workers)
            
            self.logger.debug(f"Server socket created and bound to {self.host}:{self.port}")
            
            self._server_thread = threading.Thread(target=self._accept_connections, daemon=True)
            self._server_thread.start()
            
            self.logger.debug(f"Server thread started: {self._server_thread.is_alive()}")
            
            self.logger.debug(f"Server listening on {self.host}:{self.port}")
            
        except Exception as e:
            self.logger.error(f"服务器启动失败: {e}")
            raise CommunicationError(f"服务器启动失败: {e}")
    
    def _accept_connections(self):
        """接受客户端连接"""
        self.logger.debug("Server is ready to accept connections")
        connection_count = 0
        while self.is_running:
            try:
                if self.server_socket:
                    self.logger.debug(f"Waiting for connection... (attempt {connection_count + 1})")
                    conn, addr = self.server_socket.accept()
                    connection_count += 1
                    self.logger.debug(f"Accepted connection #{connection_count} from {addr}")
                    threading.Thread(
                        target=self._handle_client_connection, 
                        args=(conn, addr), 
                        daemon=True
                    ).start()
            except socket.timeout:
                # 超时是正常的，继续循环
                self.logger.debug("Accept timeout, continuing...")
                continue
            except Exception as e:
                if self.is_running:
                    self.logger.error(f"Error accepting connection: {e}")
                break
        self.logger.debug(f"Accept loop exited, total connections accepted: {connection_count}")
    
    def _handle_client_connection(self, conn: socket.socket, addr: Tuple[str, int]):
        """处理客户端连接"""
        # 客户端连接由服务端管理，不需要存储在连接字典中
        # 因为客户端会主动连接服务端，服务端只需要处理接收的消息
        client_id = f"{addr[0]}:{addr[1]}"
        self.logger.info(f"New client connection: {client_id}")
        
        try:
            while self.is_running:
                data = self._receive_message(conn)
                if data:
                    self.logger.debug(f"Received message from {client_id}: {data.get('type', 'unknown')}")
                    self._process_received_message(data, client_id)
                # 注意：不要在data为None时break，因为可能只是超时
        except Exception as e:
            self.logger.error(f"Error handling client {client_id}: {e}")
        finally:
            try:
                conn.close()
            except:
                pass
            self.logger.info(f"Client disconnected: {client_id}")
    
    def connect_to_server(self, server_host: str, server_port: int) -> bool:
        """客户端连接到服务端"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            sock.connect((server_host, server_port))
            
            # 客户端连接成功后，设置运行状态
            self.is_running = True
            
            server_id = f"{server_host}:{server_port}"
            with self._connection_lock:
                self.connections[server_id] = sock
                self.logger.debug(f"Connection stored: {server_id}, total connections: {len(self.connections)}")
            
            # 启动消息接收线程
            threading.Thread(
                target=self._handle_server_connection, 
                args=(sock, server_id), 
                daemon=True
            ).start()
            
            self.logger.info(f"Connected to server: {server_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to server {server_host}:{server_port}: {e}")
            return False
    
    def _handle_server_connection(self, conn: socket.socket, server_id: str):
        """处理服务端连接"""
        self.logger.debug(f"Starting server connection handler for {server_id}")
        try:
            while self.is_running:
                data = self._receive_message(conn)
                if data:
                    self._process_received_message(data, server_id)
                # 注意：不要在data为None时break，因为可能只是超时
        except Exception as e:
            self.logger.error(f"Error handling server connection: {e}")
        finally:
            with self._connection_lock:
                if server_id in self.connections:
                    del self.connections[server_id]
                    self.logger.debug(f"Removed connection {server_id}, remaining: {list(self.connections.keys())}")
            try:
                conn.close()
            except:
                pass
            self.logger.debug(f"Server connection handler exiting for {server_id}")

    def send_message_to_server(self, message: Dict[str, Any]) -> bool:
        """发送消息到服务端"""
        server_config = self._get_config_value('server', {})
        if hasattr(server_config, 'get'):
            server_host = server_config.get('host', self.host)
            server_port = server_config.get('port', self.port)
        elif hasattr(server_config, 'host'):
            server_host = getattr(server_config, 'host', self.host)
            server_port = getattr(server_config, 'port', self.port)
        else:
            server_host = self.host
            server_port = self.port
        
        server_id = f"{server_host}:{server_port}"
        
        return self._send_message(server_id, message)
    
    def send_message_to_client(self, client_id: str, message: Dict[str, Any]) -> bool:
        """发送消息到客户端"""
        return self._send_message(client_id, message)
    
    def _send_message(self, target_id: str, message: Dict[str, Any]) -> bool:
        """发送消息到指定目标"""
        try:
            with self._connection_lock:
                conn = self.connections.get(target_id)
            
            if not conn:
                # 如果是客户端要连接服务端
                if self.role == "client" and ":" in target_id:
                    host, port = target_id.split(":")
                    if self.connect_to_server(host, int(port)):
                        # 稍微等待一下让连接稳定
                        time.sleep(0.1)
                        with self._connection_lock:
                            conn = self.connections.get(target_id)
                            self.logger.debug(f"After reconnect, connections: {list(self.connections.keys())}")
                
                if not conn:
                    self.logger.warning(f"No connection to {target_id}")
                    return False
            
            # 序列化消息
            data = json.dumps(message).encode('utf-8')
            length = len(data)
            
            # 发送消息长度和数据
            conn.sendall(length.to_bytes(4, byteorder='big'))
            conn.sendall(data)
            
            self.logger.debug(f"Message sent to {target_id}: {message.get('type', 'unknown')}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send message to {target_id}: {e}")
            return False
    
    def _receive_message(self, conn: socket.socket) -> Optional[Dict[str, Any]]:
        """接收消息"""
        try:
            # 设置短超时以避免阻塞太久
            conn.settimeout(1.0)
            
            # 接收消息长度
            length_data = self._receive_exact(conn, 4)
            if not length_data:
                return None

            length = int.from_bytes(length_data, byteorder='big')
            
            # 接收消息数据
            data = self._receive_exact(conn, length)
            if not data:
                return None

            # 反序列化消息
            message = json.loads(data.decode('utf-8'))
            return message
            
        except socket.timeout:
            # 超时是正常的，不记录错误
            return None
        except Exception as e:
            self.logger.error(f"Failed to receive message: {e}")
            return None

    def _receive_exact(self, conn: socket.socket, length: int) -> Optional[bytes]:
        """接收指定长度的数据"""
        data = b''
        while len(data) < length:
            try:
                chunk = conn.recv(length - len(data))
                if not chunk:
                    return None
                data += chunk
            except socket.timeout:
                # 超时返回None，让上层处理
                return None
            except Exception as e:
                self.logger.error(f"Error receiving data: {e}")
                return None
        return data
    
    def _process_received_message(self, message: Dict[str, Any], sender_id: str):
        """处理接收到的消息"""
        message_type = message.get('type', 'unknown')
        
        # 查找消息处理器
        handler = self.message_handlers.get(message_type)
        if handler:
            try:
                handler(message)
            except Exception as e:
                self.logger.error(f"Error processing message {message_type}: {e}")
        else:
            self.logger.debug(f"No handler for message type: {message_type}")
    
    def register_message_handler(self, message_type: str, handler: callable):
        """注册消息处理器"""
        self.message_handlers[message_type] = handler
        self.logger.debug(f"Registered handler for message type: {message_type}")


class AdaptiveCommunicationManager:
    """
    自适应通信管理器
    
    根据配置自动选择通信模式：
    - 同IP:端口 -> 同进程模拟通信
    - 不同IP:端口 -> 真实网络通信
    """
    
    def __init__(self, config: DictConfig, role: str, component_id: str):
        self.config = config
        self.role = role
        self.component_id = component_id
        self.logger = logger.bind(component=f"AdaptiveComm[{role}:{component_id}]")
        
        # 通信模式选择
        self.communication_mode = self._determine_communication_mode()
        
        # 创建具体的通信管理器
        if self.communication_mode == "local":
            self.comm_manager = SimpleCommunicationManager(config, role, component_id)
            self.logger.debug("Using local (in-process) communication")
        else:
            self.comm_manager = NetworkCommunicationManager(config, role, component_id)
            self.logger.debug("Using network communication")
    
    def _determine_communication_mode(self) -> str:
        """
        确定通信模式
        
        Returns:
            str: "local" 或 "network"
        """
        try:
            # 获取当前组件的端点信息
            current_host = self._get_config_value('host', 'localhost')
            current_port = self._get_config_value('port', 8080)
            current_endpoint = f"{current_host}:{current_port}"
            
            # 检查是否显式指定了模式
            explicit_mode = self._get_config_value('mode', None)
            if explicit_mode:
                if explicit_mode.lower() == 'network':
                    self.logger.debug(f"Explicit network mode specified for {self.role}")
                    return "network"
                elif explicit_mode.lower() == 'local':
                    self.logger.debug(f"Explicit local mode specified for {self.role}")
                    return "local"
            
            # 对于客户端，检查服务端配置
            if self.role == "client":
                server_config = self._get_config_value('server', {})
                if server_config:
                    if hasattr(server_config, 'get'):
                        server_host = server_config.get('host', current_host)
                        server_port = server_config.get('port', current_port)
                    elif hasattr(server_config, 'host'):
                        server_host = getattr(server_config, 'host', current_host)
                        server_port = getattr(server_config, 'port', current_port)
                    else:
                        server_host = current_host
                        server_port = current_port
                    
                    server_endpoint = f"{server_host}:{server_port}"
                    
                    # 检查是否为同一端点
                    if current_endpoint == server_endpoint:
                        self.logger.debug(f"Same endpoint detected ({current_endpoint}), using local communication")
                        return "local"
                    else:
                        self.logger.debug(f"Different endpoints: current={current_endpoint}, server={server_endpoint}, using network communication")
                        return "network"
            
            # 对于服务端，使用更智能的默认策略
            if self.role == "server":
                # 检查是否为标准的网络端口配置
                if current_port != 8080 or current_host not in ['localhost', '127.0.0.1']:
                    self.logger.debug(f"Server with non-default configuration ({current_endpoint}), using network communication")
                    return "network"
                else:
                    # 默认情况下，如果没有明确指定，服务端使用本地模式
                    self.logger.debug(f"Server with default configuration ({current_endpoint}), using local communication")
                    return "local"
            
            # 默认策略：localhost:8080 -> local，其他 -> network
            if current_host in ['localhost', '127.0.0.1'] and self._is_default_port(current_port):
                return "local"
            else:
                return "network"
                
        except Exception as e:
            self.logger.warning(f"Error determining communication mode: {e}, defaulting to local")
            return "local"
    
    def _get_config_value(self, key: str, default=None):
        """获取配置值，从communication子配置中查找"""
        # 首先尝试从communication子配置中获取
        comm_config = None
        if hasattr(self.config, 'get'):
            comm_config = self.config.get('communication', {})
        elif hasattr(self.config, 'communication'):
            comm_config = self.config.communication
        
        if comm_config:
            if hasattr(comm_config, 'get'):
                return comm_config.get(key, default)
            elif hasattr(comm_config, key):
                return getattr(comm_config, key, default)
        
        # 如果communication中没有，从顶级配置中获取
        if hasattr(self.config, 'get'):
            return self.config.get(key, default)
        elif hasattr(self.config, key):
            return getattr(self.config, key, default)
        else:
            return default
    
    def _is_default_port(self, port: int) -> bool:
        """检查是否为默认端口"""
        return port == 8080  # 可以配置为多个默认端口
    
    def start(self):
        """启动通信管理器"""
        if hasattr(self.comm_manager, 'start'):
            self.comm_manager.start()
    
    def stop(self):
        """停止通信管理器"""
        if hasattr(self.comm_manager, 'stop'):
            self.comm_manager.stop()
    
    def register_message_handler(self, message_type: str, handler: callable):
        """注册消息处理器"""
        self.comm_manager.register_message_handler(message_type, handler)
    
    def send_message_to_server(self, message: Dict[str, Any]):
        """发送消息到服务端"""
        return self.comm_manager.send_message_to_server(message)
    
    def send_message_to_client(self, client_id: str, message: Dict[str, Any]):
        """发送消息到客户端"""
        return self.comm_manager.send_message_to_client(client_id, message)
    
    def broadcast_to_clients(self, message: Dict[str, Any]):
        """广播消息到所有客户端"""
        if hasattr(self.comm_manager, 'broadcast_to_客户端'):
            return self.comm_manager.broadcast_to_clients(message)
        else:
            self.logger.warning("Broadcast not supported in current communication mode")
    
    @property
    def is_local_mode(self) -> bool:
        """是否为本地模式"""
        return self.communication_mode == "local"
    
    @property
    def is_network_mode(self) -> bool:
        """是否为网络模式"""
        return self.communication_mode == "network"
    
    def get_communication_info(self) -> Dict[str, Any]:
        """获取通信信息"""
        return {
            'mode': self.communication_mode,
            'role': self.role,
            'component_id': self.component_id,
            'host': self._get_config_value('host', 'localhost'),
            'port': self._get_config_value('port', 8080),
            'is_local': self.is_local_mode,
            'is_network': self.is_network_mode
        }


def create_adaptive_communication_manager(config: DictConfig, role: str, component_id: str) -> AdaptiveCommunicationManager:
    """
    创建自适应通信管理器的工厂函数
    
    Args:
        config: 通信配置
        role: 角色 ("server" 或 "client")
        component_id: 组件ID
        
    Returns:
        AdaptiveCommunicationManager: 自适应通信管理器实例
    """
    return AdaptiveCommunicationManager(config, role, component_id)
