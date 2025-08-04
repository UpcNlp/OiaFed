# fedcl/communication/communication_manager.py
"""
通信管理器模块

实现FedCL框架的高级通信管理功能，包括数据发送、接收、模型广播、
更新收集、客户端管理和心跳监控等。
"""

import threading
import time
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from dataclasses import dataclass
from datetime import datetime
import torch
import torch.nn as nn
from loguru import logger
from omegaconf import DictConfig

from .communication_handler import CommunicationHandler
from .message_protocol import Message
from .exceptions import CommunicationError, ConnectionTimeoutError


@dataclass
class ClientInfo:
    """客户端信息"""
    client_id: str
    address: Tuple[str, int]
    last_heartbeat: float
    status: str = "active"  # active, inactive, disconnected
    total_messages: int = 0
    total_bytes: int = 0


class HeartbeatMonitor:
    """心跳监控器"""
    
    def __init__(self, manager: 'CommunicationManager', interval: float = 30.0):
        self.manager = manager
        self.interval = interval
        self.running = False
        self.thread: Optional[threading.Thread] = None
    
    def start(self) -> None:
        """启动心跳监控"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.thread.start()
            logger.debug(f"Heartbeat monitor started with interval {self.interval}s")
    
    def stop(self) -> None:
        """停止心跳监控"""
        if self.running:
            self.running = False
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=2.0)
                if self.thread.is_alive():
                    logger.warning("Heartbeat monitor thread did not stop gracefully")
            logger.debug("Heartbeat monitor 已停止")
    
    def _monitor_loop(self) -> None:
        """监控循环"""
        while self.running:
            try:
                self.manager._check_client_heartbeats()
                # 分割睡眠时间，以便更快响应停止信号
                sleep_time = self.interval
                while sleep_time > 0 and self.running:
                    chunk = min(0.1, sleep_time)  # 每次最多睡眠0.1秒
                    time.sleep(chunk)
                    sleep_time -= chunk
            except Exception as e:
                logger.error(f"Error in heartbeat monitor: {e}")
                # 错误情况下也要分割睡眠
                for _ in range(10):  # 1秒 = 10 * 0.1秒
                    if not self.running:
                        break
                    time.sleep(0.1)


class CommunicationManager:
    """
    通信管理器
    
    提供高级的通信管理功能，包括数据发送、接收、模型广播、
    更新收集、客户端管理和心跳监控等。
    """
    
    def __init__(self, config: DictConfig, handler: CommunicationHandler) -> None:
        """
        初始化通信管理器
        
        Args:
            config: 通信配置
            handler: 通信处理器
        """
        self.config = config
        self.handler = handler
        
        # 配置参数
        self.host = config.get('host', '0.0.0.0')
        self.port = config.get('port', 8080)
        self.connection_timeout = config.get('timeouts', {}).get('connection', 30.0)
        self.send_timeout = config.get('timeouts', {}).get('send', 10.0)
        self.receive_timeout = config.get('timeouts', {}).get('receive', 30.0)
        self.heartbeat_interval = config.get('heartbeat', {}).get('interval', 30.0)
        self.heartbeat_timeout = config.get('heartbeat', {}).get('timeout', 10.0)
        
        # 客户端管理
        self._clients: Dict[str, ClientInfo] = {}
        self._client_lock = threading.Lock()
        
        # 心跳监控
        self._heartbeat_monitor = HeartbeatMonitor(self, self.heartbeat_interval)
        
        # 异步支持
        self._executor = ThreadPoolExecutor(max_workers=20)
        
        # 消息缓存
        self._pending_updates: Dict[str, List[Dict[str, torch.Tensor]]] = {}
        self._update_lock = threading.Lock()
        
        logger.debug(f"CommunicationManager initialized on {self.host}:{self.port}")
    
    def send_data(self, source: str, target: str, data: Any, data_type: str) -> None:
        """
        发送数据
        
        Args:
            source: 发送方ID
            target: 接收方ID
            data: 要发送的数据
            data_type: 数据类型
        """
        try:
            # 创建消息
            message = Message(
                message_id=f"{source}_{target}_{int(time.time() * 1000)}",
                message_type=data_type,
                sender=source,
                receiver=target,
                timestamp=datetime.now(),
                data=data,
                metadata={'data_type': data_type},
                checksum=""
            )
            
            # 计算校验和
            message.checksum = message.calculate_checksum()
            
            # 发送消息
            success = self.handler.send_message(target, message)
            
            if success:
                # 更新客户端统计
                with self._client_lock:
                    if target in self._clients:
                        self._clients[target].total_messages += 1
                        
                logger.debug(f"Data sent from {source} to {target}, type: {data_type}")
            else:
                raise CommunicationError(f"Failed to send data from {source} to {target}")
                
        except Exception as e:
            logger.error(f"Error sending data from {source} to {target}: {e}")
            raise CommunicationError(f"Failed to send data: {e}")
    
    def receive_data(self, source: str, data_type: str, timeout: float = 30.0) -> Any:
        """
        接收数据
        
        Args:
            source: 发送方ID
            data_type: 期望的数据类型
            timeout: 超时时间
            
        Returns:
            Any: 接收到的数据
        """
        try:
            message = self.handler.receive_message(source, timeout)
            
            # 验证消息类型
            if message.message_type != data_type:
                logger.warning(f"Received unexpected message type: {message.message_type}, expected: {data_type}")
            
            # 验证校验和
            if not message.validate_checksum():
                logger.warning(f"Message checksum validation failed from {source}")
            
            # 更新客户端统计
            with self._client_lock:
                if source in self._clients:
                    self._clients[source].total_messages += 1
                    self._clients[source].last_heartbeat = time.time()
            
            logger.debug(f"Data received from {source}, type: {data_type}")
            return message.data
            
        except Exception as e:
            logger.error(f"Error receiving data from {source}: {e}")
            raise CommunicationError(f"Failed to receive data: {e}")
    
    def broadcast_model(self, model: torch.nn.Module, targets: List[str]) -> Dict[str, bool]:
        """
        广播模型
        
        Args:
            model: 要广播的模型
            targets: 目标客户端列表
            
        Returns:
            Dict[str, bool]: 每个目标的发送结果
        """
        try:
            # 序列化模型
            model_data = self.handler.serializer.serialize_model(model)
            
            # 创建模型消息
            message = Message(
                message_id=f"broadcast_model_{int(time.time() * 1000)}",
                message_type="model_update",
                sender="server",
                receiver="broadcast",
                timestamp=datetime.now(),
                data=model_data,
                metadata={'model_size': len(model_data)},
                checksum=""
            )
            
            # 计算校验和
            message.checksum = message.calculate_checksum()
            
            # 广播消息
            results = self.handler.broadcast_message(message, targets)
            
            successful_count = sum(1 for success in results.values() if success)
            logger.debug(f"Model broadcast completed: {successful_count}/{len(targets)} successful")
            
            return results
            
        except Exception as e:
            logger.error(f"Error broadcasting model: {e}")
            raise CommunicationError(f"Failed to broadcast model: {e}")
    
    def collect_updates(self, sources: List[str], timeout: float = 60.0) -> List[Dict[str, torch.Tensor]]:
        """
        收集更新
        
        Args:
            sources: 源客户端列表
            timeout: 超时时间
            
        Returns:
            List[Dict[str, torch.Tensor]]: 收集到的更新列表
        """
        try:
            updates = []
            start_time = time.time()
            
            # 并行收集更新
            futures = {}
            for source in sources:
                future = self._executor.submit(self._collect_single_update, source, timeout)
                futures[source] = future
            
            # 收集结果
            for source, future in futures.items():
                try:
                    remaining_time = max(0.1, timeout - (time.time() - start_time))
                    update = future.result(timeout=remaining_time)
                    if update is not None:
                        updates.append(update)
                        logger.debug(f"Update collected from {source}")
                    else:
                        logger.warning(f"No update received from {source}")
                except Exception as e:
                    logger.error(f"Error collecting update from {source}: {e}")
            
            logger.info(f"Collected {len(updates)}/{len(sources)} updates")
            return updates
            
        except Exception as e:
            logger.error(f"Error collecting updates: {e}")
            raise CommunicationError(f"Failed to collect updates: {e}")
    
    def establish_connection(self, client_id: str, address: Tuple[str, int]) -> bool:
        """
        建立连接
        
        Args:
            client_id: 客户端ID
            address: 客户端地址
            
        Returns:
            bool: 连接是否成功建立
        """
        try:
            # 建立连接
            connection = self.handler.establish_connection(address)
            
            # 注册客户端
            with self._client_lock:
                self._clients[client_id] = ClientInfo(
                    client_id=client_id,
                    address=address,
                    last_heartbeat=time.time(),
                    status="active"
                )
            
            logger.debug(f"Connection established with client {client_id} at {address}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to establish connection with {client_id} at {address}: {e}")
            return False
    
    def handle_client_disconnect(self, client_id: str) -> None:
        """
        处理客户端断开
        
        Args:
            client_id: 客户端ID
        """
        try:
            with self._client_lock:
                if client_id in self._clients:
                    self._clients[client_id].status = "disconnected"
                    logger.info(f"Client {client_id} marked as disconnected")
            
            # 关闭连接
            try:
                self.handler.close_connection(client_id)
            except Exception as e:
                logger.warning(f"Error closing connection for client {client_id}: {e}")
            
        except Exception as e:
            logger.error(f"Error handling disconnect for client {client_id}: {e}")
    
    def get_active_clients(self) -> List[str]:
        """
        获取活跃客户端
        
        Returns:
            List[str]: 活跃客户端ID列表
        """
        with self._client_lock:
            return [client_id for client_id, info in self._clients.items() 
                   if info.status == "active"]
    
    def ping_client(self, client_id: str) -> bool:
        """
        ping客户端
        
        Args:
            client_id: 客户端ID
            
        Returns:
            bool: ping是否成功
        """
        try:
            # 发送ping消息
            ping_message = Message(
                message_id=f"ping_{client_id}_{int(time.time() * 1000)}",
                message_type="ping",
                sender="server",
                receiver=client_id,
                timestamp=datetime.now(),
                data={"ping": True},
                metadata={},
                checksum=""
            )
            ping_message.checksum = ping_message.calculate_checksum()
            
            success = self.handler.send_message(client_id, ping_message)
            
            if success:
                # 等待pong响应
                try:
                    response = self.handler.receive_message(client_id, timeout=self.heartbeat_timeout)
                    if response.message_type == "pong":
                        with self._client_lock:
                            if client_id in self._clients:
                                self._clients[client_id].last_heartbeat = time.time()
                        return True
                except ConnectionTimeoutError:
                    pass
            
            return False
            
        except Exception as e:
            logger.debug(f"Ping failed for client {client_id}: {e}")
            return False
    
    def start_heartbeat_monitor(self, interval: float = 30.0) -> None:
        """
        启动心跳监控
        
        Args:
            interval: 监控间隔
        """
        self._heartbeat_monitor.interval = interval
        self._heartbeat_monitor.start()
    
    def stop_heartbeat_monitor(self) -> None:
        """停止心跳监控"""
        self._heartbeat_monitor.stop()
    
    def _collect_single_update(self, source: str, timeout: float) -> Optional[Dict[str, torch.Tensor]]:
        """收集单个客户端的更新"""
        try:
            data = self.receive_data(source, "model_update", timeout)
            
            # 反序列化模型更新
            if isinstance(data, bytes):
                # 假设这是序列化的模型数据
                # 这里需要根据实际的序列化格式进行处理
                return {"serialized_data": data}
            elif isinstance(data, dict):
                # 如果已经是字典格式的张量数据
                return data
            else:
                logger.warning(f"Unexpected update data type from {source}: {type(data)}")
                return None
                
        except Exception as e:
            logger.error(f"Error collecting update from {source}: {e}")
            return None
    
    def _check_client_heartbeats(self) -> None:
        """检查客户端心跳"""
        current_time = time.time()
        timeout_threshold = self.heartbeat_interval * 3  # 3个心跳周期为超时
        
        with self._client_lock:
            for client_id, client_info in list(self._clients.items()):
                if client_info.status == "active":
                    time_since_heartbeat = current_time - client_info.last_heartbeat
                    
                    if time_since_heartbeat > timeout_threshold:
                        # 尝试ping客户端
                        if not self.ping_client(client_id):
                            # ping失败，标记为不活跃
                            client_info.status = "inactive"
                            logger.warning(f"Client {client_id} marked as inactive due to heartbeat timeout")
    
    def get_client_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        获取客户端统计信息
        
        Returns:
            Dict[str, Dict[str, Any]]: 客户端统计信息
        """
        with self._client_lock:
            stats = {}
            for client_id, client_info in self._clients.items():
                stats[client_id] = {
                    'address': f"{client_info.address[0]}:{client_info.address[1]}",
                    'status': client_info.status,
                    'last_heartbeat': client_info.last_heartbeat,
                    'total_messages': client_info.total_messages,
                    'total_bytes': client_info.total_bytes,
                    'time_since_heartbeat': time.time() - client_info.last_heartbeat
                }
            return stats
    
    def shutdown(self) -> None:
        """关闭通信管理器"""
        logger.info("Shutting down CommunicationManager")
        
        # 停止心跳监控
        self.stop_heartbeat_monitor()
        
        # 断开所有客户端
        client_ids = []
        with self._client_lock:
            client_ids = list(self._clients.keys())
            if client_ids:
                logger.info(f"Disconnecting {len(client_ids)} 客户端")
            # 在锁内标记所有客户端为断开状态
            for client_id in client_ids:
                if client_id in self._clients:
                    self._clients[client_id].status = "disconnected"
            self._clients.clear()
        
        # 在锁外关闭连接，避免死锁
        for client_id in client_ids:
            try:
                self.handler.close_connection(client_id)
            except Exception as e:
                logger.warning(f"Error closing connection for client {client_id}: {e}")
        
        # 关闭通信处理器
        try:
            self.handler.shutdown()
        except Exception as e:
            logger.error(f"Error shutting down handler: {e}")
        
        # 关闭线程池
        try:
            self._executor.shutdown(wait=False)  # 不等待，避免卡住
            # 等待最多2秒
            for i in range(20):  # 2秒 = 20 * 0.1秒
                if self._executor._threads:
                    time.sleep(0.1)
                else:
                    break
        except Exception as e:
            logger.error(f"Error shutting down executor: {e}")
        
        logger.info("CommunicationManager shutdown 完成")


class SimpleCommunicationManager:
    """
    简化的通信管理器
    
    用于同进程内的客户端-服务器通信，支持实际的消息传递
    """
    
    # 全局消息中心，用于不同实例间的通信
    _message_center = {
        'server_instance': None,
        'client_instances': {},
        'server_messages': [],
        'client_messages': {}
    }
    _center_lock = threading.Lock()
    
    def __init__(self, config: Optional[DictConfig] = None, role: str = "client", component_id: str = "unknown"):
        """
        初始化简化通信管理器
        
        Args:
            config: 通信配置（可选）
            role: 角色 ("server" 或 "client")
            component_id: 组件ID
        """
        self.config = config or {}
        self.role = role
        self.component_id = component_id
        self.logger = logger.bind(component=f"SimpleCommunicationManager[{role}:{component_id}]")
        self.message_handlers = {}
        
        # 注册到全局消息中心
        with self._center_lock:
            if role == "server":
                self._message_center['server_instance'] = self
                self._message_center['server_messages'] = []
            else:
                self._message_center['client_instances'][component_id] = self
                self._message_center['client_messages'][component_id] = []
        
        self.logger.debug(f"SimpleCommunicationManager initialized: {role}:{component_id}")
    
    def register_message_handler(self, message_type: str, handler):
        """注册消息处理器"""
        self.message_handlers[message_type] = handler
        self.logger.debug(f"Registered handler for message type: {message_type}")
    
    def send_message_to_server(self, message: Dict[str, Any]):
        """发送消息到服务端"""
        with self._center_lock:
            server_instance = self._message_center['server_instance']
            if server_instance:
                # 添加到服务器消息队列
                self._message_center['server_messages'].append(message)
                # 记录详细日志
                msg_type = message.get('type', 'unknown')
                sender = message.get('sender', 'unknown')
                self.logger.debug(f"Message sent to server: type='{msg_type}', sender='{sender}', full_message={message}")
                # 立即处理消息
                server_instance._handle_received_message(message)
            else:
                self.logger.warning("No server instance available")
    
    def send_message_to_client(self, client_id: str, message: Dict[str, Any]):
        """发送消息到指定客户端"""
        with self._center_lock:
            client_instance = self._message_center['client_instances'].get(client_id)
            if client_instance:
                # 添加到客户端消息队列
                if client_id not in self._message_center['client_messages']:
                    self._message_center['client_messages'][client_id] = []
                self._message_center['client_messages'][client_id].append(message)
                # 立即处理消息
                client_instance._handle_received_message(message)
                self.logger.debug(f"Message sent to client {client_id}: {message.get('type', 'unknown')}")
            else:
                self.logger.warning(f"Client {client_id} not found")
    
    def broadcast_to_clients(self, message: Dict[str, Any]):
        """广播消息到所有客户端"""
        with self._center_lock:
            for client_id in self._message_center['client_instances']:
                self.send_message_to_client(client_id, message)
    
    def _handle_received_message(self, message: Dict[str, Any]):
        """处理接收到的消息"""
        message_type = message.get('type', 'unknown')
        
        # 查找对应的处理器
        handler = self.message_handlers.get(message_type)
        if handler:
            try:
                # 调用处理器
                if callable(handler):
                    handler(message)
                else:
                    self.logger.warning(f"Handler for {message_type} is not callable")
            except Exception as e:
                self.logger.error(f"Error handling message {message_type}: {e}")
        else:
            self.logger.debug(f"No handler found for message type: {message_type}")
    
    def get_server_messages(self) -> List[Dict[str, Any]]:
        """获取服务端收到的消息"""
        with self._center_lock:
            return self._message_center['server_messages'].copy()
    
    def get_client_messages(self, client_id: str) -> List[Dict[str, Any]]:
        """获取指定客户端收到的消息"""
        with self._center_lock:
            return self._message_center['client_messages'].get(client_id, []).copy()
    
    def clear_messages(self):
        """清空所有消息"""
        with self._center_lock:
            self._message_center['server_messages'].clear()
            for client_id in self._message_center['client_messages']:
                self._message_center['client_messages'][client_id].clear()
    
    @classmethod
    def reset_message_center(cls):
        """重置消息中心（用于测试）"""
        with cls._center_lock:
            cls._message_center = {
                'server_instance': None,
                'client_instances': {},
                'server_messages': [],
                'client_messages': {}
            }
