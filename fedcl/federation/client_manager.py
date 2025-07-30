# fedcl/federation/client_manager.py
"""
客户端管理器实现

负责管理联邦学习系统中的客户端，包括注册、状态管理、选择策略、
通信协调等功能。支持真联邦和伪联邦两种模式。
"""

from enum import Enum
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import random
import time
from pathlib import Path
from omegaconf import DictConfig
from loguru import logger

from ..core.exceptions import ConfigurationError
from ..communication.communication_manager import CommunicationManager


class ClientStatus(Enum):
    """客户端状态枚举"""
    DISCONNECTED = "disconnected"
    CONNECTED = "connected"
    TRAINING = "training"
    READY = "ready"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class ClientInfo:
    """客户端信息数据类"""
    client_id: str
    status: ClientStatus = ClientStatus.DISCONNECTED
    last_seen: datetime = field(default_factory=datetime.now)
    capabilities: Dict[str, Any] = field(default_factory=dict)
    statistics: Dict[str, Any] = field(default_factory=dict)
    failure_count: int = 0
    selected_count: int = 0
    registration_time: datetime = field(default_factory=datetime.now)


class ClientManager:
    """
    客户端管理器
    
    负责管理联邦学习系统中的所有客户端，包括注册管理、状态跟踪、
    客户端选择、通信协调和故障处理等功能。
    
    Attributes:
        config: 客户端管理配置
        clients: 客户端信息映射
        selection_strategy: 客户端选择策略
        communication_manager: 通信管理器（真联邦模式）
    """
    
    def __init__(self, config: DictConfig) -> None:
        """
        初始化客户端管理器
        
        Args:
            config: 配置参数
            
        Raises:
            ConfigurationError: 配置参数无效时抛出
        """
        if not isinstance(config, DictConfig):
            raise ConfigurationError("Invalid configuration provided")
            
        self.config = config
        self.clients: Dict[str, ClientInfo] = {}
        self.selection_strategy = config.get("selection_strategy", "random")
        
        # 真联邦模式需要通信管理器
        self.communication_manager: Optional[CommunicationManager] = None
        if config.get("federation_mode", "pseudo") == "real":
            self.communication_manager = CommunicationManager(
                config.get("communication", {})
            )
        
        # 配置参数
        self.max_clients = config.get("max_clients", 100)
        self.client_timeout = config.get("client_timeout", 300)  # 5分钟
        self.max_failures = config.get("max_failures", 3)
        
        logger.info(f"Initialized ClientManager with {self.selection_strategy} selection strategy")
    
    def register_client(self, client_id: str, client_info: Dict[str, Any]) -> bool:
        """
        注册客户端
        
        Args:
            client_id: 客户端唯一标识
            client_info: 客户端信息，包括能力、资源等
            
        Returns:
            注册是否成功
        """
        try:
            if len(self.clients) >= self.max_clients:
                logger.warning(f"Maximum clients limit ({self.max_clients}) reached")
                return False
                
            if client_id in self.clients:
                logger.warning(f"Client {client_id} already registered, updating info")
                self.clients[client_id].capabilities.update(client_info)
                self.clients[client_id].last_seen = datetime.now()
                self.clients[client_id].status = ClientStatus.CONNECTED
            else:
                self.clients[client_id] = ClientInfo(
                    client_id=client_id,
                    status=ClientStatus.CONNECTED,
                    capabilities=client_info.copy()
                )
                
            logger.info(f"Client {client_id} registered successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register client {client_id}: {e}")
            return False
    
    def unregister_client(self, client_id: str) -> bool:
        """
        注销客户端
        
        Args:
            client_id: 客户端唯一标识
            
        Returns:
            注销是否成功
        """
        try:
            if client_id not in self.clients:
                logger.warning(f"Client {client_id} not found")
                return False
                
            # 更新状态为断开连接
            self.clients[client_id].status = ClientStatus.DISCONNECTED
            self.clients[client_id].last_seen = datetime.now()
            
            logger.info(f"Client {client_id} unregistered successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unregister client {client_id}: {e}")
            return False
    
    def get_active_clients(self) -> List[str]:
        """
        获取活跃客户端列表
        
        Returns:
            活跃客户端ID列表
        """
        active_clients = []
        current_time = datetime.now()
        
        for client_id, client_info in self.clients.items():
            # 检查客户端是否超时
            time_diff = current_time - client_info.last_seen
            if time_diff.total_seconds() > self.client_timeout:
                if client_info.status != ClientStatus.TIMEOUT:
                    client_info.status = ClientStatus.TIMEOUT
                    logger.warning(f"Client {client_id} timed out")
                continue
                
            # 只返回连接状态的客户端
            if client_info.status in [ClientStatus.CONNECTED, ClientStatus.READY]:
                active_clients.append(client_id)
                
        return active_clients
    
    def select_clients_for_round(self, num_clients: int, round_id: int) -> List[str]:
        """
        选择参与当前轮次的客户端
        
        Args:
            num_clients: 需要选择的客户端数量
            round_id: 当前轮次ID
            
        Returns:
            被选中的客户端ID列表
        """
        active_clients = self.get_active_clients()
        
        if len(active_clients) == 0:
            logger.warning("No active clients available for selection")
            return []
            
        # 限制选择数量
        num_clients = min(num_clients, len(active_clients))
        
        if self.selection_strategy == "random":
            selected = self._random_selection(active_clients, num_clients)
        elif self.selection_strategy == "round_robin":
            selected = self._round_robin_selection(active_clients, num_clients, round_id)
        elif self.selection_strategy == "capability_based":
            selected = self._capability_based_selection(active_clients, num_clients)
        else:
            logger.warning(f"Unknown selection strategy: {self.selection_strategy}, using random")
            selected = self._random_selection(active_clients, num_clients)
            
        # 更新选择统计
        for client_id in selected:
            self.clients[client_id].selected_count += 1
            
        logger.info(f"Selected {len(selected)} clients for round {round_id}: {selected}")
        return selected
    
    def _random_selection(self, active_clients: List[str], num_clients: int) -> List[str]:
        """随机选择策略"""
        return random.sample(active_clients, num_clients)
    
    def _round_robin_selection(self, active_clients: List[str], 
                             num_clients: int, round_id: int) -> List[str]:
        """轮询选择策略"""
        # 根据选择次数排序，优先选择被选择次数少的客户端
        sorted_clients = sorted(active_clients, 
                              key=lambda x: self.clients[x].selected_count)
        return sorted_clients[:num_clients]
    
    def _capability_based_selection(self, active_clients: List[str], 
                                  num_clients: int) -> List[str]:
        """基于能力的选择策略"""
        # 根据客户端能力评分排序
        def capability_score(client_id: str) -> float:
            capabilities = self.clients[client_id].capabilities
            score = 0.0
            score += capabilities.get("cpu_count", 1) * 0.3
            score += capabilities.get("memory_gb", 4) * 0.3
            score += capabilities.get("network_bandwidth", 100) * 0.2
            score += (1.0 / (self.clients[client_id].failure_count + 1)) * 0.2
            return score
            
        sorted_clients = sorted(active_clients, 
                              key=capability_score, reverse=True)
        return sorted_clients[:num_clients]
    
    def update_client_status(self, client_id: str, status: ClientStatus) -> None:
        """
        更新客户端状态
        
        Args:
            client_id: 客户端ID
            status: 新状态
        """
        if client_id not in self.clients:
            logger.warning(f"Client {client_id} not found")
            return
            
        old_status = self.clients[client_id].status
        self.clients[client_id].status = status
        self.clients[client_id].last_seen = datetime.now()
        
        logger.debug(f"Client {client_id} status changed: {old_status} -> {status}")
    
    def get_client_info(self, client_id: str) -> Dict[str, Any]:
        """
        获取客户端信息
        
        Args:
            client_id: 客户端ID
            
        Returns:
            客户端信息字典
        """
        if client_id not in self.clients:
            return {}
            
        client_info = self.clients[client_id]
        return {
            "client_id": client_info.client_id,
            "status": client_info.status.value,
            "last_seen": client_info.last_seen.isoformat(),
            "capabilities": client_info.capabilities,
            "statistics": client_info.statistics,
            "failure_count": client_info.failure_count,
            "selected_count": client_info.selected_count,
            "registration_time": client_info.registration_time.isoformat()
        }
    
    def broadcast_to_clients(self, message: Any, targets: List[str]) -> Dict[str, bool]:
        """
        向客户端广播消息
        
        Args:
            message: 要广播的消息
            targets: 目标客户端列表
            
        Returns:
            每个客户端的发送结果
        """
        results = {}
        
        if self.communication_manager is None:
            # 伪联邦模式，模拟广播
            logger.info(f"Pseudo-federation mode: simulating broadcast to {len(targets)} clients")
            for client_id in targets:
                if client_id in self.clients:
                    results[client_id] = True
                    self.clients[client_id].last_seen = datetime.now()
                else:
                    results[client_id] = False
        else:
            # 真联邦模式，使用通信管理器
            for client_id in targets:
                try:
                    success = self.communication_manager.send_message(client_id, message)
                    results[client_id] = success
                    if success and client_id in self.clients:
                        self.clients[client_id].last_seen = datetime.now()
                except Exception as e:
                    logger.error(f"Failed to send message to client {client_id}: {e}")
                    results[client_id] = False
                    
        logger.info(f"Broadcast completed: {sum(results.values())}/{len(targets)} successful")
        return results
    
    def collect_from_clients(self, data_type: str, sources: List[str], 
                           timeout: float = 60.0) -> Dict[str, Any]:
        """
        从客户端收集数据
        
        Args:
            data_type: 数据类型
            sources: 源客户端列表
            timeout: 超时时间
            
        Returns:
            从各客户端收集的数据
        """
        collected_data = {}
        
        if self.communication_manager is None:
            # 伪联邦模式，模拟数据收集
            logger.info(f"Pseudo-federation mode: simulating data collection from {len(sources)} clients")
            for client_id in sources:
                if client_id in self.clients:
                    # 模拟收集到的数据
                    collected_data[client_id] = {
                        "data_type": data_type,
                        "timestamp": datetime.now().isoformat(),
                        "client_id": client_id,
                        "simulated": True
                    }
                    self.clients[client_id].last_seen = datetime.now()
        else:
            # 真联邦模式，使用通信管理器
            for client_id in sources:
                try:
                    data = self.communication_manager.receive_data(
                        client_id, data_type, timeout
                    )
                    if data is not None:
                        collected_data[client_id] = data
                        if client_id in self.clients:
                            self.clients[client_id].last_seen = datetime.now()
                except Exception as e:
                    logger.error(f"Failed to collect data from client {client_id}: {e}")
                    
        logger.info(f"Data collection completed: {len(collected_data)}/{len(sources)} successful")
        return collected_data
    
    def get_client_statistics(self, client_id: str) -> Dict[str, Any]:
        """
        获取客户端统计信息
        
        Args:
            client_id: 客户端ID
            
        Returns:
            客户端统计信息
        """
        if client_id not in self.clients:
            return {}
            
        client_info = self.clients[client_id]
        current_time = datetime.now()
        uptime = current_time - client_info.registration_time
        
        return {
            "client_id": client_id,
            "status": client_info.status.value,
            "uptime_seconds": uptime.total_seconds(),
            "failure_count": client_info.failure_count,
            "selected_count": client_info.selected_count,
            "last_seen": client_info.last_seen.isoformat(),
            "statistics": client_info.statistics.copy()
        }
    
    def set_client_selection_strategy(self, strategy: str) -> None:
        """
        设置客户端选择策略
        
        Args:
            strategy: 选择策略名称
        """
        valid_strategies = ["random", "round_robin", "capability_based"]
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy: {strategy}. Valid strategies: {valid_strategies}")
            
        self.selection_strategy = strategy
        logger.info(f"Client selection strategy updated to: {strategy}")
    
    def handle_client_failure(self, client_id: str, error: Exception) -> None:
        """
        处理客户端故障
        
        Args:
            client_id: 故障客户端ID
            error: 故障异常
        """
        if client_id not in self.clients:
            logger.warning(f"Unknown client {client_id} reported failure")
            return
            
        client_info = self.clients[client_id]
        client_info.failure_count += 1
        client_info.status = ClientStatus.ERROR
        client_info.last_seen = datetime.now()
        
        logger.error(f"Client {client_id} failure #{client_info.failure_count}: {error}")
        
        # 检查是否超过最大故障次数
        if client_info.failure_count >= self.max_failures:
            logger.warning(f"Client {client_id} exceeded max failures, marking as disconnected")
            client_info.status = ClientStatus.DISCONNECTED
            
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        获取客户端管理器汇总统计
        
        Returns:
            汇总统计信息
        """
        total_clients = len(self.clients)
        status_counts = {}
        for status in ClientStatus:
            status_counts[status.value] = sum(
                1 for client in self.clients.values() 
                if client.status == status
            )
            
        active_clients = len(self.get_active_clients())
        
        return {
            "total_clients": total_clients,
            "active_clients": active_clients,
            "status_distribution": status_counts,
            "selection_strategy": self.selection_strategy,
            "federation_mode": self.config.get("federation_mode", "pseudo")
        }
