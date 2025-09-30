"""
MOE-FedCL 连接池管理
moe_fedcl/connection/pool.py
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import weakref

from ..types import Connection, ConnectionStatus, ConnectionType
from ..exceptions import ConnectionError, MOEFedCLError


class PoolStrategy(Enum):
    """连接池策略"""
    FIFO = "fifo"  # 先进先出
    LIFO = "lifo"  # 后进先出
    LEAST_RECENTLY_USED = "lru"  # 最近最少使用
    ROUND_ROBIN = "round_robin"  # 轮询
    LOAD_BALANCED = "load_balanced"  # 负载均衡


class ConnectionHealthStatus(Enum):
    """连接健康状态"""
    HEALTHY = "healthy"
    WARNING = "warning"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class PoolConfiguration:
    """连接池配置"""
    max_connections: int = 100
    min_connections: int = 0
    max_idle_time: float = 300.0  # 最大空闲时间（秒）
    connection_timeout: float = 30.0  # 连接超时时间
    health_check_interval: float = 60.0  # 健康检查间隔
    strategy: PoolStrategy = PoolStrategy.LEAST_RECENTLY_USED
    enable_preemptive_creation: bool = True  # 启用预创建连接
    enable_auto_scaling: bool = True  # 启用自动伸缩
    scaling_threshold: float = 0.8  # 伸缩阈值


@dataclass
class ConnectionMetrics:
    """连接指标"""
    connection_id: str
    created_time: datetime
    last_used: datetime
    usage_count: int = 0
    error_count: int = 0
    average_response_time: float = 0.0
    total_response_time: float = 0.0
    health_status: ConnectionHealthStatus = ConnectionHealthStatus.UNKNOWN
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConnectionHealthChecker:
    """连接健康检查器"""
    
    def __init__(self, check_timeout: float = 5.0):
        self.check_timeout = check_timeout
    
    async def check_connection_health(self, connection: Connection) -> ConnectionHealthStatus:
        """检查连接健康状态"""
        try:
            # 基础状态检查
            if connection.status == ConnectionStatus.DISCONNECTED:
                return ConnectionHealthStatus.UNHEALTHY
            elif connection.status == ConnectionStatus.ERROR:
                return ConnectionHealthStatus.UNHEALTHY
            elif connection.status in [ConnectionStatus.CONNECTED, ConnectionStatus.ACTIVE]:
                # 可以添加更多的健康检查逻辑，比如ping测试
                return await self._perform_ping_check(connection)
            else:
                return ConnectionHealthStatus.WARNING
                
        except Exception as e:
            print(f"Health check failed for connection {connection.connection_id}: {e}")
            return ConnectionHealthStatus.UNHEALTHY
    
    async def _perform_ping_check(self, connection: Connection) -> ConnectionHealthStatus:
        """执行ping检查"""
        try:
            # 这里可以实现具体的ping逻辑
            # 暂时基于连接的最后活动时间进行判断
            now = datetime.now()
            last_activity = connection.last_activity
            
            if (now - last_activity).total_seconds() < 30:
                return ConnectionHealthStatus.HEALTHY
            elif (now - last_activity).total_seconds() < 120:
                return ConnectionHealthStatus.WARNING
            else:
                return ConnectionHealthStatus.UNHEALTHY
                
        except Exception:
            return ConnectionHealthStatus.UNKNOWN


class ConnectionPool:
    """连接池管理器"""
    
    def __init__(self, 
                 config: PoolConfiguration,
                 connection_factory: Optional[Callable] = None):
        """
        初始化连接池
        
        Args:
            config: 连接池配置
            connection_factory: 连接工厂函数
        """
        self.config = config
        self.connection_factory = connection_factory
        
        # 连接存储
        self.connections: Dict[str, Connection] = {}
        self.connection_metrics: Dict[str, ConnectionMetrics] = {}
        
        # 可用连接队列（不同策略有不同的管理方式）
        self.available_connections: List[str] = []
        self.busy_connections: Set[str] = set()
        
        # 连接映射（按源-目标分组）
        self.connection_groups: Dict[str, List[str]] = {}  # "source->target" -> [connection_ids]
        
        # 健康检查
        self.health_checker = ConnectionHealthChecker()
        
        # 异步任务
        self._cleanup_task: Optional[asyncio.Task] = None
        self._health_check_task: Optional[asyncio.Task] = None
        self._scaling_task: Optional[asyncio.Task] = None
        
        # 统计信息
        self.pool_stats = {
            "total_created": 0,
            "total_destroyed": 0,
            "total_requests": 0,
            "total_hits": 0,
            "total_misses": 0
        }
        
        # 锁
        self._lock = asyncio.Lock()
        self._running = False
    
    async def start(self):
        """启动连接池"""
        self._running = True
        
        # 预创建最小连接数
        if self.config.min_connections > 0:
            await self._ensure_min_connections()
        
        # 启动后台任务
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        
        if self.config.enable_auto_scaling:
            self._scaling_task = asyncio.create_task(self._auto_scaling_loop())
    
    async def stop(self):
        """停止连接池"""
        self._running = False
        
        # 停止后台任务
        for task in [self._cleanup_task, self._health_check_task, self._scaling_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # 关闭所有连接
        await self.close_all_connections()
    
    async def get_connection(self, 
                           source: str, 
                           target: str, 
                           connection_type: ConnectionType = ConnectionType.BUSINESS_RPC) -> Connection:
        """获取连接"""
        self.pool_stats["total_requests"] += 1
        
        async with self._lock:
            # 生成连接组键
            group_key = self._get_group_key(source, target, connection_type)
            
            # 尝试从现有连接中找到可用的
            connection = await self._find_available_connection(group_key)
            
            if connection:
                self.pool_stats["total_hits"] += 1
                await self._mark_connection_busy(connection.connection_id)
                return connection
            
            # 没有可用连接，检查是否可以创建新连接
            if len(self.connections) >= self.config.max_connections:
                # 尝试清理一些连接
                await self._cleanup_idle_connections()
                
                # 如果还是达到上限，尝试从最少使用的连接中选择一个
                if len(self.connections) >= self.config.max_connections:
                    connection = await self._evict_connection_for_reuse(group_key)
                    if connection:
                        self.pool_stats["total_hits"] += 1
                        await self._mark_connection_busy(connection.connection_id)
                        return connection
                    else:
                        raise ConnectionError("Connection pool is full and no connections can be evicted")
            
            # 创建新连接
            self.pool_stats["total_misses"] += 1
            connection = await self._create_connection(source, target, connection_type, group_key)
            await self._mark_connection_busy(connection.connection_id)
            
            return connection
    
    async def release_connection(self, connection_id: str):
        """释放连接"""
        async with self._lock:
            if connection_id in self.connections:
                await self._mark_connection_available(connection_id)
    
    async def close_connection(self, connection_id: str) -> bool:
        """关闭连接"""
        async with self._lock:
            if connection_id not in self.connections:
                return False
            
            connection = self.connections[connection_id]
            
            # 更新连接状态
            connection.status = ConnectionStatus.DISCONNECTED
            
            # 从各种集合中移除
            self._remove_connection_from_collections(connection_id)
            
            # 删除连接和指标
            del self.connections[connection_id]
            if connection_id in self.connection_metrics:
                del self.connection_metrics[connection_id]
            
            self.pool_stats["total_destroyed"] += 1
            
            return True
    
    async def close_all_connections(self):
        """关闭所有连接"""
        async with self._lock:
            connection_ids = list(self.connections.keys())
            
            for connection_id in connection_ids:
                await self.close_connection(connection_id)
    
    def get_pool_statistics(self) -> Dict[str, Any]:
        """获取连接池统计信息"""
        return {
            "total_connections": len(self.connections),
            "available_connections": len(self.available_connections),
            "busy_connections": len(self.busy_connections),
            "connection_groups": len(self.connection_groups),
            "max_connections": self.config.max_connections,
            "min_connections": self.config.min_connections,
            "pool_utilization": len(self.connections) / self.config.max_connections,
            "hit_rate": (self.pool_stats["total_hits"] / 
                        max(self.pool_stats["total_requests"], 1)),
            "statistics": self.pool_stats.copy(),
            "health_distribution": self._get_health_distribution()
        }
    
    def get_connection_info(self, connection_id: str) -> Optional[Dict[str, Any]]:
        """获取连接信息"""
        if connection_id not in self.connections:
            return None
        
        connection = self.connections[connection_id]
        metrics = self.connection_metrics.get(connection_id)
        
        return {
            "connection": {
                "connection_id": connection.connection_id,
                "source_id": connection.source_id,
                "target_id": connection.target_id,
                "connection_type": connection.connection_type.value,
                "status": connection.status.value,
                "created_time": connection.created_time.isoformat(),
                "last_activity": connection.last_activity.isoformat()
            },
            "metrics": {
                "usage_count": metrics.usage_count if metrics else 0,
                "error_count": metrics.error_count if metrics else 0,
                "average_response_time": metrics.average_response_time if metrics else 0,
                "health_status": metrics.health_status.value if metrics else "unknown"
            } if metrics else None,
            "is_busy": connection_id in self.busy_connections
        }
    
    async def _find_available_connection(self, group_key: str) -> Optional[Connection]:
        """查找可用连接"""
        if group_key not in self.connection_groups:
            return None
        
        group_connection_ids = self.connection_groups[group_key]
        available_in_group = [cid for cid in group_connection_ids 
                             if cid in self.available_connections]
        
        if not available_in_group:
            return None
        
        # 根据策略选择连接
        selected_id = self._select_connection_by_strategy(available_in_group)
        return self.connections.get(selected_id)
    
    def _select_connection_by_strategy(self, connection_ids: List[str]) -> str:
        """根据策略选择连接"""
        if not connection_ids:
            return None
        
        if self.config.strategy == PoolStrategy.FIFO:
            # 选择最早创建的
            return min(connection_ids, 
                      key=lambda cid: self.connections[cid].created_time)
        
        elif self.config.strategy == PoolStrategy.LIFO:
            # 选择最晚创建的
            return max(connection_ids,
                      key=lambda cid: self.connections[cid].created_time)
        
        elif self.config.strategy == PoolStrategy.LEAST_RECENTLY_USED:
            # 选择最少最近使用的
            return min(connection_ids,
                      key=lambda cid: self.connection_metrics[cid].last_used 
                                     if cid in self.connection_metrics
                                     else datetime.min)
        
        elif self.config.strategy == PoolStrategy.ROUND_ROBIN:
            # 轮询选择（简化实现）
            return connection_ids[0]
        
        elif self.config.strategy == PoolStrategy.LOAD_BALANCED:
            # 负载均衡：选择使用次数最少的
            return min(connection_ids,
                      key=lambda cid: self.connection_metrics[cid].usage_count
                                     if cid in self.connection_metrics else 0)
        
        else:
            return connection_ids[0]
    
    async def _create_connection(self, 
                               source: str, 
                               target: str, 
                               connection_type: ConnectionType,
                               group_key: str) -> Connection:
        """创建新连接"""
        if self.connection_factory:
            connection = await self.connection_factory(source, target, connection_type)
        else:
            # 默认连接创建
            connection = Connection(
                source_id=source,
                target_id=target,
                connection_type=connection_type,
                status=ConnectionStatus.CONNECTED,
                created_time=datetime.now(),
                last_activity=datetime.now()
            )
        
        # 添加到连接池
        self.connections[connection.connection_id] = connection
        
        # 创建指标
        metrics = ConnectionMetrics(
            connection_id=connection.connection_id,
            created_time=connection.created_time,
            last_used=datetime.now()
        )
        self.connection_metrics[connection.connection_id] = metrics
        
        # 添加到组
        if group_key not in self.connection_groups:
            self.connection_groups[group_key] = []
        self.connection_groups[group_key].append(connection.connection_id)
        
        self.pool_stats["total_created"] += 1
        
        return connection
    
    async def _mark_connection_busy(self, connection_id: str):
        """标记连接为忙碌"""
        if connection_id in self.available_connections:
            self.available_connections.remove(connection_id)
        
        self.busy_connections.add(connection_id)
        
        # 更新指标
        if connection_id in self.connection_metrics:
            metrics = self.connection_metrics[connection_id]
            metrics.usage_count += 1
            metrics.last_used = datetime.now()
        
        # 更新连接活动时间
        if connection_id in self.connections:
            self.connections[connection_id].last_activity = datetime.now()
            self.connections[connection_id].status = ConnectionStatus.ACTIVE
    
    async def _mark_connection_available(self, connection_id: str):
        """标记连接为可用"""
        if connection_id in self.busy_connections:
            self.busy_connections.remove(connection_id)
        
        if connection_id not in self.available_connections:
            # 根据策略插入到适当位置
            if self.config.strategy == PoolStrategy.LIFO:
                self.available_connections.append(connection_id)
            else:
                self.available_connections.insert(0, connection_id)
        
        # 更新连接状态
        if connection_id in self.connections:
            self.connections[connection_id].status = ConnectionStatus.IDLE
            self.connections[connection_id].last_activity = datetime.now()
    
    def _remove_connection_from_collections(self, connection_id: str):
        """从各种集合中移除连接"""
        if connection_id in self.available_connections:
            self.available_connections.remove(connection_id)
        
        if connection_id in self.busy_connections:
            self.busy_connections.remove(connection_id)
        
        # 从组中移除
        for group_connections in self.connection_groups.values():
            if connection_id in group_connections:
                group_connections.remove(connection_id)
        
        # 清理空组
        empty_groups = [k for k, v in self.connection_groups.items() if not v]
        for group_key in empty_groups:
            del self.connection_groups[group_key]
    
    async def _cleanup_idle_connections(self):
        """清理空闲连接"""
        now = datetime.now()
        idle_threshold = timedelta(seconds=self.config.max_idle_time)
        
        connections_to_close = []
        
        for connection_id in list(self.available_connections):
            if connection_id in self.connections:
                connection = self.connections[connection_id]
                idle_time = now - connection.last_activity
                
                if idle_time > idle_threshold:
                    connections_to_close.append(connection_id)
        
        for connection_id in connections_to_close:
            await self.close_connection(connection_id)
        
        return len(connections_to_close)
    
    async def _evict_connection_for_reuse(self, group_key: str) -> Optional[Connection]:
        """为重用而驱逐连接"""
        # 首先尝试从其他组中找到最少使用的连接
        all_available = [cid for cid in self.available_connections 
                        if cid not in self.connection_groups.get(group_key, [])]
        
        if all_available:
            # 选择最少使用的连接进行驱逐
            selected_id = min(all_available,
                            key=lambda cid: self.connection_metrics[cid].usage_count
                                           if cid in self.connection_metrics else 0)
            
            return self.connections.get(selected_id)
        
        return None
    
    def _get_group_key(self, source: str, target: str, connection_type: ConnectionType) -> str:
        """生成连接组键"""
        return f"{source}->{target}:{connection_type.value}"
    
    async def _ensure_min_connections(self):
        """确保最小连接数"""
        current_count = len(self.connections)
        
        if current_count < self.config.min_connections:
            needed = self.config.min_connections - current_count
            
            # 预创建连接（这里简化为创建通用连接）
            for i in range(needed):
                try:
                    connection = await self._create_connection(
                        f"pool_{i}", f"default_{i}", 
                        ConnectionType.BUSINESS_RPC,
                        f"pool_{i}->default_{i}:business_rpc"
                    )
                    await self._mark_connection_available(connection.connection_id)
                except Exception as e:
                    print(f"Failed to create preemptive connection: {e}")
                    break
    
    def _get_health_distribution(self) -> Dict[str, int]:
        """获取健康状态分布"""
        distribution = {status.value: 0 for status in ConnectionHealthStatus}
        
        for metrics in self.connection_metrics.values():
            status = metrics.health_status.value
            distribution[status] += 1
        
        return distribution
    
    # ==================== 后台任务 ====================
    
    async def _cleanup_loop(self):
        """清理循环"""
        while self._running:
            try:
                cleaned = await self._cleanup_idle_connections()
                if cleaned > 0:
                    print(f"Cleaned up {cleaned} idle connections")
                
                await asyncio.sleep(60)  # 每分钟清理一次
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Connection pool cleanup error: {e}")
                await asyncio.sleep(60)
    
    async def _health_check_loop(self):
        """健康检查循环"""
        while self._running:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.config.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Connection pool health check error: {e}")
                await asyncio.sleep(self.config.health_check_interval)
    
    async def _perform_health_checks(self):
        """执行健康检查"""
        unhealthy_connections = []
        
        for connection_id, connection in list(self.connections.items()):
            try:
                health_status = await self.health_checker.check_connection_health(connection)
                
                if connection_id in self.connection_metrics:
                    self.connection_metrics[connection_id].health_status = health_status
                
                if health_status == ConnectionHealthStatus.UNHEALTHY:
                    unhealthy_connections.append(connection_id)
                    
            except Exception as e:
                print(f"Health check failed for connection {connection_id}: {e}")
                unhealthy_connections.append(connection_id)
        
        # 关闭不健康的连接
        for connection_id in unhealthy_connections:
            await self.close_connection(connection_id)
        
        if unhealthy_connections:
            print(f"Removed {len(unhealthy_connections)} unhealthy connections")
    
    async def _auto_scaling_loop(self):
        """自动伸缩循环"""
        while self._running:
            try:
                await self._perform_auto_scaling()
                await asyncio.sleep(30)  # 每30秒检查一次
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Auto scaling error: {e}")
                await asyncio.sleep(30)
    
    async def _perform_auto_scaling(self):
        """执行自动伸缩"""
        current_connections = len(self.connections)
        busy_connections = len(self.busy_connections)
        
        # 计算利用率
        utilization = busy_connections / max(current_connections, 1)
        
        # 扩展条件：利用率高于阈值且未达到最大连接数
        if (utilization > self.config.scaling_threshold and 
            current_connections < self.config.max_connections):
            
            # 预创建一些连接
            new_connections = min(5, self.config.max_connections - current_connections)
            await self._create_preemptive_connections(new_connections)
        
        # 收缩条件：利用率很低且超过最小连接数
        elif (utilization < 0.2 and 
              current_connections > self.config.min_connections):
            
            # 清理一些空闲连接
            await self._cleanup_idle_connections()
    
    async def _create_preemptive_connections(self, count: int):
        """预创建连接"""
        for i in range(count):
            try:
                connection = await self._create_connection(
                    f"preemptive_{i}", f"pool_{i}",
                    ConnectionType.BUSINESS_RPC,
                    f"preemptive_{i}->pool_{i}:business_rpc"
                )
                await self._mark_connection_available(connection.connection_id)
            except Exception as e:
                print(f"Failed to create preemptive connection: {e}")
                break