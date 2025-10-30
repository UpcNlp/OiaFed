"""
连接管理器 - 处理客户端连接的建立、维护和清理
fedcl/connection/manager.py
"""

import asyncio
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timedelta

from ..communication.base import CommunicationManagerBase
from ..communication.layer_event import LayerEventHandler
from ..types import Connection, ConnectionStatus, ConnectionType, CommunicationConfig
from ..exceptions import ConnectionError
from ..utils.auto_logger import get_comm_logger
from ..exceptions import ConnectionError


class ConnectionPool:
    """连接池管理"""
    
    def __init__(self, max_connections: int = 100, connection_timeout: float = 300.0):
        self.max_connections = max_connections
        self.connection_timeout = connection_timeout
        self.connections: Dict[str, Connection] = {}
        self.connection_usage: Dict[str, datetime] = {}
        self._lock = asyncio.Lock()
        
        # 清理任务
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def get_connection(self, source: str, target: str, connection_type: ConnectionType = ConnectionType.BUSINESS_RPC) -> Connection:
        """获取连接"""
        connection_key = self._get_connection_key(source, target, connection_type)
        
        async with self._lock:
            # 检查是否存在有效连接
            if connection_key in self.connections:
                connection = self.connections[connection_key]
                if connection.status in [ConnectionStatus.CONNECTED, ConnectionStatus.ACTIVE]:
                    # 更新使用时间
                    self.connection_usage[connection_key] = datetime.now()
                    connection.last_activity = datetime.now()
                    return connection
            
            # 检查连接数限制
            if len(self.connections) >= self.max_connections:
                await self._cleanup_idle_connections()
                
                if len(self.connections) >= self.max_connections:
                    raise ConnectionError("Connection pool limit reached")
            
            # 创建新连接
            connection = Connection(
                source_id=source,
                target_id=target,
                connection_type=connection_type,
                status=ConnectionStatus.CONNECTING,
                created_time=datetime.now(),
                last_activity=datetime.now()
            )
            
            self.connections[connection_key] = connection
            self.connection_usage[connection_key] = datetime.now()
            
            return connection
    
    async def update_connection_status(self, connection_id: str, status: ConnectionStatus):
        """更新连接状态"""
        async with self._lock:
            for connection in self.connections.values():
                if connection.connection_id == connection_id:
                    connection.status = status
                    connection.last_activity = datetime.now()
                    break
    
    async def release_connection(self, connection_id: str):
        """释放连接"""
        async with self._lock:
            keys_to_remove = []
            for key, connection in self.connections.items():
                if connection.connection_id == connection_id:
                    connection.status = ConnectionStatus.DISCONNECTED
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.connections[key]
                if key in self.connection_usage:
                    del self.connection_usage[key]
    
    async def _cleanup_idle_connections(self):
        """清理空闲连接"""
        now = datetime.now()
        timeout_threshold = timedelta(seconds=self.connection_timeout)
        
        keys_to_remove = []
        for key, last_used in self.connection_usage.items():
            if now - last_used > timeout_threshold:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            if key in self.connections:
                self.connections[key].status = ConnectionStatus.DISCONNECTED
                del self.connections[key]
            if key in self.connection_usage:
                del self.connection_usage[key]
    
    def _get_connection_key(self, source: str, target: str, connection_type: ConnectionType) -> str:
        """生成连接键"""
        return f"{source}->{target}:{connection_type.value}"
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """获取连接池统计"""
        status_counts = {}
        for connection in self.connections.values():
            status = connection.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "total_connections": len(self.connections),
            "max_connections": self.max_connections,
            "status_distribution": status_counts,
            "usage_count": len(self.connection_usage)
        }
    
    async def start_cleanup_task(self):
        """启动清理任务"""
        if not self._cleanup_task:
            self._running = True
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def stop_cleanup_task(self):
        """停止清理任务"""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
    
    async def _cleanup_loop(self):
        """清理循环"""
        while self._running:
            try:
                await self._cleanup_idle_connections()
                await asyncio.sleep(60)  # 每分钟清理一次
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Connection cleanup error: {e}")
                await asyncio.sleep(60)


class MessageRouter:
    """消息路由器"""
    
    def __init__(self):
        self.routing_rules: List[Dict[str, Any]] = []
        self.route_statistics: Dict[str, int] = {}
        self._lock = asyncio.Lock()
    
    def add_routing_rule(self, 
                        rule_id: str,
                        source_pattern: str = "*",
                        target_pattern: str = "*", 
                        message_type_pattern: str = "*",
                        priority: int = 0,
                        action: str = "FORWARD") -> str:
        """添加路由规则"""
        rule = {
            "rule_id": rule_id,
            "source_pattern": source_pattern,
            "target_pattern": target_pattern,
            "message_type_pattern": message_type_pattern,
            "priority": priority,
            "action": action,
            "enabled": True,
            "created_time": datetime.now(),
            "usage_count": 0
        }
        
        # 按优先级插入
        inserted = False
        for i, existing_rule in enumerate(self.routing_rules):
            if priority > existing_rule["priority"]:
                self.routing_rules.insert(i, rule)
                inserted = True
                break
        
        if not inserted:
            self.routing_rules.append(rule)
        
        return rule_id
    
    def remove_routing_rule(self, rule_id: str) -> bool:
        """删除路由规则"""
        for i, rule in enumerate(self.routing_rules):
            if rule["rule_id"] == rule_id:
                del self.routing_rules[i]
                return True
        return False
    
    async def route_message(self, source: str, target: str, message_type: str, data: Any) -> Dict[str, Any]:
        """路由消息"""
        routing_result = {
            "original_target": target,
            "routed_targets": [],
            "actions_taken": [],
            "rules_applied": []
        }
        
        async with self._lock:
            for rule in self.routing_rules:
                if not rule["enabled"]:
                    continue
                
                # 检查规则匹配
                if (self._match_pattern(source, rule["source_pattern"]) and
                    self._match_pattern(target, rule["target_pattern"]) and
                    self._match_pattern(message_type, rule["message_type_pattern"])):
                    
                    # 应用规则
                    action = rule["action"]
                    
                    if action == "FORWARD":
                        routing_result["routed_targets"].append(target)
                        routing_result["actions_taken"].append("FORWARD")
                        
                    elif action == "DUPLICATE":
                        routing_result["routed_targets"].extend([target, f"{target}_copy"])
                        routing_result["actions_taken"].append("DUPLICATE")
                        
                    elif action == "FILTER":
                        routing_result["actions_taken"].append("FILTER")
                        # 消息被过滤，不转发
                        break
                        
                    elif action == "TRANSFORM":
                        # 可以在这里添加数据转换逻辑
                        routing_result["routed_targets"].append(target)
                        routing_result["actions_taken"].append("TRANSFORM")
                    
                    # 更新规则使用统计
                    rule["usage_count"] += 1
                    routing_result["rules_applied"].append(rule["rule_id"])
                    
                    # 更新全局统计
                    self.route_statistics[rule["rule_id"]] = rule["usage_count"]
        
        # 如果没有规则匹配，使用默认转发
        if not routing_result["actions_taken"]:
            routing_result["routed_targets"] = [target]
            routing_result["actions_taken"] = ["DEFAULT_FORWARD"]
        
        return routing_result
    
    def _match_pattern(self, value: str, pattern: str) -> bool:
        """模式匹配"""
        if pattern == "*":
            return True
        
        if pattern.startswith("*") and pattern.endswith("*"):
            # 包含匹配
            return pattern[1:-1] in value
        elif pattern.startswith("*"):
            # 后缀匹配
            return value.endswith(pattern[1:])
        elif pattern.endswith("*"):
            # 前缀匹配
            return value.startswith(pattern[:-1])
        else:
            # 精确匹配
            return value == pattern
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """获取路由统计"""
        return {
            "total_rules": len(self.routing_rules),
            "enabled_rules": sum(1 for r in self.routing_rules if r["enabled"]),
            "rule_statistics": self.route_statistics,
            "rules_by_priority": sorted(self.routing_rules, key=lambda x: x["priority"], reverse=True)
        }


class ConnectionManager(LayerEventHandler):
    """连接管理器 - 第3层：连接管理层"""
    
    def __init__(self, 
                 communication_manager: CommunicationManagerBase,
                 config: CommunicationConfig,
                 upper_layer: Optional[LayerEventHandler] = None):
        super().__init__(upper_layer)
        self.communication_manager = communication_manager
        self.config = config
        self.logger = get_comm_logger("connection_manager")
        
        # 初始化组件
        self.pool = ConnectionPool(
            max_connections=config.max_clients * 2,  # 每客户端可能有多个连接
            connection_timeout=config.heartbeat_timeout * 2
        )
        self.router = MessageRouter()
        
        # 设置默认路由规则
        self._setup_default_routing_rules()
        
        # 连接事件处理器
        self.connection_event_handlers: Dict[str, List[Callable]] = {}
        
        self._running = False
    
    def _setup_default_routing_rules(self):
        """设置默认路由规则"""
        # 业务消息直接转发
        self.router.add_routing_rule(
            "business_forward",
            source_pattern="*",
            target_pattern="*",
            message_type_pattern="business_*",
            priority=10,
            action="FORWARD"
        )
        
        # 控制消息直接转发
        self.router.add_routing_rule(
            "control_forward",
            source_pattern="*",
            target_pattern="*", 
            message_type_pattern="control_*",
            priority=10,
            action="FORWARD"
        )
    
    async def route_message(self, source: str, target: str, message_type: str, data: Any) -> Any:
        """路由消息"""
        try:
            # 获取或创建连接
            connection = await self.pool.get_connection(source, target)
            
            # 更新连接状态为活跃
            await self.pool.update_connection_status(connection.connection_id, ConnectionStatus.ACTIVE)
            
            # 路由消息
            routing_result = await self.router.route_message(source, target, message_type, data)
            
            # 检查是否被过滤
            if "FILTER" in routing_result["actions_taken"]:
                return {"status": "filtered", "routing_result": routing_result}
            
            # 发送到路由目标
            results = {}
            for routed_target in routing_result["routed_targets"]:
                try:
                    result = await self.communication_manager.send_business_message(
                        routed_target, message_type, data
                    )
                    results[routed_target] = {"success": True, "result": result}
                except Exception as e:
                    results[routed_target] = {"success": False, "error": str(e)}
            
            # 更新连接状态为空闲
            await self.pool.update_connection_status(connection.connection_id, ConnectionStatus.IDLE)
            
            return {
                "status": "routed",
                "routing_result": routing_result,
                "delivery_results": results
            }
            
        except Exception as e:
            raise ConnectionError(f"Message routing failed: {str(e)}")
    
    async def handle_connection_event(self, event_type: str, connection_data: Dict[str, Any]):
        """处理连接事件"""
        if event_type in self.connection_event_handlers:
            for handler in self.connection_event_handlers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(connection_data)
                    else:
                        handler(connection_data)
                except Exception as e:
                    self.logger.error(f"Connection event handler error: {e}")
    
    def register_connection_event_handler(self, event_type: str, handler: Callable) -> str:
        """注册连接事件处理器"""
        if event_type not in self.connection_event_handlers:
            self.connection_event_handlers[event_type] = []
        
        handler_id = f"{event_type}_{len(self.connection_event_handlers[event_type])}"
        self.connection_event_handlers[event_type].append(handler)
        return handler_id
    
    async def check_connection_health(self):
        """检查连接健康状态"""
        pool_stats = self.pool.get_pool_stats()
        
        # 检查是否有太多断开的连接
        disconnected_count = pool_stats["status_distribution"].get("disconnected", 0)
        total_count = pool_stats["total_connections"]
        
        if total_count > 0:
            disconnected_ratio = disconnected_count / total_count
            if disconnected_ratio > 0.5:  # 50%以上连接断开
                await self.handle_connection_event("HIGH_DISCONNECTION_RATE", {
                    "disconnected_count": disconnected_count,
                    "total_count": total_count,
                    "ratio": disconnected_ratio
                })
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """获取管理器统计"""
        return {
            "connection_pool": self.pool.get_pool_stats(),
            "message_router": self.router.get_routing_stats(),
            "event_handlers": {
                event_type: len(handlers) 
                for event_type, handlers in self.connection_event_handlers.items()
            },
            "running": self._running
        }
    
    async def start(self):
        """启动连接管理器"""
        self._running = True
        
        # 启动连接池清理任务
        await self.pool.start_cleanup_task()
        
        # 注册连接事件处理
        self.register_connection_event_handler(
            "CONNECTION_LOST",
            self._handle_connection_lost
        )
        
        self.register_connection_event_handler(
            "CONNECTION_ESTABLISHED", 
            self._handle_connection_established
        )
    
    async def stop(self):
        """停止连接管理器"""
        self._running = False
        
        # 停止连接池清理任务
        await self.pool.stop_cleanup_task()
    
    async def cleanup(self):
        """清理连接管理器资源"""
        # 清理所有连接
        for connection_id in [conn.connection_id for conn in self.pool.connections.values()]:
            await self.pool.release_connection(connection_id)
        
        # 清理事件处理器
        self.connection_event_handlers.clear()
    
    async def _handle_connection_lost(self, connection_data: Dict[str, Any]):
        """处理连接丢失"""
        print(f"Connection lost: {connection_data}")
        
        # 可以在这里添加重连逻辑
        # await self.attempt_reconnection(connection_data)
    
    async def _handle_connection_established(self, connection_data: Dict[str, Any]):
        """处理连接建立"""
        print(f"Connection established: {connection_data}")
    
    # ==================== LayerEventHandler 实现 ====================
    
    def handle_layer_event(self, event_type: str, event_data: Dict[str, Any]):
        """处理来自下层的事件"""
        self.logger.debug(f"[第3层-连接管理层] 收到事件: {event_type}, 数据: {event_data}")
        
        if event_type == "CLIENT_REGISTERED":
            # 客户端注册成功，创建连接建立事件
            self.logger.debug(f"[第3层-连接管理层] 处理客户端注册事件")
            self._handle_client_registration(event_data)
        elif event_type == "CLIENT_DISCONNECTED":
            # 客户端断开连接
            self.logger.indebugfo(f"[第3层-连接管理层] 处理客户端断开事件")
            self._handle_client_disconnection(event_data)
        else:
            self.logger.warning(f"[第3层-连接管理层] 未知事件类型: {event_type}")
    
    def _handle_client_registration(self, event_data: Dict[str, Any]):
        """处理客户端注册事件"""
        client_id = event_data.get("client_id")
        if client_id:
            self.logger.debug(f"[第3层-连接管理层] 客户端[{client_id}]注册成功，建立连接...")
            
            # 向上传递连接建立事件到业务通信层
            connection_event = {
                "client_id": client_id,
                "connection_info": event_data.get("connection_info", {}),
                "timestamp": event_data.get("timestamp")
            }
            self.logger.debug(f"[第3层-连接管理层] 向上传递CONNECTION_ESTABLISHED事件: {client_id}")
            self.propagate_to_upper("CONNECTION_ESTABLISHED", connection_event)
    
    def _handle_client_disconnection(self, event_data: Dict[str, Any]):
        """处理客户端断开事件"""
        client_id = event_data.get("client_id")
        if client_id:
            self.logger.debug(f"[第3层-连接管理层] 客户端[{client_id}]断开连接...")
            
            # 向上传递连接断开事件到业务通信层
            disconnection_event = {
                "client_id": client_id,
                "reason": event_data.get("reason", "unknown"),
                "timestamp": event_data.get("timestamp")
            }
            self.logger.debug(f"[第3层-连接管理层] 向上传递CONNECTION_LOST事件: {client_id}")
            self.propagate_to_upper("CONNECTION_LOST", disconnection_event)