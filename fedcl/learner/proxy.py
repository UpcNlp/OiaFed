"""
MOE-FedCL 服务端代理实现  
moe_fedcl/learner/proxy.py
"""

import asyncio
import functools
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, Callable, List

from ..communication.base import CommunicationManagerBase
from ..connection.manager import ConnectionManager
from ..exceptions import CommunicationError, ClientNotFoundError
from ..types import (
    TrainingRequest, ModelData, TrainingResult,
    EvaluationResult, MetricsData, ConnectionStatus, TrainingResponse
)
from ..utils.auto_logger import get_sys_logger


class ProxyConfig:
    """代理配置"""
    
    def __init__(self,
                 default_timeout: float = 120.0,
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 connection_check_interval: float = 30.0,
                 enable_caching: bool = True,
                 cache_ttl: float = 60.0,
                 enable_dynamic_calls: bool = True,
                 method_whitelist: Optional[List[str]] = None,
                 method_blacklist: Optional[List[str]] = None):
        self.default_timeout = default_timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.connection_check_interval = connection_check_interval
        self.enable_caching = enable_caching
        self.cache_ttl = cache_ttl
        self.enable_dynamic_calls = enable_dynamic_calls
        self.method_whitelist = method_whitelist or []
        self.method_blacklist = method_blacklist or [
            '__', 'eval', 'exec', 'compile', 'open', 'file', 'input', 
            'raw_input', 'reload', '__import__', 'delattr', 'setattr',
            'globals', 'locals', 'vars', 'dir', 'help', 'copyright', 
            'credits', 'license', 'quit', 'exit'
        ]


class ProxyStatistics:
    """代理统计信息"""
    
    def __init__(self):
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_response_time = 0.0
        self.last_request_time: Optional[datetime] = None
        self.last_response_time: Optional[datetime] = None
        self.request_history: List[Dict[str, Any]] = []
        
        # 方法统计
        self.method_stats: Dict[str, Dict[str, Any]] = {}
    
    def record_request(self, method: str, success: bool, response_time: float, error: str = None):
        """记录请求统计"""
        self.total_requests += 1
        self.last_request_time = datetime.now()
        
        if success:
            self.successful_requests += 1
            self.last_response_time = datetime.now()
        else:
            self.failed_requests += 1
        
        self.total_response_time += response_time
        
        # 方法级统计
        if method not in self.method_stats:
            self.method_stats[method] = {
                "count": 0,
                "success_count": 0,
                "fail_count": 0,
                "total_time": 0.0,
                "avg_time": 0.0
            }
        
        stats = self.method_stats[method]
        stats["count"] += 1
        stats["total_time"] += response_time
        stats["avg_time"] = stats["total_time"] / stats["count"]
        
        if success:
            stats["success_count"] += 1
        else:
            stats["fail_count"] += 1
        
        # 历史记录（限制长度）
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "method": method,
            "success": success,
            "response_time": response_time,
            "error": error
        }
        
        self.request_history.append(history_entry)
        if len(self.request_history) > 100:  # 保留最近100条
            self.request_history = self.request_history[-100:]
    
    def get_average_response_time(self) -> float:
        """获取平均响应时间"""
        if self.total_requests == 0:
            return 0.0
        return self.total_response_time / self.total_requests
    
    def get_success_rate(self) -> float:
        """获取成功率"""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests


class LearnerProxy:
    """服务端代理统一类 - 封装对客户端的远程调用"""
    
    def __init__(self,
                 client_id: str,
                 communication_manager: CommunicationManagerBase,
                 connection_manager: ConnectionManager,
                 config: Optional[ProxyConfig] = None):
        """
        初始化服务端代理
        
        Args:
            client_id: 客户端唯一标识
            communication_manager: 通信管理器
            connection_manager: 连接管理器
            config: 代理配置
        """
        self.client_id = client_id
        self.communication_manager = communication_manager
        self.connection_manager = connection_manager
        self.config = config or ProxyConfig()
        
        # 统计信息
        self.statistics = ProxyStatistics()
        
        # 缓存
        self._cache: Dict[str, Any] = {} if self.config.enable_caching else None
        self._cache_timestamps: Dict[str, datetime] = {} if self.config.enable_caching else None
        
        # 订阅管理
        self._subscriptions: Dict[str, Dict[str, Any]] = {}
        
        # 连接状态
        self._last_ping_time: Optional[datetime] = None
        self._connection_status = ConnectionStatus.DISCONNECTED
        self._connection_check_task: Optional[asyncio.Task] = None
        
        # 启动连接检查
        if self.config.connection_check_interval > 0:
            self._connection_check_task = asyncio.create_task(self._connection_check_loop())

        self.logger = get_sys_logger()
    
    # ==================== 核心业务方法 ====================
    
    async def train(self, training_params: Dict[str, Any]) -> TrainingResult:
        """远程调用客户端训练
        
        Args:
            training_params: 训练参数
            
        Returns:
            TrainingResult: 训练结果
            
        Raises:
            CommunicationError: 通信失败
            TimeoutError: 调用超时
        """
        return await self._call_remote_method("train", training_params)
    
    async def evaluate(self, evaluation_params: Dict[str, Any]) -> EvaluationResult:
        """远程调用客户端评估
        
        Args:
            evaluation_params: 评估参数
            
        Returns:
            EvaluationResult: 评估结果
        """
        return await self._call_remote_method("evaluate", evaluation_params)
    
    async def get_model(self) -> ModelData:
        """远程获取客户端模型
        
        Returns:
            ModelData: 客户端本地模型
        """
        cache_key = "model"
        
        # 检查缓存
        if self.config.enable_caching and self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        result = await self._call_remote_method("get_model", {})
        
        # 更新缓存
        if self.config.enable_caching:
            self._cache[cache_key] = result
            self._cache_timestamps[cache_key] = datetime.now()
        
        return result
    
    async def set_model(self, model_data: ModelData) -> bool:
        """远程设置客户端模型
        
        Args:
            model_data: 要设置的模型数据
            
        Returns:
            bool: 设置是否成功
        """
        result = await self._call_remote_method("set_model", {"model_data": model_data})
        
        # 清除模型缓存
        if self.config.enable_caching and "model" in self._cache:
            del self._cache["model"]
            del self._cache_timestamps["model"]
        
        return result.get("model_updated", False)
    
    async def get_metrics(self) -> MetricsData:
        """远程获取客户端指标
        
        Returns:
            MetricsData: 客户端指标数据
        """
        cache_key = "metrics"
        
        # 检查缓存
        if self.config.enable_caching and self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        result = await self._call_remote_method("get_info", {})
        
        # 更新缓存
        if self.config.enable_caching:
            self._cache[cache_key] = result
            self._cache_timestamps[cache_key] = datetime.now()
        
        return result
    
    # ==================== 动态方法调用支持 ====================
    
    def __getattr__(self, method_name: str):
        """动态方法调用 - 实现 proxy.any_method() 的魔术方法
        
        当访问不存在的属性时，返回一个可调用对象来执行远程方法调用
        
        Args:
            method_name: 要调用的远程方法名
            
        Returns:
            Callable: 可调用对象，支持 async 调用
            
        Examples:
            # 调用远程learner的自定义方法
            result = await proxy.custom_training_method(param1, param2, key=value)
            
            # 调用远程learner的任意方法  
            data = await proxy.get_custom_data()
            status = await proxy.check_custom_status()
        """
        # 检查是否启用动态调用
        if not self.config.enable_dynamic_calls:
            raise AttributeError(f"Dynamic method calls are disabled. Method '{method_name}' not found.")
        
        # 安全检查：方法名黑名单
        if self._is_method_blocked(method_name):
            raise AttributeError(f"Method '{method_name}' is blocked for security reasons.")
        
        # 安全检查：方法名白名单（如果设置了）
        if self.config.method_whitelist and method_name not in self.config.method_whitelist:
            raise AttributeError(f"Method '{method_name}' is not in the whitelist.")
        
        # 返回一个包装的异步调用函数
        return self._create_dynamic_method(method_name)
    
    def _is_method_blocked(self, method_name: str) -> bool:
        """检查方法是否被安全黑名单阻止"""
        for blocked_pattern in self.config.method_blacklist:
            if blocked_pattern in method_name.lower():
                return True
        return False
    
    def _create_dynamic_method(self, method_name: str):
        """创建动态方法的包装器"""
        @functools.wraps(self._call_dynamic_method)
        async def dynamic_method(*args, **kwargs):
            """动态方法调用包装器
            
            Args:
                *args: 位置参数
                **kwargs: 关键字参数
                
            Returns:
                Any: 远程方法的返回值
                
            Raises:
                CommunicationError: 通信失败
                TimeoutError: 调用超时
                AttributeError: 远程方法不存在
            """
            return await self._call_dynamic_method(method_name, args, kwargs)
        
        # 添加方法信息用于调试
        dynamic_method.__name__ = method_name
        dynamic_method.__doc__ = f"Dynamic remote call to {self.client_id}.{method_name}()"
        
        return dynamic_method
    
    async def _call_dynamic_method(self, method_name: str, args: tuple, kwargs: dict) -> Any:
        """执行动态方法调用
        
        Args:
            method_name: 方法名
            args: 位置参数
            kwargs: 关键字参数
            
        Returns:
            Any: 远程方法返回值
        """
        # 构造参数字典
        parameters = {
            'args': list(args),  # 转换为可序列化的list
            'kwargs': kwargs
        }
        
        # 使用特殊的方法名标识这是动态调用
        dynamic_method_name = f"__dynamic_call__{method_name}"
        
        try:
            result = await self._call_remote_method(dynamic_method_name, parameters)
            return result
        except CommunicationError as e:
            # 如果是方法不存在的错误，转换为AttributeError
            if "not found" in str(e).lower() or "does not exist" in str(e).lower():
                raise AttributeError(f"Remote method '{method_name}' does not exist on {self.client_id}")
            raise

    def _setup_request_handlers(self):
        """设置请求处理器"""
        self._request_handlers = {
            "get_model": self.handle_get_model_request,
            "set_model": self.handle_set_model_request,
            "ping": self.handle_ping_request
        }

    async def handle_ping_request(self, request: TrainingRequest) -> TrainingResponse:
        """处理ping请求"""
        return TrainingResponse(
            request_id=request.request_id,
            client_id=request.client_id,
            success=True,
            result={
                "status": "alive",
                "timestamp": datetime.now().isoformat(),
                "registration_status": self._registration_status.value
            }
        )
    
    # ==================== 标准远程方法调用 ====================
    
    async def call_remote_method(self, method_name: str, **kwargs) -> Any:
        """通用远程方法调用
        
        Args:
            method_name: 方法名
            **kwargs: 方法参数
            
        Returns:
            Any: 方法返回值
        """
        return await self._call_remote_method(method_name, kwargs)
    
    async def _call_remote_method(self, method_name: str, parameters: Dict[str, Any]) -> Any:
        """内部远程方法调用实现"""
        print(f"[PROXY] 开始调用远程方法: {method_name}, 客户端: {self.client_id}")
        print(f"[PROXY] 调用参数: {parameters}")
        
        start_time = datetime.now()
        
        for attempt in range(self.config.max_retries + 1):
            try:
                # 创建请求
                request = TrainingRequest(
                    request_id=str(uuid.uuid4()),
                    client_id=self.client_id,
                    method_name=method_name,
                    parameters=parameters,
                    timeout=self.config.default_timeout
                )
                
                print(f"[PROXY] 创建请求: {request.__dict__}")
                
                # 发送请求并等待响应
                response_data = await self.connection_manager.route_message(
                    source="server",  # 服务端标识
                    target=self.client_id,
                    message_type="business_request",  # 修改为匹配Stub注册的处理器
                    data=request.__dict__
                )
                
                print(f"[PROXY] 收到响应数据: {response_data}")
                
                # 检查路由结果
                if response_data.get("status") == "filtered":
                    raise CommunicationError("Request was filtered by routing rules")
                
                # 获取实际响应
                delivery_results = response_data.get("delivery_results", {})
                if self.client_id in delivery_results:
                    client_result = delivery_results[self.client_id]
                    if client_result.get("success"):
                        response_result = client_result.get("result")
                        
                        print(f"[PROXY] 客户端响应结果: {response_result}")
                        
                        # 记录成功统计
                        response_time = (datetime.now() - start_time).total_seconds()
                        self.statistics.record_request(method_name, True, response_time)
                        
                        # 解析响应
                        if isinstance(response_result, dict):
                            if response_result.get("success", True):
                                return response_result.get("result")
                            else:
                                raise CommunicationError(
                                    f"Remote method failed: {response_result.get('error_message', 'Unknown error')}"
                                )
                        else:
                            return response_result
                    else:
                        error_msg = client_result.get("error", "Unknown delivery error")
                        raise CommunicationError(f"Message delivery failed: {error_msg}")
                else:
                    raise ClientNotFoundError(f"Client {self.client_id} not found in delivery results")
                
            except Exception as e:
                error_msg = str(e)
                
                # 记录失败统计
                response_time = (datetime.now() - start_time).total_seconds()
                self.statistics.record_request(method_name, False, response_time, error_msg)
                
                # 最后一次尝试，抛出异常
                if attempt == self.config.max_retries:
                    raise CommunicationError(f"Remote call failed after {self.config.max_retries + 1} attempts: {error_msg}")
                
                # 等待重试
                if self.config.retry_delay > 0:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))  # 指数退避
    
    # ==================== 推送订阅方法 ====================
    
    def subscribe_training_events(self, callback: Callable) -> str:
        """订阅训练事件
        
        Args:
            callback: 事件回调函数
            
        Returns:
            str: 订阅ID
        """
        return self._subscribe_event("training_events", callback)
    
    def subscribe_model_updates(self, callback: Callable) -> str:
        """订阅模型更新事件
        
        Args:
            callback: 事件回调函数
            
        Returns:
            str: 订阅ID
        """
        return self._subscribe_event("model_updates", callback)
    
    def _subscribe_event(self, event_type: str, callback: Callable) -> str:
        """内部事件订阅实现"""
        subscription_id = str(uuid.uuid4())
        
        if event_type not in self._subscriptions:
            self._subscriptions[event_type] = {}
        
        self._subscriptions[event_type][subscription_id] = {
            "callback": callback,
            "created_time": datetime.now(),
            "call_count": 0
        }
        
        # 注册到传输层事件处理器
        if hasattr(self.communication_manager.transport, 'register_event_handler'):
            self.communication_manager.transport.register_event_handler(
                event_type,
                lambda source, data: self._handle_subscription_event(event_type, source, data)
            )
        
        return subscription_id


    async def _handle_subscription_event(self, event_type: str, source: str, data: Any):
        """处理订阅事件"""
        if source != self.client_id:
            return  # 只处理来自该客户端的事件
        
        if event_type in self._subscriptions:
            for subscription_id, subscription in self._subscriptions[event_type].items():
                try:
                    callback = subscription["callback"]
                    subscription["call_count"] += 1
                    
                    if asyncio.iscoroutinefunction(callback):
                        await callback(data)
                    else:
                        callback(data)
                        
                except Exception as e:
                    print(f"Subscription callback error: {e}")
    
    def unsubscribe(self, subscription_id: str) -> bool:
        """取消订阅
        
        Args:
            subscription_id: 订阅ID
            
        Returns:
            bool: 是否成功取消
        """
        for event_type, subscriptions in self._subscriptions.items():
            if subscription_id in subscriptions:
                del subscriptions[subscription_id]
                return True
        return False
    
    def list_subscriptions(self) -> List[str]:
        """列出所有订阅
        
        Returns:
            List[str]: 订阅ID列表
        """
        subscription_ids = []
        for subscriptions in self._subscriptions.values():
            subscription_ids.extend(subscriptions.keys())
        return subscription_ids
    
    # ==================== 状态管理方法 ====================
    
    async def get_connection_status(self) -> ConnectionStatus:
        """获取连接状态
        
        Returns:
            ConnectionStatus: 当前连接状态
        """
        return self._connection_status
    
    async def ping(self) -> float:
        """ping客户端测试连接
        
        Returns:
            float: 响应时间(秒)
        """
        start_time = datetime.now()
        
        try:
            await self._call_remote_method("ping", {})
            response_time = (datetime.now() - start_time).total_seconds()
            
            self._last_ping_time = datetime.now()
            self._connection_status = ConnectionStatus.ACTIVE
            
            return response_time
            
        except Exception as e:
            self._connection_status = ConnectionStatus.ERROR
            raise CommunicationError(f"Ping failed: {str(e)}")
    
    def get_proxy_statistics(self) -> Dict[str, Any]:
        """获取代理统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        return {
            "client_id": self.client_id,
            "total_requests": self.statistics.total_requests,
            "successful_requests": self.statistics.successful_requests,
            "failed_requests": self.statistics.failed_requests,
            "success_rate": self.statistics.get_success_rate(),
            "average_response_time": self.statistics.get_average_response_time(),
            "last_request_time": self.statistics.last_request_time.isoformat() if self.statistics.last_request_time else None,
            "last_response_time": self.statistics.last_response_time.isoformat() if self.statistics.last_response_time else None,
            "connection_status": self._connection_status.value,
            "last_ping_time": self._last_ping_time.isoformat() if self._last_ping_time else None,
            "method_statistics": self.statistics.method_stats,
            "cache_size": len(self._cache) if self._cache else 0,
            "active_subscriptions": sum(len(subs) for subs in self._subscriptions.values())
        }
    
    def is_client_ready(self) -> bool:
        """检查客户端是否就绪
        
        Returns:
            bool: 客户端是否就绪
        """
        return self._connection_status in [ConnectionStatus.CONNECTED, ConnectionStatus.ACTIVE]
    
    # ==================== 缓存管理 ====================
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """检查缓存是否有效"""
        if not self.config.enable_caching or not self._cache:
            return False
        
        if cache_key not in self._cache or cache_key not in self._cache_timestamps:
            return False
        
        cache_age = datetime.now() - self._cache_timestamps[cache_key]
        return cache_age.total_seconds() < self.config.cache_ttl
    
    def clear_cache(self, cache_key: str = None):
        """清除缓存
        
        Args:
            cache_key: 要清除的缓存键，None表示清除所有
        """
        if not self.config.enable_caching or not self._cache:
            return
        
        if cache_key:
            if cache_key in self._cache:
                del self._cache[cache_key]
            if cache_key in self._cache_timestamps:
                del self._cache_timestamps[cache_key]
        else:
            self._cache.clear()
            self._cache_timestamps.clear()
    
    # ==================== 连接管理 ====================
    
    async def _connection_check_loop(self):
        """连接检查循环"""
        while True:
            try:
                await asyncio.sleep(self.config.connection_check_interval)
                
                # 执行ping检查
                await self.ping()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Connection check error for client {self.client_id}: {e}")
                self._connection_status = ConnectionStatus.ERROR
    
    # ==================== 生命周期方法 ====================
    
    async def connect(self) -> bool:
        """连接到客户端
        
        Returns:
            bool: 连接是否成功
        """
        try:
            # 尝试ping测试连接
            await self.ping()
            self._connection_status = ConnectionStatus.CONNECTED
            return True
            
        except Exception as e:
            print(f"Failed to connect to client {self.client_id}: {e}")
            self._connection_status = ConnectionStatus.ERROR
            return False
    
    async def disconnect(self) -> bool:
        """断开与客户端的连接
        
        Returns:
            bool: 断开是否成功
        """
        try:
            self._connection_status = ConnectionStatus.DISCONNECTED
            
            # 清除缓存
            if self.config.enable_caching:
                self.clear_cache()
            
            # 清除订阅
            self._subscriptions.clear()
            
            return True
            
        except Exception as e:
            print(f"Failed to disconnect from client {self.client_id}: {e}")
            return False
    
    async def cleanup(self) -> None:
        """清理代理资源"""
        # 停止连接检查任务
        if self._connection_check_task:
            self._connection_check_task.cancel()
            try:
                await self._connection_check_task
            except asyncio.CancelledError:
                pass
        
        # 断开连接
        await self.disconnect()
        
        # 清理统计信息
        self.statistics = ProxyStatistics()
        
        print(f"LearnerProxy for client {self.client_id} cleaned up")