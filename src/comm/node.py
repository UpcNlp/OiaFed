"""
Node 通信节点核心实现
"""

import asyncio
import time
import traceback
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set

from .config import NodeConfig, MethodOptions
from .message import (
    Message,
    MessageType,
    MessageContext,
    ConnectionInfo,
    ConnectionStatus,
    ErrorInfo,
)
from .exceptions import (
    NodeError,
    NodeNotConnectedError,
    NodeDisconnectedError,
    CallTimeoutError,
    RemoteExecutionError,
    SerializationError,
    InterceptorAbort,
)
from .transport import Transport, create_transport
from .serialization import SerializerRegistry, Serializer
from .interceptor import InterceptorChain, LoggingInterceptor, AuthInterceptor, Interceptor

# 使用 loguru 统一日志（会继承调用方的 node_id 和 log_type）
from ..infra.logging import get_logger
# logger 将在 __init__ 中绑定 node_id

# 类型定义
MethodHandler = Callable[[Any, MessageContext], Awaitable[Any]]
EventHandler = Callable[..., Any]


@dataclass
class MethodEntry:
    """方法注册条目"""
    handler: MethodHandler
    options: MethodOptions = field(default_factory=MethodOptions)


@dataclass
class PendingRequest:
    """等待响应的请求"""
    future: asyncio.Future
    method: str
    created_at: float
    timeout_at: float
    target: str


class Node:
    """
    通信节点
    
    核心职责：
    1. 方法注册与路由
    2. 消息发送与接收
    3. 连接管理
    4. 请求-响应匹配
    
    Example:
        async def handle_train(payload, ctx):
            return {"status": "completed"}
        
        node = Node("my_node", config)
        node.register("train", handle_train)
        await node.start()
    """
    
    def __init__(
        self,
        node_id: str,
        config: Optional[NodeConfig] = None,
    ):
        # 节点标识
        self.node_id: str = node_id
        self.config: NodeConfig = config or NodeConfig(node_id=node_id)

        # 创建绑定了 node_id 和 log_type 的 logger
        self.logger = get_logger(node_id=node_id, log_type="system")

        # 确保 config.node_id 与参数一致
        if self.config.node_id and self.config.node_id != node_id:
            self.self.logger.warning(
                f"Config node_id '{self.config.node_id}' differs from parameter '{node_id}', "
                f"using parameter value"
            )
        self.config.node_id = node_id
        
        # 核心表
        self._method_table: Dict[str, MethodEntry] = {}
        self._pending_table: Dict[str, PendingRequest] = {}
        self._connection_table: Dict[str, ConnectionInfo] = {}

        # 服务对象表（用于 bind）
        self._services: Dict[str, Any] = {}

        # 事件处理器
        self._event_handlers: Dict[str, List[EventHandler]] = defaultdict(list)
        
        # 组件（延迟初始化）
        self._transport: Optional[Transport] = None
        self._serializer_registry: Optional[SerializerRegistry] = None
        self._interceptor_chain: Optional[InterceptorChain] = None
        
        # 状态
        self._running: bool = False
        self._accepting_requests: bool = True  # 是否接受新请求

        # In-flight请求跟踪
        self._in_flight_requests: int = 0
        self._in_flight_lock: asyncio.Lock = asyncio.Lock()

        # 后台任务
        self._background_tasks: List[asyncio.Task] = []
    
    # ==================== 事件 API ====================
    
    def on(self, event: str, handler: EventHandler) -> "Node":
        """
        注册事件处理器，支持链式调用
        
        事件:
            "connect"    - handler(node_id: str, address: Optional[str])
            "disconnect" - handler(node_id: str, reason: str)
        
        Example:
            node.on("connect", lambda nid, addr: print(f"Connected: {nid}"))
        """
        self._event_handlers[event].append(handler)
        return self
    
    def off(self, event: str, handler: Optional[EventHandler] = None) -> "Node":
        """移除事件处理器，不传 handler 则移除该事件所有处理器"""
        if handler is None:
            self._event_handlers[event].clear()
        else:
            try:
                self._event_handlers[event].remove(handler)
            except ValueError:
                pass
        return self
    
    async def _emit(self, event: str, *args, **kwargs) -> None:
        """
        触发事件

        统一处理同步和异步事件处理器：
        - 异步处理器：直接 await
        - 同步处理器：在线程池中执行，避免阻塞事件循环
        """
        for handler in self._event_handlers[event]:
            try:
                # 判断是否为协程函数
                if asyncio.iscoroutinefunction(handler):
                    # 异步处理器：直接 await
                    await handler(*args, **kwargs)
                else:
                    # 同步处理器：在线程池中执行
                    await asyncio.to_thread(handler, *args, **kwargs)
            except Exception as e:
                self.logger.error(f"Event handler error [{event}]: {e}")
    
    # ==================== 方法注册 ====================
    
    def register(
        self,
        method: str,
        handler: MethodHandler,
        options: Optional[MethodOptions] = None,
    ) -> None:
        """
        注册方法处理器

        Args:
            method: 方法名
            handler: 处理函数，签名为 async def handler(payload, context) -> result
            options: 方法选项

        Example:
            async def handle_train(payload, ctx):
                epochs = payload["epochs"]
                return {"status": "completed", "loss": 0.01}

            node.register("train", handle_train)
        """
        if method in self._method_table:
            raise ValueError(f"Method '{method}' already registered")

        self._method_table[method] = MethodEntry(
            handler=handler,
            options=options or MethodOptions(),
        )
        self.logger.debug(f"Registered method: {method}")
    
    def unregister(self, method: str) -> None:
        """注销方法"""
        self._method_table.pop(method, None)
        self.logger.debug(f"Unregistered method: {method}")
    
    def has_method(self, method: str) -> bool:
        """检查方法是否已注册"""
        return method in self._method_table

    def bind(self, service: Any, name: str = "default") -> None:
        """
        暴露本地服务对象

        将服务对象的所有公开方法自动注册为可调用的远程方法。
        远程节点可以通过 Proxy 调用 service 的方法。

        Args:
            service: 要暴露的对象（如 Learner 实例）
            name: 服务名称（用于区分多个服务）

        Example:
            class Learner:
                async def fit(self, config):
                    return {"loss": 0.1}

                async def evaluate(self, config):
                    return {"accuracy": 0.9}

            learner = Learner()
            node.bind(learner, name="learner")

            # 远程可以调用: node.call(target, "fit", {...})
        """
        if name in self._services:
            raise ValueError(f"Service '{name}' already bound")

        self._services[name] = service

        # 收集要注册的方法名列表
        methods_to_register = []

        # 自动注册服务对象的所有公开方法
        for method_name in dir(service):
            # 跳过私有方法和魔术方法
            if method_name.startswith("_"):
                continue

            method = getattr(service, method_name)

            # 只注册可调用的方法
            if not callable(method):
                continue

            # 检查是否是协程函数
            is_async = asyncio.iscoroutinefunction(method)

            # 跳过既不是协程也不是普通函数的（如属性等）
            if not is_async and not callable(method):
                continue

            methods_to_register.append(method_name)

            # 创建处理器闭包
            if is_async:
                # 异步方法：直接调用
                async def handler(payload: Any, ctx: MessageContext, m=method) -> Any:
                    """调用服务对象的异步方法"""
                    if isinstance(payload, dict):
                        # 检查是否是标准格式 {"args": (), "kwargs": {}}
                        if "args" in payload or "kwargs" in payload:
                            args = payload.get("args", ())
                            kwargs = payload.get("kwargs", {})
                            return await m(*args, **kwargs)
                        else:
                            # 普通字典，作为 **kwargs 传递
                            return await m(**payload)
                    else:
                        # 单一参数
                        return await m(payload)
            else:
                # 同步方法：包装成协程
                async def handler(payload: Any, ctx: MessageContext, m=method) -> Any:
                    """调用服务对象的同步方法（包装为协程）"""
                    if isinstance(payload, dict):
                        # 检查是否是标准格式 {"args": (), "kwargs": {}}
                        if "args" in payload or "kwargs" in payload:
                            args = payload.get("args", ())
                            kwargs = payload.get("kwargs", {})
                            # 在线程池中执行同步方法
                            return await asyncio.to_thread(m, *args, **kwargs)
                        else:
                            # 普通字典，作为 **kwargs 传递
                            return await asyncio.to_thread(m, **payload)
                    else:
                        # 单一参数
                        return await asyncio.to_thread(m, payload)

            # 注册方法（Memory 模式默认使用 pickle 序列化器）
            from .config import MethodOptions
            # 检查是否是 Memory 模式
            is_memory_mode = (
                hasattr(self, 'config') and
                hasattr(self.config, 'transport') and
                self.config.transport.mode == 'memory'
            )
            options = MethodOptions(serializer='pickle') if is_memory_mode else None
            self.register(method_name, handler, options=options)
            if is_memory_mode:
                self.logger.debug(f"Registered method '{method_name}' with pickle serializer (Memory mode)")

        self.logger.debug(f"服务 '{name}' 已绑定，暴露方法: {methods_to_register}")

    def unbind(self, name: str = "default") -> None:
        """
        解除服务绑定

        Args:
            name: 服务名称
        """
        service = self._services.pop(name, None)
        if service:
            # 注销所有相关方法
            for method_name in dir(service):
                if not method_name.startswith("_") and callable(getattr(service, method_name)):
                    self.unregister(method_name)
            self.logger.debug(f"Service '{name}' unbound")
    
    # ==================== 消息发送 ====================
    
    async def call(
        self,
        target: str,
        method: str,
        payload: Any,
        timeout: Optional[float] = None,
        metadata: Optional[Dict[str, str]] = None,
        serializer: Optional[str] = None,
        use_smart_wait: bool = True,  # 新增：是否使用智能等待（基于状态）
    ) -> Any:
        """
        调用远程方法（同步等待响应）

        Args:
            target: 目标节点 ID
            method: 方法名
            payload: 请求参数
            timeout: 超时时间（秒），None 使用 MethodOptions 或默认值
            metadata: 附加元数据
            serializer: 序列化器名称（如 "json", "pickle"），None 使用默认值
            use_smart_wait: 是否使用智能等待（基于节点状态），默认True

        Timeout 优先级:
            显式参数 > MethodOptions.timeout > NodeConfig.default_timeout

        Returns:
            远程方法的返回值

        Raises:
            CallTimeoutError: 调用超时
            NodeNotConnectedError: 目标节点未连接
            RemoteExecutionError: 远程执行失败

        Example:
            result = await node.call(
                target="participant_1",
                method="train",
                payload={" epochs": 10, "lr": 0.001},
                timeout=300.0,
            )
        """
        # DEBUG: 记录调用开始
        self.logger.debug(f"[RPC-SEND] {target}.{method}(), payload={payload.keys()}")

        # 1. 检查连接
        if not self._is_connected(target):
            raise NodeNotConnectedError(target)

        # 2. 确定超时（优先级：显式 > MethodOptions > NodeConfig）
        if timeout is None:
            entry = self._method_table.get(method)
            if entry and entry.options.timeout is not None:
                timeout = entry.options.timeout
            else:
                timeout = self.config.default_timeout

        # 3. 序列化 payload
        ser = self._get_serializer(method, serializer)
        try:
            payload_bytes = ser.serialize(payload)
        except Exception as e:
            raise SerializationError(f"Failed to serialize payload: {e}")

        # 4. 构造消息
        message_id = str(uuid.uuid4())
        message = Message(
            id=message_id,
            type=MessageType.REQUEST,
            method=method,
            source=self.node_id,
            target=target,
            payload=payload_bytes,
            correlation_id=None,
            timestamp=time.time_ns(),
            metadata=metadata or {},
            error=None,
        )

        self.logger.debug(f"构造消息: id={message_id}, target={target}, method={method}")

        # 5. 创建 Future 并注册到 pending_table
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        now = time.time()
        pending = PendingRequest(
            future=future,
            method=method,
            created_at=now,
            timeout_at=now + timeout,
            target=target,
        )
        self._pending_table[message_id] = pending

        try:
            # 6. 通过 Interceptor 链处理
            message = await self._interceptor_chain.execute_outbound(message, self.node_id)

            # 7. 发送消息
            self.logger.debug(f"发送消息: id={message_id}, target={target}, method={method}")
            await self._transport.send(message)
            self.logger.debug(f"消息已发送: id={message_id}")

            # 8. 等待响应（智能等待或固定超时）
            if use_smart_wait and method not in ["_fed_get_property", "_heartbeat"]:
                # 使用智能等待（基于状态）
                response: Message = await self._smart_wait_for_response(
                    future=future,
                    target=target,
                    method=method,
                    message_id=message_id,
                    initial_timeout=timeout
                )
            else:
                # 使用固定超时（用于内部方法和快速调用）
                response: Message = await asyncio.wait_for(future, timeout=timeout)

            self.logger.debug(f"收到响应: id={message_id}, has_error={response.error is not None}")

            # 9. 检查错误
            if response.error:
                raise RemoteExecutionError(
                    code=response.error.code,
                    message=response.error.message,
                    details=response.error.details,
                )

            # 10. 反序列化结果
            result = ser.deserialize(response.payload)
            self.logger.debug(f"[RPC-RECV] ← {target}.{method}() = {type(result).__name__}")
            return result

        except asyncio.TimeoutError:
            self.logger.error(f"调用超时: target={target}, method={method}, timeout={timeout}")
            raise CallTimeoutError(
                msg_id=message_id,
                method=method,
                target=target,
                timeout=timeout,
            )
        finally:
            # 11. 清理 pending_table
            self._pending_table.pop(message_id, None)

    async def _smart_wait_for_response(
        self,
        future: asyncio.Future,
        target: str,
        method: str,
        message_id: str,
        initial_timeout: float
    ) -> Message:
        """
        智能等待响应：基于远程节点状态而非固定超时

        策略：
        1. 周期性检查future是否完成（响应已到达）
        2. 如果未完成，查询目标节点的状态
        3. 如果节点处于工作状态（TRAINING/EVALUATING/AGGREGATING），继续等待
        4. 如果节点处于IDLE状态但响应未到，可能有问题，继续等待但记录警告
        5. 如果节点UNHEALTHY，抛出超时
        6. 使用initial_timeout作为基准，但在节点工作时可以延长

        Args:
            future: 等待的Future对象
            target: 目标节点ID
            method: 调用的方法名
            message_id: 消息ID
            initial_timeout: 初始超时时间（秒）

        Returns:
            Message: 响应消息

        Raises:
            asyncio.TimeoutError: 真正的超时（节点异常或无响应）
        """
        check_interval = 5.0  # 每5秒检查一次状态
        start_time = time.time()
        last_status_check = start_time
        last_logged_state = None
        max_idle_wait = 30.0  # 节点IDLE状态下最多额外等待30秒
        idle_start_time = None

        while True:
            try:
                # 尝试等待响应（带短超时）
                response = await asyncio.wait_for(future, timeout=check_interval)
                return response

            except asyncio.TimeoutError:
                # 检查是否需要查询状态
                now = time.time()
                elapsed = now - start_time

                # 查询远程节点状态
                try:
                    # 调用远程的_fed_get_property获取state属性（使用固定超时，避免递归）
                    state_result = await asyncio.wait_for(
                        self.call(
                            target,
                            "_fed_get_property",
                            {"property_name": "state"},  # 注意：是"state"而不是"_state"
                            use_smart_wait=False  # 关键：避免递归
                        ),
                        timeout=3.0
                    )
                    remote_state = state_result if isinstance(state_result, str) else "unknown"

                except Exception as e:
                    # 无法获取状态，可能节点已崩溃
                    self.logger.warning(
                        f"[{self.node_id}] 无法获取 {target} 的状态: {e}, "
                        f"已等待 {elapsed:.1f}s"
                    )
                    remote_state = "unknown"

                # 根据状态决定是否继续等待
                if remote_state in ["training", "evaluating", "aggregating"]:
                    # 节点正在工作，继续等待
                    if remote_state != last_logged_state:
                        self.logger.info(
                            f"[{self.node_id}] {target}.{method}() 仍在执行中 "
                            f"(状态: {remote_state}, 已等待 {elapsed:.1f}s), 继续等待..."
                        )
                        last_logged_state = remote_state
                    idle_start_time = None  # 重置idle计时
                    continue

                elif remote_state == "idle":
                    # 节点已经IDLE，但响应还没到，可能在网络传输中
                    if idle_start_time is None:
                        idle_start_time = now
                        self.logger.debug(
                            f"[{self.node_id}] {target} 已进入IDLE状态，"
                            f"等待响应到达..."
                        )

                    idle_elapsed = now - idle_start_time
                    if idle_elapsed > max_idle_wait:
                        # IDLE状态等待太久，认为出问题了
                        self.logger.error(
                            f"[{self.node_id}] {target}.{method}() 调用超时: "
                            f"节点已IDLE {idle_elapsed:.1f}s 但响应未到达"
                        )
                        raise asyncio.TimeoutError()
                    continue

                elif remote_state == "unhealthy":
                    # 节点不健康，停止等待
                    self.logger.error(
                        f"[{self.node_id}] {target} 状态不健康，停止等待"
                    )
                    raise asyncio.TimeoutError()

                else:
                    # unknown或其他状态，使用初始超时作为兜底
                    if elapsed > initial_timeout * 2:  # 允许2倍的初始超时
                        self.logger.error(
                            f"[{self.node_id}] {target}.{method}() 调用超时: "
                            f"已等待 {elapsed:.1f}s (状态: {remote_state})"
                        )
                        raise asyncio.TimeoutError()
                    continue

    async def notify(
        self,
        target: str,
        method: str,
        payload: Any,
        metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        发送通知（不等待响应）
        
        Args:
            target: 目标节点 ID
            method: 方法名
            payload: 通知内容
            metadata: 附加元数据
            
        Example:
            await node.notify(
                target="coordinator",
                method="_heartbeat",
                payload={"status": "alive"},
            )
        """
        if not self._is_connected(target):
            raise NodeNotConnectedError(target)
        
        serializer = self._get_serializer(method)
        try:
            payload_bytes = serializer.serialize(payload)
        except Exception as e:
            raise SerializationError(f"Failed to serialize payload: {e}")
        
        message = Message(
            id=str(uuid.uuid4()),
            type=MessageType.NOTIFY,
            method=method,
            source=self.node_id,
            target=target,
            payload=payload_bytes,
            correlation_id=None,
            timestamp=time.time_ns(),
            metadata=metadata or {},
            error=None,
        )
        
        message = await self._interceptor_chain.execute_outbound(message, self.node_id)
        await self._transport.send(message)
    
    async def broadcast(
        self,
        targets: List[str],
        method: str,
        payload: Any,
        timeout: Optional[float] = None,
        wait_for_all: bool = True,
        min_responses: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        广播调用多个对等节点

        根据传输模式自动选择实现：
        - Memory 模式：串行调用（转换为多个单独调用）
        - gRPC 模式：并行调用（真正的网络广播）

        Args:
            targets: 目标节点 ID 列表
            method: 方法名
            payload: 参数（建议使用字典格式 {"args": (), "kwargs": {}}）
            timeout: 超时时间（秒）
            wait_for_all: 是否等待所有节点返回，False 时结合 min_responses 使用
            min_responses: 最少需要的响应数，达到后即可返回（仅在 wait_for_all=False 时生效）

        Returns:
            结果字典 {node_id: result}
            如果某个节点调用失败，对应的 value 为 Exception 对象

        Example:
            # 广播到所有连接的节点
            results = await node.broadcast(
                targets=["learner_1", "learner_2"],
                method="fit",
                payload={"args": (), "kwargs": {"epochs": 5}},
            )

            # 灵活广播：等待至少 3 个节点响应，超时 30 秒
            results = await node.broadcast(
                targets=all_learners,
                method="fit",
                payload={"args": (), "kwargs": {"epochs": 5}},
                timeout=30,
                wait_for_all=False,
                min_responses=3,
            )

            # 处理结果
            for node_id, result in results.items():
                if isinstance(result, Exception):
                    print(f"{node_id} failed: {result}")
                else:
                    print(f"{node_id} succeeded: {result}")
        """
        # 委托给 Transport 层处理
        return await self._transport.broadcast(
            targets, method, payload, timeout, wait_for_all, min_responses
        )

    
    # ==================== 消息接收与路由 ====================
    
    async def on_message(self, message: Message) -> Optional[Message]:
        """
        消息入口（Transport 调用此方法）

        Args:
            message: 接收到的消息

        Returns:
            对于 REQUEST，返回 RESPONSE
            对于 NOTIFY/RESPONSE，返回 None
        """
        # DEBUG: 记录消息到达节点
        self.logger.debug(f"on_message: type={message.type.value}, from={message.source}, method={message.method}, msg_id={message.id}")

        # 通过 Interceptor 链处理入站消息
        try:
            message = await self._interceptor_chain.execute_inbound(message, self.node_id)
        except InterceptorAbort as e:
            self.logger.warning(f"loggerInterceptor 中止消息: method={message.method}, error={e.error_info}")
            if message.type == MessageType.REQUEST:
                return self._create_error_response(message, e.error_info)
            return None

        # 根据消息类型分发
        if message.type == MessageType.REQUEST:
            self.logger.debug(f"logger分发到 _handle_request: method={message.method}")
            return await self._handle_request(message)
        elif message.type == MessageType.RESPONSE:
            self.logger.debug(f"logger分发到 _handle_response: correlation_id={message.correlation_id}")
            await self._handle_response(message)
            return None
        elif message.type == MessageType.NOTIFY:
            self.logger.debug(f"logger分发到 _handle_notify: method={message.method}")
            await self._handle_notify(message)
            return None

        return None
    
    async def _handle_request(self, message: Message) -> Message:
        """处理 REQUEST 消息（带 in-flight 跟踪）"""

        # DEBUG: 记录接收到请求
        self.logger.debug(f"[RPC-HANDLE] ← {message.source}.{message.method}() msg_id={message.id}")

        # 检查是否接受新请求
        if not self._accepting_requests:
            self.logger.warning(f"拒绝请求（节点关闭中）: method={message.method}")
            return self._create_error_response(
                message,
                ErrorInfo(
                    code="NODE_STOPPING",
                    message="Node is shutting down, not accepting new requests",
                ),
            )

        # 增加 in-flight 计数
        async with self._in_flight_lock:
            self._in_flight_requests += 1

        try:
            # 1. 查找 handler
            entry = self._method_table.get(message.method)
            if not entry:
                self.logger.error(f"方法未注册: method={message.method}")
                return self._create_error_response(
                    message,
                    ErrorInfo(
                        code="METHOD_NOT_FOUND",
                        message=f"Method '{message.method}' not registered",
                    ),
                )

            self.logger.debug(f"找到方法处理器: method={message.method}")

            # 2. 反序列化 payload
            try:
                serializer = self._get_serializer(message.method)
                payload = serializer.deserialize(message.payload)
                self.logger.debug(f"payload 反序列化成功: method={message.method}, payload_type={type(payload).__name__}")
            except Exception as e:
                self.logger.exception(f"Deserialization error for method {message.method}")
                return self._create_error_response(
                    message,
                    ErrorInfo(
                        code="SERIALIZATION_ERROR",
                        message=str(e),
                    ),
                )

            # 3. 构造上下文
            context = MessageContext(
                message=message,
                source=message.source,
                metadata=message.metadata,
            )

            # DEBUG: 打印payload的格式
            if message.method == "set_weights":
                self.logger.debug(f"[DEBUG] set_weights payload type: {type(payload)}")
                if isinstance(payload, dict):
                    self.logger.debug(f"[DEBUG] set_weights payload keys: {payload.keys()}")
                    if "args" in payload:
                        self.logger.debug(f"[DEBUG] set_weights args: type={type(payload['args'])}, len={len(payload.get('args', ()))}")
                        if payload.get('args'):
                            self.logger.debug(f"[DEBUG] set_weights first arg type: {type(payload['args'][0])}")
                    if "kwargs" in payload:
                        self.logger.debug(f"[DEBUG] set_weights kwargs keys: {payload.get('kwargs', {}).keys()}")

            # 4. 执行 handler
            try:
                self.logger.debug(f"[RPC-EXEC] 执行 {message.method}()")
                result = await entry.handler(payload, context)
                self.logger.debug(f"[RPC-DONE] {message.method}() → {type(result).__name__}")
            except Exception as e:
                self.logger.error(f"handler执行失败: method={message.method}, error={e}")
                error_info = ErrorInfo(
                    code="EXECUTION_ERROR",
                    message=str(e),
                    stack_trace=traceback.format_exc() if self.config.debug else None,
                )
                return self._create_error_response(message, error_info)

            # 5. 构造响应
            try:
                result_bytes = serializer.serialize(result)
                # 调试：检查序列化后的字节
                if len(result_bytes) == 0:
                    self.logger.error(f"Serialized result is empty! method={message.method}, result={result}, result_type={type(result)}")
                self.logger.debug(f"结果序列化成功: method={message.method}, size={len(result_bytes)} bytes")
            except Exception as e:
                self.logger.exception(f"Serialization error for method {message.method}, result_type={type(result)}")
                return self._create_error_response(
                    message,
                    ErrorInfo(code="SERIALIZATION_ERROR", message=str(e)),
                )
        finally:
            # 减少 in-flight 计数
            async with self._in_flight_lock:
                self._in_flight_requests -= 1

        response = Message(
            id=str(uuid.uuid4()),
            type=MessageType.RESPONSE,
            method=message.method,
            source=self.node_id,
            target=message.source,
            payload=result_bytes,
            correlation_id=message.id,
            timestamp=time.time_ns(),
            metadata={},
            error=None,
        )

        self.logger.debug(f"发送响应: to={message.source}, method={message.method}, correlation_id={message.id}")
        return response
    
    async def _handle_response(self, message: Message) -> None:
        """处理 RESPONSE 消息"""

        # DEBUG: 记录响应处理
        self.logger.debug(f"_handle_response: from={message.source}, correlation_id={message.correlation_id}")

        pending = self._pending_table.get(message.correlation_id)
        if not pending:
            self.logger.warning(
                f"Orphan response received: correlation_id={message.correlation_id}"
            )
            return

        # 完成 Future
        if not pending.future.done():
            self.logger.debug(f"设置 Future 结果: correlation_id={message.correlation_id}, method={pending.method}")
            pending.future.set_result(message)
        else:
            self.logger.warning(f"Future 已完成: correlation_id={message.correlation_id}")
    
    async def _handle_notify(self, message: Message) -> None:
        """处理 NOTIFY 消息"""
        
        entry = self._method_table.get(message.method)
        if not entry:
            self.logger.warning(f"No handler for notify: method={message.method}")
            return
        
        try:
            serializer = self._get_serializer(message.method)
            payload = serializer.deserialize(message.payload)
            
            context = MessageContext(
                message=message,
                source=message.source,
                metadata=message.metadata,
            )
            
            await entry.handler(payload, context)
            
        except Exception as e:
            self.logger.error(f"Error handling notify {message.method}: {e}")
    
    def _create_error_response(
        self,
        request: Message,
        error: ErrorInfo,
    ) -> Message:
        """创建错误响应"""
        return Message(
            id=str(uuid.uuid4()),
            type=MessageType.RESPONSE,
            method=request.method,
            source=self.node_id,
            target=request.source,
            payload=b"",
            correlation_id=request.id,
            timestamp=time.time_ns(),
            metadata={},
            error=error,
        )
    
    # ==================== 连接管理 ====================
    
    async def connect(
        self,
        target: str,
        address: Optional[str] = None,
        retry_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        连接到目标节点（支持重试）

        Args:
            target: 目标节点 ID
            address: 目标地址（gRPC 模式需要）
            retry_config: 重试配置字典

        Example:
            # Memory 模式
            await node.connect("other_node")

            # gRPC 模式
            await node.connect("other_node", "192.168.1.100:50051")

            # 带重试配置
            await node.connect("other_node", retry_config={
                "enabled": True,
                "max_retries": 10,
                "retry_interval": 2.0,
            })
        """
        await self._transport.connect(target, address, retry_config)
        
        # 发送握手通知
        # 优先使用 advertised_address，否则从 listen 或 transport 配置构造
        handshake_address = self.config.advertised_address
        if not handshake_address:
            # 尝试从 listen 配置获取
            if hasattr(self.config, 'listen') and self.config.listen:
                host = self.config.listen.get("host", "localhost")
                port = self.config.listen.get("port", None)
                if not port:
                    raise ValueError("Listen port must be specified in config for handshake address")
                self.logger.debug(f"使用 listen 配置构造握手地址: host={host}, port={port}")
            else:
                # 否则使用 transport 配置
                host = self.config.transport.grpc.host
                port = self.config.transport.grpc.port
                self.logger.debug(f"使用 transport 配置构造握手地址: host={host}, port={port}")

            # 如果 host 是 0.0.0.0，替换为 localhost（本地开发）
            if host == "0.0.0.0":
                host = "localhost"
            handshake_address = f"{host}:{port}"

        self.logger.debug(f"发送握手通知到 {target}，告知地址: {handshake_address}")

        try:
            await self.notify(target, "_handshake", {
                "node_id": self.node_id,
                "address": handshake_address,
            })
        except Exception as e:
            self.logger.debug(f"Handshake to {target} failed (may be expected): {e}")
    
    async def disconnect(self, target: str) -> None:
        """断开与目标节点的连接"""
        await self._transport.disconnect(target)
    
    def get_connected_nodes(self) -> List[str]:
        """获取所有已连接节点的 ID"""
        return [
            node_id
            for node_id, info in self._connection_table.items()
            if info.status == ConnectionStatus.CONNECTED
        ]
    
    def is_connected(self, node_id: str) -> bool:
        """检查是否已连接到指定节点"""
        return self._is_connected(node_id)

    async def wait_for_connections(self, min_peers: int, timeout: float = 60) -> List[str]:
        """
        等待最少数量的对等节点连接

        Args:
            min_peers: 最少节点数
            timeout: 超时时间（秒）

        Returns:
            已连接的节点 ID 列表

        Raises:
            TimeoutError: 超时未达到最少节点数

        Example:
            # 等待至少 2 个节点连接
            await node.wait_for_connections(min_peers=2, timeout=120)

            # 获取连接的节点
            peers = node.get_connected_nodes()
        """
        self.logger.debug(f"等待节点连接: 需要至少 {min_peers} 个节点，超时 {timeout}秒")
        start = time.time()
        while True:
            connected = self.get_connected_nodes()
            if len(connected) >= min_peers:
                self.logger.debug(f"已达到最少连接数: {len(connected)}/{min_peers}，已连接节点: {connected}")
                return connected

            # 检查超时
            elapsed = time.time() - start
            if elapsed > timeout:
                raise TimeoutError(
                    f"Timeout waiting for connections: {len(connected)}/{min_peers} "
                    f"after {elapsed:.1f}s"
                )

            # 等待一小段时间再检查
            await asyncio.sleep(0.5)
    
    def _is_connected(self, node_id: str) -> bool:
        """内部方法：检查连接状态"""
        info = self._connection_table.get(node_id)
        return info is not None and info.status == ConnectionStatus.CONNECTED
    
    # Transport 回调
    
    async def _on_connect(self, node_id: str, address: Optional[str]) -> None:
        """连接建立回调"""
        now = time.time()

        self._connection_table[node_id] = ConnectionInfo(
            node_id=node_id,
            address=address,
            status=ConnectionStatus.CONNECTED,
            connected_at=now,
            last_active=now,
            metadata={},
        )

        self.logger.debug(f"已连接到节点 '{node_id}' (地址: {address})")

        # 触发事件
        await self._emit("connect", node_id, address)
    
    async def _on_disconnect(self, node_id: str, reason: str) -> None:
        """连接断开回调"""

        # 更新连接表
        if node_id in self._connection_table:
            self._connection_table[node_id].status = ConnectionStatus.DISCONNECTED

        # 清理该节点的 pending 请求
        failed_requests = [
            (msg_id, pending)
            for msg_id, pending in self._pending_table.items()
            if pending.target == node_id
        ]

        for msg_id, pending in failed_requests:
            if not pending.future.done():
                pending.future.set_exception(
                    NodeDisconnectedError(node_id, reason)
                )
            self._pending_table.pop(msg_id, None)

        self.logger.info(f"已断开与节点 '{node_id}' 的连接: {reason}")

        # 触发事件
        await self._emit("disconnect", node_id, reason)
    
    # ==================== 生命周期管理 ====================
    
    async def start(self) -> None:
        """
        启动节点

        Example:
            node = Node("my_node", config)
            node.register("train", handle_train)
            await node.start()
        """
        if self._running:
            raise RuntimeError("Node is already running")

        self.logger.debug(f"正在启动节点: {self.node_id}")

        # 保存主事件循环引用（用于控制线程跨线程调用）
        self._main_loop = asyncio.get_event_loop()

        # 初始化组件
        self.logger.debug("初始化序列化器注册表")
        self._serializer_registry = SerializerRegistry(self.config.serialization)
        self._interceptor_chain = InterceptorChain()

        # 添加日志拦截器
        if self.config.interceptors.logging:
            self.logger.debug("添加日志拦截器")
            self._interceptor_chain.add(LoggingInterceptor(node_id=self.node_id))

        # 添加认证拦截器
        if self.config.interceptors.auth:
            self.logger.debug("添加认证拦截器")
            auth_config = self.config.interceptors.auth_config
            auth_interceptor = AuthInterceptor(
                token=auth_config.token,
            )
            self._interceptor_chain.add(auth_interceptor)

        # 创建 Transport
        self.logger.debug(f"创建传输层: mode={self.config.transport.mode}")
        self._transport = create_transport(self.config.transport, self)

        # 注册内置处理器
        self.logger.debug("注册内置处理器")
        self._register_builtin_handlers()

        # 启动 Transport
        self.logger.debug("启动传输层")
        await self._transport.start()

        # 启动后台任务
        self.logger.debug("启动后台任务")
        self._start_background_tasks()

        self._running = True
        self.logger.debug(f"节点 '{self.node_id}' 启动完成")

    async def _wait_for_in_flight_requests(self, timeout: float = 5.0) -> bool:
        """
        等待所有 in-flight 请求完成

        Args:
            timeout: 超时时间（秒）

        Returns:
            是否所有请求都完成（True）或超时（False）
        """
        import time
        start = time.time()

        while True:
            async with self._in_flight_lock:
                count = self._in_flight_requests

            if count == 0:
                self.logger.debug(f"所有 in-flight 请求已完成")
                return True

            elapsed = time.time() - start
            if elapsed >= timeout:
                self.logger.warning(
                    f"等待 in-flight 请求超时: "
                    f"仍有 {count} 个请求未完成"
                )
                return False

            # 每 100ms 检查一次
            await asyncio.sleep(0.1)

    async def stop(self) -> None:
        """
        停止节点（分阶段关闭）

        阶段1: 停止接受新请求
        阶段2: 等待 in-flight 请求完成
        阶段3: 停止后台任务
        阶段4: 取消 pending 请求
        阶段5: 停止 Transport（包括控制线程）
        """
        if not self._running:
            return

        self.logger.debug(f"开始停止节点...")

        # 阶段1: 停止接受新请求
        self._accepting_requests = False
        self.logger.debug(f"已停止接受新请求")

        # 阶段2: 等待 in-flight 请求完成（最多等待5秒）
        await self._wait_for_in_flight_requests(timeout=5.0)

        # 阶段3: 停止后台任务
        self._stop_background_tasks()
        self.logger.debug(f"后台任务已停止")

        # 阶段4: 取消所有 pending 请求
        for msg_id, pending in list(self._pending_table.items()):
            if not pending.future.done():
                pending.future.cancel()
        self._pending_table.clear()
        self.logger.debug(f"Pending 请求已清理")

        # 阶段5: 停止 Transport（包括控制线程）
        if self._transport:
            await self._transport.stop()
            self.logger.debug(f"Transport 已停止")

        self._running = False
        self.logger.debug(f"节点已完全停止")
    
    def _register_builtin_handlers(self) -> None:
        """注册框架内置处理器"""
        # 握手处理器
        if "_handshake" not in self._method_table:
            self.register("_handshake", self._builtin_handshake_handler)
        
        # 心跳处理器
        if "_heartbeat" not in self._method_table:
            self.register("_heartbeat", self._builtin_heartbeat_handler)
    
    async def _builtin_handshake_handler(
        self,
        payload: Dict[str, Any],
        ctx: MessageContext,
    ) -> None:
        """处理握手，自动回连"""
        source_id = payload.get("node_id")
        source_addr = payload.get("address")

        self.logger.debug(f"收到握手: from={source_id}, address={source_addr}")

        # 如果还没连接，则回连（仅 gRPC 模式）
        if (
            source_addr
            and source_id
            and not self.is_connected(source_id)
            and self.config.transport.mode == "grpc"
        ):
            self.logger.debug(f"准备自动回连到 {source_id} (地址: {source_addr})")
            try:
                await self._transport.connect(source_id, source_addr)
                self.logger.debug(f"自动回连成功: {source_id}")
            except Exception as e:
                self.logger.warning(f"自动回连失败: {source_id}, error={e}")
    
    async def _builtin_heartbeat_handler(
        self,
        payload: Dict[str, Any],
        ctx: MessageContext,
    ) -> None:
        """处理心跳，更新 last_active"""
        source = ctx.source
        if source in self._connection_table:
            self._connection_table[source].last_active = time.time()
    
    def _start_background_tasks(self) -> None:
        """启动后台任务"""
        # 超时检查任务
        task = asyncio.create_task(self._timeout_checker_loop())
        self._background_tasks.append(task)

        # 心跳任务（已迁移到 GRPCTransport 控制线程）
        # Memory 模式无需心跳，gRPC 模式由 GRPCTransport 内部处理
        # if self.config.heartbeat.enabled:
        #     task = asyncio.create_task(self._heartbeat_loop())
        #     self._background_tasks.append(task)
    
    def _stop_background_tasks(self) -> None:
        """停止后台任务"""
        for task in self._background_tasks:
            task.cancel()
        self._background_tasks.clear()
    
    async def _timeout_checker_loop(self) -> None:
        """检查超时的请求"""
        while self._running:
            try:
                await asyncio.sleep(1.0)
                
                now = time.time()
                expired = [
                    msg_id
                    for msg_id, pending in self._pending_table.items()
                    if now > pending.timeout_at and not pending.future.done()
                ]
                
                for msg_id in expired:
                    pending = self._pending_table.pop(msg_id, None)
                    if pending and not pending.future.done():
                        timeout_duration = pending.timeout_at - pending.created_at
                        pending.future.set_exception(
                            CallTimeoutError(
                                msg_id=msg_id,
                                method=pending.method,
                                target=pending.target,
                                timeout=timeout_duration,
                            )
                        )
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Timeout checker error: {e}")
    
    async def _heartbeat_loop(self) -> None:
        """
        心跳循环：发送 + 检测超时

        超时处理策略（宽限期机制）：
        1. 第一次超时：进入宽限期（30秒），记录警告日志
        2. 宽限期内恢复：清除宽限期，重置计数器
        3. 宽限期结束仍超时：真正断开连接

        这样可以避免因网络波动导致的频繁断开重连。
        """
        interval = self.config.heartbeat.interval
        timeout = self.config.heartbeat.timeout
        grace_period = 30.0  # 宽限期30秒

        while self._running:
            try:
                await asyncio.sleep(interval)

                now = time.time()

                # 1. 检查所有连接
                for node_id, info in list(self._connection_table.items()):
                    if info.status != ConnectionStatus.CONNECTED:
                        continue

                    time_since_active = now - info.last_active

                    # 情况1: 连接正常（在超时时间内）
                    if time_since_active <= timeout:
                        # 如果之前在宽限期，现在恢复了，清除宽限期状态
                        if info.grace_period_end is not None:
                            self.logger.info(f"Connection recovered: {node_id} (was in grace period)")
                            info.grace_period_end = None
                            info.timeout_count = 0
                        continue

                    # 情况2: 第一次超时，进入宽限期
                    if info.grace_period_end is None:
                        info.grace_period_end = now + grace_period
                        info.timeout_count = 1
                        self.logger.warning(
                            f"Heartbeat timeout detected: {node_id} "
                            f"(silent for {time_since_active:.1f}s, timeout={timeout}s). "
                            f"Entering grace period ({grace_period}s)"
                        )
                        continue

                    # 情况3: 宽限期内，继续等待
                    if now < info.grace_period_end:
                        info.timeout_count += 1
                        self.logger.debug(
                            f"Still in grace period: {node_id} "
                            f"(timeout_count={info.timeout_count}, "
                            f"remaining={info.grace_period_end - now:.1f}s)"
                        )
                        continue

                    # 情况4: 宽限期结束仍超时，真正断开
                    self.logger.error(
                        f"Heartbeat timeout (grace period expired): {node_id} "
                        f"(silent for {time_since_active:.1f}s, timeout_count={info.timeout_count})"
                    )
                    await self._handle_connection_timeout(node_id)

                # 2. 发送心跳到所有存活连接
                connected_nodes = self.get_connected_nodes()
                if connected_nodes:
                    await self.broadcast(
                        targets=connected_nodes,
                        method="_heartbeat",
                        payload={"timestamp": now},
                    )
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Heartbeat loop error: {e}")
    
    async def _handle_connection_timeout(self, node_id: str) -> None:
        """处理连接超时"""
        # 1. 标记为断开
        if node_id in self._connection_table:
            self._connection_table[node_id].status = ConnectionStatus.DISCONNECTED
        
        # 2. 关闭底层连接
        try:
            await self._transport.disconnect(node_id)
        except Exception as e:
            self.logger.debug(f"Error disconnecting {node_id}: {e}")
        
        # 3. 触发 _on_disconnect
        await self._on_disconnect(node_id, "heartbeat timeout")
    
    # ==================== 辅助方法 ====================
    
    def _get_serializer(self, method: str, serializer_name: Optional[str] = None) -> Serializer:
        """获取指定方法的序列化器

        优先级: 显式指定 > 代码注册的 MethodOptions > YAML 配置 > 默认
        """
        # 0. 显式指定
        if serializer_name:
            result = self._serializer_registry.get(serializer_name)
            self.logger.debug(f"Method '{method}': using explicit serializer '{serializer_name}'")
            return result

        # 1. 代码注册的 MethodOptions
        entry = self._method_table.get(method)
        if entry and entry.options.serializer:
            result = self._serializer_registry.get(entry.options.serializer)
            self.logger.debug(f"Method '{method}': using MethodOptions serializer '{entry.options.serializer}'")
            return result

        # 2. YAML 配置 / 默认
        result = self._serializer_registry.get_for_method(method)
        # self.logger.debug(f"Method '{method}': using default/config serializer '{result.name}' (config.default={self.config.serialization.default})")
        return result
    
    # ==================== 公开扩展 API ====================
    
    def register_serializer(self, serializer: Serializer) -> None:
        """
        注册自定义序列化器
        
        Args:
            serializer: 序列化器实例，需要实现 name, serialize(), deserialize()
            
        Example:
            class MsgPackSerializer(Serializer):
                @property
                def name(self) -> str:
                    return "msgpack"
                
                def serialize(self, data: Any) -> bytes:
                    import msgpack
                    return msgpack.packb(data)
                
                def deserialize(self, data: bytes) -> Any:
                    import msgpack
                    return msgpack.unpackb(data)
            
            node.register_serializer(MsgPackSerializer())
        """
        if self._serializer_registry is None:
            raise RuntimeError("Node not started, call start() first")
        self._serializer_registry.register(serializer)
    
    def add_interceptor(self, interceptor: Interceptor, prepend: bool = False) -> None:
        """
        添加拦截器
        
        Args:
            interceptor: 拦截器实例
            prepend: 是否添加到最外层（最先执行）
            
        执行顺序遵循"洋葱模型"：
        - 添加顺序: [A, B, C]
        - 执行顺序: C.before -> B.before -> A.before -> handler -> A.after -> B.after -> C.after
            
        Example:
            class MetricsInterceptor(Interceptor):
                async def intercept_inbound(self, message, context, next_handler):
                    start = time.time()
                    result = await next_handler(message)
                    duration = time.time() - start
                    record_metric(message.method, duration)
                    return result
                    
                async def intercept_outbound(self, message, context, next_handler):
                    return await next_handler(message)
            
            node.add_interceptor(MetricsInterceptor())
        """
        if self._interceptor_chain is None:
            raise RuntimeError("Node not started, call start() first")
        self._interceptor_chain.add(interceptor, prepend=prepend)
    
    def remove_interceptor(self, interceptor: Interceptor) -> bool:
        """
        移除拦截器
        
        Args:
            interceptor: 要移除的拦截器实例
            
        Returns:
            是否成功移除
        """
        if self._interceptor_chain is None:
            return False
        return self._interceptor_chain.remove(interceptor)
    
    # ==================== 上下文管理器 ====================
    
    async def __aenter__(self) -> "Node":
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop()
