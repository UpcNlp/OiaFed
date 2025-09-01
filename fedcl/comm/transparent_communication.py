"""
透明通信模块 - 进程内通信实现

提供进程内的轻量透明通信实现，用于本地模拟与单机测试。
使用模块级全局队列在同一进程内不同节点间传递消息。

主要功能：
- TransparentCommunication: 透明通信实现
- Message: 消息数据结构
- CommunicationMode: 通信模式枚举
- NetworkConfig: 网络配置
- BaseCommunicationBackend: 基础通信后端抽象
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional
import threading
import time
import uuid
import queue
from loguru import logger

# ============ 基本数据结构 ============

@dataclass
class Message:
    id: str
    sender: str
    receiver: str
    message_type: str
    payload: Any
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class CommunicationMode(Enum):
    LOCAL_MEMORY = "local_memory"
    PROCESS = "process"
    NETWORK_TCP = "network_tcp"


@dataclass
class NetworkConfig:
    host: str = "localhost"
    port: int = 8080
    max_connections: int = 100
    timeout_seconds: float = 30.0


# ============ 进程内全局队列（用于模拟） ============
# 全局映射：node_id -> Queue
_global_message_queues: Dict[str, "queue.Queue[Message]"] = {}
_global_queues_lock = threading.Lock()


def _get_queue(node_id: str) -> "queue.Queue[Message]":
    with _global_queues_lock:
        if node_id not in _global_message_queues:
            _global_message_queues[node_id] = queue.Queue()
        return _global_message_queues[node_id]


# ============ 基础通信后端类（类型契约） ============
class BaseCommunicationBackend:
    """基础后端抽象 - 仅作类型契约和轻量默认实现"""

    def start(self) -> bool:
        return True

    def stop(self) -> bool:
        return True

    def send_model_update(self, target: str, message: Dict[str, Any]) -> bool:
        return False

    def send_message(self, target: str, message: Message) -> bool:
        return False

    def receive_message(self, timeout: float = 1.0) -> Optional[Message]:
        return None

    def register_handler(self, message_type: str, handler: Callable[[Message], None]) -> None:
        pass

    def get_stats(self) -> Dict[str, Any]:
        return {}

    def call_learner_method(self, client_id: str, learner_name: str, method_name: str, *args, **kwargs) -> Any:
        raise NotImplementedError()


# ============ 透明通信实现（进程内模拟） ============
class TransparentCommunication(BaseCommunicationBackend):
    """
    透明通信（本地内存模拟）

    设计要点：
    - 使用全局队列进行点对点消息传递
    - 支持同步 API（start/stop/send/receive），便于在非async 线程中调用
    - 支持 handler 注册：收到消息时在后台线程回调 handler
    - 提供 call_learner_method 的请求/响应支持（基于 request_id 关联）
    """

    def __init__(self, node_id: str, mode: CommunicationMode = CommunicationMode.LOCAL_MEMORY,
                 config: Optional[NetworkConfig] = None, is_server: bool = False):
        self.node_id = node_id
        self.mode = mode
        self.config = config
        self.is_server = is_server

        self.logger = logger.bind(component="TransparentCommunication", node=node_id)

        # 本节点收到消息队列
        self._queue = _get_queue(self.node_id)

        # handlers: message_type -> callable
        self._handlers: Dict[str, Callable[[Message], None]] = {}

        # response futures: request_id -> queue.Queue
        self._response_queues: Dict[str, "queue.Queue[Any]"] = {}
        self._response_lock = threading.Lock()

        # 后台轮询线程
        self._poll_thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> bool:
        if self._running:
            return True
        self._running = True
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()
        self.logger.info("TransparentCommunication started")
        return True

    def stop(self) -> bool:
        self._running = False
        if self._poll_thread:
            self._poll_thread.join(timeout=1.0)
        self.logger.info("TransparentCommunication stopped")
        return True

    def _poll_loop(self):
        while self._running:
            try:
                # 等待短时间以便快速响应
                msg: Message = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                # 如果是对某个请求的响应，且存在等待队列，则放入响应队列
                request_id = None
                if isinstance(msg.payload, dict):
                    request_id = msg.payload.get("request_id")

                if request_id:
                    with self._response_lock:
                        resp_q = self._response_queues.get(request_id)
                        if resp_q:
                            resp_q.put(msg.payload)

                # 调用注册的 handler（若存在）
                handler = self._handlers.get(msg.message_type)
                if handler:
                    try:
                        handler(msg)
                    except Exception as e:
                        self.logger.error(f"handler error for {msg.message_type}: {e}")
                else:
                    # 未注册 handler, 记录调试信息
                    self.logger.debug(f"No handler for message_type={msg.message_type}, from={msg.sender}")

            except Exception as e:
                self.logger.error(f"_poll_loop error: {e}")

    def send_model_update(self, target: str, message: Dict[str, Any]) -> bool:
        # 兼容分布式实现中用到的高层接口
        msg = Message(
            id=str(uuid.uuid4()),
            sender=self.node_id,
            receiver=target,
            message_type=message.get("message_type", "model_update"),
            payload=message.get("payload", message),
            timestamp=time.time(),
            metadata=message.get("metadata", {}),
        )
        return self.send_message(target, msg)

    def send_message(self, target: str, message: Message) -> bool:
        try:
            q = _get_queue(target)
            q.put(message)
            self.logger.debug(f"sent message {message.message_type} -> {target}")
            return True
        except Exception as e:
            self.logger.error(f"send_message error: {e}")
            return False

    def receive_message(self, timeout: float = 1.0) -> Optional[Message]:
        try:
            msg = self._queue.get(timeout=timeout)
            return msg
        except queue.Empty:
            return None

    def register_handler(self, message_type: str, handler: Callable[[Message], None]) -> None:
        self._handlers[message_type] = handler
        self.logger.debug(f"registered handler for {message_type}")

    def get_stats(self) -> Dict[str, Any]:
        return {
            "node": self.node_id,
            "mode": self.mode.value,
            "queued_messages": _get_queue(self.node_id).qsize()
        }

    def call_learner_method(self, client_id: str, learner_name: str, method_name: str, *args, **kwargs) -> Any:
        """
        基于 request/response 的简易远程方法调用：
        - 发送 message 且包含 request_id
        - 在本地创建响应队列并阻塞等待（带超时）

        注意：这只在同一进程内有效（本适配器用于本地模拟）。
        """
        request_id = str(uuid.uuid4())
        payload = {
            "request_id": request_id,
            "method": method_name,
            "args": args,
            "kwargs": kwargs,
            "learner_name": learner_name
        }
        msg = Message(
            id=request_id,
            sender=self.node_id,
            receiver=client_id,
            message_type="learner_method_call",
            payload=payload,
            timestamp=time.time()
        )

        # 创建响应队列
        resp_q: "queue.Queue[Any]" = queue.Queue()
        with self._response_lock:
            self._response_queues[request_id] = resp_q

        try:
            sent = self.send_message(client_id, msg)
            if not sent:
                raise RuntimeError("failed to send learner_method_call")

            # 等待响应（默认超时 30s）
            try:
                resp = resp_q.get(timeout=30.0)
            except queue.Empty:
                raise TimeoutError("call_learner_method timeout")

            # 响应 payload 约定：{'request_id': id, 'result': <any>, 'error': <str?>}
            if isinstance(resp, dict) and resp.get("error"):
                raise RuntimeError(resp.get("error"))
            return resp.get("result") if isinstance(resp, dict) else resp

        finally:
            with self._response_lock:
                self._response_queues.pop(request_id, None)


# ============ 模块导出 ============
__all__ = [
    "Message",
    "TransparentCommunication",
    "CommunicationMode",
    "NetworkConfig",
    "BaseCommunicationBackend",
]
