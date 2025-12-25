"""
内存传输层实现
"""

import asyncio
import copy
from typing import ClassVar, Dict, Set, Optional, List, Any, TYPE_CHECKING

from .base import Transport
from ..config import MemoryTransportConfig
from ..message import Message
from ..exceptions import NodeNotConnectedError
from .retry import connect_with_retry, create_retry_config, RetryConfig

if TYPE_CHECKING:
    from ..node import Node

# 使用 loguru 统一日志（会继承调用方的 node_id 和 log_type）
from ...infra.logging import get_logger
# logger 将在 __init__ 中绑定 node_id


class MemoryTransport(Transport):
    """
    内存传输层
    
    特点：
    - 同进程内直接调用
    - 无序列化开销（可选）
    - 用于单元测试和算法调试
    """
    
    # 全局节点注册表
    _registry: ClassVar[Dict[str, "MemoryTransport"]] = {}
    
    def __init__(self, node: "Node", config: Optional[MemoryTransportConfig] = None):
        self.node = node
        self.config = config or MemoryTransportConfig()
        self.node_id = node.node_id
        self._connections: Set[str] = set()
        self._running = False

        # 创建绑定了 node_id 的 logger
        self.logger = get_logger(node_id=self.node_id, log_type="system")
    
    async def start(self) -> None:
        """启动传输层，注册到全局注册表"""
        MemoryTransport._registry[self.node_id] = self
        self._running = True
        self.logger.info(f"MemoryTransport started for node '{self.node_id}'")
    
    async def stop(self) -> None:
        """停止传输层，从全局注册表移除"""
        self._running = False
        MemoryTransport._registry.pop(self.node_id, None)
        self._connections.clear()
        self.logger.info(f"MemoryTransport stopped for node '{self.node_id}'")
    
    async def send(self, message: Message) -> None:
        """发送消息到目标节点"""
        # DEBUG: 记录发送消息
        self.logger.debug(f"[MemoryTransport:{self.node_id}] send: target={message.target}, method={message.method}, msg_id={message.id}")

        target = MemoryTransport._registry.get(message.target)
        if not target:
            self.logger.error(f"[MemoryTransport:{self.node_id}] 目标节点不在注册表: target={message.target}, registry={list(MemoryTransport._registry.keys())}")
            raise NodeNotConnectedError(message.target)

        self.logger.debug(f"[MemoryTransport:{self.node_id}] 找到目标节点: target={message.target}")

        # 可选：深拷贝以模拟网络行为
        if not self.config.zero_copy:
            message = copy.deepcopy(message)

        # 可选：模拟网络延迟
        if self.config.simulate_latency and self.config.latency_ms > 0:
            await asyncio.sleep(self.config.latency_ms / 1000)

        # 直接调用目标节点的消息处理
        self.logger.debug(f"[MemoryTransport:{self.node_id}] 调用目标节点的 on_message: target={message.target}")
        response = await target.node.on_message(message)
        self.logger.debug(f"[MemoryTransport:{self.node_id}] 目标节点返回响应: target={message.target}, has_response={response is not None}")

        # 如果是 REQUEST，将响应发回
        if response:
            # 响应也可能需要深拷贝
            if not self.config.zero_copy:
                response = copy.deepcopy(response)
            self.logger.debug(f"[MemoryTransport:{self.node_id}] 发送响应回本地节点: correlation_id={response.correlation_id}")
            await self.node.on_message(response)
    
    async def connect(
        self,
        node_id: str,
        address: Optional[str] = None,
        retry_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        连接到目标节点（支持重试）

        Args:
            node_id: 目标节点 ID
            address: 地址（Memory 模式忽略）
            retry_config: 重试配置字典
        """
        # 创建重试配置
        retry_cfg = create_retry_config(retry_config)

        # 使用重试逻辑连接
        await connect_with_retry(
            self._connect_once,
            node_id,
            address,
            retry_cfg,
            self.logger
        )

    async def _connect_once(self, node_id: str, address: Optional[str] = None) -> None:
        """单次连接尝试（内部方法）"""
        if node_id not in MemoryTransport._registry:
            raise NodeNotConnectedError(f"Node '{node_id}' not found in registry, all nodes: {list(MemoryTransport._registry.keys())}")

        if node_id in self._connections:
            self.logger.debug(f"Already connected to '{node_id}'")
            return

        self._connections.add(node_id)
        await self.node._on_connect(node_id, None)

        # 双向连接：通知对方也建立连接
        other = MemoryTransport._registry[node_id]
        if self.node_id not in other._connections:
            other._connections.add(self.node_id)
            await other.node._on_connect(self.node_id, None)

        self.logger.debug(f"Node '{self.node_id}' connected to '{node_id}'")
    
    async def disconnect(self, node_id: str) -> None:
        """断开与目标节点的连接"""
        if node_id not in self._connections:
            return
        
        self._connections.discard(node_id)
        await self.node._on_disconnect(node_id, "manual disconnect")
        
        # 双向断开
        other = MemoryTransport._registry.get(node_id)
        if other and self.node_id in other._connections:
            other._connections.discard(self.node_id)
            await other.node._on_disconnect(self.node_id, "peer disconnect")
        
        self.logger.debug(f"Node '{self.node_id}' disconnected from '{node_id}'")
    
    def is_connected(self, node_id: str) -> bool:
        """检查是否已连接到指定节点"""
        return node_id in self._connections

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
        广播调用多个目标节点（串行调用）

        Memory 模式下无真正的广播，逐个串行调用
        注意: Memory 模式下 wait_for_all 和 min_responses 参数被忽略（总是等待所有节点）

        Args:
            targets: 目标节点 ID 列表
            method: 方法名
            payload: 参数（字典格式）
            timeout: 超时时间（秒）
            wait_for_all: 是否等待所有节点返回（Memory 模式下忽略，总是 True）
            min_responses: 最少需要的响应数（Memory 模式下忽略）

        Returns:
            结果字典 {node_id: result}
            如果某个节点调用失败，对应的 value 为 Exception 对象
        """
        results = {}

        for target_id in targets:
            try:
                # 串行调用单个节点
                result = await self._call_single(target_id, method, payload, timeout)
                results[target_id] = result
            except Exception as e:
                # 记录异常并抛出，显示完整堆栈跟踪
                self.logger.exception(f"Broadcast to {target_id} failed,error: {e}, method: {method}, payload: {payload.keys()}")
                # 记录异常到结果中，继续下一个
                results[target_id] = e
                # 第一个失败就抛出异常
                raise e

        return results

    async def _call_single(
        self,
        target_id: str,
        method: str,
        payload: Any,
        timeout: Optional[float] = None,
    ) -> Any:
        """
        单个节点调用（内部方法）

        Args:
            target_id: 目标节点 ID
            method: 方法名
            payload: 参数
            timeout: 超时时间

        Returns:
            调用结果
        """
        # 检查连接
        if not self.is_connected(target_id):
            raise NodeNotConnectedError(target_id)

        # 获取目标节点
        target = MemoryTransport._registry.get(target_id)
        if not target:
            raise NodeNotConnectedError(target_id)

        # 构造 REQUEST 消息
        from ..message import Message, MessageType
        import uuid
        import time
        import pickle

        # Memory 模式：需要序列化 payload（使用 pickle）
        # 虽然是内存传输，但 Node 层期望 payload 是 bytes
        try:
            payload_bytes = pickle.dumps(payload)
        except Exception as e:
            self.logger.error(f"Failed to serialize payload for method={method}: {e}")
            raise

        message = Message(
            id=str(uuid.uuid4()),
            type=MessageType.REQUEST,
            source=self.node_id,
            target=target_id,
            method=method,
            payload=payload_bytes,  # 序列化后的字节
            correlation_id=None,
            timestamp=time.time_ns(),
            metadata={},
            error=None,
        )

        # 可选：深拷贝
        if not self.config.zero_copy:
            message = copy.deepcopy(message)

        # 可选：模拟网络延迟
        if self.config.simulate_latency and self.config.latency_ms > 0:
            await asyncio.sleep(self.config.latency_ms / 1000)

        # 直接调用目标节点的消息处理
        response = await target.node.on_message(message)

        if response:
            # 可选：深拷贝响应
            if not self.config.zero_copy:
                response = copy.deepcopy(response)

            # 检查是否有错误
            if response.type == MessageType.RESPONSE:
                # 反序列化 payload（使用 pickle）
                import pickle
                if isinstance(response.payload, bytes):
                    if len(response.payload) == 0:
                        # 空 payload，说明返回值为 None 或序列化失败
                        self.logger.error(f"Empty payload in response for method={method}")
                        raise EOFError(f"Empty response payload for method={method}")
                    try:
                        return pickle.loads(response.payload)
                    except EOFError as e:
                        self.logger.error(f"Failed to unpickle response for method={method}, payload_len={len(response.payload)}")
                        raise
                return response.payload
            elif response.type == MessageType.ERROR:
                # 抛出远程异常
                from ..exceptions import RemoteExecutionError
                raise RemoteExecutionError(
                    response.payload.get("message", "Remote execution failed"),
                    response.payload.get("code", "REMOTE_ERROR"),
                )

        return None
    
    @classmethod
    def clear_registry(cls) -> None:
        """清空全局注册表（用于测试）"""
        cls._registry.clear()
