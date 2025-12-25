"""
通信节点（抽象层）

Node 是通信抽象层，负责：
1. 创建和管理 node_comm 的生命周期
2. 封装底层 node_comm 的通信细节
3. 提供统一的 call/broadcast 接口
4. 提供健康状态查询接口

重要：node_comm 是 Node 的内部实现细节，外部组件不应感知
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from node_comm import CommNode

from ..config import NodeCommConfig, TransportConfig


class Node:
    """
    通信节点（抽象层 + node_comm 管理者）

    职责：
    1. 创建和管理 node_comm 的生命周期
    2. 封装底层 node_comm 的通信细节
    3. 提供统一的 call/broadcast 接口
    4. 提供健康状态查询接口

    重要：node_comm 是 Node 的内部实现细节，外部组件不应感知

    Example:
        from federation import load_config
        from federation.core import Node

        config = load_config("configs/trainer.yaml")
        comm_config = config.get_comm_config()
        
        node = Node(comm_config)
        await node.initialize()
        await node.start()
    """

    def __init__(self, config: NodeCommConfig):
        """
        初始化节点

        Args:
            config: NodeCommConfig 实例，包含通信相关配置
        """
        self._config = config
        self._node_id = config.node_id

        # node_comm 实例（延迟创建）
        self._comm: Optional["CommNode"] = None

        # 状态标志
        self._is_initialized = False
        self._is_started = False

    # ========== 属性访问器 ==========

    @property
    def node_id(self) -> str:
        """获取节点 ID"""
        return self._node_id

    @property
    def config(self) -> NodeCommConfig:
        """获取配置对象"""
        return self._config

    @property
    def is_initialized(self) -> bool:
        """检查节点是否已初始化"""
        return self._is_initialized

    @property
    def is_started(self) -> bool:
        """检查节点是否已启动"""
        return self._is_started

    @property
    def transport_mode(self) -> str:
        """获取传输模式"""
        return self._config.transport_mode

    @property
    def listen_port(self) -> Optional[int]:
        """获取监听端口"""
        return self._config.listen_port

    @property
    def listen_host(self) -> str:
        """获取监听地址"""
        return self._config.listen_host

    # ========== 生命周期管理 ==========

    async def initialize(self):
        """
        初始化 Node（创建 node_comm）

        此方法负责：
        1. 根据配置创建 node_comm 实例
        2. 不启动通信（启动在 start() 中）
        """
        if self._is_initialized:
            return

        # 从 node_comm 包导入
        from ..comm import Node as CommNode, NodeConfig as CommNodeConfig
        from ..comm.config import (
            _parse_transport_config,
            _parse_serialization_config,
            _parse_heartbeat_config,
        )

        # 将 TransportConfig 转换为 node_comm 需要的格式
        # *** 关键修复：将 heartbeat 配置传递给 grpc ***
        transport_dict = {
            "mode": self._config.transport.mode,
            "grpc": {
                "host": self._config.transport.grpc.host,
                "port": self._config.transport.grpc.port,
                "max_message_size": self._config.transport.grpc.max_message_size,
                # 添加 heartbeat 配置（包含 critical_peers）
                "heartbeat": self._config.heartbeat or {},
            },
        }

        # 解析配置（将字典转换为 node_comm 配置对象）
        transport_config = _parse_transport_config(transport_dict)
        serialization_config = _parse_serialization_config(
            self._config.serialization or {}
        )
        heartbeat_config = _parse_heartbeat_config(
            self._config.heartbeat or {}
        )

        # 创建 node_comm 的 NodeConfig
        comm_node_config = CommNodeConfig(
            node_id=self._config.node_id,
            debug=self._config.debug,
            default_timeout=self._config.default_timeout,
            advertised_address=self._config.advertised_address,
            listen=self._config.listen,
            transport=transport_config,
            serialization=serialization_config,
            heartbeat=heartbeat_config,
        )

        # 创建 node_comm.Node 实例
        self._comm = CommNode(self._node_id, comm_node_config)

        self._is_initialized = True

    async def start(self):
        """
        启动 Node（启动 node_comm）

        此方法负责：
        1. 启动底层通信（监听端口）
        2. 启动心跳机制
        """
        if not self._is_initialized:
            await self.initialize()

        if self._is_started:
            return

        # 启动 node_comm
        await self._comm.start()

        self._is_started = True

    async def stop(self):
        """
        停止 Node（停止 node_comm）

        此方法负责：
        1. 停止心跳
        2. 断开连接
        3. 关闭监听端口
        """
        if not self._is_started:
            return

        if self._comm:
            await self._comm.stop()

        self._is_started = False

    # ========== 事件系统（转发到 node_comm）==========

    def on(self, event: str, handler):
        """
        注册事件处理器（转发到 node_comm）

        Args:
            event: 事件名称（如 "connect", "disconnect"）
            handler: 事件处理函数
        """
        if not self._is_initialized:
            raise RuntimeError("Node is not initialized. Call initialize() first.")

        self._comm.on(event, handler)

    # ========== 核心通信接口（封装 node_comm）==========

    async def call(
        self,
        target_id: str,
        method: str,
        payload: Any = None,
        timeout: Optional[float] = None,
    ) -> Any:
        """
        调用远程节点的方法

        Args:
            target_id: 目标节点 ID
            method: 方法名
            payload: 请求负载
            timeout: 超时时间（秒）

        Returns:
            远程方法返回值
        """
        if not self._is_started:
            raise RuntimeError("Node is not started. Call start() first.")

        return await self._comm.call(
            target_id,
            method,
            payload,
            timeout=timeout,
        )

    async def broadcast(
        self,
        target_ids: List[str],
        method: str,
        *args,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        广播调用多个节点

        Args:
            target_ids: 目标节点 ID 列表
            method: 方法名
            *args: 位置参数
            timeout: 超时时间
            **kwargs: 关键字参数

        Returns:
            {node_id: result, ...}
        """
        if not self._is_started:
            raise RuntimeError("Node is not started. Call start() first.")

        # 打包参数为payload
        payload = {"args": args, "kwargs": kwargs}

        # 并发调用所有目标节点
        tasks = [
            self.call(target_id, method, payload, timeout=timeout)
            for target_id in target_ids
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        return dict(zip(target_ids, results))

    # ========== 处理器绑定接口 ==========

    def bind(self, handler: Any, name: str = "service"):
        """
        绑定请求处理器（如 Learner 实例）

        Args:
            handler: 服务处理对象
            name: 服务名称（命名空间）
        """
        if not self._is_initialized:
            raise RuntimeError("Node is not initialized. Call initialize() first.")

        self._comm.bind(handler, name=name)

    def register(self, method_name: str, handler: Any):
        """
        注册单个方法处理器

        Args:
            method_name: 方法名
            handler: 方法处理函数
        """
        if not self._is_initialized:
            raise RuntimeError("Node is not initialized. Call initialize() first.")

        self._comm.register(method_name, handler)

    async def connect(
        self,
        target_id: str,
        address: Optional[str] = None,
        retry_config: Optional[Dict[str, Any]] = None,
    ):
        """
        连接到远程节点

        Args:
            target_id: 目标节点 ID
            address: 连接地址（Memory 模式下可选）
            retry_config: 重试配置
        """
        if not self._is_started:
            raise RuntimeError("Node is not started. Call start() first.")

        await self._comm.connect(target_id, address, retry_config=retry_config)

    async def wait_for_connections(self, min_peers: int, timeout: float = 120):
        """
        等待足够数量的对等节点连接

        Args:
            min_peers: 最小连接数
            timeout: 超时时间（秒）
        """
        if not self._is_started:
            raise RuntimeError("Node is not started. Call start() first.")

        await self._comm.wait_for_connections(min_peers, timeout)

    def get_connected_nodes(self) -> List[str]:
        """
        获取已连接的节点 ID 列表

        Returns:
            节点 ID 列表
        """
        if not self._is_started:
            return []

        return self._comm.get_connected_nodes()

    def is_connected(self, peer_id: str) -> bool:
        """检查是否已连接到指定节点"""
        return peer_id in self.get_connected_nodes()

    # ========== 健康状态查询接口 ==========

    def get_peer_health(self, peer_id: str) -> Dict[str, Any]:
        """
        获取对等节点的健康状态

        Args:
            peer_id: 对等节点 ID

        Returns:
            健康信息字典
        """
        if not self._is_started:
            return {"status": "unknown", "reason": "node_not_started"}

        transport = self._comm._transport

        if not hasattr(transport, "get_peer_status"):
            return {"status": "unknown", "reason": "transport_no_heartbeat"}

        peer_status = transport.get_peer_status(peer_id)

        last_hb = 0
        if hasattr(transport, "_state"):
            last_hb = transport._state.get("last_heartbeat", {}).get(peer_id, 0)

        latency = (time.time() - last_hb) * 1000 if last_hb > 0 else -1

        return {
            "status": peer_status.get("health", "unknown"),
            "last_heartbeat": last_hb,
            "latency_ms": latency,
        }

    def get_all_peers_health(self) -> Dict[str, Dict[str, Any]]:
        """获取所有对等节点的健康状态"""
        if not self._is_started:
            return {}

        peers = self.get_connected_nodes()
        return {peer_id: self.get_peer_health(peer_id) for peer_id in peers}

    # ========== 调试信息 ==========

    def get_info(self) -> Dict[str, Any]:
        """获取节点信息"""
        return {
            "node_id": self._node_id,
            "transport_mode": self.transport_mode,
            "listen_port": self.listen_port,
            "listen_host": self.listen_host,
            "is_initialized": self._is_initialized,
            "is_started": self._is_started,
            "connected_peers": self.get_connected_nodes(),
        }

    def __repr__(self) -> str:
        """字符串表示"""
        status = "started" if self._is_started else ("initialized" if self._is_initialized else "created")
        return f"Node(id={self._node_id}, mode={self.transport_mode}, status={status})"
