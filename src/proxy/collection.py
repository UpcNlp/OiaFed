"""
Proxy 集合（动态版）

提供：
- 单个访问（索引、迭代）
- 广播访问
- 健康过滤
- 状态查询
- 动态更新（监听连接变化）
"""

from typing import List, Dict, Any, Optional, TYPE_CHECKING
import asyncio

if TYPE_CHECKING:
    from ..core.node import Node

from .proxy import Proxy


class ProxyCollection:
    """
    代理集合（动态版）

    特性：
    1. 动态更新：监听 Node 的 connect/disconnect 事件
    2. 状态管理：标记可用/不可用，不删除代理
    3. 健康过滤：get_healthy_proxies()
    4. 状态查询：get_all_states()
    5. 广播调用：broadcast()

    设计原则：
    - 失联的客户端不删除，只标记为不可用
    - 支持客户端动态加入（自动创建代理）
    - 支持客户端重新连接（恢复可用状态）
    """

    def __init__(self, node: "Node", target_ids: List[str], auto_discover: bool = True):
        """
        初始化代理集合

        Args:
            node: 本地 Node 实例
            target_ids: 初始目标节点 ID 列表
            auto_discover: 是否自动发现新连接的节点（默认 True）
        """
        self._node = node
        self._auto_discover = auto_discover

        # 代理字典：{node_id: Proxy}
        self._proxies: Dict[str, Proxy] = {}

        # 可用状态：{node_id: bool}
        self._available: Dict[str, bool] = {}

        # 初始化代理
        for tid in target_ids:
            self._proxies[tid] = Proxy(tid, node)
            self._available[tid] = True  # 假设初始都可用

        # 注册事件监听器（监听连接变化）
        if auto_discover:
            self._register_event_handlers()

    # ========== 事件处理 ==========

    def _register_event_handlers(self):
        """注册 Node 事件监听器"""
        self._node.on("connect", self._on_peer_connect)
        self._node.on("disconnect", self._on_peer_disconnect)

    def _on_peer_connect(self, peer_id: str, address: str):
        """
        处理节点连接事件

        Args:
            peer_id: 连接的节点 ID
            address: 连接地址
        """
        if peer_id in self._proxies:
            # 已存在，标记为可用
            self._available[peer_id] = True
        else:
            # 新节点，创建代理
            self._proxies[peer_id] = Proxy(peer_id, self._node)
            self._available[peer_id] = True

    def _on_peer_disconnect(self, peer_id: str, reason: str):
        """
        处理节点断开事件

        Args:
            peer_id: 断开的节点 ID
            reason: 断开原因
        """
        # 不删除代理，只标记为不可用
        if peer_id in self._proxies:
            self._available[peer_id] = False

    # ========== 基本访问 ==========

    def __iter__(self):
        """迭代访问所有代理（包括不可用的）"""
        return iter(self._proxies.values())

    def __len__(self):
        """返回集合大小（所有代理，包括不可用的）"""
        return len(self._proxies)

    def __getitem__(self, key):
        """
        索引访问

        Args:
            key: 可以是整数索引或节点ID字符串
        """
        if isinstance(key, int):
            # 整数索引：按插入顺序访问
            return list(self._proxies.values())[key]
        elif isinstance(key, str):
            # 字符串：按节点ID访问
            return self._proxies[key]
        else:
            raise TypeError(f"Index must be int or str, not {type(key)}")

    # ========== 动态管理 ==========

    def add_proxy(self, node_id: str, available: bool = True) -> Proxy:
        """
        手动添加代理

        Args:
            node_id: 节点 ID
            available: 是否可用（默认 True）

        Returns:
            创建的 Proxy 实例
        """
        if node_id not in self._proxies:
            self._proxies[node_id] = Proxy(node_id, self._node)
        self._available[node_id] = available
        return self._proxies[node_id]

    def mark_available(self, node_id: str):
        """标记节点为可用"""
        if node_id in self._proxies:
            self._available[node_id] = True

    def mark_unavailable(self, node_id: str):
        """标记节点为不可用"""
        if node_id in self._proxies:
            self._available[node_id] = False

    def is_available(self, node_id: str) -> bool:
        """检查节点是否可用"""
        return self._available.get(node_id, False)

    def get_available_proxies(self) -> List[Proxy]:
        """获取所有可用的代理"""
        return [
            proxy for node_id, proxy in self._proxies.items()
            if self._available.get(node_id, False)
        ]

    def get_unavailable_proxies(self) -> List[Proxy]:
        """获取所有不可用的代理"""
        return [
            proxy for node_id, proxy in self._proxies.items()
            if not self._available.get(node_id, False)
        ]

    def get_all_proxies(self) -> List[Proxy]:
        """获取所有代理（包括不可用的）"""
        return list(self._proxies.values())

    def get_proxy_ids(self) -> List[str]:
        """获取所有代理的节点 ID"""
        return list(self._proxies.keys())

    def get_available_ids(self) -> List[str]:
        """获取所有可用代理的节点 ID"""
        return [
            node_id for node_id in self._proxies.keys()
            if self._available.get(node_id, False)
        ]

    # ========== 广播调用 ==========

    async def broadcast(
        self,
        method: str,
        *args,
        only_available: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        广播调用代理

        通过 Node.broadcast() 而非直接访问 node_comm

        Args:
            method: 方法名
            *args: 位置参数
            only_available: 是否只调用可用的代理（默认 True）
            **kwargs: 关键字参数（可以包含timeout）

        Returns:
            {node_id: result, ...}
        """
        # 提取timeout参数（如果存在）
        timeout = kwargs.pop('timeout', None)

        # 选择目标代理
        if only_available:
            target_ids = self.get_available_ids()
        else:
            target_ids = list(self._proxies.keys())

        if not target_ids:
            return {}

        # 通过 Node 层广播
        return await self._node.broadcast(
            target_ids,
            method,
            *args,
            timeout=timeout,
            **kwargs
        )

    # ========== 健康过滤 ==========

    def get_healthy_proxies(self, only_available: bool = True) -> List[Proxy]:
        """
        获取所有健康的代理

        通过每个 Proxy 的 is_healthy 属性（内部查询 Node）

        Args:
            only_available: 是否只返回可用的代理（默认 True）

        Returns:
            健康的代理列表
        """
        healthy = []
        proxies = self.get_available_proxies() if only_available else self.get_all_proxies()

        for proxy in proxies:
            # Proxy 内部通过 self._node.get_peer_health() 查询
            if proxy.is_healthy:
                healthy.append(proxy)
        return healthy

    # ========== 状态查询 ==========

    async def get_all_states(self, only_available: bool = False) -> Dict[str, str]:
        """
        获取所有代理的状态（并发查询）

        每个 Proxy 内部通过 self._node.call() 查询

        Args:
            only_available: 是否只查询可用的代理（默认 False）

        Returns:
            {proxy_id: state, ...}
        """
        async def _get_state(proxy):
            try:
                return await proxy.state
            except Exception as e:
                return f"error: {e}"

        proxies = self.get_available_proxies() if only_available else self.get_all_proxies()
        tasks = [_get_state(p) for p in proxies]
        states = await asyncio.gather(*tasks)

        return {
            proxy._target_id: state
            for proxy, state in zip(proxies, states)
        }

    def get_proxies_by_state(self, state: str, only_available: bool = True) -> List[Proxy]:
        """
        按状态过滤代理（需要远程调用）

        Args:
            state: 目标状态（idle / training / evaluating）
            only_available: 是否只从可用代理中筛选（默认 True）

        Returns:
            处于该状态的代理列表
        """
        async def _check_state(proxy):
            try:
                proxy_state = await proxy.state
                return proxy_state == state
            except Exception:
                return False

        # 获取要检查的代理
        proxies = self.get_available_proxies() if only_available else self.get_all_proxies()

        # 需要在异步上下文中运行
        loop = asyncio.get_event_loop()
        tasks = [_check_state(p) for p in proxies]

        if loop.is_running():
            # 如果已经在事件循环中，使用 gather
            results = asyncio.create_task(asyncio.gather(*tasks))
        else:
            # 否则使用 run_until_complete
            results = loop.run_until_complete(asyncio.gather(*tasks))

        return [
            proxy for proxy, is_match in zip(proxies, results)
            if is_match
        ]

    # ========== 统计信息 ==========

    def get_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
        available_count = len(self.get_available_proxies())
        unavailable_count = len(self.get_unavailable_proxies())

        return {
            "total": len(self._proxies),
            "available": available_count,
            "unavailable": unavailable_count,
            "node_ids": list(self._proxies.keys()),
            "available_ids": self.get_available_ids(),
        }

    # ========== 辅助方法 ==========

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"ProxyCollection(total={stats['total']}, "
            f"available={stats['available']}, "
            f"unavailable={stats['unavailable']})"
        )

    def __str__(self) -> str:
        stats = self.get_stats()
        return (
            f"ProxyCollection: {stats['available']}/{stats['total']} available"
        )
