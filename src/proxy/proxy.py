"""
远程节点代理（透明访问机制）

Proxy 提供透明的远程访问能力：
- 属性访问：await proxy.num_samples → 远程查询
- 方法调用：await proxy.fit(config) → RPC 调用
- 健康检查：proxy.is_healthy → 本地查询
- 智能缓存：减少重复远程查询
"""

import asyncio
from typing import Any, Dict, Optional, TYPE_CHECKING
import time

if TYPE_CHECKING:
    from ..core.node import Node


class Proxy:
    """
    远程节点的本地代理

    特性：
    1. 透明的属性访问：proxy.num_samples → 远程查询
    2. 透明的方法调用：proxy.fit(config) → RPC 调用
    3. 本地健康检查：proxy.health → 查询 Transport 状态
    4. 智能缓存：减少重复的远程查询

    重要：通过 Node 层进行所有通信，不直接访问 node_comm
    """

    # 本地属性（不需要远程查询）
    _LOCAL_ATTRIBUTES = {
        "_target_id", "_node", "_cache", "_cache_timestamp",
        "_cache_ttl", "_exposed_properties", "_exposed_methods"
    }

    def __init__(self, target_id: str, node: "Node"):
        """
        初始化代理

        Args:
            target_id: 目标节点 ID
            node: 本地 Node 实例（NOT node_comm！）
        """
        object.__setattr__(self, "_target_id", target_id)
        object.__setattr__(self, "_node", node)

        # 缓存机制
        object.__setattr__(self, "_cache", {})
        object.__setattr__(self, "_cache_timestamp", {})
        object.__setattr__(self, "_cache_ttl", 5.0)  # 缓存 5 秒

        # 暴露的属性和方法（首次访问时从远程获取）
        object.__setattr__(self, "_exposed_properties", None)
        object.__setattr__(self, "_exposed_methods", None)

    # ========== 核心机制：透明访问 ==========

    def __getattr__(self, name: str) -> Any:
        """
        属性访问拦截器

        逻辑：
        1. 如果是特殊属性（health），本地处理
        2. 如果在缓存中且未过期，返回缓存
        3. 否则判断是属性还是方法：
           - 属性：返回一个协程，远程查询属性值
           - 方法：返回一个可调用对象，调用时执行 RPC
        """
        # 特殊属性：本地处理
        if name == "health":
            return self._get_health()

        if name == "is_healthy":
            return self._get_health().get("status") == "healthy"

        # 检查缓存
        if self._is_cached(name):
            return self._cache[name]

        # 返回远程访问器
        return _RemoteAttributeAccessor(self, name)

    def __setattr__(self, name: str, value: Any):
        """禁止直接设置属性（防止误用）"""
        if name in self._LOCAL_ATTRIBUTES:
            object.__setattr__(self, name, value)
        else:
            raise AttributeError(
                f"Cannot set attribute '{name}' on Proxy. "
                f"Proxy is read-only for remote attributes."
            )

    # ========== 内部方法：通过 Node 进行远程操作 ==========

    async def _get_remote_property(self, property_name: str) -> Any:
        """
        远程查询属性值

        通过 Node.call() 而非直接访问 node_comm

        Args:
            property_name: 属性名

        Returns:
            属性值
        """
        try:
            # ✅ 通过 Node 层调用
            result = await self._node.call(
                self._target_id,
                "_fed_get_property",
                {"property_name": property_name}
            )

            # 缓存结果
            self._update_cache(property_name, result)

            return result
        except Exception as e:
            raise AttributeError(
                f"Failed to get remote property '{property_name}' "
                f"from {self._target_id}: {e}"
            )

    async def _call_remote_method(self, method_name: str, *args, **kwargs) -> Any:
        """
        远程调用方法

        通过 Node.call() 而非直接访问 node_comm

        Args:
            method_name: 方法名
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            方法返回值
        """
        try:
            # 打包 payload
            payload = {"args": args, "kwargs": kwargs}

            # ✅ 通过 Node 层调用
            return await self._node.call(
                target_id=self._target_id,
                method=method_name,
                payload=payload,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to call remote method '{method_name}' "
                f"on {self._target_id}: {e}"
            )

    # ========== 本地查询：通过 Node 查询健康状态 ==========

    def _get_health(self) -> Dict[str, Any]:
        """
        获取健康状态（通过 Node，而非直接访问 Transport）

        Returns:
            健康信息字典：
            {
                "status": "healthy" / "unhealthy" / "unknown",
                "last_heartbeat": timestamp,
                "latency_ms": float
            }
        """
        # ✅ 通过 Node 提供的接口
        return self._node.get_peer_health(self._target_id)

    # ========== 缓存管理 ==========

    def _is_cached(self, name: str) -> bool:
        """检查缓存是否有效"""
        if name not in self._cache:
            return False

        timestamp = self._cache_timestamp.get(name, 0)
        return (time.time() - timestamp) < self._cache_ttl

    def _update_cache(self, name: str, value: Any):
        """更新缓存"""
        self._cache[name] = value
        self._cache_timestamp[name] = time.time()

    def invalidate_cache(self, name: Optional[str] = None):
        """
        使缓存失效

        Args:
            name: 属性名（None 表示清空所有缓存）
        """
        if name is None:
            self._cache.clear()
            self._cache_timestamp.clear()
        else:
            self._cache.pop(name, None)
            self._cache_timestamp.pop(name, None)

    # ========== 便捷方法 ==========

    def __repr__(self) -> str:
        return f"Proxy(target={self._target_id})"


class _RemoteAttributeAccessor:
    """
    远程属性访问器（内部类）

    用于延迟判断属性访问还是方法调用

    使用场景：
    1. 属性访问：await proxy.num_samples → 触发 __await__()
    2. 方法调用：await proxy.fit(config) → 触发 __call__()
    """

    def __init__(self, proxy: Proxy, name: str):
        self._proxy = proxy
        self._name = name

    def __await__(self):
        """
        支持 await 语法（用于属性访问）

        使用方式：
            value = await proxy.num_samples  # ← 触发此方法
        """
        return self._proxy._get_remote_property(self._name).__await__()

    def __call__(self, *args, **kwargs):
        """
        支持函数调用语法（用于方法调用）

        使用方式：
            result = await proxy.fit(config)  # ← 触发此方法
        """
        return self._proxy._call_remote_method(self._name, *args, **kwargs)

    def __repr__(self) -> str:
        return f"<RemoteAttribute '{self._name}' of {self._proxy._target_id}>"
