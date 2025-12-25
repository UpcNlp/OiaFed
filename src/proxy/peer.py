"""
单个对等节点的代理

提供动态方法调用能力
"""

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from node_comm import Node


class PeerProxy:
    """
    单个对等节点的代理

    支持动态方法调用（通过 __getattr__）

    Example:
        proxy = PeerProxy(node, "learner_1")

        # 动态调用任意方法
        result = await proxy.fit({"epochs": 5})
        result = await proxy.evaluate({})
        result = await proxy.custom_method(x, y, z=100)

        # 获取节点 ID
        print(proxy.peer_id)
    """

    def __init__(self, node: "Node", peer_id: str, service_name: str = "default"):
        """
        初始化代理

        Args:
            node: Node 实例
            peer_id: 对等节点 ID
            service_name: 服务名称（用于标识，暂时不做过滤）
        """
        # 使用 object.__setattr__ 避免触发 __setattr__
        object.__setattr__(self, "_node", node)
        object.__setattr__(self, "_peer_id", peer_id)
        object.__setattr__(self, "_service_name", service_name)

    def __getattr__(self, method_name: str):
        """
        动态方法调用

        当访问不存在的属性时，创建一个异步函数来调用远程方法

        Args:
            method_name: 方法名

        Returns:
            异步函数，调用时会通过 node.call() 远程调用

        Example:
            proxy = PeerProxy(node, "learner_1")

            # 调用 fit 方法
            result = await proxy.fit({"epochs": 5})

            # 等价于：
            # result = await node.call(
            #     target="learner_1",
            #     method="fit",
            #     payload={"args": ({"epochs": 5},), "kwargs": {}}
            # )
        """
        # 避免访问内部属性时递归
        if method_name.startswith("_"):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{method_name}'")

        async def _method(*args, **kwargs):
            """远程方法调用包装"""
            # 构造 payload
            payload = {"args": args, "kwargs": kwargs}

            # 获取 Node 实例
            node = object.__getattribute__(self, "_node")

            # 调用单个节点（通过 node_comm）
            # 不显式指定 serializer，让 Node 层根据配置选择
            result = await node.call(
                target=object.__getattribute__(self, "_peer_id"),
                method=method_name,
                payload=payload,
            )
            return result

        return _method

    @property
    def peer_id(self) -> str:
        """获取节点 ID"""
        return object.__getattribute__(self, "_peer_id")

    @property
    def node(self) -> "Node":
        """获取 Node 实例"""
        return object.__getattribute__(self, "_node")

    def __repr__(self) -> str:
        return f"PeerProxy(peer_id='{self.peer_id}')"

    def __str__(self) -> str:
        return f"PeerProxy({self.peer_id})"
