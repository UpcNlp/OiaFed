"""
传输层抽象基类
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..message import Message


class Transport(ABC):
    """传输层抽象基类"""
    
    @abstractmethod
    async def start(self) -> None:
        """启动传输层"""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """停止传输层"""
        pass
    
    @abstractmethod
    async def send(self, message: "Message") -> None:
        """发送消息"""
        pass
    
    @abstractmethod
    async def connect(self, node_id: str, address: Optional[str] = None, retry_config: Optional[Dict[str, Any]] = None) -> None:
        """连接到目标节点"""
        pass
    
    @abstractmethod
    async def disconnect(self, node_id: str) -> None:
        """断开与目标节点的连接"""
        pass
    
    @abstractmethod
    def is_connected(self, node_id: str) -> bool:
        """检查是否已连接到指定节点"""
        pass

    @abstractmethod
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
        广播调用多个目标节点

        根据传输模式自动选择实现：
        - Memory 模式：串行调用（转换为多个单独调用）
        - gRPC 模式：并行调用（真正的网络广播）

        Args:
            targets: 目标节点 ID 列表
            method: 方法名
            payload: 参数（字典格式）
            timeout: 超时时间（秒）
            wait_for_all: 是否等待所有节点返回，False 时结合 min_responses 使用
            min_responses: 最少需要的响应数，达到后即可返回（仅在 wait_for_all=False 时生效）

        Returns:
            结果字典 {node_id: result}
            如果某个节点调用失败，对应的 value 为 Exception 对象
        """
        pass
