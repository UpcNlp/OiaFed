"""
序列化器抽象基类
"""

from abc import ABC, abstractmethod
from typing import Any


class Serializer(ABC):
    """序列化器基类"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """序列化器名称"""
        pass
    
    @abstractmethod
    def serialize(self, obj: Any) -> bytes:
        """序列化对象为字节"""
        pass
    
    @abstractmethod
    def deserialize(self, data: bytes) -> Any:
        """反序列化字节为对象"""
        pass
