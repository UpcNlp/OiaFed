"""
序列化器注册表
"""

from typing import Dict, Optional

from .base import Serializer
from .json_serializer import JsonSerializer
from .pickle_serializer import PickleSerializer
from ..config import SerializationConfig


class SerializerRegistry:
    """序列化器注册表"""
    
    def __init__(self, config: Optional[SerializationConfig] = None):
        self.config = config or SerializationConfig()
        self._serializers: Dict[str, Serializer] = {}
        
        # 注册内置序列化器
        self.register(JsonSerializer())
        self.register(PickleSerializer())
    
    def register(self, serializer: Serializer) -> None:
        """注册序列化器"""
        self._serializers[serializer.name] = serializer
    
    def get(self, name: str) -> Serializer:
        """获取序列化器"""
        if name not in self._serializers:
            raise ValueError(f"Unknown serializer: {name}")
        return self._serializers[name]
    
    def get_for_method(self, method: str) -> Serializer:
        """
        根据方法名获取序列化器
        
        查找优先级：
        1. 精确匹配方法名
        2. 通配符 "*"
        3. 默认序列化器
        """
        # 1. 精确匹配
        method_config = self.config.methods.get(method)
        if method_config:
            return self.get(method_config.serializer)
        
        # 2. 通配符匹配
        wildcard_config = self.config.methods.get("*")
        if wildcard_config:
            return self.get(wildcard_config.serializer)
        
        # 3. 默认
        return self.get(self.config.default)
    
    def has(self, name: str) -> bool:
        """检查序列化器是否存在"""
        return name in self._serializers
    
    def list(self) -> list:
        """列出所有已注册的序列化器名称"""
        return list(self._serializers.keys())
