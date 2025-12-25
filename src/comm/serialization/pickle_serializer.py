"""
Pickle 序列化器
"""

import pickle
import sys
from typing import Any

from .base import Serializer


class PickleSerializer(Serializer):
    """Pickle 序列化器"""

    def __init__(self):
        super().__init__()
        # 添加模块别名：federation -> src
        # 这样即使序列化时使用 federation.xxx，反序列化时也能找到 src.xxx
        if 'federation' not in sys.modules and 'src' in sys.modules:
            sys.modules['federation'] = sys.modules['src']

    @property
    def name(self) -> str:
        return "pickle"

    def serialize(self, obj: Any) -> bytes:
        return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)

    def deserialize(self, data: bytes) -> Any:
        if not data:
            return None
        return pickle.loads(data)
