"""
JSON 序列化器
"""

import json
from typing import Any

from .base import Serializer


class JsonSerializer(Serializer):
    """JSON 序列化器"""
    
    @property
    def name(self) -> str:
        return "json"
    
    def serialize(self, obj: Any) -> bytes:
        return json.dumps(obj, ensure_ascii=False).encode("utf-8")
    
    def deserialize(self, data: bytes) -> Any:
        if not data:
            return None
        try:
            return json.loads(data.decode("utf-8"))
        except UnicodeDecodeError as e:
            # 如果无法解码为UTF-8，可能是错误地传入了pickle数据
            raise ValueError(
                f"无法将数据解码为UTF-8。这可能意味着数据是用pickle序列化的，"
                f"但被错误地当作JSON处理。请检查序列化器配置。"
                f"原始错误: {e}"
            )
