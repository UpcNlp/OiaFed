"""
Pickle 序列化器
"""

import pickle
import sys
from typing import Any

from .base import Serializer


class PickleSerializer(Serializer):
    """Pickle 序列化器"""

    @property
    def name(self) -> str:
        return "pickle"

    def serialize(self, obj: Any) -> bytes:
        """序列化对象为字节流"""
        return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)

    def deserialize(self, data: bytes) -> Any:
        """从字节流反序列化对象"""
        if not data:
            return None
        try:
            return pickle.loads(data)
        except ModuleNotFoundError as e:
            # 调试：分析 pickle 数据
            print(f"=== Pickle 反序列化失败: {e} ===")
            import pickletools
            import io
            output = io.StringIO()
            pickletools.dis(data, output)
            # 只打印包含 federation 的行
            for line in output.getvalue().split('\n'):
                if 'federation' in line.lower():
                    print(line)
            raise