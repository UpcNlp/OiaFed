"""
配置类型定义（向后兼容）

注意：此文件是 schema.py 的兼容层。
实际定义位于 schema.py，推荐直接使用：
    from oiafed.config import NodeConfig, ...
    
或：
    from oiafed.config.schema import NodeConfig, ...

此文件保留仅为向后兼容，将来可能被移除。
"""

# 从 schema.py 重导出所有内容
from .schema import *
from .schema import __all__