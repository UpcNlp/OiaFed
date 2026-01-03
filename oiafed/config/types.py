"""
配置类型定义（向后兼容）

注意：此文件是兼容层。
实际定义位于各个模块，推荐直接使用：
    from oiafed.config import NodeConfig, GrpcConfig, ...

此文件保留仅为向后兼容，将来可能被移除。
"""

# 从各模块重导出所有内容
from .transport import *
from .comm import *
from .logging_config import *
from .tracker import *
from .component import *
from .node import *