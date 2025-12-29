"""
论文注册表模块

提供论文定义的加载、查询和配置生成功能。

使用方式:
    from src.papers import get_registry
    
    registry = get_registry()
    
    # 列出所有论文
    papers = registry.list_all()
    
    # 获取论文详情
    paper = registry.get("fot")
    
    # 生成配置
    configs = registry.generate_node_configs("fot", override={"learner": {"lr": 0.001}})
"""

from .loader import (
    # 数据类
    ParamDef,
    PaperDef,
    
    # 注册表
    PaperRegistry,
    get_registry,
    reload_registry,
)


__all__ = [
    "ParamDef",
    "PaperDef",
    "PaperRegistry",
    "get_registry",
    "reload_registry",
]
