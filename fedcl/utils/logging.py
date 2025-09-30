"""
MOE-FedCL 日志工具
moe_fedcl/utils/logging.py
"""

from .auto_logger import get_logger as _get_auto_logger
import logging


def get_logger(name: str, log_type: str = "sys"):
    """获取日志记录器
    
    Args:
        name: logger名称 
        log_type: 日志类型 (sys/comm/train)
        
    Returns:
        Logger: 日志记录器实例
    """
    try:
        # 尝试使用auto_logger
        return _get_auto_logger(log_type, name)
    except Exception:
        # fallback到标准logging
        return logging.getLogger(name)
