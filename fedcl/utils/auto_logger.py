"""
简化的日志自动分流系统
utils/auto_logger.py
使用同一个logger实例，通过loguru的过滤器功能自动分流到不同文件：
- 通信日志：comm/
- 训练日志：train/  
- 系统日志：sys/

使用方法：
```python
from fedcl.utils.auto_logger import setup_auto_logging, get_logger

# 设置自动分流日志
setup_auto_logging()

# 获取logger并添加标记
comm_logger = get_logger("comm", "client_1")
train_logger = get_logger("train", "client_1") 

# 使用
comm_logger.info("连接建立")  # -> comm/client_1.log
train_logger.info("开始训练")  # -> train/client_1.log
```
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger


class AutoLogger:
    """自动分流日志记录器"""
    
    def __init__(self, experiment_date: Optional[str] = None, base_path: str = "logs"):
        self.base_path = Path(base_path)
        self.experiment_date = experiment_date or datetime.now().strftime("%Y%m%d-%H-%M-%S")
        
        # 创建目录结构
        self.exp_dir = self.base_path / f"exp_{self.experiment_date}"
        self.comm_dir = self.exp_dir / "comm"
        self.train_dir = self.exp_dir / "train"
        self.sys_dir = self.exp_dir / "sys"
        
        for dir_path in [self.comm_dir, self.train_dir, self.sys_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 记录已设置的处理器
        self._handlers: Dict[str, int] = {}
        self._setup_handlers()
        
        logger.info(f"自动分流日志系统已初始化: {self.exp_dir}")
    
    def _setup_handlers(self):
        """设置所有日志处理器"""
        # 通信日志处理器 - 统一到sys.log，稍后按节点过滤
        comm_handler = logger.add(
            self.comm_dir / "sys.log",
            format="{time} | {level} | {name}:{function}:{line} - {message}",
            level="DEBUG",
            rotation="10 MB",
            retention="30 days",
            compression="zip",
            filter=lambda record: record.get("extra", {}).get("log_type") == "comm"
        )
        self._handlers['comm'] = comm_handler
        
        # 训练日志处理器 - 统一到sys.log，稍后按节点过滤  
        train_handler = logger.add(
            self.train_dir / "sys.log",
            format="{time} | {level} | {name}:{function}:{line} - {message}",
            level="DEBUG",
            rotation="10 MB",
            retention="30 days",
            compression="zip",
            filter=lambda record: record.get("extra", {}).get("log_type") == "train"
        )
        self._handlers['train'] = train_handler
        
        # 系统日志处理器 - 统一文件
        sys_handler = logger.add(
            self.sys_dir / "system.log",
            format="{time} | {level} | {name}:{function}:{line} - {message}",
            level="DEBUG",
            rotation="10 MB",
            retention="30 days",
            compression="zip",
            filter=lambda record: record.get("extra", {}).get("log_type") == "sys"
        )
        self._handlers['sys'] = sys_handler
    
    def get_logger(self, log_type: str, node_id: str):
        """获取带标记的logger"""
        # 为新节点创建专用的日志文件处理器
        handler_key = f"{log_type}_{node_id}"
        if handler_key not in self._handlers:
            if log_type == "comm":
                log_file = self.comm_dir / f"{node_id}.log"
            elif log_type == "train":
                log_file = self.train_dir / f"{node_id}.log"
            else:
                log_file = self.sys_dir / f"{node_id}.log"
            
            # 为该节点创建专用的文件处理器
            handler_id = logger.add(
                log_file,
                format="{time} | {level} | {name}:{function}:{line} - {message}",
                level="DEBUG",
                rotation="10 MB",
                retention="30 days",
                compression="zip",
                filter=lambda record: (
                    record.get("extra", {}).get("log_type") == log_type and
                    record.get("extra", {}).get("node_id") == node_id
                )
            )
            self._handlers[handler_key] = handler_id
        
        return logger.bind(log_type=log_type, node_id=node_id)
    
    def cleanup(self):
        """清理日志处理器"""
        for handler_id in self._handlers.values():
            logger.remove(handler_id)
        self._handlers.clear()


# 全局日志管理器实例
_auto_logger: Optional[AutoLogger] = None


def setup_auto_logging(experiment_date: Optional[str] = None) -> AutoLogger:
    """设置自动分流日志系统"""
    global _auto_logger
    if _auto_logger is None:
        _auto_logger = AutoLogger(experiment_date)
    return _auto_logger


def get_logger(log_type: str, node_id: str):
    """获取指定类型和节点的logger
    
    Args:
        log_type: 日志类型 ('comm', 'train', 'sys')
        node_id: 节点ID
    
    Returns:
        绑定了标记的logger实例
    """
    if _auto_logger is None:
        raise RuntimeError("请先调用 setup_auto_logging() 初始化日志系统")
    
    return _auto_logger.get_logger(log_type, node_id)


def cleanup_logging():
    """清理日志系统"""
    global _auto_logger
    if _auto_logger:
        _auto_logger.cleanup()
        _auto_logger = None


# 便捷函数
def get_comm_logger(node_id: str):
    """获取通信日志记录器"""
    return get_logger("comm", node_id)


def get_train_logger(node_id: str):
    """获取训练日志记录器"""
    return get_logger("train", node_id)


def get_sys_logger():
    """获取系统日志记录器"""
    return get_logger("sys", "system")
