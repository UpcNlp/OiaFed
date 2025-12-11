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

    def __init__(
        self,
        experiment_date: Optional[str] = None,
        base_path: str = "logs",
        experiment_name: Optional[str] = None,
        node_role: Optional[str] = None,
        node_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        experiment_config: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            experiment_date: 实验日期标识（自动生成时间戳）
            base_path: 日志基础路径（默认 "logs"）
            experiment_name: 实验名称（用于目录名）
            node_role: 节点角色（"server" 或 "client"）
            node_id: 节点ID
            config: 日志配置字典，支持以下键：
                - console_enabled: 是否启用控制台输出（默认True）
                - console_level: 控制台日志级别（默认INFO）
                - file_level: 文件日志级别（默认DEBUG）
                - format: 日志格式
                - rotation: 日志轮转大小
                - retention: 日志保留时间
                - compression: 压缩格式
            experiment_config: 实验配置字典（数据集、算法、模型等配置信息）
        """
        self.base_path = Path(base_path)
        self.experiment_name = experiment_name or "experiment"
        self.node_role = node_role  # "server" or "client"
        self.node_id = node_id
        self.timestamp = experiment_date or datetime.now().strftime("%Y%m%d_%H%M%S")

        # 合并默认配置
        default_config = {
            'console_enabled': True,
            'console_level': 'INFO',     # 控制台日志级别
            'file_level': 'DEBUG',       # 文件日志级别
            'format': '{time} | {level} | {name}:{function}:{line} - {message}',
            'rotation': '10 MB',
            'retention': '30 days',
            'compression': 'zip'
        }
        self.config = {**default_config, **(config or {})}

        # 创建新的三文件夹结构: runtime/, training/, configs/
        self.run_dir = self.base_path / self.experiment_name / f"run_{self.timestamp}"
        self.exp_dir = self.run_dir  # 保持兼容性

        self.runtime_dir = self.run_dir / "runtime"
        self.training_dir = self.run_dir / "training"
        self.configs_dir = self.run_dir / "configs"

        # 创建目录
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.training_dir.mkdir(parents=True, exist_ok=True)
        self.configs_dir.mkdir(parents=True, exist_ok=True)

        # 用于存储每个节点的 handler
        self._node_handlers: Dict[str, Dict[str, int]] = {}  # {node_id: {runtime: id, training: id}}

        # 记录已设置的处理器
        self._handlers: Dict[str, int] = {}

        # 如果禁用控制台输出，移除 loguru 的默认 handler
        if not self.config['console_enabled']:
            logger.remove()  # 移除所有默认 handler（包括控制台输出）

        self._setup_handlers()

        logger.info(f"日志系统已初始化: {self.exp_dir} (控制台: {self.config['console_enabled']}, 控制台级别: {self.config['console_level']}, 文件级别: {self.config['file_level']})")
    
    def _setup_handlers(self):
        """设置日志处理器 - 不再自动添加，改为按需添加"""
        # 不再自动添加全局 handler，改为在 get_logger() 中按节点添加
        pass

    def get_logger(self, log_type: str = "sys", node_id: Optional[str] = None):
        """获取 logger，支持节点分离和日志类型分离

        Args:
            log_type: 日志类型 ("sys"/"runtime" 或 "train"/"training")
            node_id: 节点ID，如果提供则为该节点创建独立日志文件

        Returns:
            logger 实例（带有 node_id 和 log_type 上下文）
        """
        # 如果没有提供 node_id，返回全局 logger
        if not node_id:
            return logger

        # 标准化 log_type
        if log_type in ("sys", "comm", "runtime"):
            log_type = "runtime"
        elif log_type in ("train", "training"):
            log_type = "training"
        else:
            log_type = "runtime"  # 默认为 runtime

        # 如果该节点的该类型日志还没有 handler，创建一个
        if node_id not in self._node_handlers:
            self._node_handlers[node_id] = {}

        if log_type not in self._node_handlers[node_id]:
            # 确定日志文件路径
            if log_type == "runtime":
                log_file = self.runtime_dir / f"{node_id}.log"
            else:  # training
                log_file = self.training_dir / f"{node_id}.log"

            # 添加该节点的 handler（带 filter）
            handler_id = logger.add(
                log_file,
                format=self.config['format'],
                level=self.config['file_level'],
                filter=lambda record, nid=node_id, lt=log_type: (
                    record["extra"].get("node_id") == nid and
                    record["extra"].get("log_type") == lt
                ),
                rotation=self.config['rotation'],
                retention=self.config['retention'],
                compression=self.config['compression']
            )
            self._node_handlers[node_id][log_type] = handler_id

        # 返回绑定了 node_id 和 log_type 的 logger
        return logger.bind(node_id=node_id, log_type=log_type)
    
    def cleanup(self):
        """清理日志处理器"""
        # 清理所有 handler
        for handler_id in self._handlers.values():
            logger.remove(handler_id)
        self._handlers.clear()

        # 清理节点 handler
        for node_handlers in self._node_handlers.values():
            for handler_id in node_handlers.values():
                logger.remove(handler_id)
        self._node_handlers.clear()

    def save_config_snapshot(self, node_id: str, config_dict: dict):
        """保存配置快照到 configs/ 目录

        Args:
            node_id: 节点ID（如 "server", "client_0"）
            config_dict: 配置字典
        """
        import yaml
        config_file = self.configs_dir / f"{node_id}.yaml"

        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"配置快照已保存: {config_file}")
        except Exception as e:
            logger.error(f"保存配置快照失败: {e}")


# 全局日志管理器实例
_auto_logger: Optional[AutoLogger] = None


def setup_auto_logging(
    experiment_date: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    experiment_config: Optional[Dict[str, Any]] = None,
    experiment_name: Optional[str] = None,
    node_role: Optional[str] = None,
    node_id: Optional[str] = None
) -> AutoLogger:
    """
    设置自动日志系统

    Args:
        experiment_date: 实验日期标识
        config: 日志配置字典，支持以下键：
            - console_enabled: 是否启用控制台输出（默认True）
            - console_level: 控制台日志级别（默认INFO）
            - file_level: 文件日志级别（默认DEBUG）
            - format: 日志格式
            - rotation: 日志轮转大小
            - retention: 日志保留时间
            - compression: 压缩格式
        experiment_config: 实验配置字典（数据集、算法、模型等配置信息）
        experiment_name: 实验名称
        node_role: 节点角色（"server" 或 "client"）
        node_id: 节点ID
    """
    global _auto_logger
    if _auto_logger is None:
        # 从 config 中提取 base_dir
        base_dir = "logs"
        if config and 'base_dir' in config:
            base_dir = config['base_dir']

        _auto_logger = AutoLogger(
            experiment_date,
            base_path=base_dir,
            experiment_name=experiment_name,
            node_role=node_role,
            node_id=node_id,
            config=config,
            experiment_config=experiment_config
        )
    return _auto_logger


def get_logger(log_type: str = "sys", node_id: Optional[str] = None):
    """获取 logger（简化版）

    Args:
        log_type: 日志类型（为了兼容性保留）
        node_id: 节点ID（为了兼容性保留）

    Returns:
        logger 实例
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


# 便捷函数（简化版，为了兼容性保留）
def get_comm_logger(node_id: Optional[str] = None):
    """获取通信日志记录器（简化版，实际上都写入 runtime.log）"""
    return get_logger("comm", node_id)


def get_train_logger(node_id: Optional[str] = None):
    """获取训练日志记录器（简化版，实际上都写入 runtime.log）"""
    return get_logger("train", node_id)


def get_sys_logger():
    """获取系统日志记录器（简化版，实际上都写入 runtime.log）"""
    return get_logger("sys", "system")


def save_config_snapshot(node_id: str, config_dict: dict):
    """保存配置快照

    Args:
        node_id: 节点ID（如 "server", "client_0"）
        config_dict: 配置字典
    """
    global _auto_logger
    if _auto_logger:
        _auto_logger.save_config_snapshot(node_id, config_dict)
    else:
        raise RuntimeError("请先调用 setup_auto_logging() 初始化日志系统")
