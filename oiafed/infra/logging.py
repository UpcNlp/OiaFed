"""
简化的日志自动分流系统

使用同一个logger实例，通过loguru的过滤器功能自动分流到不同文件：
- runtime/: 系统运行日志（通信、连接、错误等）
- training/: 训练日志（训练过程、指标等）

使用方法：
```python
from oiafed..infra.logging import setup_logging, get_logger
from oiafed..infra.log_config import LogConfig

# 方式1: 使用 LogConfig
config = LogConfig(level="DEBUG", log_dir="./logs", exp_name="mnist")
setup_logging(node_id="trainer_1", log_config=config)

# 方式2: 向后兼容，直接传参数
setup_logging(node_id="trainer_1", level="DEBUG", log_dir="./logs", exp_name="mnist")

# 获取 logger
runtime_logger = get_logger("trainer_1", "runtime")
train_logger = get_logger("trainer_1", "training")

# 使用
runtime_logger.info("连接建立")  # -> runtime/trainer_1.log
train_logger.info("开始训练")    # -> training/trainer_1.log
```
"""

from loguru import logger as _base_logger
from pathlib import Path
from typing import Dict, Optional, Literal, Any, Union
from datetime import datetime
import sys

from ..config import LogConfig


class AutoLogger:
    """自动分流日志记录器"""

    def __init__(
        self,
        log_config: LogConfig,
        experiment_date: Optional[str] = None,
    ):
        """
        Args:
            log_config: LogConfig 日志配置对象
            experiment_date: 实验日期标识（向后兼容，优先使用 log_config.run_name）
        """
        self.log_config = log_config
        self.base_path = Path(log_config.log_dir)
        self.experiment_name = log_config.exp_name or "default"
        # 优先使用 log_config.run_name，其次是 experiment_date 参数，最后自动生成
        self.timestamp = log_config.run_name or experiment_date or datetime.now().strftime("%Y%m%d_%H%M%S")

        # 从 LogConfig 构建配置字典（用于内部逻辑）
        self.config = {
            'console_enabled': log_config.console,
            'console_level': log_config.console_level,
            'level': log_config.level,
            'format': log_config.format,  # 统一格式（控制台会自动添加颜色）
            'rotation': log_config.rotation,
            'retention': log_config.retention,
            'compression': log_config.compression,
            'diagnose': log_config.diagnose,
        }

        # 创建目录结构: logs/exp_name/run_timestamp/{runtime,training}
        self.run_dir = self.base_path / self.experiment_name / f"run_{self.timestamp}"
        self.runtime_dir = self.run_dir / "runtime"
        self.training_dir = self.run_dir / "training"

        # 创建目录
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.training_dir.mkdir(parents=True, exist_ok=True)

        # 用于存储每个节点的 handler
        self._node_handlers: Dict[str, Dict[str, int]] = {}  # {node_id: {runtime: id, training: id}}

        # 第一次初始化时移除默认 handler（无论 console 是否启用都要移除，避免重复输出）
        _base_logger.remove()

    def setup_node_logging(
        self,
        node_id: str,
        level: Optional[str] = None,
        console: Optional[bool] = None,
        console_level: Optional[str] = None,
    ):
        """
        为指定节点设置日志

        Args:
            node_id: 节点 ID
            level: 文件日志级别（默认使用 config 中的 level）
            console: 是否输出到控制台（默认使用 config 中的 console_enabled）
            console_level: 控制台日志级别（默认使用 config 中的 console_level）
        """
        # 检查是否已初始化
        if node_id in self._node_handlers:
            _base_logger.warning(f"节点 {node_id} 的日志系统已初始化，跳过重复初始化")
            return

        # 使用配置或参数
        file_level = level or self.config['level']
        console_enabled = console if console is not None else self.config['console_enabled']
        console_log_level = console_level or self.config['console_level']

        # 确定日志文件路径
        runtime_log = self.runtime_dir / f"{node_id}.log"
        training_log = self.training_dir / f"{node_id}.log"

        # 初始化该节点的 handlers
        self._node_handlers[node_id] = {}

        # 将格式字符串转换为带颜色的控制台格式
        # 策略：直接构建完整的带颜色的格式字符串（基于默认格式）
        console_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "{extra[node_id]} | "
            "<cyan>{name}:{function}:{line}</cyan> - "
            "<level>{message}</level>"
        )

        # 1. Runtime 日志 - 控制台（添加 R- 前缀）
        r_console_format = console_format.replace("{extra[node_id]}", "R-{extra[node_id]}")
        if console_enabled:
            handler_id = _base_logger.add(
                sys.stdout,
                format=r_console_format,
                level=console_log_level,
                colorize=True,
                diagnose=self.config['diagnose'],
                filter=lambda record: (
                    record["extra"].get("log_type") == "runtime" and
                    record["extra"].get("node_id") == node_id
                ),
            )
            self._node_handlers[node_id]["runtime_console"] = handler_id

        # 2. Runtime 日志 - 文件（添加 R- 前缀）
        r_file_format = self.config['format'].replace("{extra[node_id]}", "R-{extra[node_id]}")
        handler_id = _base_logger.add(
            str(runtime_log),
            format=r_file_format,
            level=file_level,
            rotation=self.config['rotation'],
            retention=self.config['retention'],
            compression=self.config['compression'],
            encoding="utf-8",
            diagnose=self.config['diagnose'],
            filter=lambda record: (
                record["extra"].get("log_type") == "runtime" and
                record["extra"].get("node_id") == node_id
            ),
        )
        self._node_handlers[node_id]["runtime_file"] = handler_id
        # 3. Training 日志 - 控制台（添加 T- 前缀）
        t_console_format = console_format.replace("{extra[node_id]}", "T-{extra[node_id]}")
        if console_enabled:
            handler_id = _base_logger.add(
                sys.stdout,
                format=t_console_format,
                level=console_log_level,
                colorize=True,
                diagnose=self.config['diagnose'],
                filter=lambda record: (
                    record["extra"].get("log_type") == "training" and
                    record["extra"].get("node_id") == node_id
                ),
            )
            self._node_handlers[node_id]["training_console"] = handler_id

        # 4. Training 日志 - 文件（添加 T- 前缀）
        t_file_format = self.config['format'].replace("{extra[node_id]}", "T-{extra[node_id]}")
        handler_id = _base_logger.add(
            str(training_log),
            format=t_file_format,
            level=file_level,
            rotation=self.config['rotation'],
            retention=self.config['retention'],
            compression=self.config['compression'],
            encoding="utf-8",
            diagnose=self.config['diagnose'],
            filter=lambda record: (
                record["extra"].get("log_type") == "training" and
                record["extra"].get("node_id") == node_id
            ),
        )
        self._node_handlers[node_id]["training_file"] = handler_id

        # 输出初始化信息
        runtime_logger = _base_logger.bind(node_id=node_id, log_type="runtime")
        runtime_logger.debug(f"节点 {node_id} 日志系统初始化完成")
        runtime_logger.debug(f"运行日志: {runtime_log}")
        runtime_logger.debug(f"训练日志: {training_log}")

    def get_logger(
        self,
        log_type: str = "runtime",
        node_id: Optional[str] = None,
    ):
        """
        获取 logger

        Args:
            log_type: 日志类型 ("runtime"/"system" 或 "training"/"train")
            node_id: 节点ID（可选）

        Returns:
            logger 实例（带有 node_id 和 log_type 上下文）
        """
        # 标准化 log_type
        if log_type in ("sys", "comm", "system"):
            log_type = "runtime"
        elif log_type in ("train","training"):
            log_type = "training"

        # 如果没有提供 node_id，返回全局 logger
        if not node_id:
            return _base_logger.bind(log_type=log_type)

        # 如果该节点还没有 handler，为其创建
        if node_id not in self._node_handlers:
            self.setup_node_logging(node_id)

        # 返回绑定了 node_id 和 log_type 的 logger
        return _base_logger.bind(node_id=node_id, log_type=log_type)

    def cleanup(self):
        """清理日志处理器"""
        for node_handlers in self._node_handlers.values():
            for handler_id in node_handlers.values():
                try:
                    _base_logger.remove(handler_id)
                except ValueError:
                    pass
        self._node_handlers.clear()


# ========== 全局单例 ==========
_auto_logger: Optional[AutoLogger] = None


def setup_logging(
    node_id: str,
    log_config: Optional[LogConfig] = None,
    # 向后兼容参数（如果没有提供 log_config）
    level: str = "INFO",
    log_dir: str = "./logs",
    console: bool = True,
    console_level: Optional[str] = None,
    diagnose: bool = False,
    exp_name: Optional[str] = None,
) -> None:
    """
    初始化日志系统（每个节点调用一次）

    Args:
        node_id: 节点 ID
        log_config: LogConfig 配置对象（推荐使用）
        level: 文件日志级别（向后兼容，如果提供了log_config则忽略）
        log_dir: 日志根目录（向后兼容）
        console: 是否输出到控制台（向后兼容）
        console_level: 控制台日志级别（向后兼容）
        diagnose: 是否显示详细诊断信息（向后兼容）
        exp_name: 实验名称（向后兼容）

    Examples:
        # 方式1: 使用 LogConfig（推荐）
        from oiafed..infra.log_config import LogConfig

        config = LogConfig(level="DEBUG", exp_name="mnist_fedavg")
        setup_logging(node_id="trainer", log_config=config)

        # 方式2: 向后兼容，直接传参数
        setup_logging(node_id="trainer", level="DEBUG", exp_name="mnist_fedavg")
    """
    global _auto_logger

    # 第一次调用时创建 AutoLogger 实例
    if _auto_logger is None:
        # 如果没有提供 LogConfig，从参数创建一个
        if log_config is None:
            log_config = LogConfig(
                level=level,
                log_dir=log_dir,
                console=console,
                console_level=console_level or level,
                diagnose=diagnose,
                exp_name=exp_name,
            )

        # 使用 LogConfig 创建 AutoLogger
        _auto_logger = AutoLogger(log_config=log_config)

    # 为该节点设置日志
    _auto_logger.setup_node_logging(
        node_id=node_id,
        level=_auto_logger.log_config.level,
        console=_auto_logger.log_config.console,
        console_level=_auto_logger.log_config.console_level,
    )


def get_logger(
    node_id: str,
    log_type: Literal["runtime", "training", "system", "train"] = "runtime",
):
    """
    获取指定类型的 logger

    Args:
        node_id: 节点 ID
        log_type: 日志类型
            - "runtime" / "system": 运行日志（连接、错误、调试）
            - "training" / "train": 训练日志（训练过程、指标）

    Returns:
        绑定了 node_id 和 log_type 的 logger 实例

    Examples:
        # 运行日志
        runtime_logger = get_logger("trainer", "runtime")
        runtime_logger.info("连接成功")

        # 训练日志
        train_logger = get_logger("trainer", "training")
        train_logger.info("Epoch 1/10 - Loss: 0.5")
    """
    if _auto_logger is None:
        raise RuntimeError(
            f"日志系统未初始化\n"
            f"请先调用 setup_logging(node_id='{node_id}')"
        )

    return _auto_logger.get_logger(log_type=log_type, node_id=node_id)


# ========== 便捷函数（兼容旧接口）==========

def get_system_logger(node_id: str):
    """获取系统日志 logger（便捷函数）"""
    return get_logger(node_id, "runtime")


def get_training_logger(node_id: str):
    """获取训练日志 logger（便捷函数）"""
    return get_logger(node_id, "training")


def get_comm_logger(node_id: str):
    """获取通信日志 logger（便捷函数，等同于 runtime）"""
    return get_logger(node_id, "runtime")


def get_train_logger(node_id: str):
    """获取训练日志 logger（便捷函数）"""
    return get_logger(node_id, "training")


def get_module_logger(module_name: str, node_id: Optional[str] = None, log_type: str = "runtime"):
    """
    获取模块级 logger（用于库内部模块）

    这个函数返回一个简单的 logger，不依赖于 setup_logging()
    适用于不需要节点级日志分离的场景

    Args:
        module_name: 模块名（通常是 __name__）
        node_id: 节点 ID（可选）
        log_type: 日志类型（runtime/training）

    Returns:
        logger 实例

    Examples:
        # 库内部模块使用
        logger = get_module_logger(__name__)
        logger.info("Module loaded")
    """
    # 简化模块名显示
    display_module = module_name.replace("oiafed..", "").replace("federation.", "")

    # 如果没有指定 node_id，返回一个不带过滤的 logger
    if node_id is None:
        return _base_logger.bind(module=display_module)

    # 标准化 log_type
    if log_type in ("system", "sys", "comm"):
        log_type = "runtime"
    elif log_type in ("train",):
        log_type = "training"

    # 返回绑定了上下文的 logger
    return _base_logger.bind(module=display_module, node_id=node_id, log_type=log_type)


def cleanup_logging():
    """清理日志系统"""
    global _auto_logger
    if _auto_logger:
        _auto_logger.cleanup()
        _auto_logger = None