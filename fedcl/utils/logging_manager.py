# fedcl/utils/logging_manager.py
"""
联邦学习日志管理器

实现日志分离功能，支持：
1. 服务端日志独立记录
2. 每个客户端日志独立记录
3. 每个辅助模型日志独立记录
4. 全局日志汇总
5. 日志级别和格式配置
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime
from loguru import logger
import threading
from contextlib import contextmanager


class FederatedLoggingManager:
    """联邦学习日志管理器"""
    
    def __init__(self, 
                 log_base_dir: str = "./logs",
                 experiment_name: Optional[str] = None,
                 enable_console: bool = True,
                 global_log_level: str = "INFO"):
        """
        初始化日志管理器
        
        Args:
            log_base_dir: 日志基础目录
            experiment_name: 实验名称，用于创建子目录
            enable_console: 是否启用控制台输出
            global_log_level: 全局日志级别
        """
        self.log_base_dir = Path(log_base_dir)
        self.experiment_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.enable_console = enable_console
        self.global_log_level = global_log_level
        
        # 创建实验专用日志目录
        self.experiment_log_dir = self.log_base_dir / self.experiment_name
        self.experiment_log_dir.mkdir(parents=True, exist_ok=True)
        
        # 日志文件路径
        self.server_log_file = self.experiment_log_dir / "server.log"
        self.global_log_file = self.experiment_log_dir / "federated_training.log"
        self.clients_log_dir = self.experiment_log_dir / "客户端"
        self.auxiliary_log_dir = self.experiment_log_dir / "auxiliary_models"
        
        # 创建子目录
        self.clients_log_dir.mkdir(exist_ok=True)
        self.auxiliary_log_dir.mkdir(exist_ok=True)
        
        # 日志器映射
        self.loggers: Dict[str, Any] = {}
        
        # 线程本地存储，用于上下文感知的日志
        self.local = threading.local()
        
        # 初始化全局日志器
        self._setup_global_logger()
        
        print(f"📁 Federated logging initialized: {self.experiment_log_dir}")
    
    def _setup_global_logger(self):
        """设置全局日志器"""
        # 清除默认的loguru配置
        logger.remove()
        
        # 全局日志文件配置 - 只记录服务器相关日志，不包含客户端日志
        logger.add(
            self.global_log_file,
            level=self.global_log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                   "<level>{message}</level>",
            rotation="10 MB",
            retention="7 days",
            compression="zip",
            enqueue=True,  # 线程安全
            backtrace=True,
            diagnose=True,
            filter=lambda record: not record["extra"].get("client_id") and not record["extra"].get("auxiliary_id")  # 过滤掉客户端和辅助模型日志
        )
        
        # 控制台输出（可选）
        if self.enable_console:
            logger.add(
                sys.stdout,
                level=self.global_log_level,
                format="<green>{time:HH:mm:ss}</green> | "
                       "<level>{level: <8}</level> | "
                       "<cyan>{extra[component]}</cyan> | "
                       "<level>{message}</level>",
                filter=self._console_filter,
                colorize=True
            )
        
        # 设置默认上下文
        logger.configure(extra={"component": "全局"})
    
    def _console_filter(self, record):
        """控制台输出过滤器，添加组件标识"""
        if "component" not in record["extra"]:
            record["extra"]["component"] = "全局"
        return True
    
    def get_server_logger(self, server_id: str = "main_server"):
        """
        获取服务端日志器
        
        Args:
            server_id: 服务端ID
            
        Returns:
            logger: 服务端专用日志器
        """
        logger_key = f"server_{server_id}"
        
        if logger_key not in self.loggers:
            # 添加服务端专用日志文件
            server_log_file = self.experiment_log_dir / f"server_{server_id}.log"
            logger.add(
                server_log_file,
                level=self.global_log_level,
                format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                       "<level>{level: <8}</level> | "
                       "<blue>服务器[{extra[server_id]}]</blue> | "
                       "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                       "<level>{message}</level>",
                filter=lambda record: record["extra"].get("server_id") == server_id,
                rotation="5 MB",
                retention="7 days",
                enqueue=True
            )
            
            # 创建服务端专用日志器，同时绑定component和server_id
            self.loggers[logger_key] = logger.bind(
                component=f"服务器[{server_id}]",
                server_id=server_id
            )
            
            self.loggers[logger_key].info(f"服务器日志器初始化完成: {server_id}")
        
        return self.loggers[logger_key]
    
    def get_client_logger(self, client_id: str):
        """
        获取客户端日志器
        
        Args:
            client_id: 客户端ID
            
        Returns:
            logger: 客户端专用日志器
        """
        logger_key = f"client_{client_id}"
        
        if logger_key not in self.loggers:
            # 添加客户端专用日志文件
            client_log_file = self.clients_log_dir / f"{client_id}.log"
            logger.add(
                client_log_file,
                level=self.global_log_level,
                format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                       "<level>{level: <8}</level> | "
                       "<yellow>客户端[{extra[client_id]}]</yellow> | "
                       "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                       "<level>{message}</level>",
                filter=lambda record: record["extra"].get("client_id") == client_id,
                rotation="2 MB",
                retention="7 days",
                enqueue=True
            )
            
            # 创建客户端专用日志器，同时绑定component和client_id
            self.loggers[logger_key] = logger.bind(
                component=f"客户端[{client_id}]",
                client_id=client_id
            )
            
            self.loggers[logger_key].info(f"客户端日志器初始化完成: {client_id}")
        
        return self.loggers[logger_key]
    

    def get_auxiliary_logger(self, auxiliary_id: str):
        """
        获取辅助模型日志器
        
        Args:
            auxiliary_id: 辅助模型ID
            
        Returns:
            logger: 辅助模型专用日志器
        """
        logger_key = f"auxiliary_{auxiliary_id}"
        
        if logger_key not in self.loggers:
            # 添加辅助模型专用日志文件
            auxiliary_log_file = self.auxiliary_log_dir / f"{auxiliary_id}.log"
            logger.add(
                auxiliary_log_file,
                level=self.global_log_level,
                format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                       "<level>{level: <8}</level> | "
                       "<magenta>辅助模型[{extra[auxiliary_id]}]</magenta> | "
                       "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                       "<level>{message}</level>",
                filter=lambda record: record["extra"].get("auxiliary_id") == auxiliary_id,
                rotation="2 MB",
                retention="7 days",
                enqueue=True
            )
            
            # 创建辅助模型专用日志器，同时绑定component和auxiliary_id
            self.loggers[logger_key] = logger.bind(
                component=f"辅助模型[{auxiliary_id}]",
                auxiliary_id=auxiliary_id
            )
            
            self.loggers[logger_key].info(f"辅助模型日志器初始化完成: {auxiliary_id}")
        
        return self.loggers[logger_key]

    @contextmanager
    def log_context(self, component_type: str, component_id: str):
        """
        日志上下文管理器，自动切换到指定组件的日志器
        
        Args:
            component_type: 组件类型 ('server', 'client', 'auxiliary')
            component_id: 组件ID
        """
        # 获取对应的日志器
        if component_type == "server":
            component_logger = self.get_server_logger(component_id)
        elif component_type == "client":
            component_logger = self.get_client_logger(component_id)
        elif component_type == "auxiliary":
            component_logger = self.get_auxiliary_logger(component_id)
        else:
            component_logger = logger.bind(component=f"UNKNOWN[{component_id}]")
        
        # 保存当前线程的日志器
        old_logger = getattr(self.local, 'current_logger', None)
        self.local.current_logger = component_logger
        
        try:
            yield component_logger
        finally:
            # 恢复之前的日志器
            self.local.current_logger = old_logger
    
    def get_current_logger(self):
        """获取当前线程的日志器"""
        return getattr(self.local, 'current_logger', logger)
    
    def log_training_start(self, round_id: int, participants: Dict[str, list]):
        """记录训练轮次开始"""
        logger.info(f"🚀 Training Round {round_id} Started")
        logger.info(f"   Clients: {participants.get('客户端', [])}")
        logger.info(f"   Auxiliary Models: {participants.get('auxiliary', [])}")
    
    def log_training_complete(self, round_id: int, results: Dict[str, Any]):
        """记录训练轮次完成"""
        success_rate = results.get('success_rate', 0.0)
        duration = results.get('round_duration', 0.0)
        
        logger.info(f"✅ Training Round {round_id} Completed")
        logger.info(f"   Success Rate: {success_rate:.2%}")
        logger.info(f"   Duration: {duration:.2f}s")
    
    def log_component_status(self, component_type: str, component_id: str, status: str, details: str = ""):
        """记录组件状态变化"""
        component_logger = None
        
        if component_type == "server":
            component_logger = self.get_server_logger(component_id)
        elif component_type == "client":
            component_logger = self.get_client_logger(component_id)
        elif component_type == "auxiliary":
            component_logger = self.get_auxiliary_logger(component_id)
        
        if component_logger:
            component_logger.info(f"Status: {status} | {details}")
    
    def log_error(self, component_type: str, component_id: str, error: Exception, context: str = ""):
        """记录错误信息"""
        component_logger = None
        
        if component_type == "server":
            component_logger = self.get_server_logger(component_id)
        elif component_type == "client":
            component_logger = self.get_client_logger(component_id)
        elif component_type == "auxiliary":
            component_logger = self.get_auxiliary_logger(component_id)
        
        if component_logger:
            component_logger.error(f"❌ Error in {context}: {str(error)}")
            component_logger.exception(error)
    
    def create_training_summary_log(self, experiment_results: Dict[str, Any]):
        """创建训练总结日志"""
        summary_file = self.experiment_log_dir / "training_summary.log"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("FEDERATED TRAINING SUMMARY\n")
            f.write("=" * 60 + "\n")
            f.write(f"Experiment: {self.experiment_name}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 基本信息
            f.write("Basic Information:\n")
            f.write(f"  Total Rounds: {experiment_results.get('total_轮次', 'N/A')}\n")
            f.write(f"  Completed Rounds: {experiment_results.get('完成_轮次', 'N/A')}\n")
            f.write(f"  Success Rate: {experiment_results.get('success_rate', 0.0):.2%}\n")
            f.write(f"  Total Duration: {experiment_results.get('total_duration', 'N/A')}\n\n")
            
            # 参与者信息
            participants = experiment_results.get('participants', {})
            f.write("Participants:\n")
            f.write(f"  Clients: {len(participants.get('客户端', []))}\n")
            f.write(f"  Auxiliary Models: {len(participants.get('auxiliary', []))}\n\n")
            
            # 每轮结果
            round_results = experiment_results.get('round_results', [])
            if round_results:
                f.write("轮次-by-Round Results:\n")
                for i, result in enumerate(round_results):
                    f.write(f"  Round {i}: Success Rate {result.get('success_rate', 0):.2%}, "
                           f"Duration {result.get('round_duration', 0):.2f}s\n")
        
        logger.info(f"📊 Training summary saved to: {summary_file}")
    
    def get_log_files_info(self) -> Dict[str, str]:
        """获取所有日志文件信息"""
        log_files = {}
        
        # 全局日志
        log_files["global"] = str(self.global_log_file)
        
        # 服务端日志
        for log_file in self.experiment_log_dir.glob("server_*.log"):
            server_id = log_file.stem.replace("server_", "")
            log_files[f"server_{server_id}"] = str(log_file)
        
        # 客户端日志
        for log_file in self.clients_log_dir.glob("*.log"):
            client_id = log_file.stem
            log_files[f"client_{client_id}"] = str(log_file)
        
        # 辅助模型日志
        for log_file in self.auxiliary_log_dir.glob("*.log"):
            auxiliary_id = log_file.stem
            log_files[f"auxiliary_{auxiliary_id}"] = str(log_file)
        
        return log_files
    
    def cleanup(self):
        """清理日志器"""
        logger.info("🧹 Cleaning up logging manager")
        
        # 刷新所有日志
        for component_logger in self.loggers.values():
            try:
                # loguru的complete()方法等待所有日志写入完成
                pass
            except:
                pass
        
        self.loggers.clear()


# 全局日志管理器实例
_global_logging_manager: Optional[FederatedLoggingManager] = None


def initialize_federated_logging(log_base_dir: str = "./logs",
                                experiment_name: Optional[str] = None,
                                enable_console: bool = True,
                                global_log_level: str = "INFO") -> FederatedLoggingManager:
    """
    初始化全局联邦学习日志管理器
    
    Args:
        log_base_dir: 日志基础目录
        experiment_name: 实验名称
        enable_console: 是否启用控制台输出
        global_log_level: 全局日志级别
        
    Returns:
        FederatedLoggingManager: 日志管理器实例
    """
    global _global_logging_manager
    
    _global_logging_manager = FederatedLoggingManager(
        log_base_dir=log_base_dir,
        experiment_name=experiment_name,
        enable_console=enable_console,
        global_log_level=global_log_level
    )
    
    return _global_logging_manager


def get_federated_logging_manager() -> Optional[FederatedLoggingManager]:
    """获取全局日志管理器"""
    return _global_logging_manager


def get_component_logger(component_type: str, component_id: str):
    """
    获取组件日志器的便利函数
    
    Args:
        component_type: 组件类型 ('server', 'client', 'auxiliary')
        component_id: 组件ID
        
    Returns:
        logger: 组件专用日志器
    """
    if _global_logging_manager is None:
        # 如果没有初始化，使用默认配置
        initialize_federated_logging()
    
    if component_type == "server":
        return _global_logging_manager.get_server_logger(component_id)
    elif component_type == "client":
        return _global_logging_manager.get_client_logger(component_id)
    elif component_type == "auxiliary":
        return _global_logging_manager.get_auxiliary_logger(component_id)
    else:
        return logger.bind(component=f"UNKNOWN[{component_id}]")


# 装饰器：自动日志上下文
def with_component_logging(component_type: str, component_id: str):
    """
    装饰器：为函数添加组件日志上下文
    
    Args:
        component_type: 组件类型
        component_id: 组件ID
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            if _global_logging_manager:
                with _global_logging_manager.log_context(component_type, component_id):
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator


# 使用示例
"""
# 1. 初始化日志管理器
log_manager = initialize_federated_logging(
    log_base_dir="./federated_logs",
    experiment_name="mnist_federated_experiment",
    enable_console=True,
    global_log_level="INFO"
)

# 2. 获取不同组件的日志器
server_logger = log_manager.get_server_logger("main_server")
client_logger = log_manager.get_client_logger("client_001")
aux_logger = log_manager.get_auxiliary_logger("generator_model")

# 3. 使用日志器
server_logger.info("Server started and ready to accept connections")
client_logger.info("Starting local training")
aux_logger.info("Generating synthetic data")

# 4. 使用日志上下文
with log_manager.log_context("client", "client_002") as logger:
    logger.info("训练开始")
    logger.info("Epoch 1 完成")
    logger.info("训练完成")

# 5. 使用装饰器
@with_component_logging("server", "main_server")
def coordinate_training_round():
    logger.info("Coordinating training round")

# 6. 便利函数
client_logger = get_component_logger("client", "client_003")
client_logger.debug("Client initialized")
"""
