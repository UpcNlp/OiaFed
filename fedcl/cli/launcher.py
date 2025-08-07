#!/usr/bin/env python3
"""
FedCL 启动器模块

提供 FedCL 联邦学习的启动和管理功能
"""

import os
import sys
import signal
import time
import threading
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any
from loguru import logger

from ..experiment.experiment import FedCLExperiment
from ..utils.improved_logging_manager import initialize_improved_logging


class FedCLLauncher:
    """FedCL 启动器"""
    
    def __init__(self, config_path: str, daemon: bool = False):
        self.config_path = Path(config_path)
        self.daemon = daemon
        self.experiment: Optional[FedCLExperiment] = None
        self.running = False
        self.threads = []
        
        # 设置信号处理
        self._setup_signal_handlers()
        
        # 如果是后台模式，设置守护进程
        if self.daemon:
            self._daemonize()
    
    def _setup_signal_handlers(self):
        """设置信号处理器"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down gracefully...")
            self.shutdown()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # 终止信号
        
        # 在Unix系统上设置更多信号处理
        if hasattr(signal, 'SIGHUP'):
            signal.signal(signal.SIGHUP, signal_handler)  # 挂起信号
    
    def _daemonize(self):
        """将进程转为守护进程"""
        try:
            # 第一次fork
            pid = os.fork()
            if pid > 0:
                # 父进程退出
                sys.exit(0)
        except OSError as e:
            logger.error(f"First fork failed: {e}")
            sys.exit(1)
        
        # 脱离父进程
        os.chdir("/")
        os.setsid()
        os.umask(0)
        
        try:
            # 第二次fork
            pid = os.fork()
            if pid > 0:
                # 第一个子进程退出
                sys.exit(0)
        except OSError as e:
            logger.error(f"Second fork failed: {e}")
            sys.exit(1)
        
        # 重定向标准输入输出
        sys.stdout.flush()
        sys.stderr.flush()
        
        # 重定向到日志文件
        log_dir = Path("./logs/daemon")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        with open(log_dir / "stdout.log", 'a') as f:
            os.dup2(f.fileno(), sys.stdout.fileno())
        with open(log_dir / "stderr.log", 'a') as f:
            os.dup2(f.fileno(), sys.stderr.fileno())
        
        # 写入PID文件
        pid_file = log_dir / "fedcl.pid"
        with open(pid_file, 'w') as f:
            f.write(str(os.getpid()))
        
        logger.info(f"Daemon started with PID: {os.getpid()}")
    
    def setup_logging(self):
        """设置日志系统"""
        # 移除loguru的默认处理器
        logger.remove()
        
        # 控制台日志格式
        console_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
        
        # 添加控制台日志（仅在非守护进程模式下）
        if not self.daemon:
            logger.add(
                sys.stderr,
                format=console_format,
                level="WARNING",  # 只显示WARNING以上级别的日志
                colorize=True
            )
        
        # 文件日志
        log_dir = Path("./logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 添加文件日志
        logger.add(
            log_dir / "fedcl_{time}.log",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
            level="DEBUG",
            rotation="10 MB",
            retention="7 days",
            compression="zip"
        )
        
        logger.info("FedCL Launcher logging initialized")
    
    def run(self):
        """启动联邦学习"""
        try:
            self.running = True
            logger.info(f"Starting FedCL with config: {self.config_path}")
            
            # 检查配置路径，如果不存在则创建默认配置
            if not self.config_path.exists():
                logger.warning(f"Config path not found: {self.config_path}, creating default configuration")
                try:
                    from ..config.default_configs import get_fallback_config_for_path
                    get_fallback_config_for_path(self.config_path)
                    logger.info(f"Created default configuration at: {self.config_path}")
                except ImportError:
                    logger.error("Default config generator not available")
                    raise FileNotFoundError(f"Config path not found: {self.config_path}")
            
            # 创建实验实例
            self.experiment = FedCLExperiment(
                config=str(self.config_path)
            )
            
            # 检查配置模式
            if self.experiment.config_mode == "directory":
                logger.info("Running in distributed mode (server + clients)")
                self._run_distributed_mode()
            else:
                logger.info("Running in single config mode")
                self._run_single_mode()
            
        except Exception as e:
            logger.error(f"Failed to start FedCL: {e}")
            raise
    
    def _run_distributed_mode(self):
        """运行分布式模式（服务端+客户端）"""
        logger.info("Starting distributed federation...")
        
        # 创建并启动实验线程
        experiment_thread = threading.Thread(
            target=self._run_experiment_thread,
            name="FedCL-Experiment",
            daemon=False
        )
        
        self.threads.append(experiment_thread)
        experiment_thread.start()
        
        # 等待线程完成或接收中断信号
        try:
            while self.running and experiment_thread.is_alive():
                time.sleep(0.1)
            
            if experiment_thread.is_alive():
                experiment_thread.join(timeout=10)
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
            self.shutdown()
    
    def _run_single_mode(self):
        """运行单配置模式"""
        logger.info("Starting single config federation...")
        
        # 直接在主线程运行
        self._run_experiment_thread()
    
    def _run_experiment_thread(self):
        """实验线程主函数"""
        try:
            logger.info("Experiment thread started")
            
            # 运行实验
            results = self.experiment.run()
            
            logger.success("Federation completed successfully!")
            logger.info(f"Results: {results}")
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            raise
        finally:
            self.running = False
    
    def shutdown(self):
        """优雅关闭"""
        logger.info("Shutting down FedCL Launcher...")
        self.running = False
        
        # 清理实验
        if self.experiment:
            try:
                self.experiment.cleanup()
                logger.info("Experiment cleaned up")
            except Exception as e:
                logger.error(f"Failed to cleanup experiment: {e}")
        
        # 等待所有线程结束
        for thread in self.threads:
            if thread.is_alive():
                logger.info(f"Waiting for thread {thread.name} to finish...")
                thread.join(timeout=5)
                if thread.is_alive():
                    logger.warning(f"Thread {thread.name} did not finish gracefully")
        
        logger.info("FedCL Launcher shutdown complete")
