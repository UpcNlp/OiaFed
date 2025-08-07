#!/usr/bin/env python3
"""
FedCL 主启动器 - 简化的Python脚本接口

用法:
    from fedcl.cli import launch_federation
    
    # 简单启动
    launch_federation("config.yaml")
    
    # 分布式启动
    launch_federation("configs/", daemon=False)
"""

import sys
import os
from pathlib import Path
from typing import Union, Optional, Dict, Any

from .launcher import FedCLLauncher


def launch_federation(
    config: Union[str, Path],
    daemon: bool = False,
    log_level: str = "INFO",
    working_dir: Optional[str] = None,
    experiment_id: Optional[str] = None,
    enable_checkpoint: bool = True,
    quiet: bool = False
) -> Any:
    """
    启动联邦学习
    
    Args:
        config: 配置文件路径或配置目录路径
        daemon: 是否后台运行
        log_level: 日志级别
        working_dir: 工作目录
        experiment_id: 实验ID
        enable_checkpoint: 是否启用检查点
        quiet: 静默模式
        
    Returns:
        实验结果
    """
    # 创建启动器
    launcher = FedCLLauncher(
        config_path=str(config),
        daemon=daemon
    )
    
    # 设置日志
    launcher.setup_logging()
    
    try:
        # 启动联邦学习
        launcher.run()
        
        # 返回实验结果
        if launcher.experiment:
            return launcher.experiment.get_progress()
        
    except Exception as e:
        launcher.shutdown()
        raise


def quick_start(config: Union[str, Path], **kwargs) -> Any:
    """
    快速启动联邦学习（简化接口）
    
    Args:
        config: 配置文件或目录路径
        **kwargs: 额外参数
        
    Returns:
        实验结果
    """
    return launch_federation(config, **kwargs)


# 兼容性别名
run_federation = launch_federation
start_fedcl = launch_federation
