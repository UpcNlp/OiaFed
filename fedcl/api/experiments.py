# fedcl/api/experiments.py
"""
快速启动接口

提供一行代码启动联邦学习的极简接口，专注于核心功能。
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

from loguru import logger
from omegaconf import DictConfig

from ..registry import registry
from .trainer import FederatedTrainer, TrainingResult


def train(
    learner: str,
    dataset: str = "mnist",
    num_clients: int = 3,
    num_rounds: int = 10,
    **kwargs,
) -> TrainingResult:
    """
    一行代码启动联邦学习

    Args:
        learner: 学习器名称（必须已通过@fedcl.learner注册）
        dataset: 数据集名称，支持 "mnist", "cifar10" 等
        num_clients: 客户端数量
        num_rounds: 训练轮次
        **kwargs: 其他配置参数

    Returns:
        TrainingResult: 训练结果

    Example:
        result = fedcl.train(
            learner="my_learner",
            dataset="mnist",
            num_clients=3,
            num_rounds=10
        )
    """
    logger.info("🚀 启动一键联邦训练")
    logger.info(f"📚 学习器: {learner}, 数据集: {dataset}")
    logger.info(f"👥 客户端数: {num_clients}, 训练轮次: {num_rounds}")

    # 验证学习器是否已注册
    if not registry.get_learner(learner):
        available_learners = list(registry.learners.keys())
        raise ValueError(
            f"学习器 '{learner}' 未找到。"
            f"可用的学习器: {available_learners}。"
            f"请先使用 @fedcl.learner('{learner}') 装饰器注册您的学习器。"
        )

    # 构建配置
    config = {
        "experiment_name": f"{learner}_{dataset}_experiment",
        "learner": learner,
        "dataset": dataset,
        "num_clients": num_clients,
        "num_rounds": num_rounds,
        **kwargs,
    }

    # 创建并运行训练器
    trainer = FederatedTrainer(config)
    return trainer.train()


def train_from_config(
    config: Union[str, Path, Dict[str, Any], DictConfig],
) -> TrainingResult:
    """
    从配置启动训练

    Args:
        config: 配置文件路径、配置字典或DictConfig对象

    Returns:
        TrainingResult: 训练结果

    Example:
        # 使用配置文件
        result = fedcl.train_from_config("my_config.yaml")

        # 使用配置字典
        config = {
            "learner": "my_learner",
            "dataset": "mnist",
            "num_clients": 3,
            "num_rounds": 10
        }
        result = fedcl.train_from_config(config)
    """
    trainer = FederatedTrainer(config)
    return trainer.train()


def quick_experiment(learner: str, dataset: str = "mnist", **kwargs) -> TrainingResult:
    """
    快速实验接口 - 使用默认参数快速测试

    Args:
        learner: 学习器名称
        dataset: 数据集名称
        **kwargs: 其他参数

    Returns:
        TrainingResult: 训练结果

    Example:
        # 最简单的快速测试
        result = fedcl.quick_experiment("my_learner")
    """
    logger.info(f"🧪 快速实验: {learner} on {dataset}")

    return train(
        learner=learner,
        dataset=dataset,
        num_clients=2,  # 快速实验使用较少客户端
        num_rounds=5,  # 快速实验使用较少轮次
        **kwargs,
    )
