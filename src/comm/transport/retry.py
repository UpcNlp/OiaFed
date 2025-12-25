"""
连接重试工具模块

提供连接重试逻辑，支持多种退避策略
"""

import asyncio

import time
from typing import Optional, Dict, Any, Callable, Awaitable
from dataclasses import dataclass
from ..exceptions import NodeNotConnectedError
from ...infra.logging import get_logger


class ConnectionError(Exception):
    """连接异常基类"""
    pass


class ConnectionTimeout(ConnectionError):
    """连接超时异常"""
    pass


class MaxRetriesExceeded(ConnectionError):
    """超过最大重试次数异常"""
    pass


@dataclass
class RetryConfig:
    """重试配置"""
    enabled: bool = True
    max_retries: int = 10
    retry_interval: float = 2.0
    timeout: float = 60.0
    backoff: str = "exponential"  # constant | exponential | linear
    backoff_factor: float = 1.5


async def connect_with_retry(
    connect_func: Callable[[str, Optional[str]], Awaitable[None]],
    node_id: str,
    address: Optional[str],
    retry_config: Optional[RetryConfig] = None,
    logger: Optional[Any] = None,
) -> None:
    """
    带重试的连接函数

    Args:
        connect_func: 实际的连接函数（async）
        node_id: 目标节点 ID
        address: 目标地址
        retry_config: 重试配置

    Raises:
        ConnectionTimeout: 连接超时
        MaxRetriesExceeded: 超过最大重试次数
    """
    # 如果没有配置或禁用重试，直接连接
    if not retry_config or not retry_config.enabled:
        await connect_func(node_id, address)
        return

    # 提取配置
    max_retries = retry_config.max_retries
    retry_interval = retry_config.retry_interval
    timeout = retry_config.timeout
    backoff = retry_config.backoff
    backoff_factor = retry_config.backoff_factor

    start_time = time.time()
    retry_count = 0
    current_interval = retry_interval

    while True:
        # 检查总超时
        elapsed = time.time() - start_time
        if elapsed > timeout:
            raise ConnectionTimeout(
                f"Failed to connect to {node_id} after {elapsed:.1f}s (timeout={timeout}s)"
            )

        # 检查重试次数
        if max_retries >= 0 and retry_count >= max_retries:
            raise MaxRetriesExceeded(
                f"Failed to connect to {node_id} after {retry_count} retries"
            )

        try:
            # 尝试连接
            if retry_count == 0:
                logger.info(f"Attempting to connect to {node_id}")
            else:
                logger.debug(
                    f"Retrying connection to {node_id} "
                    f"(attempt {retry_count + 1}/{max_retries if max_retries >= 0 else '∞'})"
                )

            await connect_func(node_id, address)

            # 连接成功
            if retry_count > 0:
                logger.info(f"Successfully connected to {node_id} after {retry_count} retries")
            else:
                logger.info(f"Successfully connected to {node_id}")
            return

        except (NodeNotConnectedError, ConnectionRefusedError, OSError) as e:
            # 连接失败，准备重试
            retry_count += 1

            if retry_count == 1:
                # 第一次失败，记录 info 级别
                logger.info(
                    f"Connection to {node_id} failed: {type(e).__name__}: {e}. Will retry..."
                )
            else:
                # 后续失败，记录 debug 级别
                logger.debug(
                    f"Connection to {node_id} failed: {type(e).__name__}: {e}. "
                    f"Retrying in {current_interval:.1f}s... (attempt {retry_count})"
                )

            # 等待后重试
            await asyncio.sleep(current_interval)

            # 计算下一次重试间隔（退避策略）
            if backoff == "exponential":
                current_interval = min(current_interval * backoff_factor, 60.0)  # 最大 60s
            elif backoff == "linear":
                current_interval = min(current_interval + retry_interval, 60.0)
            # constant: 不变


def create_retry_config(config_dict: Optional[Dict[str, Any]] = None) -> RetryConfig:
    """
    从字典创建重试配置

    Args:
        config_dict: 配置字典

    Returns:
        RetryConfig 实例
    """
    if not config_dict:
        return RetryConfig()

    return RetryConfig(
        enabled=config_dict.get("enabled", True),
        max_retries=config_dict.get("max_retries", 10),
        retry_interval=config_dict.get("retry_interval", 2.0),
        timeout=config_dict.get("timeout", 60.0),
        backoff=config_dict.get("backoff", "exponential"),
        backoff_factor=config_dict.get("backoff_factor", 1.5),
    )
