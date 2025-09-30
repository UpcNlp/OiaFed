"""
MOE-FedCL 重试机制工具
moe_fedcl/utils/retry.py
"""

import asyncio
import functools
import time
import random
from typing import Any, Callable, List, Optional, Type, Union
from dataclasses import dataclass
from enum import Enum

from ..exceptions import TimeoutError, MOEFedCLError


class RetryStrategy(Enum):
    """重试策略"""
    FIXED_DELAY = "fixed_delay"       # 固定延迟
    EXPONENTIAL_BACKOFF = "exponential_backoff"  # 指数退避
    LINEAR_BACKOFF = "linear_backoff"  # 线性退避
    RANDOM_JITTER = "random_jitter"    # 随机抖动
    FIBONACCI = "fibonacci"            # 斐波那契数列延迟


@dataclass
class RetryConfig:
    """重试配置"""
    max_attempts: int = 3                    # 最大重试次数
    base_delay: float = 1.0                  # 基础延迟时间(秒)
    max_delay: float = 60.0                  # 最大延迟时间(秒)  
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF  # 重试策略
    backoff_factor: float = 2.0              # 退避因子
    jitter: bool = True                      # 是否添加随机抖动
    jitter_factor: float = 0.1               # 抖动因子 (0.0-1.0)
    
    # 异常过滤
    retry_on_exceptions: Optional[List[Type[Exception]]] = None  # 重试的异常类型
    stop_on_exceptions: Optional[List[Type[Exception]]] = None   # 停止重试的异常类型
    
    # 重试条件
    retry_on_result: Optional[Callable[[Any], bool]] = None     # 根据结果决定是否重试
    
    # 回调函数
    on_retry: Optional[Callable[[int, Exception], None]] = None  # 重试时的回调
    on_success: Optional[Callable[[Any], None]] = None          # 成功时的回调
    on_failure: Optional[Callable[[Exception], None]] = None    # 最终失败时的回调


class RetryStatistics:
    """重试统计信息"""
    
    def __init__(self):
        self.total_attempts = 0
        self.successful_attempts = 0
        self.failed_attempts = 0
        self.total_delay_time = 0.0
        self.average_attempts = 0.0
        self.retry_history: List[dict] = []
    
    def record_attempt(self, attempt: int, success: bool, delay: float = 0.0, error: str = None):
        """记录重试尝试"""
        self.total_attempts += 1
        
        if success:
            self.successful_attempts += 1
        else:
            self.failed_attempts += 1
        
        self.total_delay_time += delay
        self.average_attempts = self.total_attempts / max(self.successful_attempts + self.failed_attempts, 1)
        
        # 记录历史
        self.retry_history.append({
            'attempt': attempt,
            'success': success,
            'delay': delay,
            'error': error,
            'timestamp': time.time()
        })
        
        # 限制历史记录长度
        if len(self.retry_history) > 1000:
            self.retry_history = self.retry_history[-1000:]
    
    def get_success_rate(self) -> float:
        """获取成功率"""
        if self.total_attempts == 0:
            return 0.0
        return self.successful_attempts / (self.successful_attempts + self.failed_attempts)
    
    def get_average_delay(self) -> float:
        """获取平均延迟"""
        if self.failed_attempts == 0:
            return 0.0
        return self.total_delay_time / self.failed_attempts


class RetryController:
    """重试控制器"""
    
    def __init__(self, config: RetryConfig):
        self.config = config
        self.statistics = RetryStatistics()
    
    def calculate_delay(self, attempt: int) -> float:
        """计算延迟时间
        
        Args:
            attempt: 当前尝试次数 (从1开始)
            
        Returns:
            float: 延迟时间(秒)
        """
        if self.config.strategy == RetryStrategy.FIXED_DELAY:
            delay = self.config.base_delay
            
        elif self.config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.config.base_delay * (self.config.backoff_factor ** (attempt - 1))
            
        elif self.config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.config.base_delay * attempt
            
        elif self.config.strategy == RetryStrategy.RANDOM_JITTER:
            delay = self.config.base_delay + random.uniform(0, self.config.base_delay)
            
        elif self.config.strategy == RetryStrategy.FIBONACCI:
            delay = self.config.base_delay * self._fibonacci(attempt)
            
        else:
            delay = self.config.base_delay
        
        # 限制最大延迟
        delay = min(delay, self.config.max_delay)
        
        # 添加抖动
        if self.config.jitter and self.config.jitter_factor > 0:
            jitter_range = delay * self.config.jitter_factor
            jitter = random.uniform(-jitter_range, jitter_range)
            delay = max(0, delay + jitter)
        
        return delay
    
    def _fibonacci(self, n: int) -> int:
        """计算斐波那契数列第n项"""
        if n <= 1:
            return 1
        elif n == 2:
            return 1
        else:
            a, b = 1, 1
            for _ in range(3, n + 1):
                a, b = b, a + b
            return b
    
    def should_retry(self, attempt: int, exception: Exception = None, result: Any = None) -> bool:
        """判断是否应该重试
        
        Args:
            attempt: 当前尝试次数
            exception: 发生的异常
            result: 执行结果
            
        Returns:
            bool: 是否应该重试
        """
        # 检查最大尝试次数
        if attempt >= self.config.max_attempts:
            return False
        
        # 检查停止重试的异常
        if exception and self.config.stop_on_exceptions:
            for stop_exc in self.config.stop_on_exceptions:
                if isinstance(exception, stop_exc):
                    return False
        
        # 检查重试的异常
        if exception and self.config.retry_on_exceptions:
            should_retry_exception = False
            for retry_exc in self.config.retry_on_exceptions:
                if isinstance(exception, retry_exc):
                    should_retry_exception = True
                    break
            if not should_retry_exception:
                return False
        
        # 检查结果是否需要重试
        if self.config.retry_on_result and result is not None:
            if not self.config.retry_on_result(result):
                return False
        
        return True
    
    async def execute_async(self, func: Callable, *args, **kwargs) -> Any:
        """异步执行函数并处理重试
        
        Args:
            func: 要执行的异步函数
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            Any: 函数执行结果
            
        Raises:
            Exception: 最终执行失败的异常
        """
        last_exception = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                # 执行函数
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # 检查结果是否需要重试
                if self.config.retry_on_result and self.config.retry_on_result(result):
                    if self.should_retry(attempt, result=result):
                        # 计算延迟并等待
                        delay = self.calculate_delay(attempt)
                        self.statistics.record_attempt(attempt, False, delay, f"Result requires retry: {result}")
                        
                        # 触发重试回调
                        if self.config.on_retry:
                            self.config.on_retry(attempt, Exception(f"Result requires retry: {result}"))
                        
                        if delay > 0:
                            await asyncio.sleep(delay)
                        continue
                
                # 成功
                self.statistics.record_attempt(attempt, True)
                if self.config.on_success:
                    self.config.on_success(result)
                
                return result
                
            except Exception as e:
                last_exception = e
                
                # 检查是否应该重试
                if not self.should_retry(attempt, exception=e):
                    # 记录失败
                    self.statistics.record_attempt(attempt, False, error=str(e))
                    break
                
                # 计算延迟
                delay = self.calculate_delay(attempt)
                self.statistics.record_attempt(attempt, False, delay, str(e))
                
                # 触发重试回调
                if self.config.on_retry:
                    self.config.on_retry(attempt, e)
                
                # 如果不是最后一次尝试，等待后重试
                if attempt < self.config.max_attempts:
                    if delay > 0:
                        await asyncio.sleep(delay)
        
        # 所有尝试都失败了
        if self.config.on_failure and last_exception:
            self.config.on_failure(last_exception)
        
        if last_exception:
            raise last_exception
        else:
            raise MOEFedCLError("Function execution failed without specific exception")
    
    def execute_sync(self, func: Callable, *args, **kwargs) -> Any:
        """同步执行函数并处理重试
        
        Args:
            func: 要执行的同步函数
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            Any: 函数执行结果
            
        Raises:
            Exception: 最终执行失败的异常
        """
        last_exception = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                # 执行函数
                result = func(*args, **kwargs)
                
                # 检查结果是否需要重试
                if self.config.retry_on_result and self.config.retry_on_result(result):
                    if self.should_retry(attempt, result=result):
                        # 计算延迟并等待
                        delay = self.calculate_delay(attempt)
                        self.statistics.record_attempt(attempt, False, delay, f"Result requires retry: {result}")
                        
                        # 触发重试回调
                        if self.config.on_retry:
                            self.config.on_retry(attempt, Exception(f"Result requires retry: {result}"))
                        
                        if delay > 0:
                            time.sleep(delay)
                        continue
                
                # 成功
                self.statistics.record_attempt(attempt, True)
                if self.config.on_success:
                    self.config.on_success(result)
                
                return result
                
            except Exception as e:
                last_exception = e
                
                # 检查是否应该重试
                if not self.should_retry(attempt, exception=e):
                    # 记录失败
                    self.statistics.record_attempt(attempt, False, error=str(e))
                    break
                
                # 计算延迟
                delay = self.calculate_delay(attempt)
                self.statistics.record_attempt(attempt, False, delay, str(e))
                
                # 触发重试回调
                if self.config.on_retry:
                    self.config.on_retry(attempt, e)
                
                # 如果不是最后一次尝试，等待后重试
                if attempt < self.config.max_attempts:
                    if delay > 0:
                        time.sleep(delay)
        
        # 所有尝试都失败了
        if self.config.on_failure and last_exception:
            self.config.on_failure(last_exception)
        
        if last_exception:
            raise last_exception
        else:
            raise MOEFedCLError("Function execution failed without specific exception")


# ==================== 装饰器 ====================

def retry_async(config: RetryConfig = None, 
               max_attempts: int = 3,
               base_delay: float = 1.0,
               strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF):
    """异步重试装饰器
    
    Args:
        config: 重试配置对象
        max_attempts: 最大重试次数
        base_delay: 基础延迟时间
        strategy: 重试策略
    
    Usage:
        @retry_async(max_attempts=5, base_delay=2.0)
        async def my_async_function():
            # 可能失败的异步操作
            pass
    """
    if config is None:
        config = RetryConfig(
            max_attempts=max_attempts,
            base_delay=base_delay,
            strategy=strategy
        )
    
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            controller = RetryController(config)
            return await controller.execute_async(func, *args, **kwargs)
        
        # 添加重试统计信息
        wrapper._retry_controller = lambda: RetryController(config)
        return wrapper
    
    return decorator


def retry_sync(config: RetryConfig = None,
               max_attempts: int = 3,
               base_delay: float = 1.0,
               strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF):
    """同步重试装饰器
    
    Args:
        config: 重试配置对象
        max_attempts: 最大重试次数
        base_delay: 基础延迟时间
        strategy: 重试策略
    
    Usage:
        @retry_sync(max_attempts=5, base_delay=2.0)
        def my_sync_function():
            # 可能失败的同步操作
            pass
    """
    if config is None:
        config = RetryConfig(
            max_attempts=max_attempts,
            base_delay=base_delay,
            strategy=strategy
        )
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            controller = RetryController(config)
            return controller.execute_sync(func, *args, **kwargs)
        
        # 添加重试统计信息
        wrapper._retry_controller = lambda: RetryController(config)
        return wrapper
    
    return decorator


# ==================== 便捷函数 ====================

async def retry_async_call(func: Callable, 
                          *args, 
                          config: RetryConfig = None,
                          **kwargs) -> Any:
    """异步重试调用函数
    
    Args:
        func: 要调用的函数
        *args: 位置参数
        config: 重试配置
        **kwargs: 关键字参数
        
    Returns:
        Any: 函数执行结果
    """
    if config is None:
        config = RetryConfig()
    
    controller = RetryController(config)
    return await controller.execute_async(func, *args, **kwargs)


def retry_sync_call(func: Callable,
                   *args,
                   config: RetryConfig = None,
                   **kwargs) -> Any:
    """同步重试调用函数
    
    Args:
        func: 要调用的函数
        *args: 位置参数
        config: 重试配置
        **kwargs: 关键字参数
        
    Returns:
        Any: 函数执行结果
    """
    if config is None:
        config = RetryConfig()
    
    controller = RetryController(config)
    return controller.execute_sync(func, *args, **kwargs)


def create_network_retry_config() -> RetryConfig:
    """创建适用于网络操作的重试配置"""
    return RetryConfig(
        max_attempts=5,
        base_delay=1.0,
        max_delay=30.0,
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        backoff_factor=2.0,
        jitter=True,
        jitter_factor=0.2,
        retry_on_exceptions=[
            ConnectionError,
            TimeoutError,
            OSError,  # 网络相关的系统错误
        ]
    )


def create_communication_retry_config() -> RetryConfig:
    """创建适用于通信操作的重试配置"""
    from ..exceptions import CommunicationError, TransportError
    
    return RetryConfig(
        max_attempts=3,
        base_delay=2.0,
        max_delay=15.0,
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        backoff_factor=1.5,
        jitter=True,
        jitter_factor=0.1,
        retry_on_exceptions=[
            CommunicationError,
            TransportError,
            ConnectionError,
            TimeoutError,
        ],
        on_retry=lambda attempt, error: print(f"Communication retry {attempt}: {error}")
    )


def create_training_retry_config() -> RetryConfig:
    """创建适用于训练操作的重试配置"""
    from ..exceptions import TrainingError
    
    return RetryConfig(
        max_attempts=2,  # 训练失败通常不需要太多重试
        base_delay=5.0,  # 较长的延迟给系统时间恢复
        strategy=RetryStrategy.FIXED_DELAY,
        retry_on_exceptions=[
            TrainingError,
        ],
        # 训练结果检查：如果结果为None或包含错误，则重试
        retry_on_result=lambda result: (
            result is None or 
            (isinstance(result, dict) and result.get("success") is False)
        ),
        on_retry=lambda attempt, error: print(f"Training retry {attempt}: {error}")
    )


# ==================== 重试统计分析 ====================

class RetryAnalyzer:
    """重试统计分析器"""
    
    def __init__(self):
        self.global_statistics = RetryStatistics()
        self.function_statistics: dict = {}
    
    def add_statistics(self, func_name: str, statistics: RetryStatistics):
        """添加函数的重试统计"""
        self.function_statistics[func_name] = statistics
        
        # 更新全局统计
        self.global_statistics.total_attempts += statistics.total_attempts
        self.global_statistics.successful_attempts += statistics.successful_attempts
        self.global_statistics.failed_attempts += statistics.failed_attempts
        self.global_statistics.total_delay_time += statistics.total_delay_time
    
    def get_analysis_report(self) -> dict:
        """生成重试分析报告"""
        report = {
            "global_statistics": {
                "total_attempts": self.global_statistics.total_attempts,
                "success_rate": self.global_statistics.get_success_rate(),
                "average_delay": self.global_statistics.get_average_delay(),
                "total_delay_time": self.global_statistics.total_delay_time
            },
            "function_statistics": {},
            "recommendations": []
        }
        
        # 函数级统计
        for func_name, stats in self.function_statistics.items():
            report["function_statistics"][func_name] = {
                "total_attempts": stats.total_attempts,
                "success_rate": stats.get_success_rate(),
                "average_delay": stats.get_average_delay(),
                "recent_failures": len([h for h in stats.retry_history[-10:] if not h["success"]])
            }
        
        # 生成建议
        if self.global_statistics.get_success_rate() < 0.8:
            report["recommendations"].append("Overall success rate is low, consider reviewing retry strategies")
        
        if self.global_statistics.get_average_delay() > 10.0:
            report["recommendations"].append("Average delay is high, consider optimizing retry delays")
        
        return report


# 全局重试分析器实例
global_retry_analyzer = RetryAnalyzer()