"""
Timer - 计时器

支持多个命名计时器和统计
"""

from typing import Any, Dict, List, Optional
from contextlib import contextmanager
import time


class Timer:
    """
    计时器
    
    支持多个命名计时器，可以记录多次执行时间并计算统计
    """
    
    def __init__(self):
        """初始化计时器"""
        self._timers: Dict[str, List[float]] = {}
        self._active: Dict[str, float] = {}
    
    def start(self, name: str = "default") -> "Timer":
        """
        开始计时
        
        Args:
            name: 计时器名称
            
        Returns:
            self（支持链式调用）
        """
        self._active[name] = time.perf_counter()
        return self
    
    def stop(self, name: str = "default") -> float:
        """
        停止计时
        
        Args:
            name: 计时器名称
            
        Returns:
            经过的时间（秒）
        """
        if name not in self._active:
            raise ValueError(f"Timer '{name}' not started")
        
        elapsed = time.perf_counter() - self._active[name]
        del self._active[name]
        
        # 记录到历史
        if name not in self._timers:
            self._timers[name] = []
        self._timers[name].append(elapsed)
        
        return elapsed
    
    @contextmanager
    def time(self, name: str = "default"):
        """
        上下文管理器方式计时
        
        Args:
            name: 计时器名称
            
        Example:
            with timer.time("training"):
                train()
        """
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)
    
    def elapsed(self, name: str = "default") -> Optional[float]:
        """
        获取当前正在计时的时间
        
        Args:
            name: 计时器名称
            
        Returns:
            经过的时间，如果未开始则返回 None
        """
        if name not in self._active:
            return None
        return time.perf_counter() - self._active[name]
    
    def get_last(self, name: str = "default") -> Optional[float]:
        """
        获取最后一次记录的时间
        
        Args:
            name: 计时器名称
            
        Returns:
            最后一次记录的时间
        """
        if name not in self._timers or not self._timers[name]:
            return None
        return self._timers[name][-1]
    
    def get_total(self, name: str = "default") -> float:
        """
        获取总时间
        
        Args:
            name: 计时器名称
            
        Returns:
            所有记录时间之和
        """
        if name not in self._timers:
            return 0.0
        return sum(self._timers[name])
    
    def get_mean(self, name: str = "default") -> Optional[float]:
        """
        获取平均时间
        
        Args:
            name: 计时器名称
            
        Returns:
            平均时间
        """
        if name not in self._timers or not self._timers[name]:
            return None
        return sum(self._timers[name]) / len(self._timers[name])
    
    def get_count(self, name: str = "default") -> int:
        """
        获取记录次数
        
        Args:
            name: 计时器名称
            
        Returns:
            记录次数
        """
        if name not in self._timers:
            return 0
        return len(self._timers[name])
    
    def get_stats(self, name: str = "default") -> Dict[str, Any]:
        """
        获取统计信息
        
        Args:
            name: 计时器名称
            
        Returns:
            统计字典
        """
        if name not in self._timers or not self._timers[name]:
            return {
                "count": 0,
                "total": 0.0,
                "mean": None,
                "min": None,
                "max": None,
            }
        
        times = self._timers[name]
        return {
            "count": len(times),
            "total": sum(times),
            "mean": sum(times) / len(times),
            "min": min(times),
            "max": max(times),
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有计时器的统计信息
        
        Returns:
            所有计时器的统计字典
        """
        return {name: self.get_stats(name) for name in self._timers}
    
    def reset(self, name: Optional[str] = None) -> None:
        """
        重置计时器
        
        Args:
            name: 计时器名称，None 表示重置所有
        """
        if name is None:
            self._timers.clear()
            self._active.clear()
        else:
            if name in self._timers:
                del self._timers[name]
            if name in self._active:
                del self._active[name]
    
    def format_stats(self, name: str = "default") -> str:
        """
        格式化统计信息
        
        Args:
            name: 计时器名称
            
        Returns:
            格式化字符串
        """
        stats = self.get_stats(name)
        if stats["count"] == 0:
            return f"{name}: no records"
        
        return (
            f"{name}: count={stats['count']}, "
            f"total={stats['total']:.3f}s, "
            f"mean={stats['mean']:.3f}s, "
            f"min={stats['min']:.3f}s, "
            f"max={stats['max']:.3f}s"
        )
