# fedcl/core/hook.py
"""
Hook系统核心模块

提供钩子系统的基础接口定义，包括钩子抽象基类、执行阶段定义、
优先级管理等功能。支持灵活的扩展机制和事件驱动的架构。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Callable
from enum import Enum
import time
from omegaconf import DictConfig
from loguru import logger

from .execution_context import ExecutionContext
from ..exceptions import HookExecutionError, ConfigurationError


class HookPhase(Enum):
    """
    钩子执行阶段枚举
    
    定义了钩子可以执行的所有阶段，涵盖了实验、轮次、任务、轮、批次等
    不同粒度的执行时机。
    """
    BEFORE_EXPERIMENT = "before_experiment"
    AFTER_EXPERIMENT = "after_experiment"
    BEFORE_ROUND = "before_round"
    AFTER_ROUND = "after_round"
    BEFORE_TASK = "before_task"
    AFTER_TASK = "after_task"
    BEFORE_EPOCH = "before_epoch"
    AFTER_EPOCH = "after_epoch"
    BEFORE_BATCH = "before_batch"
    
    AFTER_BATCH = "after_batch"
    ON_ERROR = "on_error"
    ON_CHECKPOINT = "on_checkpoint"
    ON_EVALUATION = "on_evaluation"


class HookPriority(Enum):
    """
    钩子优先级枚举
    
    定义了标准的优先级级别，数值越小优先级越高。
    """
    CRITICAL = 0
    HIGH = 10
    NORMAL = 50
    LOW = 100
    LOWEST = 1000


class Hook(ABC):
    """
    钩子抽象基类
    
    定义了钩子系统的基础接口，提供事件驱动的扩展机制。
    钩子可以在训练、评估、聚合等各个阶段执行自定义逻辑。
    
    Attributes:
        phase: 钩子执行阶段
        priority: 执行优先级（数值越小优先级越高）
        name: 钩子名称
        enabled: 是否启用
        execution_count: 执行次数
        total_execution_time: 总执行时间
    """
    
    def __init__(self, phase: str, priority: int = HookPriority.NORMAL.value, 
                 name: Optional[str] = None, enabled: bool = True) -> None:
        """
        初始化钩子
        
        Args:
            phase: 钩子执行阶段
            priority: 执行优先级，数值越小优先级越高
            name: 钩子名称，为None时使用类名
            enabled: 是否启用钩子
            
        Raises:
            ConfigurationError: 参数无效时抛出
        """
        if not isinstance(phase, str):
            raise ConfigurationError("Hook phase must be a string")
            
        if not isinstance(priority, int):
            raise ConfigurationError("Hook priority must be an integer")
            
        self.phase = phase
        self.priority = priority
        self.name = name or self.__class__.__name__
        self.enabled = enabled
        self.execution_count = 0
        self.total_execution_time = 0.0
        self._last_execution_time = 0.0
        
        logger.debug(f"Initialized hook '{self.name}' for phase '{self.phase}' with priority {self.priority}")
    
    @abstractmethod
    def execute(self, context: ExecutionContext, **kwargs) -> Any:
        """
        执行钩子逻辑
        
        这是钩子的核心方法，包含具体的执行逻辑。子类必须实现此方法。
        
        Args:
            context: 执行上下文，提供访问配置和状态的接口
            **kwargs: 阶段特定的关键字参数
            
        Returns:
            Any: 钩子执行结果，可以是任意类型
            
        Raises:
            HookExecutionError: 钩子执行失败时抛出
        """
        pass
    
    def validate_context(self, context: ExecutionContext) -> bool:
        """
        验证执行上下文
        
        检查执行上下文是否满足钩子执行的前提条件。
        
        Args:
            context: 执行上下文
            
        Returns:
            bool: True表示上下文有效，False表示无效
        """
        if not isinstance(context, ExecutionContext):
            logger.error(f"Invalid context type for hook '{self.name}': expected ExecutionContext")
            return False
        return True
    
    def should_execute(self, context: ExecutionContext, **kwargs) -> bool:
        """
        判断是否应该执行钩子
        
        基于当前上下文和参数判断是否应该执行钩子。子类可以重写此方法
        实现条件执行逻辑。
        
        Args:
            context: 执行上下文
            **kwargs: 额外的判断参数
            
        Returns:
            bool: True表示应该执行，False表示跳过执行
        """
        if not self.enabled:
            return False
            
        if not self.validate_context(context):
            return False
            
        return True
    
    def get_priority(self) -> int:
        """
        获取执行优先级
        
        Returns:
            int: 优先级数值，数值越小优先级越高
        """
        return self.priority
    
    def get_phase(self) -> str:
        """
        获取执行阶段
        
        Returns:
            str: 执行阶段名称
        """
        return self.phase
    
    def get_name(self) -> str:
        """
        获取钩子名称
        
        Returns:
            str: 钩子名称
        """
        return self.name
    
    def is_enabled(self) -> bool:
        """
        检查钩子是否启用
        
        Returns:
            bool: True表示启用，False表示禁用
        """
        return self.enabled
    
    def enable(self) -> None:
        """启用钩子"""
        self.enabled = True
        logger.debug(f"Hook '{self.name}' enabled")
    
    def disable(self) -> None:
        """禁用钩子"""
        self.enabled = False
        logger.debug(f"Hook '{self.name}' disabled")
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """
        获取执行统计信息
        
        Returns:
            Dict[str, Any]: 包含执行统计的字典
        """
        avg_time = (self.total_execution_time / self.execution_count 
                   if self.execution_count > 0 else 0.0)
        
        return {
            "name": self.name,
            "phase": self.phase,
            "priority": self.priority,
            "enabled": self.enabled,
            "execution_count": self.execution_count,
            "total_execution_time": self.total_execution_time,
            "average_execution_time": avg_time,
            "last_execution_time": self._last_execution_time
        }
    
    def reset_stats(self) -> None:
        """重置执行统计"""
        self.execution_count = 0
        self.total_execution_time = 0.0
        self._last_execution_time = 0.0
        logger.debug(f"Reset execution stats for hook '{self.name}'")
    
    def cleanup(self) -> None:
        """
        清理资源
        
        在钩子生命周期结束时调用，用于释放资源。子类可以重写此方法
        实现特定的清理逻辑。
        """
        logger.debug(f"Cleaning up hook '{self.name}'")
    
    def __call__(self, context: ExecutionContext, **kwargs) -> Any:
        """
        使钩子对象可调用
        
        这是一个便利方法，允许直接调用钩子对象执行钩子逻辑。
        包含了执行时间统计和错误处理。
        
        Args:
            context: 执行上下文
            **kwargs: 执行参数
            
        Returns:
            Any: 钩子执行结果
            
        Raises:
            HookExecutionError: 钩子执行失败时抛出
        """
        if not self.should_execute(context, **kwargs):
            logger.debug(f"Skipping execution of hook '{self.name}'")
            return None
        
        start_time = time.time()
        
        try:
            logger.debug(f"Executing hook '{self.name}' in phase '{self.phase}'")
            result = self.execute(context, **kwargs)
            
            # 更新执行统计
            execution_time = time.time() - start_time
            self.execution_count += 1
            self.total_execution_time += execution_time
            self._last_execution_time = execution_time
            
            logger.debug(f"Hook '{self.name}' executed successfully in {execution_time:.4f}s")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._last_execution_time = execution_time
            
            error_msg = f"Hook '{self.name}' execution failed: {str(e)}"
            logger.error(error_msg)
            raise HookExecutionError(error_msg, 
                                   error_code="HOOK_EXECUTION_FAILED",
                                   details={
                                       "hook_name": self.name,
                                       "phase": self.phase,
                                       "execution_time": execution_time,
                                       "original_error": str(e)
                                   })
    
    def __lt__(self, other: 'Hook') -> bool:
        """支持钩子排序（按优先级）"""
        if not isinstance(other, Hook):
            return NotImplemented
        return self.priority < other.priority
    
    def __eq__(self, other: 'Hook') -> bool:
        """钩子相等性比较"""
        if not isinstance(other, Hook):
            return NotImplemented
        return (self.name == other.name and 
                self.phase == other.phase and 
                self.priority == other.priority)
    
    def __repr__(self) -> str:
        """字符串表示"""
        return (f"{self.__class__.__name__}("
                f"name='{self.name}', "
                f"phase='{self.phase}', "
                f"priority={self.priority}, "
                f"enabled={self.enabled}, "
                f"executions={self.execution_count})")


class ConditionalHook(Hook, ABC):
    """
    条件钩子抽象基类
    
    扩展基础钩子，支持基于条件函数的执行控制。
    子类必须实现execute方法。
    """
    
    def __init__(self, phase: str, condition: Callable[[ExecutionContext], bool],
                 priority: int = HookPriority.NORMAL.value, 
                 name: Optional[str] = None, enabled: bool = True) -> None:
        """
        初始化条件钩子
        
        Args:
            phase: 钩子执行阶段
            condition: 条件判断函数
            priority: 执行优先级
            name: 钩子名称
            enabled: 是否启用
        """
        super().__init__(phase, priority, name, enabled)
        self.condition = condition
    
    def should_execute(self, context: ExecutionContext, **kwargs) -> bool:
        """
        重写执行判断逻辑
        
        除了基础检查外，还会调用条件函数进行判断。
        """
        if not super().should_execute(context, **kwargs):
            return False
        
        try:
            return self.condition(context)
        except Exception as e:
            logger.error(f"Condition evaluation failed for hook '{self.name}': {str(e)}")
            return False


class TimerHook(Hook, ABC):
    """
    计时器钩子抽象基类
    
    提供基于时间间隔的执行控制。
    子类必须实现execute方法。
    """
    
    def __init__(self, phase: str, interval_seconds: float,
                 priority: int = HookPriority.NORMAL.value,
                 name: Optional[str] = None, enabled: bool = True) -> None:
        """
        初始化计时器钩子
        
        Args:
            phase: 钩子执行阶段
            interval_seconds: 执行间隔（秒）
            priority: 执行优先级
            name: 钩子名称
            enabled: 是否启用
        """
        super().__init__(phase, priority, name, enabled)
        self.interval_seconds = interval_seconds
        self.last_execution_timestamp = 0.0
    
    def should_execute(self, context: ExecutionContext, **kwargs) -> bool:
        """
        基于时间间隔判断是否执行
        """
        if not super().should_execute(context, **kwargs):
            return False
        
        current_time = time.time()
        if current_time - self.last_execution_timestamp >= self.interval_seconds:
            self.last_execution_timestamp = current_time
            return True
        
        return False


class HookRegistry:
    """
    钩子注册表
    
    用于管理和组织钩子的注册表，支持按阶段和优先级组织钩子。
    """
    
    def __init__(self) -> None:
        """初始化钩子注册表"""
        self._hooks: Dict[str, List[Hook]] = {}
        self._hook_stats: Dict[str, Dict[str, Any]] = {}
        
        logger.debug("Initialized hook registry")
    
    def register_hook(self, hook: Hook) -> None:
        """
        注册钩子
        
        Args:
            hook: 要注册的钩子对象
            
        Raises:
            ConfigurationError: 钩子无效时抛出
        """
        if not isinstance(hook, Hook):
            raise ConfigurationError("Invalid hook: must be a Hook instance")
        
        phase = hook.get_phase()
        if phase not in self._hooks:
            self._hooks[phase] = []
        
        # 检查是否已存在相同的钩子
        for existing_hook in self._hooks[phase]:
            if existing_hook.get_name() == hook.get_name():
                logger.warning(f"Hook '{hook.get_name()}' already registered for phase '{phase}', replacing")
                self._hooks[phase].remove(existing_hook)
                break
        
        self._hooks[phase].append(hook)
        self._hooks[phase].sort()  # 按优先级排序
        
        logger.debug(f"Registered hook '{hook.get_name()}' for phase '{phase}' with priority {hook.get_priority()}")
    
    def get_hooks(self, phase: str) -> List[Hook]:
        """
        获取指定阶段的钩子列表
        
        Args:
            phase: 执行阶段
            
        Returns:
            List[Hook]: 按优先级排序的钩子列表
        """
        return self._hooks.get(phase, [])
    
    def remove_hook(self, phase: str, hook_name: str) -> bool:
        """
        移除钩子
        
        Args:
            phase: 执行阶段
            hook_name: 钩子名称
            
        Returns:
            bool: True表示成功移除，False表示未找到
        """
        if phase not in self._hooks:
            return False
        
        for hook in self._hooks[phase]:
            if hook.get_name() == hook_name:
                self._hooks[phase].remove(hook)
                hook.cleanup()
                logger.debug(f"Removed hook '{hook_name}' from phase '{phase}'")
                return True
        
        return False
    
    def clear_phase(self, phase: str) -> None:
        """
        清空指定阶段的所有钩子
        
        Args:
            phase: 执行阶段
        """
        if phase in self._hooks:
            for hook in self._hooks[phase]:
                hook.cleanup()
            self._hooks[phase].clear()
            logger.debug(f"Cleared all hooks for phase '{phase}'")
    
    def clear_all(self) -> None:
        """清空所有钩子"""
        for phase in self._hooks:
            for hook in self._hooks[phase]:
                hook.cleanup()
        self._hooks.clear()
        self._hook_stats.clear()
        logger.debug("Cleared all hooks from registry")
    
    def get_all_phases(self) -> List[str]:
        """
        获取所有注册的阶段
        
        Returns:
            List[str]: 阶段名称列表
        """
        return list(self._hooks.keys())
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """
        获取注册表统计信息
        
        Returns:
            Dict[str, Any]: 统计信息字典
        """
        total_hooks = sum(len(hooks) for hooks in self._hooks.values())
        phase_stats = {phase: len(hooks) for phase, hooks in self._hooks.items()}
        
        return {
            "total_hooks": total_hooks,
            "total_phases": len(self._hooks),
            "phase_stats": phase_stats,
            "phases": list(self._hooks.keys())
        }
    
    def __repr__(self) -> str:
        """字符串表示"""
        stats = self.get_registry_stats()
        return f"HookRegistry(hooks={stats['total_hooks']}, phases={stats['total_phases']})"