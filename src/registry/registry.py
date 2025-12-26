"""
组件注册系统

提供装饰器方式注册和获取组件
"""

from typing import Any, Dict, Type, Optional, List, Callable
import importlib
from ..infra import get_module_logger

logger = get_module_logger(__name__)


class Registry:
    """
    组件注册表（单例）
    
    职责：
    - 管理所有可插拔组件
    - 命名空间隔离
    - 支持装饰器注册
    
    命名空间结构: <范式>.<组件类型>.<实现名称>
    例如: federated.trainer.default, federated.aggregator.fedavg
    """
    
    _instance: Optional["Registry"] = None
    
    def __new__(cls) -> "Registry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._registry: Dict[str, Type] = {}
        return cls._instance
    
    def register(self, namespace: str) -> Callable[[Type], Type]:
        """
        注册装饰器
        
        Args:
            namespace: 命名空间（如 "federated.trainer.default"）
            
        Returns:
            装饰器函数
            
        Example:
            @register("federated.aggregator.fedavg")
            class FedAvgAggregator(Aggregator):
                ...
        """
        def decorator(cls: Type) -> Type:
            if namespace in self._registry:
                logger.warning(f"Overwriting existing registration: {namespace}")
            self._registry[namespace] = cls
            # 保存命名空间到类属性，便于调试
            cls._registry_namespace = namespace
            # logger.debug(f"Registered {cls.__name__} as {namespace}")
            return cls
        return decorator
    
    def get(self, namespace: str) -> Type:
        """
        获取已注册的类
        
        支持多种命名格式：
        - 完整命名空间: "learner.vfl.splitnn"
        - 简短名称: "vfl.splitnn" (自动尝试添加前缀)
        - 基础名称: "default" (自动尝试添加前缀)
        
        Args:
            namespace: 命名空间
            
        Returns:
            注册的类
            
        Raises:
            KeyError: 命名空间未注册
        """
        # 1. 尝试直接查找
        if namespace in self._registry:
            return self._registry[namespace]
        
        # 2. 尝试添加常见前缀
        prefixes = ['learner.', 'trainer.', 'aggregator.', 'model.', 'dataset.', 'callback.']
        for prefix in prefixes:
            full_namespace = prefix + namespace
            if full_namespace in self._registry:
                return self._registry[full_namespace]
        
        # 3. 尝试动态导入
        try:
            cls = self._resolve_namespace(namespace)
            return cls
        except (ImportError, AttributeError):
            pass
        
        raise KeyError(f"Namespace '{namespace}' not found in registry")
    
    def create(self, namespace: str, **kwargs) -> Any:
        """
        获取类并创建实例

        Args:
            namespace: 命名空间
            **kwargs: 传递给构造函数的参数

        Returns:
            实例
        """
        logger.debug(f"创建组件: {namespace}")
        logger.debug(f"传入参数: {list(kwargs.keys())}")
        cls = self.get(namespace)
        logger.debug(f"目标类: {cls}")

        try:
            instance = cls(**kwargs)
            logger.debug(f"组件创建成功: {namespace}")
            return instance
        except TypeError as e:
            logger.error(f"创建组件失败: {namespace}, 错误: {e}")
            import inspect
            sig = inspect.signature(cls.__init__)
            logger.error(f"期望的参数: {list(sig.parameters.keys())}")
            logger.error(f"实际传入的参数: {list(kwargs.keys())}")
            raise
    
    def list(self, prefix: str = "") -> List[str]:
        """
        列出命名空间
        
        Args:
            prefix: 前缀过滤
            
        Returns:
            匹配的命名空间列表
        """
        if not prefix:
            return list(self._registry.keys())
        return [ns for ns in self._registry.keys() if ns.startswith(prefix)]
    
    def exists(self, namespace: str) -> bool:
        """
        检查命名空间是否存在
        
        Args:
            namespace: 命名空间
            
        Returns:
            是否存在
        """
        return namespace in self._registry
    
    def unregister(self, namespace: str) -> bool:
        """
        注销组件
        
        Args:
            namespace: 命名空间
            
        Returns:
            是否成功注销
        """
        if namespace in self._registry:
            del self._registry[namespace]
            return True
        return False
    
    def clear(self) -> None:
        """清空注册表"""
        self._registry.clear()
    
    def _resolve_namespace(self, namespace: str) -> Type:
        """
        解析命名空间并动态导入
        
        支持两种格式:
        1. 简短格式: federated.trainer.default -> federation.builtin.trainers.default
        2. 完整格式: my_package.trainers.MyTrainer -> 直接导入
        """
        # 尝试作为完整模块路径导入
        if "." in namespace:
            parts = namespace.rsplit(".", 1)
            if len(parts) == 2:
                module_path, class_name = parts
                try:
                    module = importlib.import_module(module_path)
                    cls = getattr(module, class_name)
                    # 自动注册
                    self._registry[namespace] = cls
                    return cls
                except (ImportError, AttributeError):
                    pass
        
        raise ImportError(f"Cannot resolve namespace: {namespace}")
    
    def scan(self, package: str) -> int:
        """
        扫描包中的已注册组件
        
        Args:
            package: 包名
            
        Returns:
            发现的组件数量
        """
        count_before = len(self._registry)
        try:
            importlib.import_module(package)
        except ImportError as e:
            logger.warning(f"Failed to scan package {package}: {e}")
        return len(self._registry) - count_before


# 全局注册表实例
registry = Registry()


def register(namespace: str) -> Callable[[Type], Type]:
    """
    全局注册装饰器
    
    Example:
        from federation.registry import register
        
        @register("federated.trainer.my_trainer")
        class MyTrainer(Trainer):
            ...
    """
    return registry.register(namespace)


def get_component(namespace: str) -> Type:
    """获取已注册的组件类"""
    return registry.get(namespace)


def create_component(namespace: str, **kwargs) -> Any:
    """获取组件类并创建实例"""
    return registry.create(namespace, **kwargs)


def list_components(prefix: str = "") -> List[str]:
    """列出已注册的组件"""
    return registry.list(prefix)