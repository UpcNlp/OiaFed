# fedcl/registry/component_registry.py
"""
组件注册系统核心模块（完全修复版）

提供FedCL框架的组件注册、查询、验证等功能，支持学习器、聚合器、
评估器、钩子函数、损失函数和辅助模型的统一管理。
"""

from typing import Dict, List, Type, Callable, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from threading import RLock
from collections import defaultdict
import inspect
from loguru import logger
import time
from functools import wraps

from ..config.schema_validator import ValidationResult
from ..exceptions import (
    FedCLError
)



class ComponentRegistrationError(FedCLError):
    """组件注册错误"""
    pass


class ComponentNotFoundError(FedCLError):
    """组件未找到错误"""
    pass


class ComponentValidationError(FedCLError):
    """组件验证错误"""
    pass


class ComponentConflictError(FedCLError):
    """组件冲突错误"""
    pass


@dataclass
class ComponentMetadata:
    """组件元数据"""
    name: str
    component_type: str
    version: str = "1.0.0"
    author: str = "unknown"
    description: str = ""
    requirements: List[str] = field(default_factory=list)
    supported_features: List[str] = field(default_factory=list)
    config_schema: Dict = field(default_factory=dict)
    performance_characteristics: Dict = field(default_factory=dict)
    registration_time: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ComponentMetadata':
        """从字典创建"""
        return cls(**data)


class ComponentRegistry:
    """
    组件注册中心
    
    管理FedCL框架中的所有可插拔组件，包括学习器、聚合器、评估器、
    钩子函数、损失函数和辅助模型等。提供线程安全的注册、查询和验证功能。
    """
    
    # 支持的组件类型
    COMPONENT_TYPES = {
        'learner', 'aggregator', 'evaluator', 'hook', 
        'loss_function', 'auxiliary_model', 'dispatcher'
    }
    
    # 组件接口要求
    COMPONENT_INTERFACES = {
        'learner': 'BaseLearner',
        'aggregator': 'BaseAggregator', 
        'evaluator': 'BaseEvaluator',
        'hook': 'Hook',
        'dispatcher': 'BaseDispatcher'
    }
    
    def __init__(self):
        """初始化组件注册中心"""
        self._lock = RLock()
        
        # 组件存储
        self._learners: Dict[str, Type] = {}
        self._aggregators: Dict[str, Type] = {}
        self._evaluators: Dict[str, Type] = {}
        self._hooks: Dict[str, List[Type]] = defaultdict(list)
        self._loss_functions: Dict[str, Callable] = {}
        self._auxiliary_models: Dict[str, Type] = {}
        self._dispatchers: Dict[str, Type] = {}  # 新增：下发钩子存储
        
        # 元数据存储
        self._metadata: Dict[str, ComponentMetadata] = {}
        
        # 查询统计
        self._query_stats = {'total_queries': 0, 'cache_hits': 0}
        
        logger.debug("ComponentRegistry initialized")
    
    def register_learner(self, name: str, learner_class: Type, 
                        metadata: Optional[Dict] = None) -> Callable:
        """
        注册学习器
        
        Args:
            name: 学习器名称
            learner_class: 学习器类
            metadata: 元数据
            
        Returns:
            装饰器函数
            
        Raises:
            ComponentRegistrationError: 注册失败
        """
        def decorator(cls=None):
            if cls is None:
                cls = learner_class
                
            with self._lock:
                self._validate_registration('learner', name, cls)
                
                # 验证接口
                validation_result = self.validate_component(cls, 'learner')
                if not validation_result.is_valid:
                    error_messages = [f"{error.field}: {error.message}" for error in validation_result.errors]
                    raise ComponentValidationError(
                        f"Learner {name} validation failed: {'; '.join(error_messages)}"
                    )
                
                # 注册组件
                self._learners[name] = cls
                
                # 创建元数据
                meta = self._create_metadata(name, 'learner', cls, metadata)
                self._metadata[f"learner:{name}"] = meta
                
                logger.debug(f"Registered learner: {name}")
                return cls
                
        return decorator
    
    def register_aggregator(self, name: str, aggregator_class: Type,
                           metadata: Optional[Dict] = None) -> Callable:
        """
        注册聚合器
        
        Args:
            name: 聚合器名称
            aggregator_class: 聚合器类
            metadata: 元数据
            
        Returns:
            装饰器函数
        """
        def decorator(cls=None):
            if cls is None:
                cls = aggregator_class
                
            with self._lock:
                self._validate_registration('aggregator', name, cls)
                
                validation_result = self.validate_component(cls, 'aggregator')
                if not validation_result.is_valid:
                    error_messages = [f"{error.field}: {error.message}" for error in validation_result.errors]
                    raise ComponentValidationError(
                        f"聚合器 {name} validation failed: {'; '.join(error_messages)}"
                    )
                
                self._aggregators[name] = cls
                meta = self._create_metadata(name, 'aggregator', cls, metadata)
                self._metadata[f"aggregator:{name}"] = meta
                
                logger.debug(f"Registered aggregator: {name}")
                return cls
                
        return decorator
    
    def register_evaluator(self, name: str, evaluator_class: Type,
                          metadata: Optional[Dict] = None) -> Callable:
        """
        注册评估器
        
        Args:
            name: 评估器名称
            evaluator_class: 评估器类
            metadata: 元数据
            
        Returns:
            装饰器函数
        """
        def decorator(cls=None):
            if cls is None:
                cls = evaluator_class
                
            with self._lock:
                self._validate_registration('evaluator', name, cls)
                
                validation_result = self.validate_component(cls, 'evaluator')
                if not validation_result.is_valid:
                    error_messages = [f"{error.field}: {error.message}" for error in validation_result.errors]
                    raise ComponentValidationError(
                        f"Evaluator {name} validation failed: {'; '.join(error_messages)}"
                    )
                
                self._evaluators[name] = cls
                meta = self._create_metadata(name, 'evaluator', cls, metadata)
                self._metadata[f"evaluator:{name}"] = meta
                
                logger.debug(f"Registered evaluator: {name}")
                return cls
                
        return decorator
    
    def register_hook(self, phase: str, priority: int = 0, metadata: Optional[Dict] = None) -> Callable:
        """
        注册钩子函数
        
        Args:
            phase: 钩子阶段
            priority: 优先级
            metadata: 额外的元数据
            
        Returns:
            装饰器函数
        """
        def decorator(hook_class: Type):
            
            with self._lock:
                self._validate_registration('hook', f"{phase}:{hook_class.__name__}", hook_class)
                
                validation_result = self.validate_component(hook_class, 'hook')
                if not validation_result.is_valid:
                    error_messages = [f"{error.field}: {error.message}" for error in validation_result.errors]
                    raise ComponentValidationError(
                        f"Hook {hook_class.__name__} validation failed: {'; '.join(error_messages)}"
                    )
                
                # 将钩子添加到对应阶段，按优先级排序
                hook_info = {
                    'class': hook_class,
                    'priority': priority,
                    'phase': phase
                }
                
                self._hooks[phase].append(hook_info)
                self._hooks[phase].sort(key=lambda x: x['priority'], reverse=True)
                
                # 创建元数据
                hook_metadata = {'phase': phase, 'priority': priority}
                if metadata:
                    hook_metadata.update(metadata)
                
                meta = self._create_metadata(
                    f"{phase}:{hook_class.__name__}", 'hook', hook_class, hook_metadata
                )
                self._metadata[f"hook:{phase}:{hook_class.__name__}"] = meta
                
                logger.debug(f"Registered hook: {hook_class.__name__} for phase {phase}")
                return hook_class
                
        return decorator
    
    def register_loss_function(self, name: str, scope: str = "task") -> Callable:
        """
        注册损失函数
        
        Args:
            name: 损失函数名称
            scope: 作用域 (task/global/client)
            
        Returns:
            装饰器函数
        """
        def decorator(loss_func: Callable):
            with self._lock:
                self._validate_registration('loss_function', name, loss_func)
                
                if not self.validate_signature(loss_func, 'loss_function'):
                    raise ComponentValidationError(
                        f"Loss function {name} has invalid signature"
                    )
                
                self._loss_functions[name] = loss_func
                
                meta = self._create_metadata(
                    name, 'loss_function', loss_func,
                    {'scope': scope}
                )
                self._metadata[f"loss_function:{name}"] = meta
                
                logger.debug(f"Registered loss function: {name}")
                return loss_func
                
        return decorator
    
    def register_auxiliary_model(self, name: str, model_type: str) -> Callable:
        """
        注册辅助模型
        
        Args:
            name: 模型名称
            model_type: 模型类型
            
        Returns:
            装饰器函数
        """
        def decorator(model_class: Type):
            with self._lock:
                self._validate_registration('auxiliary_model', name, model_class)
                
                self._auxiliary_models[name] = model_class
                
                meta = self._create_metadata(
                    name, 'auxiliary_model', model_class,
                    {'model_type': model_type}
                )
                self._metadata[f"auxiliary_model:{name}"] = meta
                
                logger.debug(f"Registered auxiliary model: {name}")
                return model_class
                
        return decorator
    
    def register_dispatcher(self, name: str, dispatcher_class: Type = None,
                           metadata: Optional[Dict] = None) -> Callable:
        """
        注册下发钩子
        
        Args:
            name: 下发钩子名称
            dispatcher_class: 下发钩子类
            metadata: 元数据
            
        Returns:
            装饰器函数
            
        Raises:
            ComponentRegistrationError: 注册失败
        """
        def decorator(cls=None):
            if cls is None:
                cls = dispatcher_class
                
            with self._lock:
                self._validate_registration('dispatcher', name, cls)
                
                # 验证下发钩子类
                validation_result = self.validate_component(cls, 'dispatcher')
                if not validation_result.is_valid:
                    error_messages = [f"{error.field}: {error.message}" for error in validation_result.errors]
                    raise ComponentValidationError(
                        f"Dispatcher {name} validation failed: {'; '.join(error_messages)}"
                    )
                
                self._dispatchers[name] = cls
                meta = self._create_metadata(name, 'dispatcher', cls, metadata)
                self._metadata[f"dispatcher:{name}"] = meta
                
                logger.debug(f"Registered dispatcher: {name}")
                return cls
                
        return decorator
    
    def dispatch_hook(self, name: str, metadata: Optional[Dict] = None) -> Callable:
        """
        下发钩子装饰器，与现有装饰器保持一致的风格
        
        Args:
            name: 下发钩子名称
            metadata: 元数据
            
        Returns:
            装饰器函数
        """
        return self.register_dispatcher(name, metadata=metadata)
    
    def get_dispatcher(self, name: str) -> Type:
        """
        获取下发钩子类
        
        Args:
            name: 下发钩子名称
            
        Returns:
            下发钩子类
            
        Raises:
            ComponentNotFoundError: 下发钩子未找到
        """
        return self.get_component('dispatcher', name)
    
    def list_dispatchers(self) -> List[str]:
        """
        列出所有下发钩子
        
        Returns:
            下发钩子名称列表
        """
        return self.list_components('dispatcher')

    def get_component(self, component_type: str, name: str) -> Type:
        """
        获取组件类
        
        Args:
            component_type: 组件类型
            name: 组件名称
            
        Returns:
            组件类
            
        Raises:
            ComponentNotFoundError: 组件未找到
        """
        start_time = time.time()
        
        with self._lock:
            self._query_stats['total_queries'] += 1
            
            if component_type not in self.COMPONENT_TYPES:
                raise ValueError(f"Unsupported component type: {component_type}")
            
            # 根据类型查找组件
            component_store = getattr(self, f"_{component_type}s", None)
            if component_store is None:
                raise ValueError(f"Invalid component type: {component_type}")
            
            if name not in component_store:
                raise ComponentNotFoundError(
                    f"Component {component_type}:{name} not found"
                )
            
            component = component_store[name]
            
            # 性能检查：确保查询时间<1ms
            query_time = (time.time() - start_time) * 1000
            if query_time > 1.0:
                logger.warning(f"Component query took {query_time:.2f}ms (>1ms)")
            
            return component
    
    def list_components(self, component_type: str) -> List[str]:
        """
        列出组件
        
        Args:
            component_type: 组件类型
            
        Returns:
            组件名称列表
        """
        with self._lock:
            if component_type not in self.COMPONENT_TYPES:
                raise ValueError(f"Unsupported component type: {component_type}")
            
            if component_type == 'hook':
                # 钩子比较特殊，需要特殊处理
                return list(self._hooks.keys())
            
            component_store = getattr(self, f"_{component_type}s", {})
            return list(component_store.keys())
    
    def get_hooks(self, phase: str) -> List[Type]:
        """
        获取指定阶段的钩子类
        
        Args:
            phase: 钩子执行阶段
            
        Returns:
            List[Type]: 钩子类列表，按优先级排序
        """
        with self._lock:
            hook_infos = self._hooks.get(phase, [])
            # 返回钩子类，已按优先级排序
            return [hook_info['class'] for hook_info in hook_infos]
    
    def get_component_metadata(self, component_type: str, name: str) -> ComponentMetadata:
        """
        获取组件元数据
        
        Args:
            component_type: 组件类型
            name: 组件名称
            
        Returns:
            组件元数据
        """
        with self._lock:
            key = f"{component_type}:{name}"
            if key not in self._metadata:
                raise ComponentNotFoundError(f"Metadata for {key} not found")
            
            return self._metadata[key]
    
    def validate_signature(self, func: Callable, component_type: str) -> bool:
        """
        验证组件签名
        
        Args:
            func: 函数或方法
            component_type: 组件类型
            
        Returns:
            是否有效
        """
        try:
            sig = inspect.signature(func)
            
            if component_type == 'loss_function':
                # 损失函数应该接受预测和目标参数
                params = list(sig.parameters.keys())
                return len(params) >= 2
            
            # 其他类型的验证逻辑
            return True
            
        except Exception as e:
            logger.warning(f"Signature validation failed: {e}")
            return False
    
    def validate_component(self, component_class: Type, component_type: str) -> ValidationResult:
        """
        验证组件实现
        
        Args:
            component_class: 组件类
            component_type: 组件类型
            
        Returns:
            验证结果
        """
        result = ValidationResult()
        
        try:
            if component_type in self.COMPONENT_INTERFACES:
                required_interface = self.COMPONENT_INTERFACES[component_type]
                
                # 检查基类
                base_classes = [cls.__name__ for cls in component_class.__mro__]
                if required_interface not in base_classes:
                    result.add_error(
                        "inheritance",
                        f"Component {component_class.__name__} must inherit from {required_interface}"
                    )
                
                # 检查必要方法
                self._validate_required_methods(component_class, component_type, result)
            
            # 检查构造函数
            self._validate_constructor(component_class, result)
            
        except Exception as e:
            result.add_error("validation", f"Validation error: {str(e)}")
        
        return result
    
    def _validate_required_methods(self, component_class: Type, component_type: str, 
                                  result: ValidationResult):
        """验证必要方法"""
        required_methods = {
            'learner': ['train_task', 'evaluate_task'],
            'aggregator': ['aggregate'],
            'evaluator': ['evaluate'],
            'hook': ['execute']
        }
        
        if component_type in required_methods:
            for method_name in required_methods[component_type]:
                if not hasattr(component_class, method_name):
                    result.add_error("methods", f"Missing required method: {method_name}")
                elif not callable(getattr(component_class, method_name)):
                    result.add_error("methods", f"Method {method_name} is not callable")
    
    def _validate_constructor(self, component_class: Type, result: ValidationResult):
        """验证构造函数"""
        try:
            init_method = getattr(component_class, '__init__')
            sig = inspect.signature(init_method)
            
            # 检查是否有self参数
            params = list(sig.parameters.keys())
            if not params or params[0] != 'self':
                result.add_error("constructor", "Constructor must have 'self' as first parameter")
                
        except Exception as e:
            result.add_error("constructor", f"Constructor validation error: {str(e)}")
    
    def _validate_registration(self, component_type: str, name: str, component: Any):
        """验证注册请求"""
        if component_type not in self.COMPONENT_TYPES:
            raise ComponentRegistrationError(f"Unsupported component type: {component_type}")
        
        # 检查名称冲突
        if component_type == 'hook':
            # 钩子允许同一阶段有多个
            pass
        else:
            component_store = getattr(self, f"_{component_type}s", {})
            if name in component_store:
                raise ComponentConflictError(
                    f"Component {component_type}:{name} already registered"
                )
        
        # 检查组件是否为None
        if component is None:
            raise ComponentRegistrationError("Component cannot be None")
    
    def _create_metadata(self, name: str, component_type: str, component: Any, 
                        extra_metadata: Optional[Dict] = None) -> ComponentMetadata:
        """创建组件元数据"""
        metadata = ComponentMetadata(
            name=name,
            component_type=component_type,
            description=getattr(component, '__doc__', '') or '',
        )
        
        if extra_metadata:
            for key, value in extra_metadata.items():
                if hasattr(metadata, key):
                    setattr(metadata, key, value)
        
        return metadata
    
    # 装饰器工厂
    def learner(self, name: str, **metadata) -> Callable:
        """学习器装饰器"""
        return self.register_learner(name, None, metadata)
    
    def aggregator(self, name: str, **metadata) -> Callable:
        """聚合器装饰器"""
        return self.register_aggregator(name, None, metadata)
    
    def evaluator(self, name: str, **metadata) -> Callable:
        """评估器装饰器"""
        return self.register_evaluator(name, None, metadata)
    
    def hook(self, phase: str, priority: int = 0, enable=False, **metadata) -> Callable:
        """钩子装饰器"""
        def decorator(hook_class: Type):
            with self._lock:
                self._validate_registration('hook', f"{phase}:{hook_class.__name__}", hook_class)
                
                validation_result = self.validate_component(hook_class, 'hook')
                if not validation_result.is_valid:
                    error_messages = [f"{error.field}: {error.message}" for error in validation_result.errors]
                    raise ComponentValidationError(
                        f"Hook {hook_class.__name__} validation failed: {'; '.join(error_messages)}"
                    )
                
                # 将钩子添加到对应阶段，按优先级排序
                hook_info = {
                    'class': hook_class,
                    'priority': priority,
                    'phase': phase,
                    "enabled": enable
                }
                
                self._hooks[phase].append(hook_info)
                self._hooks[phase].sort(key=lambda x: x['priority'], reverse=True)
                
                # 创建元数据（合并额外的metadata）
                hook_metadata = {'phase': phase, 'priority': priority, 'enabled': enable}
                hook_metadata.update(metadata)
                
                meta = self._create_metadata(
                    f"{phase}:{hook_class.__name__}", 'hook', hook_class, hook_metadata
                )
                self._metadata[f"hook:{phase}:{hook_class.__name__}"] = meta
                
                logger.debug(f"Registered hook: {hook_class.__name__} for phase {phase}")
                return hook_class
                
        return decorator
    
    def loss_function(self, name: str, scope: str = "task") -> Callable:
        """损失函数装饰器"""
        return self.register_loss_function(name, scope)
    
    def auxiliary_model(self, name: str, model_type: str) -> Callable:
        """辅助模型装饰器"""
        return self.register_auxiliary_model(name, model_type)
    
    # 统计和管理
    def get_registry_stats(self) -> Dict:
        """获取注册表统计信息"""
        with self._lock:
            return {
                'learners': len(self._learners),
                'aggregators': len(self._aggregators),
                'evaluators': len(self._evaluators),
                'hooks': sum(len(hooks) for hooks in self._hooks.values()),
                'loss_functions': len(self._loss_functions),
                'auxiliary_models': len(self._auxiliary_models),
                'query_stats': self._query_stats.copy()
            }
    
    def clear_registry(self):
        """清空注册表"""
        with self._lock:
            self._learners.clear()
            self._aggregators.clear()
            self._evaluators.clear()
            self._hooks.clear()
            self._loss_functions.clear()
            self._auxiliary_models.clear()
            self._metadata.clear()
            self._query_stats = {'total_queries': 0, 'cache_hits': 0}
            
            logger.debug("Registry cleared")
    
    def __repr__(self) -> str:
        stats = self.get_registry_stats()
        return f"ComponentRegistry(components={sum(stats[k] for k in stats if k != 'query_stats')})"


# 全局注册表实例
registry = ComponentRegistry()