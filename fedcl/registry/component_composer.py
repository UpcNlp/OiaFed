# fedcl/registry/component_composer.py
"""
组件组合器模块

负责根据配置和注册的组件创建实际的组件实例，管理组件的依赖关系
和生命周期，支持依赖注入和配置驱动的组件创建。
"""

from typing import Dict, List, Any, Type, Optional, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from threading import RLock, Lock
import threading

from loguru import logger

from .component_registry import ComponentRegistry
from ..config.config_manager import DictConfig
from ..config.schema_validator import ValidationResult, ValidationError
from ..core.execution_context import ExecutionContext
from ..exceptions import (
    FedCLError, ConfigurationError, ResourceError
)
from ..core.base_learner import BaseLearner
from ..core.base_aggregator import BaseAggregator
from ..core.base_evaluator import BaseEvaluator
from ..core.hook import Hook
from ..implementations.factory import ModelFactory


logger = logger


class ComponentCompositionError(FedCLError):
    """组件组合错误"""
    pass


class DependencyResolutionError(ComponentCompositionError):
    """依赖解析错误"""
    pass


class ComponentCreationError(ComponentCompositionError):
    """组件创建错误"""
    pass


class CircularDependencyError(DependencyResolutionError):
    """循环依赖错误"""
    pass


@dataclass
class ComponentSpec:
    """组件规格说明"""
    name: str
    component_type: str
    class_name: str
    config: Dict = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    singleton: bool = True
    lazy_init: bool = False
    
    def __post_init__(self):
        """后处理初始化"""
        if not self.name:
            raise ValueError("Component name cannot be empty")
        if not self.component_type:
            raise ValueError("Component type cannot be empty")
        if not self.class_name:
            raise ValueError("Class name cannot be empty")


@dataclass
class ExperimentComponents:
    """实验组件集合"""
    learner: Any = None
    aggregator: Any = None
    evaluator: Any = None
    hooks: Dict[str, List[Any]] = field(default_factory=lambda: defaultdict(list))
    loss_functions: Dict[str, Any] = field(default_factory=dict)
    auxiliary_models: Dict[str, Any] = field(default_factory=dict)
    context: ExecutionContext = None
    
    def get_component(self, component_type: str, name: str = None) -> Any:
        """获取组件实例"""
        if component_type == 'learner':
            return self.learner
        elif component_type == 'aggregator':
            return self.aggregator
        elif component_type == 'evaluator':
            return self.evaluator
        elif component_type == 'hook':
            return self.hooks.get(name, [])
        elif component_type == 'loss_function':
            return self.loss_functions.get(name)
        elif component_type == 'auxiliary_model':
            return self.auxiliary_models.get(name)
        else:
            raise ValueError(f"Unknown component type: {component_type}")
    
    def validate(self) -> bool:
        """验证组件完整性"""
        required_components = ['learner', 'aggregator', 'evaluator']
        for comp_type in required_components:
            if getattr(self, comp_type, None) is None:
                logger.warning(f"Missing required component: {comp_type}")
                return False
        return True
    
    def get_component_summary(self) -> Dict[str, str]:
        """
        获取组件摘要信息
        
        Returns:
            组件摘要字典，包含各组件的类型和名称
        """
        summary = {}
        
        # 核心组件
        if self.learner:
            summary['learner'] = f"{type(self.learner).__name__}"
        if self.aggregator:
            summary['aggregator'] = f"{type(self.aggregator).__name__}"
        if self.evaluator:
            summary['evaluator'] = f"{type(self.evaluator).__name__}"
        
        # 钩子函数
        if self.hooks:
            hook_summary = {}
            for phase, hook_list in self.hooks.items():
                if isinstance(hook_list, list):
                    hook_names = [hook.get('name', type(hook.get('instance')).__name__) 
                                for hook in hook_list if isinstance(hook, dict)]
                else:
                    hook_names = [type(hook_list).__name__]
                hook_summary[phase] = hook_names
            summary['hooks'] = hook_summary
        
        # 辅助模型
        if self.auxiliary_models:
            summary['auxiliary_models'] = {
                name: type(model).__name__ 
                for name, model in self.auxiliary_models.items()
            }
        
        # 损失函数
        if self.loss_functions:
            summary['loss_functions'] = {
                name: getattr(func, '__name__', str(type(func)))
                for name, func in self.loss_functions.items()
            }
        
        return summary


class ComponentComposer:
    """
    组件组合器
    
    负责根据配置创建和组装实验所需的各种组件，管理组件间的依赖关系，
    支持单例模式、懒加载和依赖注入等特性。
    """
    
    def __init__(self, registry: ComponentRegistry):
        """
        初始化组件组合器
        
        Args:
            registry: 组件注册表
        """
        self.registry = registry
        self._instance_cache: Dict[str, Any] = {}
        self._dependency_graph: Dict[str, List[str]] = defaultdict(list)
        self._creation_order: List[str] = []
        self._component_specs: Dict[str, 'ComponentSpec'] = {}
        
        # 线程安全锁
        self._lock = RLock()
        self._cache_lock = Lock()
        
        logger.debug("ComponentComposer initialized")
    
    def compose_experiment(self, config: DictConfig, context: ExecutionContext) -> ExperimentComponents:
        """
        组合实验组件
        
        Args:
            config: 实验配置
            context: 执行上下文
            
        Returns:
            实验组件集合
            
        Raises:
            ComponentCompositionError: 组合失败
        """
        with self._lock:
            try:
                logger.debug("Starting experiment composition")
                
                # 清理缓存
                self._instance_cache.clear()
                self._dependency_graph.clear()
                self._creation_order.clear()
                self._component_specs.clear()
                
                # 创建组件集合
                components = ExperimentComponents(context=context)
                
                # 1. 首先创建辅助组件（在核心组件之前）
                self._create_auxiliary_components(components, context)
                
                # 2. 创建核心组件
                self._create_core_components(components, context)
                
                # 验证组件完整性
                if not components.validate():
                    raise ComponentCompositionError("Component composition validation failed")
                
                logger.debug("Experiment composition 成功完成")
                return components
                
            except Exception as e:
                logger.error(f"Experiment composition failed: {e}")
                raise ComponentCompositionError(f"Failed to compose experiment: {str(e)}") from e
    def _create_core_components(self, components: ExperimentComponents, 
                              context: ExecutionContext) -> None:
        """
        创建核心组件
        
        包括learner、aggregator和evaluator。这些组件在辅助组件之后创建，
        可以使用已创建的辅助组件。
        
        Args:
            components: 实验组件集合
            context: 执行上下文
        """
        # 创建learner
        components.learner = self.create_learner(context.config, context)
        
        # 创建aggregator
        components.aggregator = self.create_aggregator(context.config, context)
        
        # 创建evaluator
        components.evaluator = self.create_evaluator(context.config, context)

    
    
    def _create_auxiliary_components(self, components: ExperimentComponents, 
                                   context: ExecutionContext) -> None:
        """
        创建辅助组件
        
        包括辅助模型、损失函数和钩子函数。这些组件需要在核心组件之前创建，
        因为核心组件可能依赖这些辅助组件。
        
        Args:
            components: 实验组件集合
            context: 执行上下文
        """
        # 1. 创建辅助模型
        aux_model_configs = context.config.get('auxiliary_models', {})
        if aux_model_configs:
            auxiliary_models = self.create_auxiliary_models(aux_model_configs, context)
            components.auxiliary_models = auxiliary_models
            # 将辅助模型注册到上下文中，供其他组件使用
            context.set_state('auxiliary_models', auxiliary_models)
            logger.debug(f"Created {len(auxiliary_models)} auxiliary models")
        
        # 2. 创建损失函数
        loss_configs = context.config.get('loss_functions', {})
        if loss_configs:
            loss_functions = self.prepare_loss_functions(loss_configs)
            components.loss_functions = loss_functions
            context.set_state('loss_functions', loss_functions)
            logger.debug(f"Created {len(loss_functions)} loss functions")
        
        # 3. 创建钩子函数
        hooks = self.create_hooks(context.config, context)
        components.hooks = hooks
        logger.debug(f"Created hooks for {len(hooks)} phases")
    
    def create_auxiliary_models(self, model_configs: Dict[str, Dict], 
                               context: ExecutionContext) -> Dict[str, Any]:
        """
        创建辅助模型 - 简化版本
        
        只通过registry创建模型，移除ModelFactory的复杂性。
        
        Args:
            model_configs: 模型配置字典
            context: 执行上下文
            
        Returns:
            辅助模型实例字典
        """
        auxiliary_models = {}
        
        for model_name, model_config in model_configs.items():
            try:
                # 通过registry创建模型
                model_class_name = model_config.get('class', model_name)
                model_class = self.registry.get_component('auxiliary_model', model_class_name)
                
                # 准备初始化参数
                init_kwargs = self._prepare_init_kwargs(model_config, context, 'auxiliary_model')
                
                # 创建模型实例
                model_instance = model_class(**init_kwargs)
                auxiliary_models[model_name] = model_instance
                
                # 注册到上下文
                context.register_auxiliary_model(model_name, model_instance)
                
                logger.debug(f"Created auxiliary model: {model_name}")
                
            except Exception as e:
                logger.error(f"Failed to create auxiliary model {model_name}: {e}")
                raise ComponentCreationError(
                    f"Failed to create auxiliary model {model_name}: {str(e)}"
                ) from e
        
        return auxiliary_models
    
    
    
    def prepare_loss_functions(self, loss_configs: Dict[str, Dict]) -> Dict[str, Any]:
        """
        准备损失函数
        
        Args:
            loss_configs: 损失函数配置
            
        Returns:
            损失函数字典
        """
        loss_functions = {}
        
        for loss_name, loss_config in loss_configs.items():
            try:
                # 获取损失函数
                func_name = loss_config.get('function', loss_name)
                loss_function = self.registry.get_component('loss_function', func_name)
                
                # 创建损失函数包装器（如果需要配置）
                if 'params' in loss_config:
                    loss_functions[loss_name] = self._create_loss_wrapper(
                        loss_function, loss_config['params']
                    )
                else:
                    loss_functions[loss_name] = loss_function
                
                logger.debug(f"Prepared loss function: {loss_name}")
                
            except Exception as e:
                logger.error(f"Failed to prepare loss function {loss_name}: {e}")
                raise ComponentCreationError(
                    f"Failed to prepare loss function {loss_name}: {str(e)}"
                ) from e
        
        return loss_functions
    
    def create_learner(self, config: DictConfig, context: ExecutionContext) -> Any:
        """
        创建学习器
        
        Args:
            config: 学习器配置
            context: 执行上下文
            
        Returns:
            学习器实例
        """
        try:
            learner_config = config.get('learner', {})
            learner_class_name = learner_config.get('class')
            
            if not learner_class_name:
                raise ComponentCreationError("Learner class not specified in config")
            
            # 获取学习器类
            learner_class = self.registry.get_component('learner', learner_class_name)
            
            # 准备初始化参数（支持新的模型初始化方式）
            init_kwargs = self._prepare_learner_init_kwargs(learner_config, context)
            
            # 创建学习器实例
            learner = learner_class(**init_kwargs)
            
            logger.debug(f"Created learner: {learner_class_name}")
            return learner
            
        except Exception as e:
            logger.error(f"Failed to create learner: {e}")
            raise ComponentCreationError(f"Failed to create learner: {str(e)}") from e
    
    def _prepare_learner_init_kwargs(self, config: Dict, context: ExecutionContext) -> Dict[str, Any]:
        """
        为learner准备初始化参数
        
        支持新的模型初始化方式，传入auxiliary_models等参数。
        
        Args:
            config: learner配置
            context: 执行上下文
            
        Returns:
            初始化参数字典
        """
        init_kwargs = {
            'context': context,
            'config': DictConfig(config)
        }
        
        # 传入辅助模型（用于模型初始化）
        auxiliary_models = context.get_state('auxiliary_models') or {}
        if auxiliary_models:
            init_kwargs['auxiliary_models'] = auxiliary_models
            logger.debug(f"Passing {len(auxiliary_models)} auxiliary models to learner")
        
        # 传入损失函数（如果有）
        loss_functions = context.get_state('loss_functions') or {}
        if loss_functions:
            init_kwargs['loss_functions'] = loss_functions
            logger.debug(f"Passing {len(loss_functions)} loss functions to learner")
        
        return init_kwargs
    
    def create_aggregator(self, config: DictConfig, context: ExecutionContext) -> Any:
        """创建聚合器"""
        try:
            aggregator_config = config.get('aggregator', {})
            aggregator_class_name = aggregator_config.get('class')
            
            if not aggregator_class_name:
                raise ComponentCreationError("聚合器 class not specified in config")
            
            aggregator_class = self.registry.get_component('aggregator', aggregator_class_name)
            init_kwargs = self._prepare_init_kwargs(aggregator_config, context, 'aggregator')
            aggregator = aggregator_class(**init_kwargs)
            
            logger.debug(f"Created aggregator: {aggregator_class_name}")
            return aggregator
            
        except Exception as e:
            raise ComponentCreationError(f"Failed to create aggregator: {str(e)}") from e
    
    def create_evaluator(self, config: DictConfig, context: ExecutionContext) -> Any:
        """创建评估器"""
        try:
            evaluator_config = config.get('evaluator', {})
            evaluator_class_name = evaluator_config.get('class')
            
            if not evaluator_class_name:
                raise ComponentCreationError("Evaluator class not specified in config")
            
            evaluator_class = self.registry.get_component('evaluator', evaluator_class_name)
            init_kwargs = self._prepare_init_kwargs(evaluator_config, context, 'evaluator')
            evaluator = evaluator_class(**init_kwargs)
            
            logger.debug(f"Created evaluator: {evaluator_class_name}")
            return evaluator
            
        except Exception as e:
            raise ComponentCreationError(f"Failed to create evaluator: {str(e)}") from e
        
    def create_hooks(self, config: DictConfig, context: ExecutionContext) -> Dict[str, List[Any]]:
        """
        创建钩子函数
        
        Args:
            config: 配置
            context: 执行上下文
            
        Returns:
            钩子函数字典
        """
        hooks = defaultdict(list)
        
        # 从注册表获取所有钩子
        for phase, hook_list in self.registry._hooks.items():
            for hook_info in hook_list:
                try:
                    hook_class = hook_info['class']
                    priority = hook_info['priority']
                    
                    # 构建默认配置
                    hook_config = {
                        'phase': phase,
                        'priority': priority,
                        'enabled': False,  # 默认不启用
                        'name': hook_info.get('name', hook_class.__name__),
                        'class_name': hook_class.__name__
                    }
                    
                    # 检查用户配置中是否有特定的钩子配置
                    user_config = config.get('hooks', {}).get(hook_class.__name__, {})
                    hook_config.update(user_config)
                    
                    # 如果钩子被禁用，跳过
                    if not hook_config.get('enabled', False):
                        logger.debug(f"Hook {hook_class.__name__} disabled in config")
                        continue
                    
                    # 准备初始化参数
                    init_kwargs = self._prepare_init_kwargs(hook_config, context, 'hook')
                    
                    # 创建钩子实例
                    hook_instance = hook_class(**init_kwargs)
                    hooks[phase].append({
                        'instance': hook_instance,
                        'priority': priority,
                        'name': hook_class.__name__
                    })
                    
                    logger.debug(f"Created hook: {hook_class.__name__} for phase {phase}")
                    
                except Exception as e:
                    logger.error(f"Failed to create hook {hook_class.__name__}: {e}")
                    # 钩子创建失败不应该阻止整个实验
                    continue
        
        # 按优先级排序
        for phase in hooks:
            hooks[phase].sort(key=lambda x: x['priority'], reverse=True)
        
        return dict(hooks)
    
    def _parse_component_specs(self, config: DictConfig) -> Dict[str, ComponentSpec]:
        """解析组件规格"""
        specs = {}
        
        # 解析核心组件
        for component_type in ['learner', 'aggregator', 'evaluator']:
            comp_config = config.get(component_type, {})
            if comp_config:
                if isinstance(comp_config, dict):
                    class_name = comp_config.get('class')  
                    dependencies = comp_config.get('dependencies', [])
                    comp_config_ = comp_config
                else:
                    class_name = comp_config
                    dependencies = []  
                    comp_config_ = {} 
                if class_name:
                    spec = ComponentSpec(
                        name=component_type,
                        component_type=component_type,
                        class_name=class_name,
                        config=comp_config_,
                        dependencies=dependencies
                    )
                    specs[component_type] = spec
        
        # 解析辅助模型
        aux_models = config.get('auxiliary_models', {})
        for model_name, model_config in aux_models.items():
            class_name = model_config.get('class', model_name)
            spec = ComponentSpec(
                name=model_name,
                component_type='auxiliary_model',
                class_name=class_name,
                config=model_config,
                dependencies=model_config.get('dependencies', [])
            )
            specs[f"auxiliary_model:{model_name}"] = spec
        
        self._component_specs = specs
        return specs
    
    def _build_dependency_graph(self, specs: Dict[str, ComponentSpec]):
        """构建依赖图"""
        self._dependency_graph.clear()
        
        for spec_name, spec in specs.items():
            self._dependency_graph[spec_name] = spec.dependencies.copy()
    
    def _check_circular_dependencies(self):
        """检查循环依赖"""
        def has_cycle(graph):
            WHITE, GRAY, BLACK = 0, 1, 2
            color = {node: WHITE for node in graph}
            
            def dfs(node):
                if color[node] == GRAY:
                    return True  # 发现环
                if color[node] == BLACK:
                    return False
                
                color[node] = GRAY
                for neighbor in graph.get(node, []):
                    if neighbor in graph and dfs(neighbor):
                        return True
                color[node] = BLACK
                return False
            
            for node in graph:
                if color[node] == WHITE and dfs(node):
                    return True
            return False
        
        if has_cycle(self._dependency_graph):
            raise CircularDependencyError("Circular dependency detected in component graph")
    
    def _determine_creation_order(self):
        """确定创建顺序（拓扑排序）"""
        graph = self._dependency_graph.copy()
        in_degree = defaultdict(int)
        
        # 计算入度：对于依赖关系A->B（A依赖B），B不增加入度，A增加入度
        # 因为B必须先创建，A后创建
        for node in graph:
            # node依赖于graph[node]中的所有组件
            # 所以node的入度等于它的依赖数量
            in_degree[node] = len([dep for dep in graph[node] if dep in graph])
        
        # 确保所有节点都在入度表中
        for node in graph:
            if node not in in_degree:
                in_degree[node] = 0
        
        # 初始化队列：入度为0的节点（没有依赖的节点）
        queue = deque([node for node in graph if in_degree[node] == 0])
        order = []
        
        while queue:
            # 取出一个入度为0的节点（没有依赖或依赖已满足）
            node = queue.popleft()
            order.append(node)
            
            # 对于依赖当前节点的其他节点，减少它们的入度
            for other_node in graph:
                if node in graph[other_node]:  # other_node依赖当前node
                    in_degree[other_node] -= 1
                    if in_degree[other_node] == 0:
                        queue.append(other_node)
        
        if len(order) != len(graph):
            raise DependencyResolutionError("Failed to determine component creation order")
        
        self._creation_order = order
    
    def _create_components(self, context: ExecutionContext) -> ExperimentComponents:
        """创建组件 - 修复版本"""
        components = ExperimentComponents(context=context)
        
        # 重要：首先创建辅助组件（包括模型）
        self._create_additional_components(components, context)
        
        # 然后按顺序创建核心组件（它们可能依赖辅助模型）
        for spec_name in self._creation_order:
            spec = self._component_specs[spec_name]
            
            try:
                instance = self._create_component_instance(spec, context)
                self._instance_cache[spec_name] = instance
                
                # 将组件添加到结果中
                self._add_component_to_result(components, spec, instance)
                
            except Exception as e:
                logger.error(f"Failed to create component {spec_name}: {e}")
                raise ComponentCreationError(f"Failed to create component {spec_name}: {str(e)}") from e
        
        return components
    
    def _create_component_instance(self, spec: ComponentSpec, context: ExecutionContext) -> Any:
        """创建组件实例"""
        try:
            print(spec)
            # 获取组件类
            component_class = self.registry.get_component(spec.component_type, spec.class_name)
            
            # 准备初始化参数
            init_kwargs = self._prepare_init_kwargs(spec.config, context, spec.component_type)
            
            # 注入依赖
            self._inject_dependencies(init_kwargs, spec, context)
            
            # 创建实例
            instance = component_class(**init_kwargs)
            
            logger.debug(f"Created component: {spec.name} ({spec.class_name})")
            return instance
            
        except Exception as e:
            raise ComponentCreationError(f"Failed to create {spec.name}: {str(e)}") from e
    
    def _find_hook_info(self, hook_class_name: str, config: Dict) -> Optional[Dict]:
        """查找hook信息"""
        try:
            # 方案1: 如果配置中指定了phase
            specified_phase = config.get('phase')
            if specified_phase:
                for hook_info in self.registry._hooks.get(specified_phase, []):
                    if hook_info['class'].__name__ == hook_class_name:
                        return hook_info
            
            # 方案2: 如果没有指定phase，查找第一个匹配的
            for phase, hook_list in self.registry._hooks.items():
                for hook_info in hook_list:
                    if hook_info['class'].__name__ == hook_class_name:
                        return hook_info
            
            return None
        except Exception as e:
            logger.error(f"Failed to find hook info for {hook_class_name}: {e}")
            return None
    
    def _prepare_init_kwargs(self, config: Dict, context: ExecutionContext, 
                           component_type: str) -> Dict[str, Any]:
        """
        准备初始化参数
        
        Args:
            config: 组件配置
            context: 执行上下文
            component_type: 组件类型
            
        Returns:
            初始化参数字典
        """
        init_kwargs = {}
        
        # 根据组件类型准备不同的参数
        if component_type in ['learner', 'aggregator', 'evaluator']:
            # 对于核心组件，传递context和config
            init_kwargs = {
                'context': context,
                'config': DictConfig(config)
            }
        elif component_type == 'hook':
            init_kwargs.update({
                'phase': config.get('phase', 'default'),
                'priority': config.get('priority', 0),
                'enabled': config.get('enabled', False),
                'name': config.get('name', 'None')
            })
        else:
            # 对于其他组件（auxiliary_models等），传递直接配置参数
            init_kwargs.update(config.get('params', {}))
            init_kwargs['config'] = config
        
        return init_kwargs
    
    def _inject_dependencies(self, init_kwargs: Dict, spec: ComponentSpec, 
                           context: ExecutionContext):
        """注入依赖"""
        for dep_name in spec.dependencies:
            if dep_name in self._instance_cache:
                # 直接注入依赖实例
                init_kwargs[dep_name] = self._instance_cache[dep_name]
            elif dep_name in context.get_all_states():
                # 从上下文获取依赖
                init_kwargs[dep_name] = context.get_state(dep_name)
            else:
                logger.warning(f"Dependency {dep_name} not found for component {spec.name}")
    
    def _add_component_to_result(self, components: ExperimentComponents, 
                               spec: ComponentSpec, instance: Any):
        """将组件添加到结果中"""
        if spec.component_type == 'learner':
            components.learner = instance
        elif spec.component_type == 'aggregator':
            components.aggregator = instance
        elif spec.component_type == 'evaluator':
            components.evaluator = instance
        elif spec.component_type == 'auxiliary_model':
            components.auxiliary_models[spec.name] = instance
    
    def _create_additional_components(self, components: ExperimentComponents, 
                                    context: ExecutionContext):
        """创建其他类型的组件"""
        # 1. 首先创建辅助模型（在其他组件之前）
        aux_model_configs = context.config.get('auxiliary_models', {})
        if aux_model_configs:
            auxiliary_models = self.create_auxiliary_models(aux_model_configs, context)
            components.auxiliary_models = auxiliary_models
            # 重要：将辅助模型注册到上下文中，供其他组件使用
            context.set_state('auxiliary_models', auxiliary_models)
        
        # 2. 创建钩子函数
        hooks = self.create_hooks(context.config, context)
        components.hooks = hooks
        
        # 3. 创建损失函数
        loss_configs = context.config.get('loss_functions', {})
        if loss_configs:
            loss_functions = self.prepare_loss_functions(loss_configs)
            components.loss_functions = loss_functions
    
    def _create_loss_wrapper(self, loss_function: callable, params: Dict) -> callable:
        """
        创建损失函数包装器
        
        Args:
            loss_function: 原始损失函数
            params: 参数字典
            
        Returns:
            包装后的损失函数
        """
        def wrapped_loss(*args, **kwargs):
            # 合并参数
            merged_kwargs = {**params, **kwargs}
            return loss_function(*args, **merged_kwargs)
        
        return wrapped_loss
    
    def get_cached_instance(self, component_name: str) -> Optional[Any]:
        """获取缓存的实例"""
        return self._instance_cache.get(component_name)
    
    def clear_cache(self) -> None:
        """清理缓存"""
        self._instance_cache.clear()
        self._dependency_graph.clear()
        self._creation_order.clear()
        self._component_specs.clear()
        
        logger.debug("Component composer cache cleared")
    
    def get_composition_stats(self) -> Dict:
        """获取组合统计信息"""
        return {
            'cached_instances': len(self._instance_cache),
            'dependency_graph_size': len(self._dependency_graph),
            'creation_order': len(self._creation_order),
            'component_specs': len(self._component_specs)
        }
    
    def __repr__(self) -> str:
        stats = self.get_composition_stats()
        return f"ComponentComposer(cached={stats['cached_instances']}, specs={stats['component_specs']})"