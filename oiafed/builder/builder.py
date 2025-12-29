"""
组件构建器

职责：
1. 根据 NodeConfig 创建组件实例
2. 使用 registry.create(namespace, **kwargs) 实例化组件
3. 管理组件间的依赖关系
4. 支持自定义 BuildHook 扩展

设计原则：
- 只使用配置类，不使用字典操作
- registry 只提供 create(namespace, **kwargs)，builder 负责组装参数
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..config import (
    NodeConfig,
    ComponentConfig,
    DatasetConfig,
)


@dataclass
class BuildContext:
    """
    构建上下文
    
    存储配置和已构建的组件，在构建过程中传递。
    """
    config: NodeConfig
    _components: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def node_id(self) -> str:
        """获取节点 ID"""
        return self.config.node_id
    
    def set(self, name: str, component: Any) -> None:
        """注册已构建的组件"""
        self._components[name] = component
    
    def get(self, name: str, default: Any = None) -> Any:
        """获取已构建的组件"""
        return self._components.get(name, default)
    
    def has(self, name: str) -> bool:
        """检查组件是否已构建"""
        return name in self._components
    
    def get_all(self) -> Dict[str, Any]:
        """获取所有已构建的组件"""
        return self._components.copy()


class BuildHook:
    """
    构建钩子基类
    
    用于自定义组件初始化逻辑。
    """
    
    async def build(self, context: BuildContext) -> Any:
        """构建组件"""
        raise NotImplementedError


class ComponentBuilder:
    """
    组件构建器
    
    职责：
    - 根据 NodeConfig 创建组件实例
    - 使用 registry.create(namespace, **kwargs) 实例化
    - 管理组件依赖关系
    
    Example:
        from federation import load_config
        from federation.registry import registry
        
        config = load_config("trainer.yaml")
        builder = ComponentBuilder(registry)
        components = await builder.build(config)
    """
    
    # 默认构建顺序（按依赖关系排序）
    DEFAULT_BUILD_ORDER = [
        "datasets",      # 所有数据集（包含 train/test/valid）
        "model",         # 模型
        "aggregator",    # 聚合器
        "learner",       # 学习器（依赖 model, datasets）
    ]
    
    def __init__(self, registry=None):
        """
        初始化构建器
        
        Args:
            registry: 组件注册表实例（如果不提供，则导入全局 registry）
        """
        if registry is None:
            from ..registry import registry as global_registry
            self._registry = global_registry
        else:
            self._registry = registry
        
        self._hooks: Dict[str, BuildHook] = {}
        
        # 默认构建函数映射
        self._default_builders: Dict[str, Callable] = {
            "datasets": self._build_datasets,
            "model": self._build_model,
            "aggregator": self._build_aggregator,
            "learner": self._build_learner,
        }
    
    def register_hook(self, component_name: str, hook: BuildHook) -> None:
        """注册自定义构建钩子"""
        self._hooks[component_name] = hook
    
    async def build(self, config: NodeConfig) -> Dict[str, Any]:
        """
        构建所有组件
        
        Args:
            config: NodeConfig 实例
            
        Returns:
            构建好的组件字典，包含：
            - datasets: List[Dataset] 训练数据集列表
            - test_datasets: List[Dataset] 测试数据集列表  
            - model: Model 模型
            - aggregator: Aggregator 聚合器
            - learner: Learner 学习器
        """
        context = BuildContext(config=config)
        
        for component_name in self.DEFAULT_BUILD_ORDER:
            if not self._should_build(component_name, config):
                continue
            
            component = await self._build_component(component_name, context)
            
            if component is not None:
                context.set(component_name, component)
        
        return context.get_all()
    
    def _should_build(self, component_name: str, config: NodeConfig) -> bool:
        """检查是否应该构建该组件"""
        if component_name == "datasets":
            return bool(config.get_datasets())

        if component_name == "model":
            return config.model is not None

        if component_name == "aggregator":
            return config.aggregator is not None

        if component_name == "learner":
            return config.learner is not None

        return False
    
    async def _build_component(self, component_name: str, context: BuildContext) -> Any:
        """构建单个组件"""
        # 优先使用自定义钩子
        if component_name in self._hooks:
            return await self._hooks[component_name].build(context)

        # 使用默认构建函数
        if component_name in self._default_builders:
            return await self._default_builders[component_name](context)

        raise ValueError(f"No builder found for component: {component_name}")
    
    # ==================== 组件创建辅助方法 ====================
    
    def _create_component(
        self,
        category: str,
        component_config: ComponentConfig,
        **extra_kwargs,
    ) -> Any:
        """
        通用组件创建方法
        
        Args:
            category: 组件类别（如 "dataset", "model", "learner"）
            component_config: 组件配置
            **extra_kwargs: 额外参数（如依赖注入）
            
        Returns:
            创建的组件实例
        """
        # 解析 component_type
        # 支持三种格式：
        # 1. 简短格式: "mnist" -> "{category}.mnist"
        # 2. category.type 格式: "dataset.mnist" -> "dataset.mnist" (直接使用)
        # 3. 完整命名空间: "fl.horizontal.dataset.mnist" -> "fl.horizontal.dataset.mnist" (直接使用)

        component_type = component_config.type

        # 判断格式
        parts = component_type.split(".")

        if len(parts) == 1:
            # 格式1：简短格式 "mnist" -> "dataset.mnist"
            namespace = f"{category}.{component_type}"
        else:
            # 格式2和3：包含点号，直接使用
            # "dataset.mnist" -> "dataset.mnist"
            # "fl.horizontal.dataset.mnist" -> "fl.horizontal.dataset.mnist"
            namespace = component_type

        # 合并参数
        kwargs = component_config.get_args()
        kwargs.update(extra_kwargs)

        # 使用 registry 创建
        return self._registry.create(namespace, **kwargs)
    
    # ==================== 默认构建函数 ====================
    
    async def _build_datasets(self, context: BuildContext) -> Dict[str, List[Any]]:
        """
        构建所有数据集（train/test/valid），按 split 分组

        返回格式：
        {
            "train": [dataset1, dataset2, ...],
            "test": [dataset1, dataset2, ...],
            "valid": [dataset1, dataset2, ...]
        }

        每个 DatasetConfig 的 split 字段会自动注入到 args 中，
        数据集类会根据 split 参数加载对应的数据。

        Returns:
            Dict[str, List[Dataset]] 按 split 分组的数据集字典
        """
        config = context.config
        dataset_configs = config.get_datasets()  # 获取所有数据集

        if not dataset_configs:
            return {}

        # 按 split 分组
        datasets_by_split = {
            "train": [],
            "test": [],
            "valid": []
        }

        for i, ds_config in enumerate(dataset_configs):
            # 检查是否有划分配置
            if ds_config.partition:
                dataset = await self._build_partitioned_dataset(
                    ds_config,
                    context.node_id,
                    partition_index=i,
                )
            else:
                # 无划分，直接创建
                # split 参数已经在 ds_config.args 中（由 DatasetConfig.__post_init__ 注入）
                dataset = self._create_component("dataset", ds_config)

            # 根据 split 字段分组
            split = ds_config.split
            if split in datasets_by_split:
                datasets_by_split[split].append(dataset)

        return datasets_by_split
    
    async def _build_partitioned_dataset(
        self,
        ds_config: DatasetConfig,
        node_id: str,
        partition_index: int = 0,
    ) -> Any:
        """
        构建带划分的数据集

        Args:
            ds_config: 数据集配置
            node_id: 节点 ID（用于推断 partition_id）
            partition_index: 数据集索引
        """
        from ..data.partitioned_dataset import create_dataset_with_partition

        partition_config = ds_config.partition.copy()

        # 如果没有指定 partition_id，从 node_id 推断
        if "partition_id" not in partition_config:
            partition_id = self._infer_partition_id(node_id)
            partition_config["partition_id"] = partition_id

        # 准备数据集配置
        dataset_config = {
            'type': ds_config.type,
            'args': ds_config.get_args()
        }

        # 使用现有的 partitioned_dataset 模块创建带划分的数据集
        return create_dataset_with_partition(
            dataset_config=dataset_config,
            partition_config=partition_config
        )
    
    async def _build_model(self, context: BuildContext) -> Any:
        """构建模型"""
        config = context.config
        model_config = config.get_model_config()
        
        if not model_config:
            return None
        
        return self._create_component("model", model_config)
    
    async def _build_aggregator(self, context: BuildContext) -> Any:
        """构建聚合器"""
        config = context.config
        aggregator_config = config.get_aggregator_config()
        
        if not aggregator_config:
            return None
        
        return self._create_component("aggregator", aggregator_config)
    
    async def _build_learner(self, context: BuildContext) -> Any:
        """
        构建学习器

        依赖：model, datasets（dict 格式）
        """
        config = context.config
        learner_config = config.get_learner_config()

        if not learner_config:
            return None

        # 获取依赖
        model = context.get("model")
        datasets_dict = context.get("datasets", {})

        # 获取训练数据集（向后兼容旧的 data 参数）
        train_datasets = datasets_dict.get("train", [])
        train_dataset = train_datasets[0] if train_datasets else None

        # 日志警告
        from ..infra.logging import get_logger
        logger = get_logger(config.node_id, "builder")

        if model is None:
            logger.warning("Learner requires 'model', but it's not available.")

        if train_dataset is None:
            logger.warning("Learner requires 'train dataset', but it's not available.")

        # 创建 Learner
        # learner_config.args 中的参数（batch_size, learning_rate 等）应该放入 config 参数中
        # 而不是作为 kwargs 传递给 __init__
        learner_init_kwargs = {
            "model": model,
            "datasets": datasets_dict,       # 传递完整的数据集字典
            "config": learner_config.get_args(),  # learner 配置参数（batch_size, learning_rate, epochs, device 等）
            "tracker": None,                 # 由 FederatedSystem 注入
            "callbacks": None,               # 由 FederatedSystem 注入
            "node_id": config.node_id,
        }

        # 使用 registry 创建，但不传递 learner_config 的 args
        namespace = learner_config.type
        if "." not in namespace:
            namespace = f"learner.{namespace}"

        return self._registry.create(namespace, **learner_init_kwargs)
    
    def _infer_partition_id(self, node_id: str) -> int:
        """
        从 node_id 推断 partition_id
        
        Examples:
            "learner_0" -> 0
            "client_5" -> 5
        """
        if "_" in node_id:
            try:
                return int(node_id.split("_")[-1])
            except ValueError:
                return 0
        return 0
