# FedCL详细实现规划 - 类级别指导文档

## 📋 实现原则与约束

### 核心原则
1. **依赖优先**: 被依赖的类优先实现
2. **接口先行**: 抽象接口先于具体实现
3. **配置驱动**: 所有行为都应可配置
4. **测试并行**: 每个类实现的同时编写测试
5. **文档同步**: 代码与文档保持同步
6. **依赖明确**: 所有依赖关系在构造函数中明确声明
7. **接口隔离**: 依赖抽象接口而非具体实现

### 全局约束
- **Python 3.8+**: 支持类型注解和异步编程
- **线程安全**: 所有共享状态类必须线程安全
- **序列化**: 所有数据类支持pickle序列化
- **配置化**: 所有可变行为通过配置控制
- **日志集成**: 所有类集成统一日志系统

## 🔗 依赖管理规范

### 依赖注入模式
所有类的依赖关系通过构造函数注入，遵循以下模式：

```python
class DependentClass:
    def __init__(self, 
                 required_dependency: RequiredInterface,
                 optional_dependency: Optional[OptionalInterface] = None,
                 config: DictConfig = None):
        self.required_dependency = required_dependency
        self.optional_dependency = optional_dependency
        self.config = config or DictConfig({})
        
    def some_method(self):
        # 调用依赖的方法
        result = self.required_dependency.some_interface_method()
        return result
```

### 依赖关系声明
每个类的实现规划中增加以下部分：

#### **依赖声明格式:**
- **必需依赖**: 构造函数中必须传入的依赖
- **可选依赖**: 可以为None的依赖，有默认行为
- **配置依赖**: 通过配置获取的依赖
- **运行时依赖**: 运行时动态获取的依赖

#### **调用关系说明:**
- **直接调用**: 通过依赖对象直接调用方法
- **事件调用**: 通过事件系统间接调用
- **回调调用**: 通过回调函数调用
- **异步调用**: 通过异步机制调用

### 依赖生命周期管理
- **Singleton**: 全局单例，如ConfigManager
- **Scoped**: 作用域内单例，如ExecutionContext
- **Transient**: 每次创建新实例，如Task
- **Factory**: 通过工厂创建，如各种算法组件

---

## 🏗️ 第一阶段: 基础工具层

### Step 1.1: FileUtils 实现

**文档输入:**
- `目录结构.md` - 文件组织规范
- `数据与配置管理.md` - 文件操作需求

**依赖关系:**
- **必需依赖**: 无（基础工具类）
- **可选依赖**: 无
- **配置依赖**: 无
- **运行时依赖**: 无

**类职责:**
- 文件系统操作封装
- 数据序列化/反序列化
- 目录管理和清理

**调用关系说明:**
- **被调用者**: 作为工具类被其他所有需要文件操作的类调用
- **调用方式**: 静态方法调用，无需实例化
- **典型调用场景**: 
  - `ConfigManager.load_config()` 中调用 `FileUtils.load_json()`
  - `CheckpointManager.save_checkpoint()` 中调用 `FileUtils.save_pickle()`
  - `DatasetManager.cache_dataset()` 中调用 `FileUtils.ensure_dir_exists()`

**依赖注入实现:**
```python
# 无需依赖注入，纯静态工具类
class FileUtils:
    @staticmethod
    def ensure_dir_exists(path: Union[str, Path]) -> Path:
        # 实现逻辑
        pass
```

**必须实现的方法:**
```python
class FileUtils:
    @staticmethod
    def ensure_dir_exists(path: Union[str, Path]) -> Path
    @staticmethod
    def save_json(data: Dict, path: Union[str, Path], indent: int = 2) -> None
    @staticmethod
    def load_json(path: Union[str, Path]) -> Dict
    @staticmethod
    def save_pickle(obj: Any, path: Union[str, Path]) -> None
    @staticmethod
    def load_pickle(path: Union[str, Path]) -> Any
    @staticmethod
    def get_file_size(path: Union[str, Path]) -> int
    @staticmethod
    def compress_file(source: Path, target: Path) -> None
    @staticmethod
    def decompress_file(source: Path, target: Path) -> None
    @staticmethod
    def safe_remove(path: Union[str, Path]) -> bool
    @staticmethod
    def atomic_write(content: Union[str, bytes], path: Path) -> None
```

**配置要求:**
- 支持配置默认压缩级别
- 支持配置文件权限设置
- 支持配置原子写入的临时目录

**约束条件:**
- 所有文件操作必须异常安全
- 支持原子写入操作
- 必须处理文件权限问题
- 支持大文件操作（>1GB）

**验收标准:**
- [ ] 处理所有常见文件系统异常
- [ ] 支持原子写入操作
- [ ] 内存使用低于100MB处理1GB文件
- [ ] 单元测试覆盖率>95%

### Step 1.2: 基础数据结构类

**文档输入:**
- `数据与配置管理.md` - 数据结构设计
- `联邦训练核心流程图.md` - 数据流转需求

#### Task 数据结构

**必须实现的接口:**
```python
@dataclass
class Task:
    task_id: int
    data: DataLoader
    classes: List[int]
    metadata: Dict[str, Any]
    task_type: TaskType
    
    def get_class_distribution(self) -> Dict[int, int]
    def get_sample_count(self) -> int
    def get_memory_usage(self) -> float
    def validate(self) -> bool
    def to_dict(self) -> Dict
    def from_dict(cls, data: Dict) -> 'Task'
    def __eq__(self, other) -> bool
    def __hash__(self) -> int
```

**配置要求:**
- 支持任务类型枚举配置
- 支持元数据验证规则配置

**约束条件:**
- 必须支持深拷贝
- 必须是不可变对象（除了metadata）
- 必须支持序列化

#### Dataset 包装类

**必须实现的接口:**
```python
class Dataset:
    def __init__(self, name: str, data: torch.Tensor, targets: torch.Tensor, 
                 transform: Optional[Callable] = None)
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]
    def __len__(self) -> int
    def get_classes(self) -> List[int]
    def get_class_distribution(self) -> Dict[int, int]
    def apply_transform(self, transform: Callable) -> 'Dataset'
    def subset(self, indices: List[int]) -> 'Dataset'
    def split(self, ratios: List[float]) -> List['Dataset']
    def validate_integrity(self) -> bool
    def get_stats(self) -> Dict[str, Any]
```

**配置要求:**
- 支持变换管道配置
- 支持数据验证规则配置

**约束条件:**
- 与PyTorch Dataset兼容
- 支持延迟加载
- 内存使用优化

#### DataLoader 包装类

**必须实现的接口:**
```python
class DataLoader:
    def __init__(self, dataset: Dataset, batch_size: int = 32, 
                 shuffle: bool = False, num_workers: int = 0, **kwargs)
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]
    def __len__(self) -> int
    def set_epoch(self, epoch: int) -> None
    def get_batch_size(self) -> int
    def get_dataset_size(self) -> int
```

**约束条件:**
- 与PyTorch DataLoader兼容
- 支持多进程加载
- 支持动态批次大小调整

#### ExperimentResults 和 TaskResults

**必须实现的接口:**
```python
@dataclass
class TaskResults:
    task_id: int
    metrics: Dict[str, float]
    training_time: float
    memory_usage: float
    model_size: int
    convergence_step: int
    metadata: Dict[str, Any]
    
    def get_metric(self, name: str, default: Any = None) -> Any
    def to_dict(self) -> Dict
    def from_dict(cls, data: Dict) -> 'TaskResults'
    def merge_with(self, other: 'TaskResults') -> 'TaskResults'

@dataclass 
class ExperimentResults:
    experiment_id: str
    config: DictConfig
    metrics: Dict[str, List[float]]
    task_results: List[TaskResults]
    checkpoints: List[Path]
    artifacts: Dict[str, Any]
    start_time: datetime
    end_time: Optional[datetime]
    
    def save_to_file(self, path: Path) -> None
    def load_from_file(cls, path: Path) -> 'ExperimentResults'
    def generate_summary(self) -> Dict
    def add_task_result(self, result: TaskResults) -> None
    def get_best_result(self, metric: str) -> TaskResults
    def plot_learning_curves(self, save_path: Optional[Path] = None) -> None
```

**配置要求:**
- 支持结果格式配置
- 支持图表样式配置

**约束条件:**
- 必须支持增量更新
- 必须支持大规模实验结果
- 必须支持结果可视化

---

## ⚙️ 第二阶段: 配置管理系统

### Step 2.1: SchemaValidator 实现

**文档输入:**
- `数据与配置管理.md` - 配置验证设计
- `组件注册与实验初始化交互.md` - 配置验证流程

**必须实现的接口:**
```python
class SchemaValidator:
    def __init__(self, schema_path: Optional[Path] = None)
    def validate_experiment_config(self, config: Dict) -> ValidationResult
    def validate_model_config(self, config: Dict) -> ValidationResult
    def validate_data_config(self, config: Dict) -> ValidationResult
    def validate_communication_config(self, config: Dict) -> ValidationResult
    def check_required_fields(self, config: Dict, schema: Dict) -> List[str]
    def check_field_types(self, config: Dict, schema: Dict) -> List[str]
    def check_field_ranges(self, config: Dict, schema: Dict) -> List[str]
    def add_custom_validator(self, field: str, validator: Callable) -> None
    def get_validation_errors(self) -> List[ValidationError]
    def register_schema(self, name: str, schema: Dict) -> None
```

**配置要求:**
- 验证规则配置文件
- 自定义验证器注册
- 错误信息本地化配置

**约束条件:**
- 支持嵌套配置验证
- 验证错误信息必须具体明确
- 支持条件验证（字段间依赖）
- 性能要求：1000字段配置<100ms验证时间

**ValidationResult 数据结构:**
```python
@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[ValidationError]
    warnings: List[ValidationWarning]
    
    def add_error(self, field: str, message: str) -> None
    def add_warning(self, field: str, message: str) -> None
    def merge(self, other: 'ValidationResult') -> 'ValidationResult'
```

### Step 2.2: ConfigManager 实现

**文档输入:**
- `数据与配置管理.md` - 配置管理详细设计
- `组件注册与实验初始化交互.md` - 配置加载流程

**依赖关系:**
- **必需依赖**: 
  - `schema_validator: SchemaValidator` - 配置验证器
- **可选依赖**: 
  - `file_utils: Optional[Type[FileUtils]] = FileUtils` - 文件工具类（可mock）
- **配置依赖**: 无
- **运行时依赖**: 无

**调用关系说明:**
- **调用依赖的方式**:
  - `self.schema_validator.validate_experiment_config(config)` - 验证配置
  - `FileUtils.load_json(config_path)` - 加载配置文件
  - `FileUtils.save_json(config, path)` - 保存配置
- **被调用场景**:
  - `FedCLExperiment.__init__()` 中调用 `config_manager.load_config()`
  - `ComponentComposer.validate_configuration()` 中调用验证方法

**依赖注入实现:**
```python
class ConfigManager:
    def __init__(self, schema_validator: SchemaValidator):
        self.schema_validator = schema_validator
        self._config_cache: Dict[str, DictConfig] = {}
        self._config_history: List[Tuple[datetime, DictConfig]] = []
        
    def load_config(self, config_path: Path) -> DictConfig:
        # 使用依赖进行配置验证
        raw_config = FileUtils.load_json(config_path)
        validation_result = self.schema_validator.validate_experiment_config(raw_config)
        if not validation_result.is_valid:
            raise ConfigValidationError(validation_result.errors)
        return DictConfig(raw_config)
```

**测试策略:**
```python
# 单元测试时的mock方式
def test_config_manager():
    mock_validator = Mock(spec=SchemaValidator)
    mock_validator.validate_experiment_config.return_value = ValidationResult(is_valid=True)
    
    config_manager = ConfigManager(mock_validator)
    # 测试逻辑
```

**必须实现的接口:**
```python
class ConfigManager:
    def __init__(self, schema_validator: SchemaValidator)
    def load_config(self, config_path: Path) -> DictConfig
    def validate_config(self, config: DictConfig) -> ValidationResult
    def merge_configs(self, base: DictConfig, override: DictConfig) -> DictConfig
    def resolve_references(self, config: DictConfig) -> DictConfig
    def save_config(self, config: DictConfig, path: Path) -> None
    def get_nested_value(self, config: DictConfig, path: str, default: Any = None) -> Any
    def set_nested_value(self, config: DictConfig, path: str, value: Any) -> None
    def expand_environment_variables(self, config: DictConfig) -> DictConfig
    def register_config_hook(self, hook: Callable[[DictConfig], DictConfig]) -> None
    def get_config_history(self) -> List[Tuple[datetime, DictConfig]]
    def rollback_config(self, steps: int = 1) -> DictConfig
```

**配置管理器特殊要求:**
- 支持YAML、JSON、TOML格式
- 支持配置继承（base configs）
- 支持环境变量替换
- 支持配置模板和参数化
- 支持配置版本管理
- 支持热重载机制

**约束条件:**
- 配置加载必须是原子操作
- 支持配置回滚机制
- 必须检测配置循环引用
- 支持配置加密（敏感信息）

---

## 🎯 第三阶段: 核心抽象层

### Step 3.1: ExecutionContext 实现

**文档输入:**
- `核心框架层.md` - ExecutionContext接口设计
- `组件交互关键要点总结.md` - 状态管理机制
- `FedCL组件交互逻辑详解.md` - 上下文使用场景

**依赖关系:**
- **必需依赖**: 
  - `config: DictConfig` - 实验配置
  - `experiment_id: str` - 实验标识
- **可选依赖**: 
  - `communication_manager: Optional[CommunicationManager] = None` - 通信管理器
  - `metrics_logger: Optional[MetricsLogger] = None` - 度量记录器
- **配置依赖**: 
  - 状态存储后端（从config.state_storage获取）
  - 事件系统配置（从config.event_system获取）
- **运行时依赖**: 
  - 动态注册的资源和模型

**调用关系说明:**
- **调用依赖的方式**:
  - `self.communication_manager.send_data()` - 发送数据
  - `self.metrics_logger.log_scalar()` - 记录度量
  - `self.config.get('path.to.value', default)` - 获取配置
- **被调用场景**:
  - 所有业务类通过构造函数接收ExecutionContext
  - `BaseLearner.train_task()` 中调用 `context.log_metric()`
  - `CommunicationHandler` 中调用 `context.get_state()`

**依赖注入实现:**
```python
class ExecutionContext:
    def __init__(self, 
                 config: DictConfig, 
                 experiment_id: str,
                 communication_manager: Optional[CommunicationManager] = None,
                 metrics_logger: Optional[MetricsLogger] = None):
        self.config = config
        self.experiment_id = experiment_id
        self.communication_manager = communication_manager
        self.metrics_logger = metrics_logger
        
        # 内部状态管理
        self._global_state: Dict[str, Any] = {}
        self._local_states: Dict[str, Dict[str, Any]] = {}
        self._auxiliary_models: Dict[str, Any] = {}
        self._resources: Dict[str, Any] = {}
        
        # 从配置初始化
        self._init_from_config()
        
    def send_data(self, target: str, data: Any, data_type: str) -> None:
        if self.communication_manager is None:
            raise RuntimeError("CommunicationManager not available")
        self.communication_manager.send_data(
            source=self.experiment_id, 
            target=target, 
            data=data, 
            data_type=data_type
        )
        
    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        if self.metrics_logger is not None:
            self.metrics_logger.log_scalar(name, value, step or 0)
        # 总是在内部状态中记录
        self._record_metric_internally(name, value, step)
```

**依赖生命周期管理:**
- **ExecutionContext**: Scoped（实验作用域内单例）
- **CommunicationManager**: Singleton（全局单例）
- **MetricsLogger**: Scoped（实验作用域内单例）

**必须实现的接口:**
```python
class ExecutionContext:
    def __init__(self, config: DictConfig, experiment_id: str)
    
    # 状态管理
    def get_state(self, key: str, scope: str = "global") -> Any
    def set_state(self, key: str, value: Any, scope: str = "global") -> None
    def has_state(self, key: str, scope: str = "global") -> bool
    def clear_state(self, scope: str = "global") -> None
    def get_all_states(self, scope: str = "global") -> Dict[str, Any]
    
    # 模型管理
    def get_model(self, name: str) -> Any
    def register_auxiliary_model(self, name: str, model: Any) -> None
    def unregister_auxiliary_model(self, name: str) -> None
    def list_auxiliary_models(self) -> List[str]
    
    # 度量管理
    def log_metric(self, name: str, value: float, step: Optional[int] = None, 
                   scope: str = "global") -> None
    def get_metrics(self, scope: str = "global") -> Dict[str, List[float]]
    def get_metric_history(self, name: str, scope: str = "global") -> List[float]
    
    # 配置访问
    def get_config(self, path: str, default: Any = None) -> Any
    def update_config(self, path: str, value: Any) -> None
    
    # 通信接口
    def send_data(self, target: str, data: Any, data_type: str) -> None
    def request_data(self, source: str, data_type: str, timeout: float = 30.0) -> Any
    def broadcast_data(self, data: Any, data_type: str, targets: List[str]) -> None
    
    # 资源管理
    def register_resource(self, name: str, resource: Any) -> None
    def get_resource(self, name: str) -> Any
    def cleanup_resources(self) -> None
    
    # 事件系统
    def emit_event(self, event: str, data: Any = None) -> None
    def subscribe_event(self, event: str, callback: Callable) -> str
    def unsubscribe_event(self, subscription_id: str) -> None
    
    # 生命周期
    def initialize(self) -> None
    def cleanup(self) -> None
    def save_state(self, path: Path) -> None
    def load_state(self, path: Path) -> None
```

**状态管理特殊要求:**
- **线程安全**: 所有状态操作必须线程安全
- **作用域隔离**: 支持global、client、task等作用域
- **状态持久化**: 支持状态保存和恢复
- **状态监控**: 支持状态变化监听和通知

**配置要求:**
- 状态存储后端配置（内存、Redis、文件）
- 状态清理策略配置
- 事件系统配置

**约束条件:**
- 内存使用监控，防止状态无限增长
- 状态访问性能要求：<1ms
- 支持分布式状态同步
- 必须支持状态回滚

### Step 3.2: 基础抽象类实现

**文档输入:**
- `组件与扩展系统.md` - 基类接口设计
- `Hook系统与扩展机制交互.md` - Hook系统设计

#### BaseLearner 抽象基类

**必须实现的接口:**
```python
from abc import ABC, abstractmethod

class BaseLearner(ABC):
    def __init__(self, context: ExecutionContext, config: DictConfig)
    
    @abstractmethod
    def train_task(self, task_data: DataLoader) -> TaskResults:
        """训练单个任务"""
        pass
    
    @abstractmethod
    def evaluate_task(self, task_data: DataLoader) -> Dict[str, float]:
        """评估单个任务"""
        pass
    
    # 钩子方法（可选重写）
    def before_task_hook(self, task_id: int, task_data: DataLoader) -> None:
        """任务开始前的钩子"""
        pass
    
    def after_task_hook(self, task_id: int, results: TaskResults) -> None:
        """任务结束后的钩子"""
        pass
    
    def before_epoch_hook(self, epoch: int) -> None:
        """轮次开始前的钩子"""
        pass
    
    def after_epoch_hook(self, epoch: int, metrics: Dict[str, float]) -> None:
        """轮次结束后的钩子"""
        pass
    
    # 模型管理
    def get_model(self) -> torch.nn.Module:
        """获取当前模型"""
        return self.model
    
    def set_model(self, model: torch.nn.Module) -> None:
        """设置模型"""
        self.model = model
    
    def get_optimizer(self) -> torch.optim.Optimizer:
        """获取优化器"""
        return self.optimizer
    
    # 配置访问
    def get_learning_rate(self) -> float:
        return self.context.get_config("learning_rate", 0.001)
    
    def get_batch_size(self) -> int:
        return self.context.get_config("batch_size", 32)
    
    # 状态管理
    def save_learner_state(self) -> Dict[str, Any]:
        """保存学习器状态"""
        return {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "config": self.config
        }
    
    def load_learner_state(self, state: Dict[str, Any]) -> None:
        """加载学习器状态"""
        self.model.load_state_dict(state["model_state"])
        self.optimizer.load_state_dict(state["optimizer_state"])
```

**特殊要求:**
- 支持模型检查点保存和加载
- 支持训练状态恢复
- 支持动态学习率调整
- 支持早停机制

**约束条件:**
- 必须是线程安全的
- 必须支持GPU/CPU自动切换
- 内存使用必须可控

#### BaseAggregator 抽象基类

**必须实现的接口:**
```python
class BaseAggregator(ABC):
    def __init__(self, context: ExecutionContext, config: DictConfig)
    
    @abstractmethod
    def aggregate(self, client_updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """聚合客户端更新"""
        pass
    
    @abstractmethod
    def weight_updates(self, updates: List[Dict[str, torch.Tensor]]) -> List[float]:
        """计算客户端权重"""
        pass
    
    # 可选方法
    def pre_aggregate_hook(self, client_updates: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
        """聚合前处理"""
        return client_updates
    
    def post_aggregate_hook(self, aggregated_update: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """聚合后处理"""
        return aggregated_update
    
    def get_aggregation_stats(self) -> Dict[str, Any]:
        """获取聚合统计信息"""
        return {}
    
    def supports_partial_participation(self) -> bool:
        """是否支持部分客户端参与"""
        return True
    
    def adjust_for_missing_clients(self, missing_clients: List[str]) -> None:
        """调整缺失客户端的聚合策略"""
        pass
```

**特殊要求:**
- 支持异构模型聚合
- 支持加权聚合
- 支持安全聚合（差分隐私）
- 支持部分客户端参与

#### BaseEvaluator 抽象基类

**必须实现的接口:**
```python
class BaseEvaluator(ABC):
    def __init__(self, context: ExecutionContext, config: DictConfig)
    
    @abstractmethod
    def evaluate(self, model: torch.nn.Module, data: DataLoader) -> Dict[str, float]:
        """评估模型性能"""
        pass
    
    @abstractmethod
    def compute_task_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """计算任务级别指标"""
        pass
    
    # 可选方法
    def compute_forgetting_metrics(self, current_results: Dict[str, float], 
                                   historical_results: List[Dict[str, float]]) -> Dict[str, float]:
        """计算遗忘指标"""
        return {}
    
    def compute_transfer_metrics(self, source_results: Dict[str, float], 
                                target_results: Dict[str, float]) -> Dict[str, float]:
        """计算迁移学习指标"""
        return {}
    
    def generate_evaluation_report(self, results: Dict[str, float]) -> str:
        """生成评估报告"""
        return ""
    
    def supports_online_evaluation(self) -> bool:
        """是否支持在线评估"""
        return False
```

#### Hook 抽象基类

**必须实现的接口:**
```python
class Hook(ABC):
    def __init__(self, phase: str, priority: int = 0, name: Optional[str] = None)
    
    @abstractmethod
    def execute(self, context: ExecutionContext, **kwargs) -> Any:
        """执行钩子逻辑"""
        pass
    
    def validate_context(self, context: ExecutionContext) -> bool:
        """验证执行上下文"""
        return True
    
    def should_execute(self, context: ExecutionContext, **kwargs) -> bool:
        """判断是否应该执行"""
        return True
    
    def get_priority(self) -> int:
        """获取执行优先级"""
        return self.priority
    
    def get_phase(self) -> str:
        """获取执行阶段"""
        return self.phase
    
    def get_name(self) -> str:
        """获取钩子名称"""
        return self.name or self.__class__.__name__
    
    def cleanup(self) -> None:
        """清理资源"""
        pass
```

**钩子阶段定义:**
```python
class HookPhase:
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
```

---

## 🔧 第四阶段: 组件注册系统

### Step 4.1: ComponentRegistry 实现

**文档输入:**
- `用户注册组件流程.md` - 注册机制详细设计
- `核心框架层.md` - ComponentRegistry接口

**必须实现的接口:**
```python
class ComponentRegistry:
    def __init__(self)
    
    # 组件注册
    def register_learner(self, name: str, learner_class: Type[BaseLearner], 
                        metadata: Optional[Dict] = None) -> Callable:
        """注册学习器"""
        pass
    
    def register_aggregator(self, name: str, aggregator_class: Type[BaseAggregator],
                           metadata: Optional[Dict] = None) -> Callable:
        """注册聚合器"""
        pass
    
    def register_evaluator(self, name: str, evaluator_class: Type[BaseEvaluator],
                          metadata: Optional[Dict] = None) -> Callable:
        """注册评估器"""
        pass
    
    def register_hook(self, phase: str, priority: int = 0) -> Callable:
        """注册钩子"""
        pass
    
    def register_loss_function(self, name: str, scope: str = "task") -> Callable:
        """注册损失函数"""
        pass
    
    def register_auxiliary_model(self, name: str, model_type: str) -> Callable:
        """注册辅助模型"""
        pass
    
    # 组件查询
    def get_component(self, component_type: str, name: str) -> Type:
        """获取组件类"""
        pass
    
    def list_components(self, component_type: str) -> List[str]:
        """列出组件"""
        pass
    
    def get_component_metadata(self, component_type: str, name: str) -> Dict:
        """获取组件元数据"""
        pass
    
    # 组件验证
    def validate_signature(self, func: Callable, component_type: str) -> bool:
        """验证组件签名"""
        pass
    
    def validate_component(self, component_class: Type, component_type: str) -> ValidationResult:
        """验证组件实现"""
        pass
    
    # 装饰器工厂
    def learner(self, name: str, **metadata) -> Callable:
        """学习器装饰器"""
        def decorator(cls):
            self.register_learner(name, cls, metadata)
            return cls
        return decorator
    
    def aggregator(self, name: str, **metadata) -> Callable:
        """聚合器装饰器"""
        def decorator(cls):
            self.register_aggregator(name, cls, metadata)
            return cls
        return decorator
    
    def evaluator(self, name: str, **metadata) -> Callable:
        """评估器装饰器"""
        def decorator(cls):
            self.register_evaluator(name, cls, metadata)
            return cls
        return decorator
    
    def hook(self, phase: str, priority: int = 0) -> Callable:
        """钩子装饰器"""
        def decorator(cls):
            self.register_hook(phase, priority)(cls)
            return cls
        return decorator
```

**组件元数据结构:**
```python
@dataclass
class ComponentMetadata:
    name: str
    component_type: str
    version: str
    author: str
    description: str
    requirements: List[str]
    supported_features: List[str]
    config_schema: Dict
    performance_characteristics: Dict
    
    def to_dict(self) -> Dict
    def from_dict(cls, data: Dict) -> 'ComponentMetadata'
```

**特殊要求:**
- 支持组件版本管理
- 支持组件依赖检查
- 支持运行时组件注册
- 支持组件热替换

**约束条件:**
- 线程安全的注册机制
- 组件名称唯一性检查
- 接口兼容性验证
- 性能要求：查询<1ms

### Step 4.2: ComponentComposer 实现

**文档输入:**
- `组件注册与实验初始化交互.md` - 组装流程
- `FedCL组件交互逻辑详解.md` - 依赖注入机制

**依赖关系:**
- **必需依赖**: 
  - `registry: ComponentRegistry` - 组件注册中心
- **可选依赖**: 无
- **配置依赖**: 无
- **运行时依赖**: 
  - `ExecutionContext` - 在compose_experiment时传入
  - 各种组件类 - 从registry动态获取

**调用关系说明:**
- **调用依赖的方式**:
  - `self.registry.get_component(component_type, name)` - 获取组件类
  - `self.registry.get_component_metadata()` - 获取组件元数据
  - `component_class(context, config)` - 实例化组件
- **被调用场景**:
  - `ExperimentEngine.start_experiment()` 中调用 `composer.compose_experiment()`
  - `FedCLExperiment.run()` 中间接调用

**依赖注入实现:**
```python
class ComponentComposer:
    def __init__(self, registry: ComponentRegistry):
        self.registry = registry
        self._instance_cache: Dict[str, Any] = {}
        self._dependency_graph: Dict[str, List[str]] = {}
        
    def create_learner(self, config: DictConfig, context: ExecutionContext) -> BaseLearner:
        learner_name = config.learner.name
        learner_class = self.registry.get_component("learner", learner_name)
        
        # 检查依赖
        metadata = self.registry.get_component_metadata("learner", learner_name)
        self._validate_dependencies(metadata.get("dependencies", []))
        
        # 创建实例
        learner_config = config.learner.parameters
        learner = learner_class(context=context, config=learner_config)
        
        # 缓存实例
        cache_key = f"learner_{learner_name}_{id(context)}"
        self._instance_cache[cache_key] = learner
        
        return learner
        
    def compose_experiment(self, config: DictConfig, context: ExecutionContext) -> ExperimentComponents:
        # 按依赖顺序创建组件
        learner = self.create_learner(config, context)
        aggregator = self.create_aggregator(config, context)
        evaluator = self.create_evaluator(config, context)
        hooks = self._create_hooks(config, context)
        
        return ExperimentComponents(
            learner=learner,
            aggregator=aggregator,
            evaluator=evaluator,
            hooks=hooks,
            auxiliary_models={},
            loss_functions={}
        )
        
    def _validate_dependencies(self, dependencies: List[str]) -> None:
        for dep in dependencies:
            if not self.registry.get_component("any", dep):
                raise DependencyNotFoundError(f"Dependency {dep} not found")
```

**依赖解析策略:**
- **循环依赖检测**: 构建依赖图并检测环路
- **延迟实例化**: 只在需要时创建组件实例
- **实例缓存**: 相同配置的组件复用实例
- **依赖验证**: 实例化前验证所有依赖可用

**必须实现的接口:**
```python
class ComponentComposer:
    def __init__(self, registry: ComponentRegistry)
    
    def compose_experiment(self, config: DictConfig, context: ExecutionContext) -> ExperimentComponents:
        """组装实验组件"""
        pass
    
    def create_learner(self, config: DictConfig, context: ExecutionContext) -> BaseLearner:
        """创建学习器实例"""
        pass
    
    def create_aggregator(self, config: DictConfig, context: ExecutionContext) -> BaseAggregator:
        """创建聚合器实例"""
        pass
    
    def create_evaluator(self, config: DictConfig, context: ExecutionContext) -> BaseEvaluator:
        """创建评估器实例"""
        pass
    
    def create_auxiliary_models(self, model_configs: List[DictConfig], 
                               context: ExecutionContext) -> Dict[str, Any]:
        """创建辅助模型"""
        pass
    
    def prepare_loss_functions(self, loss_configs: List[DictConfig]) -> Dict[str, Callable]:
        """准备损失函数"""
        pass
    
    def resolve_dependencies(self, components: List[str]) -> List[str]:
        """解析组件依赖"""
        pass
    
    def validate_configuration(self, config: DictConfig) -> ValidationResult:
        """验证组件配置"""
        pass
    
    # 实例管理
    def get_instance_cache(self) -> Dict[str, Any]:
        """获取实例缓存"""
        pass
    
    def clear_instance_cache(self) -> None:
        """清理实例缓存"""
        pass
    
    def register_instance(self, name: str, instance: Any) -> None:
        """注册实例"""
        pass
```

**ExperimentComponents 数据结构:**
```python
@dataclass
class ExperimentComponents:
    learner: BaseLearner
    aggregator: BaseAggregator
    evaluator: BaseEvaluator
    hooks: List[Hook]
    auxiliary_models: Dict[str, Any]
    loss_functions: Dict[str, Callable]
    
    def validate(self) -> bool
    def get_component_summary(self) -> Dict[str, str]
```

**特殊要求:**
- 支持循环依赖检测
- 支持懒加载实例化
- 支持单例模式组件
- 支持组件生命周期管理

**约束条件:**
- 实例化过程必须是线程安全的
- 支持实例缓存和复用
- 错误处理和回滚机制
- 内存泄露防护

---

## 📊 第五阶段: 数据管理系统

### Step 5.1: 数据分割策略实现

**文档输入:**
- `数据与配置管理.md` - 数据处理设计
- `客户端详细训练流程图.md` - 数据分割需求

#### SplitStrategy 抽象基类

**必须实现的接口:**
```python
class SplitStrategy(ABC):
    def __init__(self, config: DictConfig)
    
    @abstractmethod
    def split_data(self, dataset: Dataset, num_clients: int) -> Dict[str, Dataset]:
        """分割数据集"""
        pass
    
    @abstractmethod
    def validate_split(self, split_data: Dict[str, Dataset]) -> bool:
        """验证分割结果"""
        pass
    
    def get_split_statistics(self, split_data: Dict[str, Dataset]) -> Dict[str, Any]:
        """获取分割统计信息"""
        pass
    
    def visualize_split(self, split_data: Dict[str, Dataset], save_path: Optional[Path] = None) -> None:
        """可视化分割结果"""
        pass
```

#### IIDSplitStrategy 实现

**必须实现的接口:**
```python
class IIDSplitStrategy(SplitStrategy):
    def __init__(self, config: DictConfig, random_seed: int = 42)
    
    def split_data(self, dataset: Dataset, num_clients: int) -> Dict[str, Dataset]:
        """IID数据分割"""
        pass
    
    def random_split(self, dataset: Dataset, ratios: List[float]) -> List[Dataset]:
        """随机分割数据集"""
        pass
    
    def stratified_split(self, dataset: Dataset, num_clients: int) -> Dict[str, Dataset]:
        """分层分割保持类别比例"""
        pass
    
    def validate_split(self, split_data: Dict[str, Dataset]) -> bool:
        """验证IID性质"""
        pass
```

#### NonIIDSplitStrategy 实现

**必须实现的接口:**
```python
class NonIIDSplitStrategy(SplitStrategy):
    def __init__(self, config: DictConfig, alpha: float = 0.5, min_samples_per_client: int = 10)
    
    def split_data(self, dataset: Dataset, num_clients: int) -> Dict[str, Dataset]:
        """Non-IID数据分割"""
        pass
    
    def dirichlet_split(self, dataset: Dataset, alpha: float, num_clients: int) -> Dict[str, Dataset]:
        """使用Dirichlet分布分割"""
        pass
    
    def pathological_split(self, dataset: Dataset, shards_per_client: int, num_clients: int) -> Dict[str, Dataset]:
        """病理性Non-IID分割"""
        pass
    
    def label_skew_split(self, dataset: Dataset, num_clients: int, num_classes_per_client: int) -> Dict[str, Dataset]:
        """标签偏斜分割"""
        pass
    
    def feature_skew_split(self, dataset: Dataset, num_clients: int) -> Dict[str, Dataset]:
        """特征偏斜分割"""
        pass
    
    def validate_split(self, split_data: Dict[str, Dataset]) -> bool:
        """验证Non-IID性质"""
        pass
```

**配置要求:**
- Dirichlet参数配置
- 最小样本数配置
- 随机种子配置
- 分割策略选择配置

**约束条件:**
- 数据完整性保证（分割后总样本数不变）
- 可重现性保证（相同配置产生相同分割）
- 内存效率（支持大数据集分割）
- 分割结果验证

### Step 5.2: DataProcessor 实现

**文档输入:**
- `数据与配置管理.md` - 数据处理管道
- `数据流图/` - 数据预处理流程

**必须实现的接口:**
```python
class DataProcessor:
    def __init__(self, config: DictConfig)
    
    def preprocess_data(self, raw_data: Dataset) -> Dataset:
        """数据预处理"""
        pass
    
    def apply_transforms(self, data: Dataset, transform_config: DictConfig) -> Dataset:
        """应用数据变换"""
        pass
    
    def create_data_loaders(self, datasets: Dict[str, Dataset], 
                           batch_size: int, shuffle: bool = True) -> Dict[str, DataLoader]:
        """创建数据加载器"""
        pass
    
    def split_data_federated(self, dataset: Dataset, num_clients: int, 
                            strategy: str = "iid") -> Dict[str, Dataset]:
        """联邦数据分割"""
        pass
    
    def balance_client_data(self, client_datasets: Dict[str, Dataset]) -> Dict[str, Dataset]:
        """平衡客户端数据"""
        pass
    
    def augment_data(self, dataset: Dataset, augmentation_config: DictConfig) -> Dataset:
        """数据增强"""
        pass
    
    def normalize_data(self, dataset: Dataset, normalization_config: DictConfig) -> Dataset:
        """数据标准化"""
        pass
    
    def validate_data_quality(self, dataset: Dataset) -> Dict[str, Any]:
        """数据质量验证"""
        pass
    
    def get_data_statistics(self, dataset: Dataset) -> Dict[str, Any]:
        """获取数据统计信息"""
        pass
```

**数据变换管道配置:**
```yaml
data_processing:
  transforms:
    - name: "resize"
      params:
        size: [224, 224]
    - name: "normalize"
      params:
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
    - name: "random_horizontal_flip"
      params:
        p: 0.5
  
  augmentation:
    enable: true
    strategies:
      - "rotation"
      - "color_jitter"
      - "gaussian_noise"
  
  validation:
    check_nan: true
    check_range: true
    check_distribution: true
```

**约束条件:**
- 支持流式数据处理
- 内存使用优化
- 支持并行处理
- 数据完整性保证

### Step 5.3: TaskGenerator 实现

**文档输入:**
- `联邦训练核心流程图.md` - 任务序列设计
- `数据流图/` - 任务生成流程

**必须实现的接口:**
```python
class TaskGenerator:
    def __init__(self, config: DictConfig, split_strategy: SplitStrategy)
    
    def generate_class_incremental_tasks(self, dataset: Dataset, 
                                        classes_per_task: int) -> List[Task]:
        """生成类增量任务"""
        pass
    
    def generate_domain_incremental_tasks(self, datasets: List[Dataset]) -> List[Task]:
        """生成域增量任务"""
        pass
    
    def generate_task_incremental_tasks(self, dataset: Dataset, 
                                       task_configs: List[DictConfig]) -> List[Task]:
        """生成任务增量任务"""
        pass
    
    def shuffle_classes(self, seed: Optional[int] = None) -> List[int]:
        """打乱类别顺序"""
        pass
    
    def validate_task_sequence(self, tasks: List[Task]) -> bool:
        """验证任务序列"""
        pass
    
    def get_task_statistics(self, tasks: List[Task]) -> Dict[str, Any]:
        """获取任务统计信息"""
        pass
    
    def visualize_task_sequence(self, tasks: List[Task], save_path: Optional[Path] = None) -> None:
        """可视化任务序列"""
        pass
    
    def create_replay_buffer(self, task: Task, buffer_size: int) -> Dataset:
        """创建重放缓冲区"""
        pass
```

**任务生成配置:**
```yaml
task_generation:
  type: "class_incremental"  # class_incremental, domain_incremental, task_incremental
  num_tasks: 10
  classes_per_task: 10
  
  class_incremental:
    shuffle_classes: true
    random_seed: 42
    
  domain_incremental:
    domains: ["mnist", "fashion_mnist", "cifar10"]
    
  task_incremental:
    task_configs:
      - name: "classification"
        classes: [0, 1, 2, 3, 4]
      - name: "detection" 
        classes: [5, 6, 7, 8, 9]
        
  replay:
    enable: true
    buffer_size_per_class: 50
    selection_strategy: "random"  # random, herding, gradient_based
```

**约束条件:**
- 任务间类别不重叠（类增量）
- 任务序列可重现
- 支持多种增量学习场景
- 任务平衡性保证

### Step 5.4: DatasetManager 实现

**文档输入:**
- `数据与配置管理.md` - 数据集管理设计
- `数据流图/` - 数据管理架构

**必须实现的接口:**
```python
class DatasetManager:
    def __init__(self, config: DictConfig, task_generator: TaskGenerator)
    
    def load_dataset(self, name: str, config: DictConfig) -> Dataset:
        """加载数据集"""
        pass
    
    def register_dataset(self, name: str, dataset: Dataset, metadata: Dict) -> None:
        """注册数据集"""
        pass
    
    def create_task_sequence(self, dataset_name: str, num_tasks: int) -> List[Task]:
        """创建任务序列"""
        pass
    
    def get_client_data(self, client_id: str, task_id: int) -> DataLoader:
        """获取客户端数据"""
        pass
    
    def cache_dataset(self, name: str, dataset: Dataset) -> None:
        """缓存数据集"""
        pass
    
    def clear_cache(self, dataset_name: Optional[str] = None) -> None:
        """清理缓存"""
        pass
    
    def get_dataset_metadata(self, name: str) -> Dict[str, Any]:
        """获取数据集元数据"""
        pass
    
    def validate_dataset(self, dataset: Dataset) -> ValidationResult:
        """验证数据集"""
        pass
    
    def download_dataset(self, name: str, download_path: Path) -> Path:
        """下载数据集"""
        pass
    
    def list_available_datasets(self) -> List[str]:
        """列出可用数据集"""
        pass
    
    def get_dataset_statistics(self, name: str) -> Dict[str, Any]:
        """获取数据集统计信息"""
        pass
```

**数据集配置:**
```yaml
datasets:
  cifar10:
    type: "torchvision"
    name: "CIFAR10" 
    root: "./data"
    download: true
    transforms:
      train: 
        - "random_horizontal_flip"
        - "random_crop"
        - "normalize"
      test:
        - "normalize"
    
  custom_dataset:
    type: "custom"
    data_path: "./custom_data"
    loader_class: "CustomDatasetLoader"
    preprocessing:
      - "resize"
      - "normalize"
      
cache:
  enable: true
  max_size: "2GB"
  strategy: "LRU"
  persist: true
```

**约束条件:**
- 支持多种数据集格式
- 智能缓存管理
- 数据完整性检查
- 并发访问安全
- 内存使用优化

---

## 🌐 第六阶段: 通信系统
通信需要支持伪联邦（即多个客户端运行在同一或者多个进程中（资源限制，运行可能是并行或着顺序执行））
### Step 6.1: 基础通信组件

**文档输入:**
- `通信与分布式系统.md` - 通信架构设计
- `通信协议交互图.md` - 协议设计

#### MessageProtocol 实现

**必须实现的接口:**
```python
class MessageProtocol:
    # 消息类型常量
    MODEL_UPDATE = "model_update"
    GLOBAL_MODEL = "global_model"
    CLIENT_READY = "client_ready"
    TRAINING_COMPLETE = "training_complete"
    HEARTBEAT = "heartbeat"
    ERROR_REPORT = "error_report"
    SHUTDOWN = "shutdown"
    
    def __init__(self, version: str = "1.0")
    
    def serialize_message(self, message_type: str, data: Any, 
                         metadata: Optional[Dict] = None) -> bytes:
        """序列化消息"""
        pass
    
    def deserialize_message(self, raw_data: bytes) -> Message:
        """反序列化消息"""
        pass
    
    def validate_message(self, message: Message) -> bool:
        """验证消息格式"""
        pass
    
    def create_message(self, message_type: str, data: Any, 
                      sender: str, receiver: str) -> Message:
        """创建消息对象"""
        pass
    
    def get_message_size(self, message: Message) -> int:
        """获取消息大小"""
        pass
    
    def compress_message(self, message: Message) -> Message:
        """压缩消息"""
        pass
    
    def decompress_message(self, message: Message) -> Message:
        """解压消息"""
        pass
```

**Message 数据结构:**
```python
@dataclass
class Message:
    message_id: str
    message_type: str
    sender: str
    receiver: str
    timestamp: datetime
    data: Any
    metadata: Dict[str, Any]
    checksum: str
    
    def validate_checksum(self) -> bool
    def calculate_checksum(self) -> str
    def to_dict(self) -> Dict
    def from_dict(cls, data: Dict) -> 'Message'
```

#### DataSerializer 实现

**必须实现的接口:**
```python
class DataSerializer:
    def __init__(self, compression_level: int = 6)
    
    def serialize_model(self, model: torch.nn.Module) -> bytes:
        """序列化模型"""
        pass
    
    def deserialize_model(self, data: bytes, model_class: Type[torch.nn.Module]) -> torch.nn.Module:
        """反序列化模型"""
        pass
    
    def serialize_tensor(self, tensor: torch.Tensor) -> bytes:
        """序列化张量"""
        pass
    
    def deserialize_tensor(self, data: bytes) -> torch.Tensor:
        """反序列化张量"""
        pass
    
    def serialize_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> bytes:
        """序列化状态字典"""
        pass
    
    def deserialize_state_dict(self, data: bytes) -> Dict[str, torch.Tensor]:
        """反序列化状态字典"""
        pass
    
    def compress_data(self, data: bytes) -> bytes:
        """压缩数据"""
        pass
    
    def decompress_data(self, compressed_data: bytes) -> bytes:
        """解压数据"""
        pass
    
    def get_compression_ratio(self, original_size: int, compressed_size: int) -> float:
        """获取压缩比"""
        pass
```

#### NetworkInterface 实现

**必须实现的接口:**
```python
class NetworkInterface:
    def __init__(self, config: DictConfig)
    
    def create_server_socket(self, host: str, port: int) -> ServerSocket:
        """创建服务器套接字"""
        pass
    
    def create_client_socket(self, host: str, port: int) -> ClientSocket:
        """创建客户端套接字"""
        pass
    
    def send_data(self, connection: Connection, data: bytes) -> bool:
        """发送数据"""
        pass
    
    def receive_data(self, connection: Connection, timeout: float = 30.0) -> bytes:
        """接收数据"""
        pass
    
    def close_connection(self, connection_id: str) -> None:
        """关闭连接"""
        pass
    
    def get_connection_status(self, connection_id: str) -> ConnectionStatus:
        """获取连接状态"""
        pass
    
    def set_socket_options(self, socket: Socket, options: Dict[str, Any]) -> None:
        """设置套接字选项"""
        pass
    
    async def async_send_data(self, connection: Connection, data: bytes) -> bool:
        """异步发送数据"""
        pass
    
    async def async_receive_data(self, connection: Connection, timeout: float = 30.0) -> bytes:
        """异步接收数据"""
        pass
```

#### Connection 和 ConnectionPool

**Connection 接口:**
```python
class Connection:
    def __init__(self, socket: Socket, connection_id: str, peer_address: Tuple[str, int])
    
    def send(self, data: bytes) -> bool:
        """发送数据"""
        pass
    
    def receive(self, timeout: float = 30.0) -> bytes:
        """接收数据"""
        pass
    
    def is_alive(self) -> bool:
        """检查连接是否活跃"""
        pass
    
    def close(self) -> None:
        """关闭连接"""
        pass
    
    def get_peer_address(self) -> Tuple[str, int]:
        """获取对端地址"""
        pass
    
    def get_connection_info(self) -> Dict[str, Any]:
        """获取连接信息"""
        pass
    
    def set_timeout(self, timeout: float) -> None:
        """设置超时时间"""
        pass
```

**ConnectionPool 接口:**
```python
class ConnectionPool:
    def __init__(self, max_connections: int = 100)
    
    def get_connection(self, client_id: str) -> Optional[Connection]:
        """获取连接"""
        pass
    
    def add_connection(self, client_id: str, connection: Connection) -> None:
        """添加连接"""
        pass
    
    def remove_connection(self, client_id: str) -> None:
        """移除连接"""
        pass
    
    def get_active_connections(self) -> List[str]:
        """获取活跃连接"""
        pass
    
    def cleanup_stale_connections(self) -> None:
        """清理失效连接"""
        pass
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """获取连接池统计"""
        pass
    
    def set_max_connections(self, max_connections: int) -> None:
        """设置最大连接数"""
        pass
```

**约束条件:**
- 支持TCP和WebSocket协议
- 支持SSL/TLS加密
- 连接重用和管理
- 自动重连机制
- 心跳检测机制

### Step 6.2: 安全和通信管理

**文档输入:**
- `错误处理与恢复机制交互.md` - 通信错误处理
- `联邦学习完整训练轮次交互.md` - 通信管理需求

#### SecurityModule 实现

**必须实现的接口:**
```python
class SecurityModule:
    def __init__(self, config: DictConfig)
    
    def encrypt_data(self, data: bytes, key: Optional[bytes] = None) -> bytes:
        """加密数据"""
        pass
    
    def decrypt_data(self, encrypted_data: bytes, key: Optional[bytes] = None) -> bytes:
        """解密数据"""
        pass
    
    def generate_key_pair(self) -> Tuple[bytes, bytes]:
        """生成密钥对"""
        pass
    
    def sign_data(self, data: bytes, private_key: bytes) -> bytes:
        """数据签名"""
        pass
    
    def verify_signature(self, data: bytes, signature: bytes, public_key: bytes) -> bool:
        """验证签名"""
        pass
    
    def authenticate_client(self, client_id: str, token: str) -> bool:
        """客户端认证"""
        pass
    
    def generate_session_key(self) -> bytes:
        """生成会话密钥"""
        pass
    
    def hash_data(self, data: bytes) -> str:
        """数据哈希"""
        pass
    
    def generate_secure_random(self, length: int) -> bytes:
        """生成安全随机数"""
        pass
```

**安全配置:**
```yaml
security:
  encryption:
    algorithm: "AES-256-GCM"
    key_rotation_interval: 3600  # seconds
    
  authentication:
    method: "token"  # token, certificate, mutual_tls
    token_lifetime: 1800  # seconds
    
  signing:
    algorithm: "RSA-PSS"
    key_size: 2048
    
  tls:
    enable: true
    version: "1.3"
    verify_certificates: true
    ca_bundle: "./certs/ca.pem"
```

#### CommunicationHandler 实现

**必须实现的接口:**
```python
class CommunicationHandler:
    def __init__(self, protocol: MessageProtocol, serializer: DataSerializer, 
                 security: SecurityModule, network: NetworkInterface)
    
    def send_message(self, target: str, message: Message) -> bool:
        """发送消息"""
        pass
    
    def receive_message(self, source: str, timeout: float = 30.0) -> Message:
        """接收消息"""
        pass
    
    def broadcast_message(self, message: Message, targets: List[str]) -> Dict[str, bool]:
        """广播消息"""
        pass
    
    def establish_connection(self, address: Tuple[str, int]) -> Connection:
        """建立连接"""
        pass
    
    def close_connection(self, connection_id: str) -> None:
        """关闭连接"""
        pass
    
    def handle_network_error(self, error: Exception, connection_id: str) -> None:
        """处理网络错误"""
        pass
    
    def set_retry_policy(self, max_retries: int, backoff_factor: float) -> None:
        """设置重试策略"""
        pass
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """获取通信统计"""
        pass
    
    async def async_send_message(self, target: str, message: Message) -> bool:
        """异步发送消息"""
        pass
    
    async def async_receive_message(self, source: str, timeout: float = 30.0) -> Message:
        """异步接收消息"""
        pass
```

#### CommunicationManager 实现

**必须实现的接口:**
```python
class CommunicationManager:
    def __init__(self, config: DictConfig, handler: CommunicationHandler)
    
    def send_data(self, source: str, target: str, data: Any, data_type: str) -> None:
        """发送数据"""
        pass
    
    def receive_data(self, source: str, data_type: str, timeout: float = 30.0) -> Any:
        """接收数据"""
        pass
    
    def broadcast_model(self, model: torch.nn.Module, targets: List[str]) -> Dict[str, bool]:
        """广播模型"""
        pass
    
    def collect_updates(self, sources: List[str], timeout: float = 60.0) -> List[Dict[str, torch.Tensor]]:
        """收集更新"""
        pass
    
    def establish_connection(self, client_id: str, address: Tuple[str, int]) -> bool:
        """建立连接"""
        pass
    
    def handle_client_disconnect(self, client_id: str) -> None:
        """处理客户端断开"""
        pass
    
    def get_active_clients(self) -> List[str]:
        """获取活跃客户端"""
        pass
    
    def ping_client(self, client_id: str) -> bool:
        """ping客户端"""
        pass
    
    def start_heartbeat_monitor(self, interval: float = 30.0) -> None:
        """启动心跳监控"""
        pass
    
    def stop_heartbeat_monitor(self) -> None:
        """停止心跳监控"""
        pass
```

**通信配置:**
```yaml
communication:
  protocol: "tcp"  # tcp, websocket
  host: "0.0.0.0"
  port: 8080
  
  timeouts:
    connection: 30.0
    send: 10.0
    receive: 30.0
    
  retry:
    max_attempts: 3
    backoff_factor: 2.0
    
  heartbeat:
    interval: 30.0
    timeout: 10.0
    
  buffer_sizes:
    send: 65536
    receive: 65536
    
  compression:
    enable: true
    level: 6
    threshold: 1024  # bytes
```

**约束条件:**
- 支持异步通信
- 自动重连和容错
- 流量控制和背压
- 通信性能监控
- 支持大消息分片传输

---

## ⚙️ 第七阶段: 执行引擎层

### Step 7.1: HookExecutor 实现

**文档输入:**
- `Hook系统与扩展机制交互.md` - 钩子执行流程
- `钩子系统执行时机图.md` - 执行时机设计

**必须实现的接口:**
```python
class HookExecutor:
    def __init__(self, registry: ComponentRegistry)
    
    def execute_hooks(self, phase: str, context: ExecutionContext, **kwargs) -> List[Any]:
        """执行指定阶段的钩子"""
        pass
    
    def register_hook(self, hook: Hook) -> str:
        """注册钩子"""
        pass
    
    def unregister_hook(self, hook_id: str) -> bool:
        """注销钩子"""
        pass
    
    def get_hooks(self, phase: str) -> List[Hook]:
        """获取指定阶段的钩子"""
        pass
    
    def prepare_hook_kwargs(self, hook: Hook, phase: str, context: ExecutionContext, 
                           **kwargs) -> Dict[str, Any]:
        """准备钩子参数"""
        pass
    
    def handle_hook_error(self, hook: Hook, error: Exception, context: ExecutionContext) -> None:
        """处理钩子错误"""
        pass
    
    def set_error_policy(self, policy: str) -> None:
        """设置错误处理策略"""
        pass
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """获取执行统计"""
        pass
    
    def enable_hook(self, hook_id: str) -> None:
        """启用钩子"""
        pass
    
    def disable_hook(self, hook_id: str) -> None:
        """禁用钩子"""
        pass
    
    def clear_hooks(self, phase: Optional[str] = None) -> None:
        """清理钩子"""
        pass
```

**钩子执行策略配置:**
```yaml
hook_execution:
  error_policy: "continue"  # continue, stop, skip_phase
  timeout: 30.0  # seconds
  parallel_execution: false
  
  logging:
    enable: true
    log_level: "INFO"
    log_performance: true
    
  monitoring:
    track_execution_time: true
    max_execution_time: 10.0  # seconds
    alert_on_slow_hooks: true
```

**约束条件:**
- 钩子执行顺序保证
- 错误隔离和恢复
- 执行性能监控
- 支持条件执行
- 支持钩子间通信

### Step 7.2: 基础Hook实现

**文档输入:**
- `组件与扩展系统.md` - Hook实现示例

#### MetricsHook 实现

**必须实现的接口:**
```python
class MetricsHook(Hook):
    def __init__(self, phase: str, priority: int = 0, metrics_config: DictConfig)
    
    def execute(self, context: ExecutionContext, **kwargs) -> None:
        """执行度量记录"""
        pass
    
    def log_training_metrics(self, context: ExecutionContext, metrics: Dict[str, float]) -> None:
        """记录训练度量"""
        pass
    
    def log_evaluation_metrics(self, context: ExecutionContext, results: Dict[str, float]) -> None:
        """记录评估度量"""
        pass
    
    def log_system_metrics(self, context: ExecutionContext) -> None:
        """记录系统度量"""
        pass
    
    def log_communication_metrics(self, context: ExecutionContext, comm_stats: Dict) -> None:
        """记录通信度量"""
        pass
    
    def should_execute(self, context: ExecutionContext, **kwargs) -> bool:
        """判断是否应该记录"""
        pass
```

#### CheckpointHook 实现

**必须实现的接口:**
```python
class CheckpointHook(Hook):
    def __init__(self, phase: str, priority: int = 0, checkpoint_config: DictConfig)
    
    def execute(self, context: ExecutionContext, **kwargs) -> None:
        """执行检查点保存"""
        pass
    
    def save_model_checkpoint(self, model: torch.nn.Module, path: Path) -> None:
        """保存模型检查点"""
        pass
    
    def save_experiment_state(self, context: ExecutionContext, path: Path) -> None:
        """保存实验状态"""
        pass
    
    def should_save_checkpoint(self, context: ExecutionContext, **kwargs) -> bool:
        """判断是否应该保存"""
        pass
    
    def cleanup_old_checkpoints(self, max_checkpoints: int) -> None:
        """清理旧检查点"""
        pass
    
    def get_checkpoint_info(self, checkpoint_path: Path) -> Dict[str, Any]:
        """获取检查点信息"""
        pass
```

### Step 7.3: TrainingEngine 实现

**文档输入:**
- `核心框架层.md` - 训练引擎设计
- `客户端详细训练流程图.md` - 训练执行流程

**依赖关系:**
- **必需依赖**: 
  - `hook_executor: HookExecutor` - 钩子执行器
  - `context: ExecutionContext` - 执行上下文
- **可选依赖**: 
  - `checkpoint_manager: Optional[CheckpointManager] = None` - 检查点管理器
  - `metrics_logger: Optional[MetricsLogger] = None` - 度量记录器
- **配置依赖**: 
  - 训练参数（从context.config.training获取）
  - 优化器配置（从context.config.optimizer获取）
- **运行时依赖**: 
  - `BaseLearner` - 通过方法参数传入
  - `DataLoader` - 通过方法参数传入

**调用关系说明:**
- **调用依赖的方式**:
  - `self.hook_executor.execute_hooks(phase, self.context, **kwargs)` - 执行钩子
  - `self.context.log_metric(name, value, step)` - 记录度量
  - `learner.train_task(task_data)` - 调用学习器训练
  - `self.checkpoint_manager.save_checkpoint(state_dict)` - 保存检查点
- **被调用场景**:
  - `LocalTrainer.train_epoch()` 中调用 `training_engine.execute_training_loop()`
  - `FederatedClient.train_local_model()` 中调用相关方法

**依赖注入实现:**
```python
class TrainingEngine:
    def __init__(self, 
                 hook_executor: HookExecutor, 
                 context: ExecutionContext,
                 checkpoint_manager: Optional[CheckpointManager] = None,
                 metrics_logger: Optional[MetricsLogger] = None):
        self.hook_executor = hook_executor
        self.context = context
        self.checkpoint_manager = checkpoint_manager
        self.metrics_logger = metrics_logger
        
        # 从配置获取训练参数
        self.training_config = context.get_config("training", {})
        self.num_epochs = self.training_config.get("num_epochs", 10)
        self.early_stopping = self.training_config.get("early_stopping", {})
        
        # 训练状态
        self._is_paused = False
        self._should_stop = False
        
    def train_task(self, task_id: int, task_data: DataLoader, learner: BaseLearner) -> TaskResults:
        # 执行训练前钩子
        self.hook_executor.execute_hooks(
            HookPhase.BEFORE_TASK, 
            self.context, 
            task_id=task_id, 
            task_data=task_data,
            learner=learner
        )
        
        try:
            # 执行训练循环
            training_metrics = self.execute_training_loop(learner, task_data, self.num_epochs)
            
            # 创建任务结果
            task_results = TaskResults(
                task_id=task_id,
                metrics=training_metrics,
                training_time=training_metrics.get("training_time", 0),
                memory_usage=training_metrics.get("memory_usage", 0),
                model_size=training_metrics.get("model_size", 0),
                convergence_step=training_metrics.get("convergence_step", -1)
            )
            
            # 执行训练后钩子
            self.hook_executor.execute_hooks(
                HookPhase.AFTER_TASK,
                self.context,
                task_id=task_id,
                results=task_results,
                learner=learner
            )
            
            return task_results
            
        except Exception as e:
            # 执行错误处理钩子
            self.hook_executor.execute_hooks(
                HookPhase.ON_ERROR,
                self.context,
                error=e,
                task_id=task_id,
                learner=learner
            )
            raise
            
    def execute_training_loop(self, learner: BaseLearner, data_loader: DataLoader, 
                             num_epochs: int) -> Dict[str, float]:
        metrics = {}
        
        for epoch in range(num_epochs):
            if self._should_stop:
                break
                
            # 执行轮次前钩子
            self.hook_executor.execute_hooks(
                HookPhase.BEFORE_EPOCH,
                self.context,
                epoch=epoch,
                learner=learner
            )
            
            # 训练一个轮次
            epoch_metrics = self._train_epoch(learner, data_loader, epoch)
            
            # 记录度量
            for metric_name, metric_value in epoch_metrics.items():
                self.context.log_metric(f"epoch_{metric_name}", metric_value, epoch)
                
            # 执行轮次后钩子
            self.hook_executor.execute_hooks(
                HookPhase.AFTER_EPOCH,
                self.context,
                epoch=epoch,
                metrics=epoch_metrics,
                learner=learner
            )
            
            # 检查早停条件
            if self._should_early_stop(epoch_metrics):
                break
                
        return metrics
```

**异常处理策略:**
- **训练错误**: 通过错误钩子处理，支持恢复策略
- **依赖缺失**: 优雅降级（如无checkpoint_manager时跳过保存）
- **资源不足**: 自动调整批次大小或停止训练

**必须实现的接口:**
```python
class TrainingEngine:
    def __init__(self, hook_executor: HookExecutor, context: ExecutionContext)
    
    def train_task(self, task_id: int, task_data: DataLoader, learner: BaseLearner) -> TaskResults:
        """训练单个任务"""
        pass
    
    def execute_training_loop(self, learner: BaseLearner, data_loader: DataLoader, 
                             num_epochs: int) -> Dict[str, float]:
        """执行训练循环"""
        pass
    
    def handle_batch(self, batch_data: Tuple[torch.Tensor, torch.Tensor], 
                    learner: BaseLearner) -> Dict[str, float]:
        """处理单个批次"""
        pass
    
    def validate_model(self, learner: BaseLearner, validation_data: DataLoader) -> Dict[str, float]:
        """验证模型"""
        pass
    
    def setup_training_environment(self, learner: BaseLearner) -> None:
        """设置训练环境"""
        pass
    
    def cleanup_training_environment(self) -> None:
        """清理训练环境"""
        pass
    
    def handle_training_error(self, error: Exception, context: Dict[str, Any]) -> bool:
        """处理训练错误"""
        pass
    
    def get_training_stats(self) -> Dict[str, Any]:
        """获取训练统计"""
        pass
    
    def pause_training(self) -> None:
        """暂停训练"""
        pass
    
    def resume_training(self) -> None:
        """恢复训练"""
        pass
    
    def stop_training(self) -> None:
        """停止训练"""
        pass
```

**训练配置:**
```yaml
training:
  num_epochs: 10
  batch_size: 32
  learning_rate: 0.001
  
  optimization:
    optimizer: "adam"
    lr_scheduler: "cosine"
    weight_decay: 0.0001
    
  early_stopping:
    enable: true
    patience: 5
    min_delta: 0.001
    
  gradient:
    clip_norm: 1.0
    accumulation_steps: 1
    
  validation:
    interval: 1  # epochs
    metric: "accuracy"
    
  checkpointing:
    save_best: true
    save_interval: 5  # epochs
```

**约束条件:**
- 支持GPU/CPU自动切换
- 内存使用监控
- 训练进度跟踪
- 异常恢复机制
- 支持分布式训练

### Step 7.4: EvaluationEngine和管理组件

**文档输入:**
- `核心框架层.md` - 评估引擎设计
- `数据与配置管理.md` - 度量和检查点管理

#### EvaluationEngine 实现

**必须实现的接口:**
```python
class EvaluationEngine:
    def __init__(self, context: ExecutionContext)
    
    def evaluate_model(self, model: torch.nn.Module, test_data: DataLoader, 
                      evaluator: BaseEvaluator) -> Dict[str, float]:
        """评估模型"""
        pass
    
    def compute_metrics(self, predictions: torch.Tensor, targets: torch.Tensor, 
                       evaluator: BaseEvaluator) -> Dict[str, float]:
        """计算度量"""
        pass
    
    def generate_reports(self, results: Dict[str, float], task_id: int) -> str:
        """生成评估报告"""
        pass
    
    def evaluate_continual_learning(self, model: torch.nn.Module, 
                                   all_task_data: List[DataLoader],
                                   evaluator: BaseEvaluator) -> Dict[str, Any]:
        """评估持续学习性能"""
        pass
    
    def compute_forgetting_metrics(self, current_results: Dict[str, float],
                                  historical_results: List[Dict[str, float]]) -> Dict[str, float]:
        """计算遗忘度量"""
        pass
    
    def visualize_results(self, results: Dict[str, Any], save_path: Path) -> None:
        """可视化结果"""
        pass
    
    def compare_models(self, model_results: List[Dict[str, float]]) -> Dict[str, Any]:
        """比较模型性能"""
        pass
```

#### MetricsLogger 实现

**必须实现的接口:**
```python
class MetricsLogger:
    def __init__(self, log_dir: Path, config: DictConfig)
    
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """记录标量度量"""
        pass
    
    def log_histogram(self, tag: str, values: torch.Tensor, step: int) -> None:
        """记录直方图"""
        pass
    
    def log_image(self, tag: str, image: torch.Tensor, step: int) -> None:
        """记录图像"""
        pass
    
    def log_text(self, tag: str, text: str, step: int) -> None:
        """记录文本"""
        pass
    
    def log_hyperparameters(self, hparams: Dict[str, Any], metrics: Dict[str, float]) -> None:
        """记录超参数"""
        pass
    
    def create_summary_writer(self, log_dir: Path) -> Any:
        """创建摘要写入器"""
        pass
    
    def flush(self) -> None:
        """刷新缓冲区"""
        pass
    
    def close(self) -> None:
        """关闭日志记录器"""
        pass
    
    def get_log_files(self) -> List[Path]:
        """获取日志文件列表"""
        pass
```

#### CheckpointManager 实现

**必须实现的接口:**
```python
class CheckpointManager:
    def __init__(self, checkpoint_dir: Path, max_checkpoints: int = 5)
    
    def save_checkpoint(self, state_dict: Dict[str, Any], metadata: Dict[str, Any]) -> Path:
        """保存检查点"""
        pass
    
    def load_checkpoint(self, checkpoint_path: Path) -> Dict[str, Any]:
        """加载检查点"""
        pass
    
    def list_checkpoints(self) -> List[CheckpointInfo]:
        """列出检查点"""
        pass
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """获取最新检查点"""
        pass
    
    def get_best_checkpoint(self, metric: str) -> Optional[Path]:
        """获取最佳检查点"""
        pass
    
    def cleanup_old_checkpoints(self) -> None:
        """清理旧检查点"""
        pass
    
    def restore_from_checkpoint(self, checkpoint_path: Path, target_object: Any) -> bool:
        """从检查点恢复"""
        pass
    
    def validate_checkpoint(self, checkpoint_path: Path) -> bool:
        """验证检查点"""
        pass
    
    def get_checkpoint_info(self, checkpoint_path: Path) -> CheckpointInfo:
        """获取检查点信息"""
        pass
```

**CheckpointInfo 数据结构:**
```python
@dataclass
class CheckpointInfo:
    path: Path
    timestamp: datetime
    round_id: int
    task_id: int
    metadata: Dict[str, Any]
    file_size: int
    metrics: Dict[str, float]
    
    def to_dict(self) -> Dict
    def from_dict(cls, data: Dict) -> 'CheckpointInfo'
```

**约束条件:**
- 支持多种日志后端（TensorBoard、W&B等）
- 检查点压缩和加密
- 原子性保存操作
- 版本兼容性检查
- 自动清理过期文件

---

## 🤝 第八阶段: 联邦学习核心

### Step 8.1: 基础联邦组件

**文档输入:**
- `联邦学习核心.md` - 联邦组件设计
- `服务器聚合详细流程图.md` - 聚合流程

注意！！！需要支持实现伪联邦（单机多个客户端）
整个系统存在两种模式：真联邦和伪联邦

#### LocalTrainer 实现

**必须实现的接口:**
```python
class LocalTrainer:
    def __init__(self, learner: BaseLearner, config: DictConfig)
    
    def train_epoch(self, model: torch.nn.Module, task_data: DataLoader) -> Dict[str, float]:
        """训练一个轮次"""
        pass
    
    def evaluate_model(self, model: torch.nn.Module, test_data: DataLoader) -> Dict[str, float]:
        """评估模型"""
        pass
    
    def compute_model_update(self, old_model: torch.nn.Module, 
                            new_model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        """计算模型更新"""
        pass
    
    def apply_model_update(self, model: torch.nn.Module, 
                          update: Dict[str, torch.Tensor]) -> torch.nn.Module:
        """应用模型更新"""
        pass
    
    def get_model_parameters(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        """获取模型参数"""
        pass
    
    def set_model_parameters(self, model: torch.nn.Module, 
                            parameters: Dict[str, torch.Tensor]) -> None:
        """设置模型参数"""
        pass
    
    def compute_gradient_norms(self, model: torch.nn.Module) -> Dict[str, float]:
        """计算梯度范数"""
        pass
    
    def get_training_stats(self) -> Dict[str, Any]:
        """获取训练统计"""
        pass
    
    def reset_optimizer(self) -> None:
        """重置优化器"""
        pass
    
    def save_training_state(self) -> Dict[str, Any]:
        """保存训练状态"""
        pass
    
    def load_training_state(self, state: Dict[str, Any]) -> None:
        """加载训练状态"""
        pass
```

#### ModelManager 实现

**必须实现的接口:**
```python
class ModelManager:
    def __init__(self, config: DictConfig, aggregator: BaseAggregator)
    
    def update_global_model(self, client_updates: List[Dict[str, torch.Tensor]]) -> torch.nn.Module:
        """更新全局模型"""
        pass
    
    def get_current_model(self) -> torch.nn.Module:
        """获取当前模型"""
        pass
    
    def set_global_model(self, model: torch.nn.Module) -> None:
        """设置全局模型"""
        pass
    
    def save_checkpoint(self, round_id: int, additional_info: Dict[str, Any] = None) -> Path:
        """保存检查点"""
        pass
    
    def load_checkpoint(self, checkpoint_path: Path) -> torch.nn.Module:
        """加载检查点"""
        pass
    
    def get_model_diff(self, old_model: torch.nn.Module, 
                      new_model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        """获取模型差异"""
        pass
    
    def apply_model_diff(self, model: torch.nn.Module, 
                        diff: Dict[str, torch.Tensor]) -> torch.nn.Module:
        """应用模型差异"""
        pass
    
    def get_model_size(self, model: torch.nn.Module) -> int:
        """获取模型大小"""
        pass
    
    def compress_model(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        """压缩模型"""
        pass
    
    def decompress_model(self, compressed_model: Dict[str, torch.Tensor]) -> torch.nn.Module:
        """解压模型"""
        pass
    
    def get_model_history(self) -> List[Dict[str, Any]]:
        """获取模型历史"""
        pass
    
    def validate_model_update(self, update: Dict[str, torch.Tensor]) -> bool:
        """验证模型更新"""
        pass
```

#### ClientManager 实现

**必须实现的接口:**
```python
class ClientManager:
    def __init__(self, config: DictConfig)
    
    def register_client(self, client_id: str, client_info: Dict[str, Any]) -> bool:
        """注册客户端"""
        pass
    
    def unregister_client(self, client_id: str) -> bool:
        """注销客户端"""
        pass
    
    def get_active_clients(self) -> List[str]:
        """获取活跃客户端"""
        pass
    
    def select_clients_for_round(self, num_clients: int, round_id: int) -> List[str]:
        """选择参与轮次的客户端"""
        pass
    
    def update_client_status(self, client_id: str, status: ClientStatus) -> None:
        """更新客户端状态"""
        pass
    
    def get_client_info(self, client_id: str) -> Dict[str, Any]:
        """获取客户端信息"""
        pass
    
    def broadcast_to_clients(self, message: Any, targets: List[str]) -> Dict[str, bool]:
        """向客户端广播消息"""
        pass
    
    def collect_from_clients(self, data_type: str, sources: List[str], 
                            timeout: float = 60.0) -> Dict[str, Any]:
        """从客户端收集数据"""
        pass
    
    def get_client_statistics(self, client_id: str) -> Dict[str, Any]:
        """获取客户端统计信息"""
        pass
    
    def set_client_selection_strategy(self, strategy: str) -> None:
        """设置客户端选择策略"""
        pass
    
    def handle_client_failure(self, client_id: str, error: Exception) -> None:
        """处理客户端故障"""
        pass
```

**ClientStatus 枚举:**
```python
from enum import Enum

class ClientStatus(Enum):
    DISCONNECTED = "disconnected"
    CONNECTED = "connected"
    TRAINING = "training"
    READY = "ready"
    ERROR = "error"
    TIMEOUT = "timeout"
```

### Step 8.2: 客户端服务端基类

**文档输入:**
- `联邦学习核心.md` - 基类接口设计
- `联邦学习完整训练轮次交互.md` - 交互协议

#### BaseClient 抽象基类

**必须实现的接口:**
```python
class BaseClient(ABC):
    def __init__(self, client_id: str, context: ExecutionContext)
    
    @abstractmethod
    def connect_to_server(self, server_address: Tuple[str, int]) -> bool:
        """连接到服务器"""
        pass
    
    @abstractmethod
    def receive_global_model(self) -> torch.nn.Module:
        """接收全局模型"""
        pass
    
    @abstractmethod
    def train_local_model(self, task_data: DataLoader) -> Dict[str, torch.Tensor]:
        """训练本地模型"""
        pass
    
    @abstractmethod
    def send_update_to_server(self, update: Dict[str, torch.Tensor]) -> bool:
        """发送更新到服务器"""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """断开连接"""
        pass
    
    # 可选实现的方法
    def get_client_info(self) -> Dict[str, Any]:
        """获取客户端信息"""
        return {
            "client_id": self.client_id,
            "status": self.status,
            "local_data_size": self.get_local_data_size()
        }
    
    def get_local_data_size(self) -> int:
        """获取本地数据大小"""
        return 0
    
    def handle_server_message(self, message: Message) -> Any:
        """处理服务器消息"""
        pass
    
    def send_heartbeat(self) -> bool:
        """发送心跳"""
        pass
    
    def get_client_capabilities(self) -> Dict[str, Any]:
        """获取客户端能力"""
        return {}
```

#### BaseServer 抽象基类

**必须实现的接口:**
```python
class BaseServer(ABC):
    def __init__(self, server_id: str, context: ExecutionContext)
    
    @abstractmethod
    def start_server(self, host: str, port: int) -> bool:
        """启动服务器"""
        pass
    
    @abstractmethod
    def select_clients(self, round_id: int) -> List[str]:
        """选择客户端"""
        pass
    
    @abstractmethod
    def broadcast_global_model(self, model: torch.nn.Module, targets: List[str]) -> Dict[str, bool]:
        """广播全局模型"""
        pass
    
    @abstractmethod
    def collect_client_updates(self, sources: List[str], timeout: float = 60.0) -> List[Dict[str, torch.Tensor]]:
        """收集客户端更新"""
        pass
    
    @abstractmethod
    def aggregate_updates(self, updates: List[Dict[str, torch.Tensor]]) -> torch.nn.Module:
        """聚合更新"""
        pass
    
    @abstractmethod
    def shutdown(self) -> None:
        """关闭服务器"""
        pass
    
    # 可选实现的方法
    def handle_client_connection(self, client_id: str) -> bool:
        """处理客户端连接"""
        return True
    
    def handle_client_disconnection(self, client_id: str) -> None:
        """处理客户端断开"""
        pass
    
    def get_server_stats(self) -> Dict[str, Any]:
        """获取服务器统计"""
        return {}
    
    def validate_client_update(self, client_id: str, update: Dict[str, torch.Tensor]) -> bool:
        """验证客户端更新"""
        return True
```

### Step 8.3: 联邦客户端实现

**文档输入:**
- `客户端详细训练流程图.md` - 客户端详细流程
- `错误处理与恢复机制交互.md` - 客户端容错

**依赖关系:**
- **必需依赖**: 
  - `client_id: str` - 客户端标识
  - `context: ExecutionContext` - 执行上下文
  - `local_trainer: LocalTrainer` - 本地训练器
  - `data_handler: DataHandler` - 数据处理器
- **可选依赖**: 
  - `communication_handler: Optional[CommunicationHandler] = None` - 通信处理器
  - `model_updater: Optional[ModelUpdater] = None` - 模型更新器
- **配置依赖**: 
  - 客户端配置（从context.config.client获取）
  - 通信配置（从context.config.communication获取）
- **运行时依赖**: 
  - `FederatedServer` - 通过网络连接交互
  - `torch.nn.Module` - 接收和发送的模型

**调用关系说明:**
- **调用依赖的方式**:
  - `self.local_trainer.train_epoch(model, task_data)` - 本地训练
  - `self.data_handler.get_task_data(task_id)` - 获取任务数据
  - `self.communication_handler.send_message(target, message)` - 发送消息
  - `self.context.log_metric(name, value)` - 记录度量
- **网络调用**:
  - `receive_global_model()` - 从服务器接收模型
  - `send_update_to_server(update)` - 向服务器发送更新
- **被调用场景**:
  - `FederationEngine.coordinate_clients()` 中协调客户端训练
  - 服务器通过网络调用客户端方法

**依赖注入实现:**
```python
class FederatedClient(BaseClient):
    def __init__(self, 
                 client_id: str, 
                 context: ExecutionContext,
                 local_trainer: LocalTrainer,
                 data_handler: DataHandler,
                 communication_handler: Optional[CommunicationHandler] = None):
        super().__init__(client_id, context)
        self.local_trainer = local_trainer
        self.data_handler = data_handler
        self.communication_handler = communication_handler or self._create_default_communication_handler()
        
        # 从配置获取客户端参数
        self.client_config = context.get_config("client", {})
        self.max_retries = self.client_config.get("max_retries", 3)
        self.timeout = self.client_config.get("timeout", 30.0)
        
        # 客户端状态
        self.status = ClientStatus.DISCONNECTED
        self.current_model: Optional[torch.nn.Module] = None
        self.server_connection: Optional[Connection] = None
        
    def connect_to_server(self, server_address: Tuple[str, int]) -> bool:
        try:
            # 使用通信处理器建立连接
            self.server_connection = self.communication_handler.establish_connection(server_address)
            
            # 发送客户端注册消息
            registration_message = Message(
                message_type=MessageProtocol.CLIENT_READY,
                sender=self.client_id,
                receiver="server",
                data=self.get_client_info()
            )
            
            success = self.communication_handler.send_message("server", registration_message)
            if success:
                self.status = ClientStatus.CONNECTED
                self.context.log_metric("client_connection_success", 1.0)
                return True
            else:
                self.status = ClientStatus.ERROR
                return False
                
        except Exception as e:
            self.context.log_metric("client_connection_error", 1.0)
            self.handle_training_error(e)
            return False
            
    def train_local_model(self, task_data: DataLoader) -> Dict[str, torch.Tensor]:
        if self.current_model is None:
            raise RuntimeError("No global model received from server")
            
        try:
            self.status = ClientStatus.TRAINING
            
            # 记录训练开始
            self.context.log_metric("client_training_start", 1.0)
            
            # 保存初始模型状态
            initial_state = copy.deepcopy(self.current_model.state_dict())
            
            # 使用本地训练器进行训练
            training_results = self.local_trainer.train_epoch(self.current_model, task_data)
            
            # 计算模型更新（差值）
            final_state = self.current_model.state_dict()
            model_update = self.compute_local_update(initial_state, final_state)
            
            # 记录训练度量
            for metric_name, metric_value in training_results.items():
                self.context.log_metric(f"client_{metric_name}", metric_value)
                
            self.status = ClientStatus.READY
            return model_update
            
        except Exception as e:
            self.status = ClientStatus.ERROR
            self.handle_training_error(e)
            raise
            
    def receive_global_model(self) -> torch.nn.Module:
        try:
            # 从服务器接收模型消息
            model_message = self.communication_handler.receive_message("server", timeout=self.timeout)
            
            if model_message.message_type != MessageProtocol.GLOBAL_MODEL:
                raise ValueError(f"Expected global model message, got {model_message.message_type}")
                
            # 反序列化模型
            model_data = model_message.data
            self.current_model = self._deserialize_model(model_data)
            
            # 验证接收的模型
            if not self.validate_received_model(self.current_model):
                raise ValueError("Received invalid model from server")
                
            self.context.log_metric("model_received", 1.0)
            return self.current_model
            
        except Exception as e:
            self.context.log_metric("model_receive_error", 1.0)
            self.handle_training_error(e)
            raise
```

**错误处理和恢复:**
- **网络错误**: 自动重连机制，指数退避策略
- **训练错误**: 状态恢复，部分更新发送
- **模型验证错误**: 请求重新发送模型
- **超时处理**: 优雅超时，状态保存

**必须实现的接口:**
```python
class FederatedClient(BaseClient):
    def __init__(self, client_id: str, context: ExecutionContext, 
                 local_trainer: LocalTrainer, data_handler: DataHandler)
    
    def connect_to_server(self, server_address: Tuple[str, int]) -> bool:
        """连接到服务器"""
        pass
    
    def receive_global_model(self) -> torch.nn.Module:
        """接收全局模型"""
        pass
    
    def train_local_model(self, task_data: DataLoader) -> Dict[str, torch.Tensor]:
        """训练本地模型"""
        pass
    
    def send_update_to_server(self, update: Dict[str, torch.Tensor]) -> bool:
        """发送更新到服务器"""
        pass
    
    def handle_task_sequence(self, tasks: List[Task]) -> List[TaskResults]:
        """处理任务序列"""
        pass
    
    def disconnect(self) -> None:
        """断开连接"""
        pass
    
    # 额外的方法
    def prepare_for_training(self, global_model: torch.nn.Module) -> None:
        """准备训练"""
        pass
    
    def compute_local_update(self, initial_model: torch.nn.Module, 
                            trained_model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        """计算本地更新"""
        pass
    
    def validate_received_model(self, model: torch.nn.Module) -> bool:
        """验证接收的模型"""
        pass
    
    def handle_training_error(self, error: Exception) -> bool:
        """处理训练错误"""
        pass
    
    def get_training_progress(self) -> Dict[str, Any]:
        """获取训练进度"""
        pass
    
    def pause_training(self) -> None:
        """暂停训练"""
        pass
    
    def resume_training(self) -> None:
        """恢复训练"""
        pass
```

### Step 8.4: 联邦服务端实现

**文档输入:**
- `服务器聚合详细流程图.md` - 服务端详细流程
- `联邦学习完整训练轮次交互.md` - 服务端协调逻辑

**依赖关系:**
- **必需依赖**: 
  - `server_id: str` - 服务器标识
  - `context: ExecutionContext` - 执行上下文
  - `client_manager: ClientManager` - 客户端管理器
  - `model_manager: ModelManager` - 模型管理器
  - `communication_manager: CommunicationManager` - 通信管理器
- **可选依赖**: 
  - `client_selector: Optional[ClientSelector] = None` - 客户端选择器
  - `aggregation_server: Optional[AggregationServer] = None` - 聚合服务器
- **配置依赖**: 
  - 服务器配置（从context.config.server获取）
  - 联邦配置（从context.config.federation获取）
- **运行时依赖**: 
  - `List[FederatedClient]` - 连接的客户端
  - `BaseAggregator` - 聚合器

**调用关系说明:**
- **调用依赖的方式**:
  - `self.client_manager.select_clients_for_round(num_clients)` - 选择客户端
  - `self.model_manager.get_current_model()` - 获取当前模型
  - `self.communication_manager.broadcast_model(model, targets)` - 广播模型
  - `self.model_manager.update_global_model(updates)` - 更新全局模型
- **网络调用**:
  - 与多个`FederatedClient`进行网络通信
  - 并行收集客户端更新
  - 处理客户端连接和断开
- **被调用场景**:
  - `FederationEngine.start_federation_round()` 中协调轮次
  - `ExperimentEngine` 中管理整个联邦学习过程

**依赖注入实现:**
```python
class FederatedServer(BaseServer):
    def __init__(self,
                 server_id: str,
                 context: ExecutionContext, 
                 client_manager: ClientManager,
                 model_manager: ModelManager,
                 communication_manager: CommunicationManager,
                 client_selector: Optional[ClientSelector] = None):
        super().__init__(server_id, context)
        self.client_manager = client_manager
        self.model_manager = model_manager
        self.communication_manager = communication_manager
        self.client_selector = client_selector or DefaultClientSelector()
        
        # 从配置获取服务器参数
        self.server_config = context.get_config("server", {})
        self.federation_config = context.get_config("federation", {})
        
        self.num_rounds = self.federation_config.get("num_rounds", 100)
        self.clients_per_round = self.federation_config.get("clients_per_round", 10)
        self.round_timeout = self.server_config.get("round_timeout", 300.0)
        
        # 服务器状态
        self.current_round = 0
        self.is_running = False
        self.round_stats: Dict[int, Dict[str, Any]] = {}
        
    def coordinate_round(self, round_id: int) -> Dict[str, Any]:
        round_start_time = time.time()
        
        try:
            # 1. 选择参与的客户端
            selected_clients = self.select_clients(round_id)
            if not selected_clients:
                raise RuntimeError("No clients available for training")
                
            # 2. 获取当前全局模型
            global_model = self.model_manager.get_current_model()
            
            # 3. 广播模型给选中的客户端
            broadcast_results = self.broadcast_global_model(global_model, selected_clients)
            successful_clients = [client_id for client_id, success in broadcast_results.items() if success]
            
            if not successful_clients:
                raise RuntimeError("Failed to broadcast model to any client")
                
            # 4. 等待客户端训练并收集更新
            client_updates = self.collect_client_updates(successful_clients, self.round_timeout)
            
            # 5. 聚合客户端更新
            if client_updates:
                updated_model = self.aggregate_updates(client_updates)
                
                # 6. 验证聚合结果
                if self.validate_aggregation_result(updated_model):
                    self.model_manager.set_global_model(updated_model)
                    
            # 7. 记录轮次统计
            round_stats = {
                "round_id": round_id,
                "selected_clients": len(selected_clients),
                "successful_broadcasts": len(successful_clients),
                "received_updates": len(client_updates),
                "round_time": time.time() - round_start_time,
                "model_accuracy": self._evaluate_current_model()
            }
            
            self.round_stats[round_id] = round_stats
            
            # 记录度量
            for metric_name, metric_value in round_stats.items():
                if isinstance(metric_value, (int, float)):
                    self.context.log_metric(f"server_{metric_name}", metric_value, round_id)
                    
            return round_stats
            
        except Exception as e:
            # 处理轮次错误
            error_stats = {
                "round_id": round_id,
                "error": str(e),
                "round_time": time.time() - round_start_time,
                "status": "failed"
            }
            self.round_stats[round_id] = error_stats
            self.context.log_metric("server_round_error", 1.0, round_id)
            raise
            
    def collect_client_updates(self, sources: List[str], timeout: float = 60.0) -> List[Dict[str, torch.Tensor]]:
        updates = []
        
        # 并行收集客户端更新
        with ThreadPoolExecutor(max_workers=len(sources)) as executor:
            # 为每个客户端提交收集任务
            future_to_client = {
                executor.submit(self._collect_single_client_update, client_id, timeout): client_id
                for client_id in sources
            }
            
            # 收集结果
            for future in as_completed(future_to_client, timeout=timeout):
                client_id = future_to_client[future]
                try:
                    update = future.result()
                    if update is not None:
                        updates.append(update)
                        self.context.log_metric("client_update_received", 1.0)
                except Exception as e:
                    self.context.log_metric("client_update_error", 1.0)
                    self.client_manager.handle_client_failure(client_id, e)
                    
        return updates
        
    def _collect_single_client_update(self, client_id: str, timeout: float) -> Optional[Dict[str, torch.Tensor]]:
        try:
            # 通过通信管理器收集客户端更新
            update_data = self.communication_manager.receive_data(
                source=client_id,
                data_type=MessageProtocol.MODEL_UPDATE,
                timeout=timeout
            )
            
            # 验证更新数据
            if self.validate_client_update(client_id, update_data):
                return update_data
            else:
                self.context.log_metric("invalid_client_update", 1.0)
                return None
                
        except TimeoutError:
            self.context.log_metric("client_timeout", 1.0)
            self.client_manager.update_client_status(client_id, ClientStatus.TIMEOUT)
            return None
        except Exception as e:
            self.client_manager.handle_client_failure(client_id, e)
            return None
```

**并发和性能优化:**
- **并行通信**: 使用线程池并行与多个客户端通信
- **异步聚合**: 支持流式聚合，不等待所有客户端
- **资源管理**: 限制并发连接数，防止资源耗尽
- **超时处理**: 多级超时机制，避免长时间阻塞

**必须实现的接口:**
```python
class FederatedServer(BaseServer):
    def __init__(self, server_id: str, context: ExecutionContext,
                 client_manager: ClientManager, model_manager: ModelManager,
                 communication_manager: CommunicationManager)
    
    def start_server(self, host: str, port: int) -> bool:
        """启动服务器"""
        pass
    
    def select_clients(self, round_id: int) -> List[str]:
        """选择客户端"""
        pass
    
    def broadcast_global_model(self, model: torch.nn.Module, targets: List[str]) -> Dict[str, bool]:
        """广播全局模型"""
        pass
    
    def collect_client_updates(self, sources: List[str], timeout: float = 60.0) -> List[Dict[str, torch.Tensor]]:
        """收集客户端更新"""
        pass
    
    def aggregate_updates(self, updates: List[Dict[str, torch.Tensor]]) -> torch.nn.Module:
        """聚合更新"""
        pass
    
    def coordinate_round(self, round_id: int) -> Dict[str, Any]:
        """协调轮次"""
        pass
    
    def shutdown(self) -> None:
        """关闭服务器"""
        pass
    
    # 额外的方法
    def handle_round_timeout(self, round_id: int, missing_clients: List[str]) -> bool:
        """处理轮次超时"""
        pass
    
    def validate_aggregation_result(self, aggregated_model: torch.nn.Module) -> bool:
        """验证聚合结果"""
        pass
    
    def get_round_statistics(self, round_id: int) -> Dict[str, Any]:
        """获取轮次统计"""
        pass
    
    def handle_partial_participation(self, available_updates: List[Dict[str, torch.Tensor]],
                                    missing_clients: List[str]) -> torch.nn.Module:
        """处理部分参与"""
        pass
    
    def schedule_next_round(self) -> None:
        """调度下一轮"""
        pass
```

---

## 🎮 第九阶段: 主控制器

### Step 9.1: FederationEngine 实现

**文档输入:**
- `宏观架构.md` - 整体控制流程
- `联邦训练核心流程图.md` - 联邦协调逻辑

**必须实现的接口:**
```python
class FederationEngine:
    def __init__(self, context: ExecutionContext, communication_manager: CommunicationManager,
                 aggregator: BaseAggregator)
    
    def coordinate_clients(self, round_id: int) -> List[str]:
        """协调客户端"""
        pass
    
    def aggregate_updates(self, client_updates: List[Dict[str, torch.Tensor]]) -> torch.nn.Module:
        """聚合客户端更新"""
        pass
    
    def broadcast_model(self, model: torch.nn.Module, targets: List[str]) -> Dict[str, bool]:
        """广播模型"""
        pass
    
    def create_federated_server(self, config: DictConfig) -> FederatedServer:
        """创建联邦服务器"""
        pass
    
    def create_federated_clients(self, num_clients: int, config: DictConfig) -> List[FederatedClient]:
        """创建联邦客户端"""
        pass
    
    def start_federation_round(self, round_id: int) -> Dict[str, Any]:
        """开始联邦轮次"""
        pass
    
    def handle_federation_error(self, error: Exception, round_id: int) -> bool:
        """处理联邦错误"""
        pass
    
    def get_federation_stats(self) -> Dict[str, Any]:
        """获取联邦统计"""
        pass
    
    def pause_federation(self) -> None:
        """暂停联邦学习"""
        pass
    
    def resume_federation(self) -> None:
        """恢复联邦学习"""
        pass
    
    def stop_federation(self) -> None:
        """停止联邦学习"""
        pass
```

#### DistributedComputationManager 实现

**必须实现的接口:**
```python
class DistributedComputationManager:
    def __init__(self, context: ExecutionContext)
    
    def create_feature_transfer_point(self, source: str, target: str, 
                                     layer: str, features: torch.Tensor) -> str:
        """创建特征传输点"""
        pass
    
    def handle_gradient_backprop(self, tensor_id: str, gradients: torch.Tensor) -> None:
        """处理梯度反向传播"""
        pass
    
    def register_virtual_tensor(self, tensor_id: str, metadata: Dict[str, Any]) -> None:
        """注册虚拟张量"""
        pass
    
    def get_virtual_tensor(self, tensor_id: str) -> Optional[torch.Tensor]:
        """获取虚拟张量"""
        pass
    
    def synchronize_gradients(self, tensor_ids: List[str]) -> None:
        """同步梯度"""
        pass
    
    def optimize_computation_graph(self, graph: Dict[str, Any]) -> Dict[str, Any]:
        """优化计算图"""
        pass
    
    def handle_tensor_communication(self, source: str, target: str, tensor: torch.Tensor) -> None:
        """处理张量通信"""
        pass
    
    def cleanup_virtual_tensors(self) -> None:
        """清理虚拟张量"""
        pass
```

### Step 9.2: ExperimentEngine 实现

**文档输入:**
- `FedCL组件交互逻辑详解.md` - 实验生命周期
- `组件注册与实验初始化交互.md` - 初始化流程

**必须实现的接口:**
```python
class ExperimentEngine:
    def __init__(self, registry: ComponentRegistry, composer: ComponentComposer, config: DictConfig)
    
    def start_experiment(self, config: DictConfig) -> ExperimentResults:
        """启动实验"""
        pass
    
    def manage_lifecycle(self) -> None:
        """管理生命周期"""
        pass
    
    def handle_checkpoints(self) -> None:
        """处理检查点"""
        pass
    
    def coordinate_engines(self) -> None:
        """协调引擎"""
        pass
    
    def monitor_experiment_progress(self) -> Dict[str, Any]:
        """监控实验进度"""
        pass
    
    def handle_experiment_error(self, error: Exception) -> bool:
        """处理实验错误"""
        pass
    
    def pause_experiment(self) -> None:
        """暂停实验"""
        pass
    
    def resume_experiment(self, checkpoint_path: Optional[Path] = None) -> None:
        """恢复实验"""
        pass
    
    def stop_experiment(self, save_results: bool = True) -> ExperimentResults:
        """停止实验"""
        pass
    
    def get_experiment_status(self) -> Dict[str, Any]:
        """获取实验状态"""
        pass
    
    def validate_experiment_config(self, config: DictConfig) -> ValidationResult:
        """验证实验配置"""
        pass
    
    def setup_experiment_environment(self) -> None:
        """设置实验环境"""
        pass
    
    def cleanup_experiment_environment(self) -> None:
        """清理实验环境"""
        pass
```

---

## 👤 第十阶段: 用户接口层

### Step 10.1: 主要用户接口

**文档输入:**
- `核心框架层.md` - 用户接口设计
- `README.md` - 用户使用需求

#### FedCLExperiment 实现

**必须实现的接口:**
```python
class FedCLExperiment:
    def __init__(self, config: Union[str, Path, DictConfig], 
                 working_dir: Optional[Path] = None, seed: Optional[int] = None)
    
    def run(self) -> ExperimentResults:
        """运行实验"""
        pass
    
    def run_sweep(self, sweep_config: DictConfig) -> List[ExperimentResults]:
        """运行参数扫描"""
        pass
    
    def resume(self, checkpoint_path: Path) -> ExperimentResults:
        """从检查点恢复"""
        pass
    
    def evaluate_checkpoint(self, checkpoint_path: Path, test_data: DataLoader) -> Dict[str, float]:
        """评估检查点"""
        pass
    
    def get_experiment_config(self) -> DictConfig:
        """获取实验配置"""
        pass
    
    def set_experiment_config(self, config: DictConfig) -> None:
        """设置实验配置"""
        pass
    
    def validate_config(self) -> ValidationResult:
        """验证配置"""
        pass
    
    def get_available_components(self) -> Dict[str, List[str]]:
        """获取可用组件"""
        pass
    
    def register_custom_component(self, component_type: str, name: str, component_class: Type) -> None:
        """注册自定义组件"""
        pass
    
    def save_config(self, path: Path) -> None:
        """保存配置"""
        pass
    
    def load_config(self, path: Path) -> None:
        """加载配置"""
        pass
```

#### QuickAPI 实现

**必须实现的接口:**
```python
class QuickAPI:
    @staticmethod
    def quick_experiment(method: str, dataset: str, num_tasks: int, 
                        num_clients: int, **kwargs) -> ExperimentResults:
        """快速实验"""
        pass
    
    @staticmethod
    def create_simple_config(method: str, dataset: str, **kwargs) -> DictConfig:
        """创建简单配置"""
        pass
    
    @staticmethod
    def list_available_methods() -> List[str]:
        """列出可用方法"""
        pass
    
    @staticmethod
    def list_available_datasets() -> List[str]:
        """列出可用数据集"""
        pass
    
    @staticmethod
    def benchmark_methods(methods: List[str], dataset: str, 
                         num_runs: int = 3) -> Dict[str, ExperimentResults]:
        """基准测试方法"""
        pass
    
    @staticmethod
    def compare_results(results: List[ExperimentResults]) -> Dict[str, Any]:
        """比较结果"""
        pass
    
    @staticmethod
    def generate_report(results: ExperimentResults, output_path: Path) -> None:
        """生成报告"""
        pass
```

---

## 🧩 第十一阶段: 算法实现

### Step 11.1: 基础算法实现

**文档输入:**
- `组件与扩展系统.md` - 算法实现指南
- 相关算法论文

#### FedAvgAggregator 实现

**必须实现的接口:**
```python
class FedAvgAggregator(BaseAggregator):
    def __init__(self, context: ExecutionContext, config: DictConfig)
    
    def aggregate(self, client_updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """FedAvg聚合"""
        pass
    
    def weight_updates(self, updates: List[Dict[str, torch.Tensor]]) -> List[float]:
        """计算权重"""
        pass
    
    def compute_weighted_average(self, updates: List[Dict[str, torch.Tensor]], 
                                weights: List[float]) -> Dict[str, torch.Tensor]:
        """计算加权平均"""
        pass
    
    def normalize_weights(self, weights: List[float]) -> List[float]:
        """标准化权重"""
        pass
```

#### AccuracyEvaluator 实现

**必须实现的接口:**
```python
class AccuracyEvaluator(BaseEvaluator):
    def __init__(self, context: ExecutionContext, config: DictConfig)
    
    def evaluate(self, model: torch.nn.Module, data: DataLoader) -> Dict[str, float]:
        """评估模型"""
        pass
    
    def compute_task_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """计算任务指标"""
        pass
    
    def compute_accuracy(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """计算准确率"""
        pass
    
    def compute_confusion_matrix(self, predictions: torch.Tensor, targets: torch.Tensor) -> np.ndarray:
        """计算混淆矩阵"""
        pass
```

### Step 11.2: 高级算法实现

#### L2PLearner 实现

**必须实现的接口:**
```python
class L2PLearner(BaseLearner):
    def __init__(self, context: ExecutionContext, config: DictConfig)
    
    def train_task(self, task_data: DataLoader) -> TaskResults:
        """训练任务"""
        pass
    
    def evaluate_task(self, task_data: DataLoader) -> Dict[str, float]:
        """评估任务"""
        pass
    
    def select_prompts(self, input_data: torch.Tensor) -> torch.Tensor:
        """选择提示"""
        pass
    
    def update_prompt_pool(self, gradients: torch.Tensor) -> None:
        """更新提示池"""
        pass
    
    def compute_prompt_similarity(self, query: torch.Tensor, prompts: torch.Tensor) -> torch.Tensor:
        """计算提示相似度"""
        pass
```

#### DDDRLearner 实现

**必须实现的接口:**
```python
class DDDRLearner(BaseLearner):
    def __init__(self, context: ExecutionContext, config: DictConfig)
    
    def train_task(self, task_data: DataLoader) -> TaskResults:
        """训练任务"""
        pass
    
    def evaluate_task(self, task_data: DataLoader) -> Dict[str, float]:
        """评估任务"""
        pass
    
    def extract_class_embeddings(self, class_data: DataLoader) -> Dict[int, torch.Tensor]:
        """提取类别嵌入"""
        pass
    
    def generate_replay_data(self, embeddings: Dict[int, torch.Tensor]) -> List[Tuple[torch.Tensor, int]]:
        """生成重放数据"""
        pass
    
    def update_diffusion_model(self, replay_data: DataLoader) -> None:
        """更新扩散模型"""
        pass
```

## 📊 配置类系统

### 全局配置结构

**必须实现的配置类:**
```python
@dataclass
class ExperimentConfig:
    name: str
    description: str
    seed: int
    working_dir: Path
    log_level: str
    
    # 组件配置
    learner: LearnerConfig
    aggregator: AggregatorConfig
    evaluator: EvaluatorConfig
    
    # 数据配置
    data: DataConfig
    
    # 联邦配置
    federation: FederationConfig
    
    # 训练配置
    training: TrainingConfig
    
    # 通信配置
    communication: CommunicationConfig
    
    # 钩子配置
    hooks: List[HookConfig]

@dataclass
class LearnerConfig:
    name: str
    type: str
    parameters: Dict[str, Any]
    
@dataclass
class DataConfig:
    dataset_name: str
    num_tasks: int
    num_clients: int
    split_strategy: str
    split_parameters: Dict[str, Any]
    
@dataclass
class FederationConfig:
    num_rounds: int
    clients_per_round: int
    client_selection_strategy: str
    aggregation_strategy: str
    
@dataclass
class TrainingConfig:
    num_epochs: int
    batch_size: int
    learning_rate: float
    optimizer: str
    scheduler: Optional[str]
```

---

## 📋 实现指导总结

## 📊 综合依赖管理策略

### 依赖注入最佳实践

#### 1. 构造函数注入模式
**所有依赖都通过构造函数明确声明:**
```python
class ExampleClass:
    def __init__(self, 
                 required_dep: RequiredInterface,
                 optional_dep: Optional[OptionalInterface] = None,
                 config: DictConfig = None):
        # 验证必需依赖
        if required_dep is None:
            raise ValueError("required_dep cannot be None")
            
        self.required_dep = required_dep
        self.optional_dep = optional_dep
        self.config = config or DictConfig({})
        
        # 从配置初始化
        self._init_from_config()
```

#### 2. 工厂模式处理复杂创建
**对于复杂的对象创建，使用工厂模式:**
```python
class ComponentFactory:
    def __init__(self, registry: ComponentRegistry, context: ExecutionContext):
        self.registry = registry
        self.context = context
        
    def create_learner(self, learner_config: DictConfig) -> BaseLearner:
        learner_class = self.registry.get_component("learner", learner_config.name)
        
        # 创建依赖
        optimizer = self._create_optimizer(learner_config.optimizer)
        scheduler = self._create_scheduler(learner_config.scheduler)
        
        # 注入依赖
        return learner_class(
            context=self.context,
            config=learner_config,
            optimizer=optimizer,
            scheduler=scheduler
        )
```

#### 3. 服务定位器模式
**对于可选和运行时依赖，使用服务定位器:**
```python
class ServiceLocator:
    _instance = None
    _services: Dict[Type, Any] = {}
    
    @classmethod
    def register(cls, service_type: Type, service_instance: Any):
        cls._services[service_type] = service_instance
        
    @classmethod
    def get(cls, service_type: Type, default: Any = None) -> Any:
        return cls._services.get(service_type, default)
        
# 使用示例
class SomeClass:
    def __init__(self, required_dep: RequiredInterface):
        self.required_dep = required_dep
        # 可选依赖通过服务定位器获取
        self.optional_service = ServiceLocator.get(OptionalService)
```

### 依赖生命周期管理

#### 1. 生命周期分类
```python
from enum import Enum

class DependencyLifecycle(Enum):
    SINGLETON = "singleton"      # 全局单例
    SCOPED = "scoped"           # 作用域内单例
    TRANSIENT = "transient"     # 每次创建新实例
    FACTORY = "factory"         # 通过工厂创建
    
# 示例实现
class DependencyContainer:
    def __init__(self):
        self._singletons: Dict[Type, Any] = {}
        self._scoped: Dict[str, Dict[Type, Any]] = {}
        self._factories: Dict[Type, Callable] = {}
        
    def register_singleton(self, service_type: Type, instance: Any):
        self._singletons[service_type] = instance
        
    def get_scoped(self, scope_id: str, service_type: Type) -> Any:
        if scope_id not in self._scoped:
            self._scoped[scope_id] = {}
        return self._scoped[scope_id].get(service_type)
```

#### 2. 具体生命周期规划
- **ConfigManager**: Singleton（全局配置管理）
- **ExecutionContext**: Scoped（实验作用域）
- **CommunicationManager**: Singleton（全局通信）
- **TrainingEngine**: Scoped（实验作用域）
- **Task**: Transient（每次创建新实例）
- **BaseLearner**: Factory（通过ComponentComposer创建）

### 循环依赖避免策略

#### 1. 依赖图验证
```python
class DependencyGraphValidator:
    def __init__(self):
        self.graph: Dict[str, List[str]] = {}
        
    def add_dependency(self, dependent: str, dependency: str):
        if dependent not in self.graph:
            self.graph[dependent] = []
        self.graph[dependent].append(dependency)
        
    def detect_cycles(self) -> List[List[str]]:
        def dfs(node, path, visited, rec_stack):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in self.graph.get(node, []):
                if neighbor in rec_stack:
                    # 找到循环
                    cycle_start = path.index(neighbor)
                    return path[cycle_start:] + [neighbor]
                elif neighbor not in visited:
                    cycle = dfs(neighbor, path, visited, rec_stack)
                    if cycle:
                        return cycle
                        
            path.pop()
            rec_stack.remove(node)
            return None
            
        visited = set()
        for node in self.graph:
            if node not in visited:
                cycle = dfs(node, [], visited, set())
                if cycle:
                    return cycle
        return []
```

#### 2. 延迟初始化模式
```python
class LazyDependency:
    def __init__(self, factory: Callable[[], Any]):
        self._factory = factory
        self._instance = None
        self._initialized = False
        
    def get(self) -> Any:
        if not self._initialized:
            self._instance = self._factory()
            self._initialized = True
        return self._instance
        
# 使用示例
class ComponentWithCircularDep:
    def __init__(self, other_component_factory: Callable[[], 'OtherComponent']):
        self._other_component = LazyDependency(other_component_factory)
        
    def use_other_component(self):
        other = self._other_component.get()
        return other.some_method()
```

### 测试策略和Mock管理

#### 1. 依赖Mock策略
```python
from unittest.mock import Mock, MagicMock
import pytest

class TestTrainingEngine:
    @pytest.fixture
    def mock_dependencies(self):
        return {
            'hook_executor': Mock(spec=HookExecutor),
            'context': Mock(spec=ExecutionContext),
            'checkpoint_manager': Mock(spec=CheckpointManager),
            'metrics_logger': Mock(spec=MetricsLogger)
        }
    
    @pytest.fixture
    def training_engine(self, mock_dependencies):
        # 配置mock行为
        mock_dependencies['context'].get_config.return_value = {
            "num_epochs": 5,
            "batch_size": 32
        }
        mock_dependencies['hook_executor'].execute_hooks.return_value = []
        
        return TrainingEngine(**mock_dependencies)
        
    def test_train_task(self, training_engine, mock_dependencies):
        # 准备测试数据
        mock_learner = Mock(spec=BaseLearner)
        mock_task_data = Mock(spec=DataLoader)
        
        # 执行测试
        result = training_engine.train_task(1, mock_task_data, mock_learner)
        
        # 验证调用
        mock_dependencies['hook_executor'].execute_hooks.assert_called()
        assert isinstance(result, TaskResults)
```

#### 2. 集成测试策略
```python
class IntegrationTestContext:
    def __init__(self):
        self.real_dependencies = {}
        self.mock_dependencies = {}
        
    def use_real_dependency(self, dep_type: Type, instance: Any):
        self.real_dependencies[dep_type] = instance
        
    def use_mock_dependency(self, dep_type: Type, mock_instance: Mock):
        self.mock_dependencies[dep_type] = mock_instance
        
    def create_component(self, component_class: Type) -> Any:
        # 根据配置注入真实或mock依赖
        constructor_params = {}
        for param_name, param_type in component_class.__init__.__annotations__.items():
            if param_type in self.real_dependencies:
                constructor_params[param_name] = self.real_dependencies[param_type]
            elif param_type in self.mock_dependencies:
                constructor_params[param_name] = self.mock_dependencies[param_type]
                
        return component_class(**constructor_params)
```

### 配置驱动的依赖注入

#### 1. 配置文件格式
```yaml
dependencies:
  # 单例服务配置
  singletons:
    config_manager:
      class: "fedcl.config.ConfigManager"
      args:
        schema_validator: "@schema_validator"
        
    communication_manager:
      class: "fedcl.communication.CommunicationManager"
      args:
        config: "@config.communication"
        
  # 作用域服务配置  
  scoped:
    experiment:
      execution_context:
        class: "fedcl.core.ExecutionContext"
        args:
          config: "@config"
          experiment_id: "${experiment.id}"
          
      training_engine:
        class: "fedcl.training.TrainingEngine"
        args:
          hook_executor: "@hook_executor"
          context: "@execution_context"
          
  # 工厂服务配置
  factories:
    learner_factory:
      class: "fedcl.factories.LearnerFactory"
      method: "create_learner"
      args:
        registry: "@component_registry"
        context: "@execution_context"
```

#### 2. 配置解析器实现
```python
class DependencyInjectionContainer:
    def __init__(self, config: DictConfig):
        self.config = config
        self._singletons: Dict[str, Any] = {}
        self._scoped_instances: Dict[str, Dict[str, Any]] = {}
        
    def resolve_dependency(self, dependency_spec: str, scope: str = "global") -> Any:
        if dependency_spec.startswith("@"):
            # 依赖引用
            dep_name = dependency_spec[1:]
            return self._get_dependency(dep_name, scope)
        elif dependency_spec.startswith("${"):
            # 配置引用
            config_path = dependency_spec[2:-1]
            return self._get_config_value(config_path)
        else:
            # 字面值
            return dependency_spec
            
    def _get_dependency(self, dep_name: str, scope: str) -> Any:
        # 检查单例
        if dep_name in self._singletons:
            return self._singletons[dep_name]
            
        # 检查作用域实例
        if scope in self._scoped_instances and dep_name in self._scoped_instances[scope]:
            return self._scoped_instances[scope][dep_name]
            
        # 创建新实例
        return self._create_instance(dep_name, scope)
```

### 性能优化策略

#### 1. 延迟加载
```python
class LazyInitializer:
    def __init__(self, initializer: Callable[[], Any]):
        self._initializer = initializer
        self._value = None
        self._initialized = False
        
    def __call__(self) -> Any:
        if not self._initialized:
            self._value = self._initializer()
            self._initialized = True
        return self._value
        
# 使用示例
class ExpensiveComponent:
    def __init__(self):
        # 延迟初始化昂贵的资源
        self._expensive_resource = LazyInitializer(self._create_expensive_resource)
        
    def _create_expensive_resource(self):
        # 昂贵的初始化逻辑
        return ExpensiveResource()
        
    def use_resource(self):
        resource = self._expensive_resource()
        return resource.do_something()
```

#### 2. 对象池管理
```python
class ObjectPool:
    def __init__(self, factory: Callable[[], Any], max_size: int = 10):
        self._factory = factory
        self._pool: List[Any] = []
        self._max_size = max_size
        self._in_use: Set[Any] = set()
        
    def acquire(self) -> Any:
        if self._pool:
            obj = self._pool.pop()
        else:
            obj = self._factory()
        self._in_use.add(obj)
        return obj
        
    def release(self, obj: Any):
        if obj in self._in_use:
            self._in_use.remove(obj)
            if len(self._pool) < self._max_size:
                # 重置对象状态
                if hasattr(obj, 'reset'):
                    obj.reset()
                self._pool.append(obj)
```

## 📋 实施指导总结

这个完善后的实现规划为每个类都明确了：

### 🎯 **明确的接口定义**
- 每个方法的签名和返回类型
- 抽象方法和可选方法的区分
- 配置参数的结构定义

### 📚 **清晰的文档依赖**
- 每个类实现需要参考的具体文档
- 设计决策的来源和依据
- 接口设计的理论基础

### 🔗 **详细的依赖关系管理**
- **依赖声明**: 构造函数中明确所有依赖
- **调用关系**: 具体说明如何调用依赖的方法
- **生命周期**: 明确每个依赖的生命周期管理
- **注入方式**: 提供具体的依赖注入实现代码
- **测试策略**: 说明如何mock依赖进行单元测试

### 🛡️ **严格的约束条件**
- 性能要求和基准
- 线程安全和并发控制
- 内存使用和资源管理
- 错误处理和容错机制

### 🧪 **完整的测试支持**
- Mock策略和示例代码
- 集成测试指导
- 依赖验证方法

## 🚀 解决的核心问题

### 1. **依赖关系明确体现**
**问题**: 如果后面的类需要依赖前面的类，怎么在步骤中体现？

**解决方案**:
- 每个类都有详细的**依赖关系**声明
- 明确区分必需依赖、可选依赖、配置依赖、运行时依赖
- 提供具体的**依赖注入实现**代码示例
- 说明依赖的**生命周期管理**策略

**示例**: `TrainingEngine`明确依赖`HookExecutor`和`ExecutionContext`，并展示如何在构造函数中注入这些依赖

### 2. **类间调用关系清晰**
**问题**: 如果现在步骤的类需要调用其他的类实现功能，该怎么体现？

**解决方案**:
- 详细的**调用关系说明**，具体到方法级别
- **被调用场景**描述，说明在什么情况下被其他类调用
- 提供**具体调用代码**示例
- 网络调用、异步调用等特殊调用方式的处理

**示例**: `FederatedClient`调用`LocalTrainer.train_epoch()`和`CommunicationHandler.send_message()`的具体实现

### 3. **依赖注入策略完整**
- **构造函数注入**: 所有依赖通过构造函数明确传入
- **工厂模式**: 复杂对象创建通过工厂处理
- **服务定位器**: 可选依赖的获取策略
- **配置驱动**: 通过配置文件控制依赖注入

### 4. **测试和Mock策略**
- 提供完整的**单元测试**mock策略
- **集成测试**的依赖管理方法
- **依赖验证**和循环依赖检测

## 🎨 **开发指导价值**

基于这个完善的规划，开发团队可以：

1. **精确实现**: 每个类都有明确的依赖要求和调用方式，避免理解偏差
2. **并行开发**: 依赖关系清晰，接口定义明确，可以多人并行开发
3. **质量保证**: 约束条件和测试策略明确，便于代码审查和质量控制
4. **AI辅助开发**: 接口和依赖关系定义详细，可以让AI生成准确的代码框架
5. **维护扩展**: 依赖管理规范统一，便于后期维护和功能扩展
6. **测试驱动**: 提供完整的测试策略，支持TDD开发模式

## 🔍 **实施建议**

1. **严格按阶段执行**: 遵循依赖优先原则，确保被依赖的类先实现
2. **接口先行**: 每个阶段先定义和实现抽象接口
3. **依赖注入**: 严格按照规划的依赖注入模式实施
4. **测试并行**: 每个类实现后立即编写对应的单元测试
5. **集成验证**: 每个阶段完成后进行集成测试验证
6. **文档同步**: 实现过程中及时更新接口文档

这个规划确保了系统的**一致性**、**可维护性**、**可测试性**和**可扩展性**，为大规模团队协作开发提供了可靠的指导，完全解决了类间依赖关系体现的问题。