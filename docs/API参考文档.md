# FedCL API 参考文档

## 核心API

### fedcl.train()

一行代码启动联邦学习。

```python
fedcl.train(
    dataset: str = "mnist",
    num_clients: int = 3,
    rounds: int = 10,
    **kwargs
) -> TrainingResult
```

**参数**:
- `dataset`: 数据集名称，默认 "mnist"
- `num_clients`: 客户端数量，默认 3
- `rounds`: 联邦学习轮数，默认 10
- `**kwargs`: 其他配置参数

**返回**: `TrainingResult` 对象

**示例**:
```python
result = fedcl.train(
    dataset="mnist",
    num_clients=5,
    rounds=20,
    local_epochs=3
)
```

### fedcl.train_from_config()

使用配置文件启动联邦学习。

```python
fedcl.train_from_config(
    config: Dict[str, Any]
) -> TrainingResult
```

**参数**:
- `config`: 配置字典

**返回**: `TrainingResult` 对象

**示例**:
```python
config = {
    "dataset": "mnist",
    "num_clients": 3,
    "rounds": 10,
    "learner": "default",
    "aggregator": "fedavg"
}
result = fedcl.train_from_config(config)
```

### fedcl.quick_experiment()

快速进行多配置实验。

```python
fedcl.quick_experiment(
    experiment_name: str,
    configs: List[Dict[str, Any]],
    **kwargs
) -> List[TrainingResult]
```

**参数**:
- `experiment_name`: 实验名称
- `configs`: 配置列表
- `**kwargs`: 其他参数

**返回**: `TrainingResult` 列表

**示例**:
```python
configs = [
    {"aggregator": "fedavg", "rounds": 10},
    {"aggregator": "fedprox", "rounds": 10},
]
results = fedcl.quick_experiment("comparison", configs)
```

## 装饰器API

### @fedcl.learner

注册自定义学习器。

```python
@fedcl.learner
class MyLearner(AbstractLearner):
    pass
```

**要求**:
- 继承 `AbstractLearner`
- 实现所有抽象方法

### @fedcl.aggregator

注册自定义聚合器。

```python
@fedcl.aggregator
class MyAggregator(AbstractAggregator):
    pass
```

**要求**:
- 继承 `AbstractAggregator`
- 实现 `aggregate` 方法

### @fedcl.evaluator

注册自定义评估器。

```python
@fedcl.evaluator
class MyEvaluator(AbstractEvaluator):
    pass
```

**要求**:
- 继承 `AbstractEvaluator`
- 实现 `evaluate` 方法

### @fedcl.trainer

注册自定义训练器。

```python
@fedcl.trainer
class MyTrainer(AbstractFederationTrainer):
    pass
```

**要求**:
- 继承 `AbstractFederationTrainer`
- 实现 `train` 和 `evaluate` 方法

## 组件管理API

### fedcl.list_components()

列出所有已注册的组件。

```python
fedcl.list_components() -> Dict[str, List[str]]
```

**返回**: 组件类型到组件名称列表的映射

**示例**:
```python
components = fedcl.list_components()
print(components)
# 输出: {'learner': ['default', 'my_learner'], 'aggregator': ['fedavg', 'fedprox']}
```

### fedcl.get_component_info()

获取组件详细信息。

```python
fedcl.get_component_info(
    component_name: str,
    component_type: Optional[str] = None
) -> Dict[str, Any]
```

**参数**:
- `component_name`: 组件名称
- `component_type`: 组件类型（可选）

**返回**: 组件信息字典

**示例**:
```python
info = fedcl.get_component_info("fedavg", "aggregator")
print(info)
```

### fedcl.clear_registry()

清除组件注册表。

```python
fedcl.clear_registry() -> None
```

**示例**:
```python
fedcl.clear_registry()
```

## 核心类

### AbstractLearner

学习器抽象基类。

```python
class AbstractLearner(ABC):
    def __init__(self, client_id: str, config: Dict[str, Any]):
        pass
    
    @abstractmethod
    async def train_epoch(self, **kwargs) -> Dict[str, Any]:
        """执行一个epoch的本地训练"""
        pass
    
    @abstractmethod
    async def evaluate(self, **kwargs) -> Dict[str, Any]:
        """执行本地评估"""
        pass
    
    @abstractmethod
    def get_model_weights(self) -> Dict[str, Any]:
        """获取模型权重"""
        pass
    
    @abstractmethod
    def set_model_weights(self, weights: Dict[str, Any]) -> None:
        """设置模型权重"""
        pass
```

### AbstractAggregator

聚合器抽象基类。

```python
class AbstractAggregator(ABC):
    @abstractmethod
    def aggregate(self, client_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """聚合客户端结果"""
        pass
```

### AbstractFederationTrainer

联邦训练器抽象基类。

```python
class AbstractFederationTrainer(ABC):
    @abstractmethod
    async def train(self, **kwargs) -> TrainingResult:
        """执行联邦训练"""
        pass
    
    @abstractmethod
    async def evaluate(self, **kwargs) -> EvaluationResult:
        """执行联邦评估"""
        pass
```

### TrainingResult

训练结果类。

```python
class TrainingResult:
    def __init__(self):
        self.accuracy: float
        self.loss: float
        self.rounds: int
        self.num_clients: int
        self.training_history: List[Dict[str, Any]]
        self.evaluation_history: List[Dict[str, Any]]
```

### EvaluationResult

评估结果类。

```python
class EvaluationResult:
    def __init__(self):
        self.accuracy: float
        self.loss: float
        self.metrics: Dict[str, float]
        self.num_samples: int
```

## 内置组件

### 学习器

#### DefaultLearner

默认学习器实现。

```python
from fedcl.methods.learners import DefaultLearner

learner = DefaultLearner(
    client_id: str,
    config: Dict[str, Any]
)
```

**配置选项**:
```python
config = {
    "model": {
        "instance": model  # torch.nn.Module 实例
    },
    "optimizer": {
        "type": "adam",  # "adam", "sgd"
        "learning_rate": 0.01
    },
    "local_epochs": 2
}
```

### 聚合器

#### FedAvgAggregator

FedAvg聚合器。

```python
from fedcl.methods.aggregators import FedAvgAggregator

aggregator = FedAvgAggregator(
    weighted: bool = True,
    device: str = "cpu"
)
```

#### FedProxAggregator

FedProx聚合器。

```python
from fedcl.methods.aggregators import FedProxAggregator

aggregator = FedProxAggregator(
    mu: float = 0.01,
    weighted: bool = True,
    device: str = "cpu"
)
```

### 训练器

#### StandardFederationTrainer

标准联邦训练器。

```python
from fedcl.methods.trainers import StandardFederationTrainer

trainer = StandardFederationTrainer(config)
```

**配置选项**:
```python
config = {
    "num_clients": 3,
    "local_epochs": 2,
    "learning_rate": 0.01,
    "batch_size": 32,
    "client_selection_ratio": 1.0,
    "min_clients": 1,
    "aggregator": "fedavg",
    "learner": "default"
}
```

## 配置参考

### 基础配置

```python
config = {
    # 数据集配置
    "dataset": "mnist",
    "data_path": "./data",
    "batch_size": 32,
    
    # 联邦学习配置
    "num_clients": 3,
    "rounds": 10,
    "local_epochs": 2,
    "client_selection_ratio": 1.0,
    "min_clients": 1,
    
    # 组件配置
    "learner": "default",
    "aggregator": "fedavg",
    "evaluator": "prototype",
    
    # 模型配置
    "model": {
        "type": "mlp",
        "input_dim": 784,
        "hidden_dims": [128, 64],
        "output_dim": 10
    },
    
    # 优化器配置
    "optimizer": {
        "type": "adam",
        "learning_rate": 0.01,
        "weight_decay": 1e-4
    }
}
```

### 高级配置

```python
config = {
    # 执行模式配置
    "execution": {
        "mode": "auto",  # "auto", "local", "pseudo", "distributed"
        "num_workers": 4,
        "timeout": 300,
        "device": "cpu"  # "cpu", "cuda"
    },
    
    # 通信配置
    "communication": {
        "transport": "auto",  # "auto", "memory", "process", "network"
        "host": "localhost",
        "port": 8080,
        "timeout": 30
    },
    
    # 数据分区配置
    "data_partition": {
        "type": "iid",  # "iid", "non_iid_label", "non_iid_quantity"
        "alpha": 0.5,  # 用于non_iid_label的Dirichlet分布参数
        "seed": 42
    },
    
    # 日志配置
    "logging": {
        "level": "INFO",  # "DEBUG", "INFO", "WARNING", "ERROR"
        "file": "fedcl.log",
        "format": "{time} | {level} | {message}"
    }
}
```

## 错误处理

### 常见异常

#### ComponentNotFoundError

组件未找到异常。

```python
class ComponentNotFoundError(Exception):
    pass
```

#### ConfigurationError

配置错误异常。

```python
class ConfigurationError(Exception):
    pass
```

#### TrainingError

训练错误异常。

```python
class TrainingError(Exception):
    pass
```

### 异常处理示例

```python
try:
    result = fedcl.train_from_config(config)
except ComponentNotFoundError as e:
    print(f"组件未找到: {e}")
except ConfigurationError as e:
    print(f"配置错误: {e}")
except TrainingError as e:
    print(f"训练错误: {e}")
```

## 版本信息

```python
import fedcl

print(fedcl.__version__)  # "0.2.0"
print(fedcl.__author__)   # "FedCL Development Team"
```

## 日志配置

```python
import logging

# 设置日志级别
logging.basicConfig(level=logging.DEBUG)

# 或者使用loguru
from loguru import logger
logger.add("fedcl.log", level="DEBUG")
```

## 性能优化

### 内存优化

```python
config = {
    "execution": {
        "device": "cuda",  # 使用GPU
        "num_workers": 4   # 多进程
    },
    "batch_size": 64,      # 增大批处理大小
    "local_epochs": 1      # 减少本地训练轮数
}
```

### 通信优化

```python
config = {
    "communication": {
        "transport": "memory",  # 使用内存传输
        "compression": True     # 启用压缩
    }
}
```

## 扩展开发

### 自定义组件示例

```python
from fedcl.api import learner
from fedcl.execution.base_learner import AbstractLearner

@learner
class CustomLearner(AbstractLearner):
    def __init__(self, client_id: str, config: Dict[str, Any]):
        super().__init__(client_id, config)
        # 初始化代码
    
    async def train_epoch(self, **kwargs):
        # 训练逻辑
        return {"model_weights": weights, "loss": loss}
    
    async def evaluate(self, **kwargs):
        # 评估逻辑
        return {"accuracy": acc, "loss": loss}
    
    def get_model_weights(self):
        return self.model.state_dict()
    
    def set_model_weights(self, weights):
        self.model.load_state_dict(weights)
```

### 组件注册检查

```python
# 检查组件是否已注册
components = fedcl.list_components()
if "my_learner" in components.get("learner", []):
    print("组件已注册")
else:
    print("组件未注册")
```
