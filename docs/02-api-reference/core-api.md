# 核心 API

Trainer、Learner、Aggregator 等核心组件的 API 参考。

---

## Trainer

训练器，服务端角色。

### 类定义

```python
class Trainer:
    def __init__(
        self,
        model: nn.Module,
        aggregator: Aggregator,
        config: Dict[str, Any],
        tracker: Optional[Tracker] = None,
        callbacks: Optional[List[Callback]] = None,
    )
```

### 核心方法

#### `run()`

```python
async def run(self) -> Dict[str, Any]:
    """
    主训练循环
    
    Returns:
        包含训练结果的字典
    """
```

#### `train_round()`

```python
async def train_round(self, round_num: int) -> RoundResult:
    """
    执行单轮训练
    
    Args:
        round_num: 当前轮次
    
    Returns:
        RoundResult: 本轮结果
    """
```

#### `select_clients()`

```python
def select_clients(self) -> List[str]:
    """
    选择参与训练的客户端
    
    Returns:
        客户端 ID 列表
    """
```

#### `broadcast_weights()`

```python
async def broadcast_weights(self, weights: Dict[str, Tensor]) -> None:
    """
    广播全局模型权重到所有客户端
    """
```

#### `collect_updates()`

```python
async def collect_updates(
    self,
    client_ids: List[str],
    config: Dict[str, Any],
) -> List[ClientUpdate]:
    """
    收集客户端更新
    
    Returns:
        ClientUpdate 列表
    """
```

### 生命周期钩子

```python
async def on_train_start(self): ...
async def on_train_end(self, result): ...
async def on_round_start(self, round_num): ...
async def on_round_end(self, round_num, result): ...
```

---

## Learner

学习器，客户端角色。

### 类定义

```python
class Learner:
    def __init__(
        self,
        model: nn.Module,
        data: Optional[Dataset] = None,
        tracker: Optional[Tracker] = None,
        callbacks: Optional[List[Callback]] = None,
        config: Optional[Dict[str, Any]] = None,
        node_id: Optional[str] = None,
    )
```

### 核心方法

#### `fit()`

```python
async def fit(
    self,
    weights: Dict[str, Tensor],
    config: Dict[str, Any],
) -> TrainResult:
    """
    执行本地训练
    
    Args:
        weights: 全局模型权重
        config: 训练配置
    
    Returns:
        TrainResult: 训练结果
    """
```

#### `train_step()`

```python
async def train_step(
    self,
    batch: Any,
    batch_idx: int,
) -> StepMetrics:
    """
    单步训练（必须实现）
    
    Args:
        batch: 数据批次
        batch_idx: 批次索引
    
    Returns:
        StepMetrics: 步骤指标
    """
```

#### `evaluate()`

```python
async def evaluate(
    self,
    config: Optional[Dict[str, Any]] = None,
) -> EvalResult:
    """
    评估模型
    
    Returns:
        EvalResult: 评估结果
    """
```

#### `get_weights()` / `set_weights()`

```python
def get_weights(self) -> Dict[str, Tensor]:
    """获取模型权重"""

def set_weights(self, weights: Dict[str, Tensor]) -> None:
    """设置模型权重"""
```

### 生命周期钩子

```python
async def setup(self, config): ...
async def teardown(self): ...
async def on_epoch_start(self, epoch): ...
async def on_epoch_end(self, epoch, metrics): ...
```

---

## Aggregator

聚合器，服务端组件。

### 类定义

```python
class Aggregator(ABC):
    @abstractmethod
    def aggregate(
        self,
        updates: List[ClientUpdate],
        global_model: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """聚合客户端更新"""
```

### 钩子方法

```python
def pre_aggregate(self, updates: List[ClientUpdate]) -> List[ClientUpdate]:
    """聚合前处理"""
    return updates

def post_aggregate(
    self,
    weights: Dict[str, Tensor],
    updates: List[ClientUpdate],
) -> Dict[str, Tensor]:
    """聚合后处理"""
    return weights
```

---

## 类型定义

### ClientUpdate

```python
@dataclass
class ClientUpdate:
    client_id: str
    weights: Dict[str, Tensor]
    num_samples: int
    metrics: Optional[Dict[str, float]] = None
    extra: Optional[Dict[str, Any]] = None
```

### TrainResult

```python
@dataclass
class TrainResult:
    weights: Dict[str, Tensor]
    num_samples: int
    metrics: TrainMetrics
    extra: Optional[Dict[str, Any]] = None
```

### EvalResult

```python
@dataclass
class EvalResult:
    num_samples: int
    metrics: Dict[str, float]
```

### StepMetrics

```python
@dataclass
class StepMetrics:
    loss: float
    num_samples: int
    metrics: Optional[Dict[str, float]] = None
```

### RoundResult

```python
@dataclass
class RoundResult:
    round_num: int
    num_clients: int
    metrics: Dict[str, float]
    eval_result: Optional[EvalResult] = None
```

---

## FederatedSystem

系统容器，管理所有组件。

```python
class FederatedSystem:
    def __init__(self, config: Dict[str, Any])
    
    async def initialize(self) -> None:
        """初始化系统"""
    
    async def run(self) -> Dict[str, Any]:
        """运行训练"""
    
    async def stop(self) -> None:
        """停止系统"""
```

---

## FederationRunner

便捷运行器。

```python
class FederationRunner:
    def __init__(
        self,
        config_path: str,
        mode: str = "serial",
        num_clients: int = 5,
    )
    
    def run_sync(self) -> Dict[str, Any]:
        """同步运行"""
    
    async def run(self) -> Dict[str, Any]:
        """异步运行"""
```

---

## 工具函数

### 模型权重操作

```python
from oiafed.utils import (
    get_model_weights,
    set_model_weights,
    average_weights,
    subtract_weights,
)

# 获取权重
weights = get_model_weights(model)

# 设置权重
set_model_weights(model, weights)

# 平均权重
avg = average_weights([w1, w2, w3], [100, 200, 150])

# 权重差
delta = subtract_weights(new_weights, old_weights)
```

### Registry 操作

```python
from oiafed import (
    register,
    get_component,
    create_component,
    list_components,
)

# 注册
@register("aggregator.my_agg")
class MyAgg(Aggregator): ...

# 获取类
cls = get_component("aggregator.my_agg")

# 创建实例
agg = create_component("aggregator.my_agg", param=1.0)

# 列出组件
all_aggs = list_components("aggregator")
```

### 配置加载

```python
from oiafed import load_config

config = load_config("experiment.yaml")
```

---

## 下一步

- [通信 API](comm-api.md)
- [算法 API](methods-api.md)
- [架构总览](../03-architecture/overview.md)
