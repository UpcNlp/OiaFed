# 基础设施 API

Tracker、Callback、Config 等基础设施组件的 API 参考。

---

## Tracker

实验追踪。

### 接口定义

```python
class Tracker(ABC):
    @abstractmethod
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """记录指标"""
    
    @abstractmethod
    def log_params(self, params: Dict[str, Any]) -> None:
        """记录参数"""
    
    @abstractmethod
    def log_artifact(self, path: str, name: Optional[str] = None) -> None:
        """记录文件"""
```

### MLflowTracker

```python
@register("tracker.mlflow")
class MLflowTracker(Tracker):
    def __init__(
        self,
        tracking_uri: str = "./mlruns",
        experiment_name: str = "default",
        run_name: Optional[str] = None,
    )
```

**配置**

```yaml
tracker:
  backends:
    - type: mlflow
      tracking_uri: ./mlruns
      experiment_name: my_experiment
```

**使用**

```bash
# 启动 UI
mlflow ui --backend-store-uri ./mlruns
```

### LoguruTracker

```python
@register("tracker.loguru")
class LoguruTracker(Tracker):
    def __init__(
        self,
        level: str = "INFO",
        log_dir: str = "./logs",
        rotation: str = "100 MB",
    )
```

**配置**

```yaml
tracker:
  backends:
    - type: loguru
      level: INFO
      log_dir: ./logs
```

### CompositeTracker

组合多个 Tracker。

```python
tracker = CompositeTracker([
    MLflowTracker(...),
    LoguruTracker(...),
])
```

---

## Callback

生命周期回调。

### 接口定义

```python
class Callback(ABC):
    async def on_train_start(self, trainer, **kwargs) -> None: ...
    async def on_train_end(self, trainer, result, **kwargs) -> None: ...
    async def on_round_start(self, trainer, round_num, **kwargs) -> None: ...
    async def on_round_end(self, trainer, round_num, result, **kwargs) -> None: ...
    async def on_epoch_start(self, learner, epoch, **kwargs) -> None: ...
    async def on_epoch_end(self, learner, epoch, metrics, **kwargs) -> None: ...
```

### EarlyStoppingCallback

```python
@register("callback.early_stopping")
class EarlyStoppingCallback(Callback):
    def __init__(
        self,
        patience: int = 10,
        metric: str = "accuracy",
        mode: str = "max",          # "max" | "min"
        min_delta: float = 0.0,
    )
```

**配置**

```yaml
callbacks:
  - type: early_stopping
    args:
      patience: 10
      metric: accuracy
      mode: max
```

### ModelCheckpointCallback

```python
@register("callback.model_checkpoint")
class ModelCheckpointCallback(Callback):
    def __init__(
        self,
        save_dir: str = "./checkpoints",
        save_best: bool = True,
        save_every: int = 10,
        metric: str = "accuracy",
        mode: str = "max",
    )
```

### LRSchedulerCallback

```python
@register("callback.lr_scheduler")
class LRSchedulerCallback(Callback):
    def __init__(
        self,
        scheduler_type: str = "step",  # "step" | "cosine" | "plateau"
        step_size: int = 10,
        gamma: float = 0.1,
    )
```

### 自定义 Callback

```python
from oiafed import Callback, register

@register("callback.my_callback")
class MyCallback(Callback):
    async def on_round_end(self, trainer, round_num, result, **kwargs):
        if round_num % 10 == 0:
            print(f"Round {round_num}: {result.metrics}")
```

---

## Config

配置管理。

### 加载配置

```python
from oiafed import load_config

# 加载单个文件
config = load_config("experiment.yaml")

# 加载目录（合并所有 yaml）
config = load_config("configs/")
```

### 配置继承

```python
from oiafed.config import merge_configs

base = load_config("base.yaml")
override = load_config("override.yaml")
merged = merge_configs(base, override)
```

### 环境变量

```python
from oiafed.config import resolve_env_vars

config = {"path": "${DATA_DIR}/data"}
resolved = resolve_env_vars(config)
# {"path": "/home/user/data"}
```

### 配置验证

```python
from oiafed.config import validate_config

try:
    validate_config(config)
except ConfigError as e:
    print(f"配置错误: {e}")
```

---

## Registry

组件注册。

### 注册

```python
from oiafed import register

@register("aggregator.my_agg")
class MyAggregator(Aggregator): ...

# 或手动注册
from oiafed import registry
registry.register("aggregator.my_agg")(MyAggregator)
```

### 获取

```python
from oiafed import get_component, create_component

# 获取类
cls = get_component("aggregator.fedavg")

# 创建实例
agg = create_component("aggregator.fedavg", weighted=True)
```

### 列出

```python
from oiafed import list_components

# 所有组件
all_components = list_components()

# 按类型过滤
aggregators = list_components("aggregator")
```

### 检查存在

```python
from oiafed import registry

if registry.exists("aggregator.my_agg"):
    print("已注册")
```

---

## Checkpoint

检查点管理。

```python
from oiafed.infra import CheckpointManager

manager = CheckpointManager(save_dir="./checkpoints")

# 保存
manager.save(
    round_num=50,
    weights=model_weights,
    metrics={"accuracy": 0.95},
)

# 加载最新
checkpoint = manager.load_latest()

# 加载指定轮次
checkpoint = manager.load(round_num=50)

# 加载最佳
checkpoint = manager.load_best(metric="accuracy", mode="max")
```

---

## Timer

计时器。

```python
from oiafed.infra import Timer

timer = Timer()

timer.start("training")
# ... 训练代码 ...
timer.stop("training")

print(timer.get("training"))  # 耗时（秒）
print(timer.summary())         # 所有计时
```

---

## 配置参考

```yaml
# Tracker
tracker:
  backends:
    - type: mlflow
      tracking_uri: ./mlruns
      experiment_name: ${exp_name}
    - type: loguru
      level: INFO
      log_dir: ./logs

# Callbacks
callbacks:
  - type: early_stopping
    args:
      patience: 10
      metric: accuracy
  - type: model_checkpoint
    args:
      save_dir: ./checkpoints
      save_best: true

# Checkpoint
checkpoint:
  enabled: true
  save_dir: ./checkpoints
  save_every: 10
  keep_last: 5
```

---

## 下一步

- [核心 API](core-api.md)
- [Callback 机制](../03-architecture/callback-system.md)
