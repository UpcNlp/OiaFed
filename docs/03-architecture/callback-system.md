# Callback 机制

OiaFed 的生命周期回调系统。

---

## 概述

Callback 允许在训练的关键节点注入自定义逻辑，无需修改核心代码。

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Loop                             │
├─────────────────────────────────────────────────────────────┤
│  on_train_start()                                           │
│      │                                                      │
│      ▼                                                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Round Loop                                          │   │
│  │  on_round_start()                                    │   │
│  │      │                                               │   │
│  │      ▼                                               │   │
│  │  ┌─────────────────────────────────────────────┐    │   │
│  │  │  Local Training (Learner)                    │    │   │
│  │  │  on_epoch_start()                            │    │   │
│  │  │      │                                       │    │   │
│  │  │      ▼                                       │    │   │
│  │  │  train_step() × batches                      │    │   │
│  │  │      │                                       │    │   │
│  │  │      ▼                                       │    │   │
│  │  │  on_epoch_end()                              │    │   │
│  │  └─────────────────────────────────────────────┘    │   │
│  │      │                                               │   │
│  │      ▼                                               │   │
│  │  on_round_end()                                      │   │
│  └─────────────────────────────────────────────────────┘   │
│      │                                                      │
│      ▼                                                      │
│  on_train_end()                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Callback 接口

```python
class Callback(ABC):
    """回调基类"""
    
    # 训练级别
    async def on_train_start(self, trainer, **kwargs) -> None:
        """训练开始"""
        pass
    
    async def on_train_end(self, trainer, result, **kwargs) -> None:
        """训练结束"""
        pass
    
    # 轮次级别
    async def on_round_start(self, trainer, round_num, **kwargs) -> None:
        """每轮开始"""
        pass
    
    async def on_round_end(self, trainer, round_num, result, **kwargs) -> None:
        """每轮结束"""
        pass
    
    # Epoch 级别（Learner）
    async def on_epoch_start(self, learner, epoch, **kwargs) -> None:
        """每个 epoch 开始"""
        pass
    
    async def on_epoch_end(self, learner, epoch, metrics, **kwargs) -> None:
        """每个 epoch 结束"""
        pass
    
    # 聚合级别
    async def on_aggregate_start(self, trainer, updates, **kwargs) -> None:
        """聚合开始"""
        pass
    
    async def on_aggregate_end(self, trainer, weights, **kwargs) -> None:
        """聚合结束"""
        pass
```

---

## 内置 Callback

### EarlyStopping

```python
@register("callback.early_stopping")
class EarlyStoppingCallback(Callback):
    """
    早停回调
    
    监控指标，无改善时停止训练
    """
    
    def __init__(
        self,
        patience: int = 10,
        metric: str = "accuracy",
        mode: str = "max",
        min_delta: float = 0.0,
    ):
        self.patience = patience
        self.metric = metric
        self.mode = mode
        self.min_delta = min_delta
        self.best = None
        self.counter = 0
    
    async def on_round_end(self, trainer, round_num, result, **kwargs):
        current = result.metrics.get(self.metric)
        
        if self._is_improvement(current):
            self.best = current
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            trainer.should_stop = True
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

### ModelCheckpoint

```python
@register("callback.model_checkpoint")
class ModelCheckpointCallback(Callback):
    """
    模型检查点
    
    定期保存模型，支持最佳模型保存
    """
    
    def __init__(
        self,
        save_dir: str = "./checkpoints",
        save_best: bool = True,
        save_every: int = 10,
        metric: str = "accuracy",
        mode: str = "max",
    ):
        ...
    
    async def on_round_end(self, trainer, round_num, result, **kwargs):
        # 定期保存
        if round_num % self.save_every == 0:
            self._save(trainer, round_num, "periodic")
        
        # 保存最佳
        if self.save_best and self._is_best(result):
            self._save(trainer, round_num, "best")
```

### LRScheduler

```python
@register("callback.lr_scheduler")
class LRSchedulerCallback(Callback):
    """
    学习率调度
    """
    
    def __init__(
        self,
        scheduler_type: str = "step",
        step_size: int = 10,
        gamma: float = 0.1,
    ):
        ...
    
    async def on_round_end(self, trainer, round_num, result, **kwargs):
        self.scheduler.step()
```

### TensorBoard

```python
@register("callback.tensorboard")
class TensorBoardCallback(Callback):
    """
    TensorBoard 日志
    """
    
    def __init__(self, log_dir: str = "./runs"):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir)
    
    async def on_round_end(self, trainer, round_num, result, **kwargs):
        for key, value in result.metrics.items():
            self.writer.add_scalar(key, value, round_num)
```

---

## 自定义 Callback

### 简单示例

```python
from oiafed import Callback, register

@register("callback.print_progress")
class PrintProgressCallback(Callback):
    async def on_round_end(self, trainer, round_num, result, **kwargs):
        acc = result.metrics.get("accuracy", 0)
        print(f"Round {round_num}: accuracy = {acc:.4f}")
```

### 复杂示例

```python
@register("callback.adaptive_client_selection")
class AdaptiveClientSelection(Callback):
    """
    自适应客户端选择
    
    根据历史性能动态调整客户端选择概率
    """
    
    def __init__(self, exploration_rate: float = 0.1):
        self.exploration_rate = exploration_rate
        self.client_scores = {}
    
    async def on_round_end(self, trainer, round_num, result, **kwargs):
        # 更新客户端得分
        for update in result.updates:
            client_id = update.client_id
            score = update.metrics.get("accuracy", 0)
            self._update_score(client_id, score)
    
    async def on_round_start(self, trainer, round_num, **kwargs):
        # 修改选择策略
        if random.random() > self.exploration_rate:
            # 利用：选择高分客户端
            trainer.client_selection_weights = self._get_weights()
```

---

## CallbackManager

管理多个 Callback。

```python
class CallbackManager:
    def __init__(self, callbacks: List[Callback]):
        self.callbacks = callbacks
    
    async def on_round_end(self, trainer, round_num, result):
        for callback in self.callbacks:
            await callback.on_round_end(trainer, round_num, result)
```

### 执行顺序

Callback 按配置顺序执行：

```yaml
callbacks:
  - type: early_stopping    # 先执行
  - type: model_checkpoint  # 后执行
```

### 中断训练

```python
class MyCallback(Callback):
    async def on_round_end(self, trainer, round_num, result, **kwargs):
        if some_condition:
            trainer.should_stop = True  # 设置停止标志
```

---

## 配置参考

```yaml
callbacks:
  - type: early_stopping
    args:
      patience: 10
      metric: accuracy
      mode: max
  
  - type: model_checkpoint
    args:
      save_dir: ./checkpoints
      save_best: true
      save_every: 10
  
  - type: lr_scheduler
    args:
      scheduler_type: cosine
      T_max: 100
  
  - type: tensorboard
    args:
      log_dir: ./runs
```

---

## 最佳实践

### 1. 保持无状态

如果需要状态，确保可序列化（用于分布式）。

### 2. 异常处理

```python
async def on_round_end(self, trainer, round_num, result, **kwargs):
    try:
        # 可能失败的操作
        self._save_to_external_service(result)
    except Exception as e:
        logger.warning(f"Callback failed: {e}")
        # 不要让 Callback 失败影响训练
```

### 3. 性能考虑

避免在 Callback 中进行耗时操作，或使用异步。

---

## 下一步

- [注册系统](registry-system.md)
- [基础设施 API](../02-api-reference/infra-api.md)
