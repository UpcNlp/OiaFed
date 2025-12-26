# 自定义算法

开发自己的联邦学习算法。

---

## 概述

OiaFed 的扩展机制：

1. **继承基类** - 所有组件继承抽象基类
2. **装饰器注册** - 使用 `@register` 注册
3. **配置引用** - YAML 中通过类型名引用

```python
from oiafed import Aggregator, register

@register("aggregator.my_algo")
class MyAggregator(Aggregator):
    def aggregate(self, updates, global_model=None):
        pass
```

```yaml
aggregator:
  type: my_algo
```

---

## 自定义 Aggregator

### 基本结构

```python
from typing import List, Any, Optional
from oiafed import Aggregator, register, ClientUpdate

@register("aggregator.weighted_median")
class WeightedMedianAggregator(Aggregator):
    """加权中位数聚合器"""
    
    def __init__(self, trim_ratio: float = 0.1):
        self.trim_ratio = trim_ratio
    
    def aggregate(
        self,
        updates: List[ClientUpdate],
        global_model: Optional[Any] = None,
    ) -> Any:
        """聚合客户端更新"""
        import torch
        
        keys = updates[0].weights.keys()
        aggregated = {}
        
        for key in keys:
            weights = [u.weights[key] for u in updates]
            counts = [u.num_samples for u in updates]
            aggregated[key] = self._weighted_median(weights, counts)
        
        return aggregated
    
    def _weighted_median(self, weights, counts):
        import torch
        total = sum(counts)
        result = torch.zeros_like(weights[0])
        for w, c in zip(weights, counts):
            result += w * (c / total)
        return result
```

### 使用配置

```yaml
aggregator:
  type: weighted_median
  args:
    trim_ratio: 0.2
```

---

## 自定义 Learner

### 基本结构

```python
from oiafed import Learner, register, StepMetrics

@register("learner.my_learner")
class MyLearner(Learner):
    
    def __init__(self, model, data=None, custom_param: float = 1.0, **kwargs):
        super().__init__(model, data, **kwargs)
        self.custom_param = custom_param
    
    async def setup(self, config):
        """初始化"""
        import torch
        self.device = config.get("device", "cuda")
        self.model.to(self.device)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=config.get("lr", 0.01)
        )
        self.criterion = torch.nn.CrossEntropyLoss()
    
    async def train_step(self, batch, batch_idx) -> StepMetrics:
        """单步训练"""
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        
        self.optimizer.zero_grad()
        output = self.model(x)
        loss = self.criterion(output, y)
        
        # 自定义正则化
        loss += self.custom_param * self._regularization()
        
        loss.backward()
        self.optimizer.step()
        
        pred = output.argmax(dim=1)
        acc = (pred == y).float().mean().item()
        
        return StepMetrics(
            loss=loss.item(),
            num_samples=len(y),
            metrics={"accuracy": acc}
        )
    
    def _regularization(self):
        import torch
        reg = 0.0
        for p in self.model.parameters():
            reg += torch.norm(p)
        return reg
```

### 继承现有 Learner

```python
from oiafed.methods.learners import DefaultLearner

@register("learner.extended_default")
class ExtendedLearner(DefaultLearner):
    
    def __init__(self, *args, extra_param: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.extra_param = extra_param
    
    async def train_step(self, batch, batch_idx):
        metrics = await super().train_step(batch, batch_idx)
        # 添加额外逻辑
        return metrics
```

---

## 自定义 Trainer

```python
from oiafed import Trainer, register, RoundResult

@register("trainer.async_trainer")
class AsyncTrainer(Trainer):
    """异步训练器"""
    
    async def train_round(self, round_num: int) -> RoundResult:
        import asyncio
        
        clients = self.select_clients()
        config = self._get_fit_config(round_num)
        
        # 异步收集（允许部分失败）
        tasks = [
            self._get_update(cid, config) 
            for cid in clients
        ]
        
        done, pending = await asyncio.wait(
            tasks,
            timeout=self.config.get("round_timeout", 60),
            return_when=asyncio.FIRST_COMPLETED
        )
        
        for task in pending:
            task.cancel()
        
        updates = [t.result() for t in done if not t.exception()]
        
        if updates:
            weights = self.aggregator.aggregate(updates)
            self.set_global_weights(weights)
            await self.broadcast_weights(weights)
        
        return RoundResult(
            round_num=round_num,
            num_clients=len(updates),
            metrics=self._aggregate_metrics(updates)
        )
```

---

## 项目组织

```
my_project/
├── my_algorithms/
│   ├── __init__.py
│   ├── aggregators.py
│   └── learners.py
├── configs/
│   └── experiment.yaml
└── run.py
```

### `__init__.py`

```python
# 导入即注册
from .aggregators import WeightedMedianAggregator
from .learners import MyLearner
```

### `run.py`

```python
import my_algorithms  # 触发注册
from oiafed import FederationRunner

runner = FederationRunner("configs/experiment.yaml")
runner.run_sync()
```

---

## 测试

```python
# tests/test_my_aggregator.py
import pytest
from oiafed import ClientUpdate
from my_algorithms import WeightedMedianAggregator

def test_aggregate():
    updates = [
        ClientUpdate(
            client_id="c0",
            weights={"layer": torch.tensor([1.0, 2.0])},
            num_samples=100
        ),
        ClientUpdate(
            client_id="c1",
            weights={"layer": torch.tensor([3.0, 4.0])},
            num_samples=200
        ),
    ]
    
    agg = WeightedMedianAggregator()
    result = agg.aggregate(updates)
    
    assert "layer" in result
```

---

## 发布为插件

### `setup.py`

```python
from setuptools import setup

setup(
    name="oiafed-my-plugin",
    version="0.1.0",
    packages=["my_algorithms"],
    install_requires=["oiafed>=0.1.0"],
    entry_points={
        "oiafed.plugins": [
            "my_plugin = my_algorithms:register_all",
        ],
    },
)
```

```bash
pip install -e .
```

---

## 下一步

- [注册系统](../03-architecture/registry-system.md)
- [核心 API](../02-api-reference/core-api.md)
- [论文复现](../05-papers/reproduction-guide.md)
