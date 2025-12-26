# 插件开发

开发可复用的 OiaFed 插件包。

---

## 概述

插件允许将自定义组件打包为独立的 Python 包，便于分发和复用。

```
my-oiafed-plugin/
├── my_plugin/
│   ├── __init__.py
│   ├── aggregators.py
│   └── learners.py
├── pyproject.toml
└── README.md
```

---

## 快速开始

### 1. 创建项目结构

```bash
mkdir my-oiafed-plugin
cd my-oiafed-plugin
mkdir my_plugin
touch my_plugin/__init__.py
```

### 2. 实现组件

```python
# my_plugin/aggregators.py
from oiafed import Aggregator, register

@register("aggregator.myplugin.robust_avg")
class RobustAvgAggregator(Aggregator):
    """鲁棒平均聚合器，剔除异常值"""
    
    def __init__(self, trim_ratio: float = 0.1):
        self.trim_ratio = trim_ratio
    
    def aggregate(self, updates, global_model=None):
        # 实现逻辑
        ...
```

```python
# my_plugin/learners.py
from oiafed import Learner, register

@register("learner.myplugin.custom")
class CustomLearner(Learner):
    """自定义学习器"""
    
    async def train_step(self, batch, batch_idx):
        # 实现逻辑
        ...
```

### 3. 导出组件

```python
# my_plugin/__init__.py
"""OiaFed 插件：我的自定义算法"""

__version__ = "0.1.0"

# 导入即注册
from .aggregators import RobustAvgAggregator
from .learners import CustomLearner

__all__ = [
    "RobustAvgAggregator",
    "CustomLearner",
]
```

### 4. 配置打包

```toml
# pyproject.toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "oiafed-myplugin"
version = "0.1.0"
description = "My custom OiaFed algorithms"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "oiafed>=0.1.0",
]

[project.entry-points."oiafed.plugins"]
myplugin = "my_plugin:register_all"

[project.urls]
Homepage = "https://github.com/user/oiafed-myplugin"
```

### 5. 安装和使用

```bash
# 开发安装
pip install -e .

# 使用
python -c "import my_plugin; print('Loaded!')"
```

```yaml
# 在配置中使用
aggregator:
  type: myplugin.robust_avg
  args:
    trim_ratio: 0.2
```

---

## 入口点注册

### 自动加载

通过 entry_points 实现自动加载：

```toml
[project.entry-points."oiafed.plugins"]
myplugin = "my_plugin:register_all"
```

```python
# my_plugin/__init__.py
def register_all():
    """注册所有组件（被 OiaFed 自动调用）"""
    from . import aggregators
    from . import learners
```

### 手动加载

```python
import my_plugin  # 导入即触发注册
from oiafed import FederationRunner

runner = FederationRunner("config.yaml")
```

---

## 组件开发

### Aggregator 模板

```python
from typing import List, Dict, Any, Optional
import torch
from oiafed import Aggregator, ClientUpdate, register

@register("aggregator.myplugin.example")
class ExampleAggregator(Aggregator):
    """
    示例聚合器
    
    Args:
        param1: 参数1说明
        param2: 参数2说明
    
    Example:
        aggregator:
          type: myplugin.example
          args:
            param1: 0.5
    """
    
    def __init__(self, param1: float = 0.5, param2: int = 10):
        self.param1 = param1
        self.param2 = param2
    
    def aggregate(
        self,
        updates: List[ClientUpdate],
        global_model: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        if not updates:
            raise ValueError("No updates to aggregate")
        
        # 聚合逻辑
        keys = updates[0].weights.keys()
        total_samples = sum(u.num_samples for u in updates)
        
        aggregated = {}
        for key in keys:
            weighted_sum = sum(
                u.weights[key] * u.num_samples
                for u in updates
            )
            aggregated[key] = weighted_sum / total_samples
        
        return aggregated
```

### Learner 模板

```python
from typing import Dict, Any, Optional
import torch
from oiafed import Learner, StepMetrics, register

@register("learner.myplugin.example")
class ExampleLearner(Learner):
    """
    示例学习器
    """
    
    def __init__(
        self,
        model,
        data=None,
        custom_param: float = 1.0,
        **kwargs,
    ):
        super().__init__(model, data, **kwargs)
        self.custom_param = custom_param
    
    async def setup(self, config: Dict[str, Any]) -> None:
        self.device = config.get("device", "cpu")
        self.model.to(self.device)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=config.get("lr", 0.01),
        )
        self.criterion = torch.nn.CrossEntropyLoss()
    
    async def train_step(
        self,
        batch: Any,
        batch_idx: int,
    ) -> StepMetrics:
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        
        self.optimizer.zero_grad()
        output = self.model(x)
        loss = self.criterion(output, y)
        loss.backward()
        self.optimizer.step()
        
        return StepMetrics(
            loss=loss.item(),
            num_samples=len(y),
            metrics={"accuracy": (output.argmax(1) == y).float().mean().item()},
        )
```

---

## 测试插件

### 测试结构

```
my-oiafed-plugin/
├── my_plugin/
├── tests/
│   ├── __init__.py
│   ├── test_aggregators.py
│   └── test_learners.py
└── pyproject.toml
```

### 测试示例

```python
# tests/test_aggregators.py
import pytest
import torch
from oiafed import ClientUpdate
from my_plugin import RobustAvgAggregator

class TestRobustAvgAggregator:
    
    def test_aggregate(self):
        agg = RobustAvgAggregator(trim_ratio=0.1)
        updates = [
            ClientUpdate("c0", {"w": torch.tensor([1.0])}, 100),
            ClientUpdate("c1", {"w": torch.tensor([2.0])}, 100),
        ]
        
        result = agg.aggregate(updates)
        assert "w" in result
```

### 运行测试

```bash
pytest tests/
```

---

## 发布到 PyPI

### 1. 构建

```bash
pip install build
python -m build
```

### 2. 上传

```bash
pip install twine
twine upload dist/*
```

### 3. 使用

```bash
pip install oiafed-myplugin
```

---

## 最佳实践

### 1. 命名规范

```python
# 包名：oiafed-{name}
# 注册名：{name}.{component}

@register("aggregator.myplugin.robust")  # ✓
@register("aggregator.robust")            # ✗ 可能冲突
```

### 2. 版本兼容

```toml
dependencies = [
    "oiafed>=0.1.0,<1.0.0",
]
```

### 3. 文档

```python
@register("aggregator.myplugin.example")
class ExampleAggregator(Aggregator):
    """
    一句话描述
    
    详细说明...
    
    Args:
        param: 参数说明
    
    Example:
        配置示例:
        
        ```yaml
        aggregator:
          type: myplugin.example
          args:
            param: 0.5
        ```
    
    References:
        论文链接（如有）
    """
```

### 4. 类型提示

```python
from typing import List, Dict, Optional
import torch

def aggregate(
    self,
    updates: List[ClientUpdate],
    global_model: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    ...
```

---

## 示例插件

### 差分隐私插件

```python
# dp_plugin/aggregators.py
from oiafed import Aggregator, register
import torch

@register("aggregator.dp.gaussian")
class GaussianMechanismAggregator(Aggregator):
    """高斯机制差分隐私聚合器"""
    
    def __init__(self, noise_multiplier: float = 1.0, clip_norm: float = 1.0):
        self.noise_multiplier = noise_multiplier
        self.clip_norm = clip_norm
    
    def aggregate(self, updates, global_model=None):
        # 裁剪
        clipped = self._clip_updates(updates)
        # 聚合
        avg = self._average(clipped)
        # 加噪
        noised = self._add_noise(avg)
        return noised
```

### 异步联邦插件

```python
# async_fl/trainers.py
from oiafed import Trainer, register

@register("trainer.async.buffered")
class BufferedAsyncTrainer(Trainer):
    """缓冲异步训练器"""
    
    def __init__(self, buffer_size: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.buffer_size = buffer_size
        self.buffer = []
    
    async def train_round(self, round_num):
        # 异步收集，不等待所有客户端
        ...
```

---

## 下一步

- [自定义算法](../01-guides/custom-algorithm.md)
- [注册系统](../03-architecture/registry-system.md)
- [测试指南](testing.md)
