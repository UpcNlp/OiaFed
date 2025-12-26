# 注册系统

OiaFed 的组件注册和发现机制。

---

## 概述

Registry 实现了**配置驱动**的核心机制：

```python
# 注册
@register("aggregator.my_algo")
class MyAggregator(Aggregator): ...

# 配置引用
aggregator:
  type: my_algo

# 框架自动解析
agg = registry.get("aggregator.my_algo")(**args)
```

---

## 工作原理

```
┌─────────────┐     导入模块     ┌─────────────┐
│   @register │ ───────────────► │   Registry  │
│   装饰器    │                  │   全局注册表 │
└─────────────┘                  └──────┬──────┘
                                        │
                                        ▼
┌─────────────┐     配置解析     ┌─────────────┐
│    YAML     │ ───────────────► │  get/create │
│   配置文件   │                  │   获取组件   │
└─────────────┘                  └─────────────┘
```

---

## 注册方式

### 装饰器注册

```python
from oiafed import register

@register("aggregator.my_algo")
class MyAggregator(Aggregator):
    pass
```

### 手动注册

```python
from oiafed import registry

class MyAggregator(Aggregator):
    pass

registry.register("aggregator.my_algo")(MyAggregator)
```

### 自动注册

模块导入时自动触发装饰器：

```python
# src/methods/aggregators/__init__.py
from .fedavg import FedAvgAggregator    # 导入即注册
from .fedprox import FedProxAggregator
```

---

## 命名空间

### 命名规范

```
<组件类型>.<实现名称>

例如:
- aggregator.fedavg
- learner.moon
- model.resnet18
- dataset.cifar10
```

### 内置命名空间

| 前缀 | 组件类型 | 示例 |
|------|----------|------|
| `aggregator.` | 聚合器 | `aggregator.fedavg` |
| `learner.` | 学习器 | `learner.moon` |
| `trainer.` | 训练器 | `trainer.default` |
| `model.` | 模型 | `model.resnet18` |
| `dataset.` | 数据集 | `dataset.cifar10` |
| `partitioner.` | 划分器 | `partitioner.dirichlet` |
| `callback.` | 回调 | `callback.early_stopping` |
| `tracker.` | 追踪器 | `tracker.mlflow` |

### 配置中的简写

```yaml
# 完整写法
aggregator:
  type: aggregator.fedavg

# 简写（推荐）
aggregator:
  type: fedavg  # 自动补全前缀
```

### 自定义命名空间

```python
# 使用项目名前缀避免冲突
@register("aggregator.myproject.custom")
class CustomAggregator(Aggregator):
    pass
```

---

## API

### 获取组件类

```python
from oiafed import get_component

# 获取类（不实例化）
AggClass = get_component("aggregator.fedavg")
agg = AggClass(weighted=True)
```

### 创建实例

```python
from oiafed import create_component

# 直接创建实例
agg = create_component("aggregator.fedavg", weighted=True)
```

### 列出组件

```python
from oiafed import list_components

# 所有组件
all_components = list_components()

# 按前缀过滤
aggregators = list_components("aggregator")
learners = list_components("learner")
```

### 检查存在

```python
from oiafed import registry

if registry.exists("aggregator.my_algo"):
    print("已注册")
```

---

## 内置组件

### Aggregator

| 类型 | 说明 |
|------|------|
| `fedavg` | 加权平均 |
| `fedprox` | 近端项 |
| `scaffold` | 控制变量 |
| `fednova` | 归一化 |
| `fedadam` | Adam 优化 |
| `fedyogi` | Yogi 优化 |
| `fedbn` | 跳过 BN |
| `feddyn` | 动态正则 |
| `fedproto` | 原型聚合 |

### Learner

| 类型 | 说明 |
|------|------|
| `default` | 标准学习器 |
| `moon` | 对比学习 |
| `fedper` | 个性化层 |
| `fedrep` | 表示学习 |
| `fedbabu` | 冻结微调 |
| `fedproto` | 原型学习 |
| `target` | TARGET 算法 |
| `fedweit` | 权重分解 |

### Model

| 类型 | 说明 |
|------|------|
| `simple_cnn` | 简单 CNN |
| `cifar10_cnn` | CIFAR-10 CNN |
| `mnist_cnn` | MNIST CNN |
| `resnet18/34/50` | ResNet 系列 |
| `mlp` | 多层感知机 |

### Dataset

| 类型 | 说明 |
|------|------|
| `mnist` | MNIST |
| `fmnist` | Fashion-MNIST |
| `cifar10` | CIFAR-10 |
| `cifar100` | CIFAR-100 |

---

## 实现细节

### Registry 类

```python
class Registry:
    """全局组件注册表"""
    
    _instance = None
    _registry: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str):
        """注册装饰器"""
        def decorator(component_class):
            if name in cls._registry:
                logger.warning(f"Overwriting: {name}")
            cls._registry[name] = component_class
            return component_class
        return decorator
    
    @classmethod
    def get(cls, name: str) -> type:
        """获取组件类"""
        # 尝试完整名称
        if name in cls._registry:
            return cls._registry[name]
        
        # 尝试自动补全前缀
        for prefix in ["aggregator.", "learner.", "model.", ...]:
            full_name = prefix + name
            if full_name in cls._registry:
                return cls._registry[full_name]
        
        raise KeyError(f"Component not found: {name}")
```

### 配置解析

```python
def create_from_config(config: Dict) -> Any:
    """从配置创建组件"""
    type_name = config["type"]
    args = config.get("args", {})
    
    component_class = registry.get(type_name)
    return component_class(**args)
```

---

## 最佳实践

### 1. 组织结构

```
my_project/
├── algorithms/
│   ├── __init__.py      # 导入所有组件
│   ├── aggregators.py
│   └── learners.py
└── run.py
```

```python
# algorithms/__init__.py
from .aggregators import *
from .learners import *
```

### 2. 命名规范

```python
# 推荐：使用项目前缀
@register("aggregator.myproject.custom")

# 不推荐：可能冲突
@register("aggregator.custom")
```

### 3. 文档化

```python
@register("aggregator.myproject.weighted_median")
class WeightedMedianAggregator(Aggregator):
    """
    加权中位数聚合器
    
    Args:
        trim_ratio: 修剪比例
    
    Example:
        aggregator:
          type: myproject.weighted_median
          args:
            trim_ratio: 0.2
    """
```

### 4. 类型提示

```python
from typing import List, Optional
from oiafed import Aggregator, ClientUpdate

@register("aggregator.myproject.custom")
class CustomAggregator(Aggregator):
    def aggregate(
        self,
        updates: List[ClientUpdate],
        global_model: Optional[dict] = None,
    ) -> dict:
        ...
```

---

## 调试

### 打印所有组件

```python
from oiafed import list_components

for name in sorted(list_components()):
    print(name)
```

### 检查注册

```python
from oiafed import registry

# 检查是否注册
assert registry.exists("aggregator.my_algo"), "未注册！"
```

### 注册冲突

重复注册会打印警告：

```
WARNING: Overwriting existing registration: aggregator.fedavg
```

---

## 下一步

- [自定义算法](../01-guides/custom-algorithm.md)
- [架构总览](overview.md)
