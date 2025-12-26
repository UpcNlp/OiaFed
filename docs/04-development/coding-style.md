# 代码规范

OiaFed 项目的代码风格指南。

---

## 工具链

| 工具 | 用途 | 配置 |
|------|------|------|
| **Black** | 代码格式化 | `pyproject.toml` |
| **isort** | import 排序 | `pyproject.toml` |
| **mypy** | 类型检查 | `pyproject.toml` |
| **pytest** | 测试 | `pytest.ini` |

---

## 格式化

### Black

```bash
# 格式化单个文件
black src/core/trainer.py

# 格式化整个项目
black src/

# 检查（不修改）
black --check src/
```

### isort

```bash
# 排序 import
isort src/

# 检查
isort --check src/
```

### 配置

```toml
# pyproject.toml
[tool.black]
line-length = 88
target-version = ['py312']

[tool.isort]
profile = "black"
line_length = 88
```

---

## 命名规范

### 文件名

```
# 小写 + 下划线
fedavg.py
my_aggregator.py
```

### 类名

```python
# PascalCase
class FedAvgAggregator:
    pass

class MyCustomLearner:
    pass
```

### 函数/方法名

```python
# snake_case
def train_round(self):
    pass

async def collect_updates(self):
    pass
```

### 常量

```python
# 大写 + 下划线
DEFAULT_TIMEOUT = 30
MAX_RETRY_ATTEMPTS = 3
```

### 私有成员

```python
class MyClass:
    def __init__(self):
        self._private_var = 1      # 内部使用
        self.__very_private = 2    # 强私有（名称修饰）
    
    def _helper_method(self):      # 内部方法
        pass
```

---

## 类型注解

### 基本类型

```python
def process(
    name: str,
    count: int,
    ratio: float,
    enabled: bool,
) -> None:
    pass
```

### 容器类型

```python
from typing import List, Dict, Tuple, Optional, Any

def aggregate(
    updates: List[ClientUpdate],
    weights: Dict[str, Tensor],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Tensor]:
    pass
```

### 可选参数

```python
def train(
    epochs: int = 5,
    lr: Optional[float] = None,  # 明确可以是 None
) -> TrainResult:
    pass
```

### 类型别名

```python
from typing import TypeAlias

Weights: TypeAlias = Dict[str, Tensor]
Metrics: TypeAlias = Dict[str, float]

def aggregate(updates: List[Weights]) -> Weights:
    pass
```

---

## 文档字符串

### 模块

```python
"""
联邦平均聚合器

实现 FedAvg 算法的服务端聚合逻辑。

论文: Communication-Efficient Learning of Deep Networks
      from Decentralized Data
"""
```

### 类

```python
class FedAvgAggregator(Aggregator):
    """
    FedAvg 聚合器
    
    使用加权平均聚合客户端更新。
    
    Attributes:
        weighted: 是否按样本数加权
    
    Example:
        >>> agg = FedAvgAggregator(weighted=True)
        >>> result = agg.aggregate(updates)
    """
```

### 方法

```python
def aggregate(
    self,
    updates: List[ClientUpdate],
    global_model: Optional[Dict[str, Tensor]] = None,
) -> Dict[str, Tensor]:
    """
    聚合客户端更新
    
    Args:
        updates: 客户端更新列表，每个包含权重和样本数
        global_model: 当前全局模型权重（可选）
    
    Returns:
        聚合后的模型权重
    
    Raises:
        ValueError: 如果 updates 为空
    
    Example:
        >>> updates = [ClientUpdate(...), ClientUpdate(...)]
        >>> new_weights = agg.aggregate(updates)
    """
```

---

## Import 顺序

```python
# 1. 标准库
import asyncio
import os
from typing import List, Dict, Optional

# 2. 第三方库
import torch
import torch.nn as nn
from loguru import logger

# 3. 本项目
from oiafed import Aggregator, register
from oiafed.core.types import ClientUpdate
```

---

## 异步代码

### async/await

```python
# 正确
async def train_round(self):
    updates = await self.collect_updates()
    return updates

# 错误：忘记 await
async def train_round(self):
    updates = self.collect_updates()  # 返回 coroutine，不是结果
```

### 并发收集

```python
async def collect_all(self, clients):
    # 并发执行
    tasks = [self.get_update(c) for c in clients]
    results = await asyncio.gather(*tasks)
    return results
```

### 超时

```python
async def get_update(self, client_id):
    try:
        return await asyncio.wait_for(
            self._do_get_update(client_id),
            timeout=30.0
        )
    except asyncio.TimeoutError:
        logger.warning(f"Timeout: {client_id}")
        return None
```

---

## 错误处理

### 自定义异常

```python
class OiaFedError(Exception):
    """基础异常"""
    pass

class ConfigError(OiaFedError):
    """配置错误"""
    pass

class CommunicationError(OiaFedError):
    """通信错误"""
    pass
```

### 异常处理

```python
def load_config(path: str) -> Dict:
    try:
        with open(path) as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise ConfigError(f"Config not found: {path}")
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML: {e}")
```

---

## 日志

```python
from loguru import logger

# 级别
logger.debug("详细调试信息")
logger.info("一般信息")
logger.warning("警告")
logger.error("错误")

# 结构化日志
logger.info(
    "Round completed",
    round_num=10,
    accuracy=0.95,
    clients=5,
)
```

---

## 运行检查

```bash
# 完整检查
make lint

# 或单独运行
black --check src/
isort --check src/
mypy src/
```

---

## 下一步

- [测试指南](testing.md)
- [贡献指南](../../CONTRIBUTING.md)
