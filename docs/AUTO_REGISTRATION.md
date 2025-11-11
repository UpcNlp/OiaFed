# 自动组件注册系统

## 概述

MOE-FedCL 框架实现了**自动组件注册**功能，用户无需手动导入或注册任何内置组件。

## 使用方式

### ✅ 现在（自动注册）

用户只需要正常导入框架：

```python
from fedcl.federated_learning import FederatedLearning

# 所有内置组件已自动注册！
# 可以直接在配置文件中使用
```

### ❌ 以前（手动导入）

```python
# 需要手动导入每个组件
from fedcl.methods.models.mnist_cnn import MNISTCNNModel
from fedcl.methods.learners.generic import GenericLearner
from fedcl.methods.trainers.generic import GenericTrainer
from fedcl.methods.datasets.mnist import MNISTFederatedDataset
# ... 还有更多

from fedcl.federated_learning import FederatedLearning
```

## 自动注册的组件

框架会自动注册以下 **19 个内置组件**：

### 模型 (1个)
- `MNIST_CNN`: MNIST CNN模型

### 学习器 (5个)
- `Generic`: 通用学习器（推荐）
- `MNISTLearner`: MNIST专用学习器
- `contrastive`: 对比学习
- `personalized_client`: 个性化学习
- `meta`: 元学习

### 训练器 (3个)
- `Generic`: 通用训练器（推荐）
- `FedAvgMNIST`: FedAvg MNIST训练器
- `default`: 标准训练器

### 聚合器 (7个)
- `fedavg`: FedAvg聚合
- `fedprox`: FedProx聚合
- `scaffold`: SCAFFOLD聚合
- `fednova`: FedNova聚合
- `fedadam`: FedAdam聚合
- `fedyogi`: FedYogi聚合
- `feddyn`: FedDyn聚合

### 数据集 (1个)
- `MNIST`: MNIST联邦数据集

### 评估器 (2个)
- `prototype`: 基于原型的评估
- `fairness`: 公平性评估

## 实现原理

### 1. 模块级导入 (`fedcl/__init__.py`)

在框架主包中自动导入 methods 模块：

```python
# fedcl/__init__.py
from . import methods  # noqa: F401
```

### 2. 子模块自动导入

每个子模块的 `__init__.py` 会导入具体实现：

```python
# fedcl/methods/models/__init__.py
from .mnist_cnn import MNISTCNNModel  # noqa: F401

# fedcl/methods/learners/__init__.py
from .generic import GenericLearner  # noqa: F401
from .mnist_learner import MNISTLearner  # noqa: F401

# fedcl/methods/trainers/__init__.py
from .generic import GenericTrainer  # noqa: F401
from .fedavg_mnist import FedAvgMNISTTrainer  # noqa: F401

# fedcl/methods/datasets/__init__.py
from .mnist import MNISTFederatedDataset  # noqa: F401
```

### 3. 装饰器自动注册

每个组件通过装饰器自动注册到全局注册表：

```python
@model(name="MNIST_CNN", version="1.0")
class MNISTCNNModel(nn.Module):
    ...

@learner(name="Generic", version="1.0")
class GenericLearner(BaseLearner):
    ...

@trainer(name="Generic", version="1.0")
class GenericTrainer(BaseTrainer):
    ...

@dataset(name="MNIST", version="1.0")
class MNISTFederatedDataset(FederatedDataset):
    ...
```

## 用户代码示例

### 最简示例

```python
from fedcl.federated_learning import FederatedLearning

# 创建并运行（组件已自动注册）
fl = FederatedLearning("configs/mnist_true_generic.yaml")
await fl.initialize()
result = await fl.run()
```

### 完整示例

```python
import asyncio
from fedcl.federated_learning import FederatedLearning

async def main():
    # 所有内置组件已自动注册
    fl = FederatedLearning("configs/distributed/experiments/iid/")

    await fl.initialize()
    result = await fl.run()

    print(f"✓ 准确率: {result.final_accuracy:.4f}")
    print(f"✓ 轮数: {result.completed_rounds}/{result.total_rounds}")

    await fl.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
```

## 添加自定义组件

用户仍然可以注册自己的自定义组件：

```python
from fedcl.api.decorators import learner
from fedcl.learner.base_learner import BaseLearner

@learner(name="MyCustomLearner", version="1.0")
class MyCustomLearner(BaseLearner):
    async def train(self, training_params):
        # 自定义训练逻辑
        ...

    async def evaluate(self, evaluation_params):
        # 自定义评估逻辑
        ...
```

然后在配置文件中使用：

```yaml
training:
  learner:
    name: "MyCustomLearner"  # 使用自定义学习器
    params:
      ...
```

## 优势

1. ✅ **开箱即用**: 无需手动配置，导入框架即可使用
2. ✅ **简洁代码**: 减少样板代码，提高可读性
3. ✅ **零学习成本**: 用户不需要了解组件注册机制
4. ✅ **灵活扩展**: 仍然支持自定义组件注册

## 测试自动注册

运行测试脚本验证组件注册：

```bash
python examples/test_auto_registration.py
```

输出示例：

```
================================================================================
自动注册的组件列表
================================================================================

提示：用户只需导入 fedcl 的任何功能，所有内置组件自动注册！
例如：from fedcl.federated_learning import FederatedLearning

✓ 已注册的模型:
  - MNIST_CNN

✓ 已注册的学习器:
  - Generic
  - MNISTLearner
  - ...

总共: 19 个组件已自动注册！

✓ 用户无需任何额外配置，开箱即用！
```

## 总结

MOE-FedCL 的自动组件注册系统让框架真正做到了**开箱即用**，用户只需要：

1. 导入 `FederatedLearning`
2. 提供配置文件
3. 运行训练

所有内置组件会自动注册，无需任何手动配置！
