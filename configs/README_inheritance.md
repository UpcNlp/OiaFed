# 配置文件继承说明

## 目录结构

```
configs/
├── base/                      # 基础配置文件
│   └── mnist_base.yaml        # MNIST基础配置
├── experiments/               # 具体实验配置（继承base）
│   ├── mnist_iid.yaml         # IID数据划分实验
│   ├── mnist_non_iid.yaml     # Non-IID数据划分实验
│   ├── mnist_long_training.yaml  # 长训练实验
│   └── mnist_minimal.yaml     # 最简配置
└── README_inheritance.md      # 本文件
```

## 配置继承功能

### 什么是配置继承？

配置继承允许你创建基础配置文件，然后让其他配置文件继承这个基础配置，只需要覆盖需要修改的部分。这样可以：
- **避免重复配置**：相同的配置只需要在base中写一次
- **提高可维护性**：修改base配置会自动影响所有继承它的配置
- **使配置更清晰**：实验配置只包含差异部分，更容易理解

### 如何使用？

#### 1. 创建基础配置

创建一个基础配置文件 `configs/base/mnist_base.yaml`，包含所有通用设置：

```yaml
# configs/base/mnist_base.yaml
experiment_name: "MNIST_Base_Config"
communication:
  mode: memory
training:
  max_rounds: 10
server:
  trainer:
    name: "Generic"
  global_model:
    name: "MNIST_CNN"
clients:
  num_clients: 3
  learner:
    name: "Generic"
  dataset:
    name: "MNIST"
```

#### 2. 创建继承配置

创建继承配置文件，使用 `extends` 字段指定要继承的基础配置：

```yaml
# configs/experiments/mnist_iid.yaml
extends: "../base/mnist_base.yaml"

# 只覆盖需要修改的部分
experiment_name: "MNIST_IID_Experiment"
clients:
  dataset:
    partition:
      strategy: "iid"
```

#### 3. 运行实验

使用简化的启动脚本：

```bash
# 运行IID实验
python examples/simple_run.py --config configs/experiments/mnist_iid.yaml

# 运行Non-IID实验
python examples/simple_run.py --config configs/experiments/mnist_non_iid.yaml
```

## 配置合并规则

ConfigLoader使用深度合并（deep merge）策略：

1. **字典类型**：递归合并
   ```yaml
   # base.yaml
   training:
     max_rounds: 10
     min_clients: 2

   # child.yaml
   extends: "base.yaml"
   training:
     max_rounds: 50  # 覆盖max_rounds
     # min_clients保持为2（从base继承）
   ```

2. **列表类型**：完全替换（不合并）
   ```yaml
   # base.yaml
   clients: [1, 2, 3]

   # child.yaml
   extends: "base.yaml"
   clients: [4, 5]  # 完全替换，不是追加
   ```

3. **其他类型**：直接覆盖

## 示例配置说明

### mnist_base.yaml
基础配置，包含：
- 通信模式：memory
- 训练轮数：10轮
- 客户端数量：3个
- 数据划分：IID

### mnist_iid.yaml
继承base，使用IID数据划分

### mnist_non_iid.yaml
继承base，修改为Non-IID划分（label_skew）

### mnist_long_training.yaml
继承base，修改多个参数：
- 训练轮数：50轮
- 学习率：0.001
- Batch size：64
- 客户端数量：5个
- 数据划分：Dirichlet

### mnist_minimal.yaml
继承base，几乎不做修改，展示最简用法

## 相对路径说明

`extends` 字段支持相对路径和绝对路径：

```yaml
# 相对路径（相对于当前配置文件）
extends: "../base/mnist_base.yaml"

# 绝对路径
extends: "/absolute/path/to/base.yaml"
```

## 多层继承

支持多层继承（继承链）：

```yaml
# configs/base/common.yaml
training:
  max_rounds: 10

# configs/base/mnist_base.yaml
extends: "common.yaml"
server:
  global_model:
    name: "MNIST_CNN"

# configs/experiments/mnist_iid.yaml
extends: "../base/mnist_base.yaml"
clients:
  dataset:
    partition:
      strategy: "iid"
```

最终 `mnist_iid.yaml` 会继承 `mnist_base.yaml` 和 `common.yaml` 的所有配置。

## 最佳实践

1. **基础配置放在 base/ 目录**：便于管理和查找
2. **具体实验放在 experiments/ 目录**：保持清晰的结构
3. **只覆盖必要的字段**：让配置文件尽可能简洁
4. **使用描述性的文件名**：如 `mnist_non_iid_5clients.yaml`
5. **添加注释说明**：解释为什么要覆盖某个参数

## 传统方式 vs 配置继承

### 传统方式（不使用继承）

```python
# 需要手动构建配置对象
python examples/run_federated_learning.py --config configs/mnist_iid.json
```

每个配置文件都包含完整的配置，导致大量重复。

### 使用配置继承

```bash
# 直接使用配置文件路径
python examples/simple_run.py --config configs/experiments/mnist_iid.yaml
```

配置文件只包含差异部分，base配置修改后所有实验自动更新。

## 故障排查

如果遇到配置加载问题：

1. **检查extends路径**：确保路径正确（相对于当前文件）
2. **检查YAML语法**：使用YAML validator验证语法
3. **查看日志**：ConfigLoader会输出详细的加载信息
4. **测试base配置**：先确保base配置本身可以正常运行
