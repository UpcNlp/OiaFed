# MOE-FedCL 配置继承和灵活部署指南

## 概述

MOE-FedCL支持配置继承和灵活部署模式，实现以下功能：

1. **配置继承**：通过`extends`字段继承基础配置，避免重复
2. **分布式配置**：每个节点一个独立的配置文件
3. **灵活部署**：支持Memory、Process和Network三种模式

## 配置文件结构

```
configs/
├── distributed/
│   ├── base/                          # 基础配置
│   │   ├── server_base.yaml           # 服务器基础配置
│   │   └── client_base.yaml           # 客户端基础配置
│   └── experiments/                   # 具体实验
│       ├── iid/                       # IID实验
│       │   ├── server.yaml            # 服务器配置（继承base）
│       │   ├── client_0.yaml          # 客户端0配置（继承base）
│       │   ├── client_1.yaml          # 客户端1配置（继承base）
│       │   └── client_2.yaml          # 客户端2配置（继承base）
│       └── non_iid/                   # Non-IID实验
│           ├── server.yaml
│           └── client_*.yaml
```

## 配置继承示例

### 基础配置（server_base.yaml）

```yaml
# configs/distributed/base/server_base.yaml
role: server
node_id: server_1

communication:
  mode: memory
  server:
    host: "127.0.0.1"
    port: 8000

training:
  max_rounds: 10
  min_clients: 2
  trainer:
    name: "Generic"
    params:
      local_epochs: 1
      learning_rate: 0.01
      batch_size: 32
  global_model:
    name: "MNIST_CNN"
    params:
      num_classes: 10
```

### 基础配置（client_base.yaml）

```yaml
# configs/distributed/base/client_base.yaml
role: client

communication:
  mode: memory
  server:
    host: "127.0.0.1"
    port: 8000

training:
  learner:
    name: "Generic"
    params:
      model:
        name: "MNIST_CNN"
        params:
          num_classes: 10
      optimizer:
        type: "SGD"
        lr: 0.01
        momentum: 0.9
      loss: "CrossEntropyLoss"
      learning_rate: 0.01
      batch_size: 32
      local_epochs: 1
  dataset:
    name: "MNIST"
    params:
      root: "./data"
      train: true
      download: true
    partition:
      strategy: "iid"
      num_clients: 3
      seed: 42
```

### 继承配置（server.yaml）

```yaml
# configs/distributed/experiments/iid/server.yaml
extends: "../../base/server_base.yaml"

node_id: "server_iid"  # 只覆盖node_id
```

### 继承配置（client_0.yaml）

```yaml
# configs/distributed/experiments/iid/client_0.yaml
extends: "../../base/client_base.yaml"

node_id: "client_0"  # 只覆盖node_id
```

## 使用方式

### 1. Memory/Process模式（推荐用于开发和测试）

在同一进程中创建所有节点（1个server + N个clients）：

```bash
# 方式A: 使用run_federated_learning.py（推荐）
python examples/run_federated_learning.py --config configs/mnist_true_generic.yaml

# 方式B: 使用run_flexible.py加载文件夹
python examples/run_flexible.py --config configs/distributed/experiments/iid/
```

**特点**：
- 所有节点在同一进程
- 适合本地开发和调试
- 配置简单，启动快速

### 2. Network模式（用于真实分布式部署）

分别启动每个节点：

```bash
# 在服务器机器上
python examples/run_flexible.py --config configs/distributed/experiments/iid/server.yaml

# 在客户端机器1上
python examples/run_flexible.py --config configs/distributed/experiments/iid/client_0.yaml

# 在客户端机器2上
python examples/run_flexible.py --config configs/distributed/experiments/iid/client_1.yaml

# 在客户端机器3上
python examples/run_flexible.py --config configs/distributed/experiments/iid/client_2.yaml
```

**特点**：
- 每个节点独立进程
- 支持跨机器部署
- 真实的分布式环境

## 配置继承规则

### 深度合并（Deep Merge）

ConfigLoader使用深度合并策略：

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

结果：
```yaml
training:
  max_rounds: 50   # 来自child
  min_clients: 2   # 来自base
```

### 列表完全替换

列表类型会完全替换，不会合并：

```yaml
# base.yaml
some_list: [1, 2, 3]

# child.yaml
extends: "base.yaml"
some_list: [4, 5]  # 完全替换，不是追加
```

结果：`some_list: [4, 5]`

### 相对路径

`extends`支持相对路径（相对于当前配置文件）：

```yaml
extends: "../../base/server_base.yaml"  # 相对路径
extends: "/absolute/path/to/base.yaml"  # 绝对路径
```

### 多层继承

支持继承链：

```yaml
# common.yaml
training:
  max_rounds: 10

# mnist_base.yaml
extends: "common.yaml"
server:
  global_model:
    name: "MNIST_CNN"

# mnist_iid.yaml
extends: "mnist_base.yaml"
dataset:
  partition:
    strategy: "iid"
```

最终`mnist_iid.yaml`会继承所有上层配置。

## 当前工作状态

### ✅ 已完成功能

1. **YAML配置支持** - ConfigLoader支持YAML格式
2. **配置继承** - 通过`extends`字段自动继承和合并
3. **FederatedLearning直接加载** - 支持传递文件路径或文件夹路径
4. **Generic Trainer/Learner** - 通用组件，无需为每个数据集编写代码
5. **示例配置文件** - 提供了完整的配置文件示例

### ✅ 已验证功能

```bash
# 这些命令已经成功运行
python examples/run_federated_learning.py --config configs/mnist_true_generic.yaml
# 结果: ✓ 98.08% accuracy

python examples/run_flexible.py --config configs/distributed/experiments/iid/
# 结果: ✓ 正确加载4个配置文件（1 server + 3 clients）
```

### ⚠️ 待解决问题

1. 客户端数据集配置需要在启动时正确创建和划分
2. TrainingConfig的某些字段可能需要更完善的传递机制

## 最佳实践

1. **使用配置继承**：将通用配置放在base目录
2. **描述性命名**：配置文件名应清晰描述用途（如`mnist_iid.yaml`）
3. **最小化覆盖**：子配置只覆盖必要的字段
4. **添加注释**：在配置文件中解释为什么要覆盖某个参数
5. **版本控制base配置**：基础配置修改会影响所有实验

## 故障排查

### 问题：配置文件加载失败

**解决方法**：
1. 检查`extends`路径是否正确
2. 使用YAML验证器检查语法
3. 查看日志中的详细错误信息

### 问题：配置继承不生效

**解决方法**：
1. 确保使用`ConfigLoader.load_with_inheritance()`
2. 检查相对路径是否正确
3. 验证YAML缩进是否正确

### 问题：Network模式连接失败

**解决方法**：
1. 检查防火墙设置
2. 确认服务器地址和端口配置正确
3. 确保服务器先于客户端启动

## 示例：创建新实验

### 步骤1：创建服务器配置

```yaml
# configs/distributed/experiments/my_exp/server.yaml
extends: "../../base/server_base.yaml"

node_id: "server_myexp"
```

### 步骤2：创建客户端配置

```yaml
# configs/distributed/experiments/my_exp/client_0.yaml
extends: "../../base/client_base.yaml"

node_id: "client_0"

# 覆盖数据划分策略（如果需要）
training:
  dataset:
    partition:
      strategy: "label_skew"
      params:
        labels_per_client: 2
```

### 步骤3：运行实验

```bash
# Memory模式（本地测试）
python examples/run_flexible.py --config configs/distributed/experiments/my_exp/

# Network模式（分布式部署）
# 服务器: python examples/run_flexible.py --config configs/distributed/experiments/my_exp/server.yaml
# 客户端: python examples/run_flexible.py --config configs/distributed/experiments/my_exp/client_0.yaml
```

## 总结

MOE-FedCL的配置系统提供了：
- **灵活性**：支持多种部署模式
- **可维护性**：配置继承减少重复
- **易用性**：直接传递配置文件路径
- **可扩展性**：轻松添加新实验配置

立即开始：选择一个示例配置，根据需要修改，然后运行！
