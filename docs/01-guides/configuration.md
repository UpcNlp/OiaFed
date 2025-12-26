# 配置系统

OiaFed 采用 YAML 配置驱动，支持继承、覆盖和模块化组织。

---

## 配置文件结构

```yaml
# ========== 基础信息 ==========
exp_name: my_experiment      # 实验名称
node_id: trainer             # 节点 ID
role: trainer                # 角色: trainer | learner

# ========== 可选：继承 ==========
extend: base.yaml            # 继承其他配置

# ========== 核心组件 ==========
trainer:
  type: default
  args: { ... }

aggregator:
  type: fedavg
  args: { ... }

learner:
  type: default
  args: { ... }

model:
  type: cifar10_cnn
  args: { ... }

# ========== 数据配置 ==========
datasets:
  - type: cifar10
    split: train
    partition: { ... }

# ========== 通信配置 ==========
transport:
  mode: auto

# ========== 追踪配置 ==========
tracker:
  backends: [...]
```

---

## 配置继承

### 单层继承

```yaml
# base.yaml
exp_name: base_experiment
trainer:
  type: default
  args:
    max_rounds: 100
    local_epochs: 5

# experiment.yaml
extend: base.yaml
exp_name: my_experiment  # 覆盖
trainer:
  args:
    max_rounds: 200      # 只覆盖这一项
```

### 多层继承

```yaml
# base.yaml → fedavg.yaml → experiment.yaml
extend: fedavg.yaml
```

### 合并规则

- 标量值：覆盖
- 字典：深度合并
- 列表：覆盖（不合并）

---

## Trainer 配置

```yaml
trainer:
  type: default              # 训练器类型
  args:
    max_rounds: 100          # 最大轮次
    local_epochs: 5          # 本地训练 epoch
    client_fraction: 1.0     # 每轮参与比例 (0.0-1.0)
    min_clients: 2           # 最少参与客户端
    eval_every: 1            # 评估频率
    checkpoint_every: 10     # 检查点频率
```

### Trainer 类型

| 类型 | 说明 |
|------|------|
| `default` | 标准联邦训练 |
| `continual` | 持续学习训练器 |
| `target` | TARGET 算法专用 |

---

## Aggregator 配置

```yaml
aggregator:
  type: fedavg
  args:
    weighted: true           # 按样本数加权
```

### 常用聚合器

```yaml
# FedProx
aggregator:
  type: fedprox
  args:
    mu: 0.01

# SCAFFOLD
aggregator:
  type: scaffold

# FedAdam
aggregator:
  type: fedadam
  args:
    lr: 0.01
    beta1: 0.9
    beta2: 0.999
```

---

## Learner 配置

```yaml
learner:
  type: default
  args:
    batch_size: 64
    lr: 0.01
    optimizer: sgd
    momentum: 0.9
    weight_decay: 0.0001
    device: auto             # auto | cpu | cuda | cuda:0
```

### 个性化学习器

```yaml
# MOON
learner:
  type: moon
  args:
    temperature: 0.5
    mu: 1.0

# FedPer
learner:
  type: fedper
  args:
    personal_layers: ["fc", "classifier"]
```

---

## Model 配置

```yaml
model:
  type: cifar10_cnn
  args:
    num_classes: 10
    dropout: 0.5
```

### 可用模型

| 类型 | 说明 |
|------|------|
| `simple_cnn` | 简单 CNN |
| `cifar10_cnn` | CIFAR-10 专用 |
| `mnist_cnn` | MNIST 专用 |
| `resnet18/34/50` | ResNet 系列 |
| `mlp` | 多层感知机 |

---

## Dataset 配置

```yaml
datasets:
  # 训练集
  - type: cifar10
    split: train
    args:
      data_dir: ./data
      download: true
    partition:
      strategy: dirichlet
      num_partitions: 10
      partition_id: ${NODE_PARTITION_ID}  # 环境变量
      config:
        alpha: 0.5
        seed: 42

  # 测试集
  - type: cifar10
    split: test
    args:
      data_dir: ./data
```

### 划分策略

```yaml
# IID
partition:
  strategy: iid
  num_partitions: 10

# Dirichlet
partition:
  strategy: dirichlet
  config:
    alpha: 0.5  # 0.1=极度异构, 10.0=接近IID

# 标签偏斜
partition:
  strategy: label_skew
  config:
    num_labels_per_client: 2
```

---

## Transport 配置

```yaml
transport:
  mode: auto                 # auto | memory | grpc

# gRPC 详细配置
listen:
  host: 0.0.0.0
  port: 50051

connect_to:
  - trainer@192.168.1.100:50051

# 重试配置
retry:
  max_attempts: 3
  backoff_strategy: exponential
  initial_delay: 1.0
```

---

## Tracker 配置

```yaml
tracker:
  backends:
    # Loguru 日志
    - type: loguru
      level: INFO
      log_dir: ./logs

    # MLflow
    - type: mlflow
      tracking_uri: ./mlruns
      experiment_name: ${exp_name}
```

---

## Callback 配置

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
```

---

## 环境变量

配置中可使用环境变量：

```yaml
exp_name: ${EXP_NAME:default_name}  # 带默认值
data_dir: ${DATA_DIR}               # 必须设置
partition_id: ${NODE_ID:0}
```

---

## 配置验证

```bash
# 验证配置
python -m oiafed validate config.yaml

# 输出解析后的完整配置
python -m oiafed show-config config.yaml
```

---

## 最佳实践

### 1. 分层组织

```
configs/
├── base.yaml           # 基础配置
├── algorithms/
│   ├── fedavg.yaml
│   ├── fedprox.yaml
│   └── scaffold.yaml
└── experiments/
    ├── cifar10_iid.yaml
    └── cifar10_noniid.yaml
```

### 2. 使用继承

```yaml
# experiments/cifar10_noniid.yaml
extend: ../algorithms/fedprox.yaml

datasets:
  - partition:
      config:
        alpha: 0.1
```

### 3. 固定随机种子

```yaml
seed: 42

partition:
  config:
    seed: 42
```

---

## 下一步

- [运行模式](running-modes.md)
- [内置算法](algorithms.md)
- [数据划分](data-partitioning.md)
