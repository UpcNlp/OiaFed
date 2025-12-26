# 快速入门

5 分钟运行你的第一个联邦学习实验。

---

## 第一个实验：FedAvg on CIFAR-10

### 步骤 1：创建配置文件

创建 `my_experiment.yaml`：

```yaml
exp_name: my_first_fl
node_id: trainer
role: trainer

trainer:
  type: default
  args:
    max_rounds: 10
    local_epochs: 5

aggregator:
  type: fedavg

learner:
  type: default
  args:
    batch_size: 64
    lr: 0.01

model:
  type: cifar10_cnn
  args:
    num_classes: 10

datasets:
  - type: cifar10
    split: train
    args:
      data_dir: ./data
      download: true
    partition:
      strategy: dirichlet
      num_partitions: 5
      config:
        alpha: 0.5

  - type: cifar10
    split: test
    args:
      data_dir: ./data
```

### 步骤 2：运行实验

```bash
python -m src.runner --config my_experiment.yaml --mode serial --num-clients 5
```

### 步骤 3：查看结果

```
Round 1/10 - accuracy: 0.4523
Round 2/10 - accuracy: 0.5678
...
Round 10/10 - accuracy: 0.7234
Training completed!
```

---

## 尝试不同算法

### FedProx（处理异构数据）

```yaml
aggregator:
  type: fedprox
  args:
    mu: 0.01
```

### MOON（对比学习）

```yaml
learner:
  type: moon
  args:
    temperature: 0.5
    mu: 1.0
```

### SCAFFOLD（修正漂移）

```yaml
aggregator:
  type: scaffold
```

---

## 尝试不同数据划分

### IID（均匀分布）

```yaml
partition:
  strategy: iid
  num_partitions: 10
```

### 严重 Non-IID

```yaml
partition:
  strategy: dirichlet
  config:
    alpha: 0.1  # 更小 = 更异构
```

---

## 运行模式

```bash
# Serial（调试）- 推荐使用 CLI
oiafed run --config config.yaml

# Parallel（多进程）
oiafed run --config config.yaml --mode parallel

# 或使用 Python 模块方式
python -m oiafed run --config config.yaml

# Distributed（多机器）
oiafed run --config trainer.yaml  # 在服务器上
oiafed run --config learner.yaml  # 在客户端上
```

---

## 编程方式

```python
from oiafed import FederationRunner

runner = FederationRunner("my_experiment.yaml")
result = runner.run_sync()

print(f"Final accuracy: {result['final_accuracy']}")
```

---

## 下一步

- [核心概念](concepts.md) - 理解框架基础
- [配置系统](../01-guides/configuration.md) - 深入配置
- [内置算法](../01-guides/algorithms.md) - 选择算法
