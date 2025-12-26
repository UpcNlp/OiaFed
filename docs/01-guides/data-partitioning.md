# 数据划分

模拟联邦学习中的 Non-IID 场景。

---

## 划分策略

| 策略 | 异构类型 | 关键参数 |
|------|----------|----------|
| **IID** | 无 | - |
| **Dirichlet** | 标签分布 | α |
| **Label Skew** | 标签数量 | num_labels |
| **Quantity Skew** | 样本数量 | ratio |

---

## IID 划分

数据均匀随机分配，每个客户端分布相同。

```yaml
partition:
  strategy: iid
  num_partitions: 10
  config:
    seed: 42
```

```
Client 0: [C0: ██, C1: ██, C2: ██, ...]
Client 1: [C0: ██, C1: ██, C2: ██, ...]
Client 2: [C0: ██, C1: ██, C2: ██, ...]
```

---

## Dirichlet 划分

最常用的 Non-IID 方法，α 控制异构程度。

```yaml
partition:
  strategy: dirichlet
  num_partitions: 10
  config:
    alpha: 0.5  # 核心参数
    seed: 42
```

### α 参数指南

| α 值 | 异构程度 | 用途 |
|------|----------|------|
| 0.1 | 极度异构 | 极端测试 |
| 0.3 | 严重异构 | 挑战实验 |
| 0.5 | 中度异构 | **常用基准** |
| 1.0 | 轻度异构 | 温和场景 |
| 10.0 | 接近 IID | 对比实验 |

```
α = 0.1 (极度异构)        α = 1.0 (轻度异构)
Client 0: [C0: ██████]    Client 0: [C0: ██, C1: ███, C2: █]
Client 1: [C1: ██████]    Client 1: [C0: ███, C1: ██, C2: ██]
```

---

## Label Skew 划分

每个客户端只有固定数量的类别。

```yaml
partition:
  strategy: label_skew
  num_partitions: 10
  config:
    num_labels_per_client: 2  # 每客户端类别数
    seed: 42
```

```
num_labels = 2
Client 0: [C0, C1]
Client 1: [C2, C3]
Client 2: [C4, C5]
```

---

## Quantity Skew 划分

样本数量不均匀。

```yaml
partition:
  strategy: quantity_skew
  num_partitions: 10
  config:
    imbalance_ratio: 0.5  # 0=均匀, 1=极不平衡
    seed: 42
```

```
Client 0: ██████████████ (大客户端)
Client 1: ████████ (中等)
Client 2: ███ (小客户端)
```

---

## 组合使用

```yaml
# 标签 + 数量 双重异构
partition:
  strategy: combined
  config:
    label_alpha: 0.5
    quantity_ratio: 0.3
```

---

## 可视化

```python
from oiafed.utils import visualize_partition

visualize_partition(
    partitions,
    targets,
    save_path="partition.png"
)
```

输出热力图显示每个客户端的类别分布。

---

## 最佳实践

### 1. 固定随机种子

```yaml
partition:
  config:
    seed: 42  # 确保可复现
```

### 2. 设置最小样本数

```yaml
partition:
  strategy: dirichlet
  config:
    alpha: 0.1
    min_samples: 100  # 防止分区过小
```

### 3. 实验对比

```yaml
# 实验组 1: IID
partition: { strategy: iid }

# 实验组 2: 轻度 Non-IID
partition: { strategy: dirichlet, config: { alpha: 1.0 } }

# 实验组 3: 中度 Non-IID
partition: { strategy: dirichlet, config: { alpha: 0.5 } }

# 实验组 4: 重度 Non-IID
partition: { strategy: dirichlet, config: { alpha: 0.1 } }
```

---

## 下一步

- [内置算法](algorithms.md) - 选择适合 Non-IID 的算法
- [配置系统](configuration.md)
