# 内置算法

OiaFed 内置 20+ 联邦学习算法，覆盖多种场景。

---

## 算法分类

```
┌────────────────────────────────────────────────────────────┐
│                    Aggregator（服务端）                     │
│  FedAvg · FedProx · SCAFFOLD · FedNova · FedAdam · ...    │
├────────────────────────────────────────────────────────────┤
│                    Learner（客户端）                        │
│  Default · MOON · FedPer · FedRep · TARGET · FedWEIT ·... │
└────────────────────────────────────────────────────────────┘
```

---

## 聚合器算法

### FedAvg

最基础的联邦平均算法。

```yaml
aggregator:
  type: fedavg
  args:
    weighted: true  # 按样本数加权
```

**适用**：IID 数据、基准对比

---

### FedProx

添加近端项正则化，缓解客户端漂移。

```yaml
aggregator:
  type: fedprox
  args:
    mu: 0.01  # 0.001-0.1
```

**适用**：Non-IID 数据、系统异构

**论文**：[Federated Optimization in Heterogeneous Networks](https://arxiv.org/abs/1812.06127)

---

### SCAFFOLD

使用控制变量修正漂移。

```yaml
aggregator:
  type: scaffold
```

**适用**：严重 Non-IID、追求快速收敛

**论文**：[SCAFFOLD: Stochastic Controlled Averaging](https://arxiv.org/abs/1910.06378)

---

### FedNova

归一化本地更新，处理训练步数不一致。

```yaml
aggregator:
  type: fednova
```

**适用**：客户端 epoch 数不同

---

### FedAdam / FedYogi

服务端自适应优化。

```yaml
aggregator:
  type: fedadam  # 或 fedyogi
  args:
    lr: 0.01
    beta1: 0.9
    beta2: 0.999
```

---

### FedBN

跳过 BatchNorm 层聚合。

```yaml
aggregator:
  type: fedbn
```

**适用**：特征分布差异大

---

## 个性化学习算法

### MOON

对比学习减少漂移。

```yaml
learner:
  type: moon
  args:
    temperature: 0.5
    mu: 1.0
```

**论文**：[Model-Contrastive Federated Learning](https://arxiv.org/abs/2103.16257)

---

### FedPer

分离共享层和个性化层。

```yaml
learner:
  type: fedper
  args:
    personal_layers: ["fc", "classifier"]
```

---

### FedRep

交替训练表示层和分类头。

```yaml
learner:
  type: fedrep
  args:
    head_epochs: 5
    body_epochs: 1
```

---

### FedBABU

冻结 body，只微调分类头。

```yaml
learner:
  type: fedbabu
  args:
    finetune_epochs: 5
```

---

### FedProto

基于原型的学习。

```yaml
aggregator:
  type: fedproto

learner:
  type: fedproto
  args:
    proto_dim: 256
```

---

## 持续学习算法

### TARGET

任务生成器方法。

```yaml
trainer:
  type: target

learner:
  type: target
  args:
    generator_type: cnn
    distill_weight: 1.0
```

---

### FedWEIT

权重分解方法。

```yaml
learner:
  type: fedweit
  args:
    decompose_ratio: 0.5
```

---

### FedKNOW

知识蒸馏方法。

```yaml
learner:
  type: fedknow
  args:
    distill_weight: 1.0
    temperature: 2.0
```

---

### GLFC / LGA

```yaml
# GLFC
learner:
  type: glfc
  args:
    local_coef: 0.5

# LGA
learner:
  type: lga
  args:
    adapter_dim: 64
```

---

## 选择指南

### 按数据分布

| 数据分布 | 推荐算法 |
|----------|----------|
| IID | FedAvg |
| 轻度 Non-IID | FedProx |
| 中度 Non-IID | SCAFFOLD / MOON |
| 重度 Non-IID | FedBN / PFL 系列 |

### 按场景

| 场景 | 推荐算法 |
|------|----------|
| 基准测试 | FedAvg |
| 需要个性化 | MOON / FedPer / FedRep |
| 数据持续变化 | TARGET / FedWEIT |
| 通信受限 | FedAvg / FedAdam |

### 决策流程

```
            IID?
           /    \
         是      否
          |       |
       FedAvg   需要个性化?
                /       \
              是         否
               |          |
          FedPer/MOON   FedProx/SCAFFOLD
```

---

## 超参数调优

| 参数 | 范围 | 建议 |
|------|------|------|
| `learning_rate` | 0.001-0.1 | 0.01 起 |
| `batch_size` | 32-256 | 64 |
| `local_epochs` | 1-10 | 5 |
| `fedprox_mu` | 0.001-0.1 | 0.01 |
| `moon_temperature` | 0.1-1.0 | 0.5 |
| `dirichlet_alpha` | 0.1-10.0 | 0.5 |

---

## 下一步

- [数据划分](data-partitioning.md)
- [自定义算法](custom-algorithm.md)
- [论文复现](../05-papers/reproduced-papers.md)
