# 复现指南

如何使用 OiaFed 复现联邦学习论文实验。

---

## 复现流程

```
1. 阅读论文 → 理解算法和实验设置
      ↓
2. 检查内置 → 是否已有实现
      ↓
3. 配置实验 → 匹配论文超参数
      ↓
4. 运行实验 → 记录结果
      ↓
5. 对比分析 → 验证复现
```

---

## 示例：复现 FedAvg

### 论文设置

论文中的关键实验设置：
- 数据集：CIFAR-10
- 模型：2-layer CNN
- 客户端数：100
- 每轮参与：10%
- 本地 epoch：5
- 批大小：50
- 学习率：0.01
- Non-IID：Dirichlet(0.5)

### 配置文件

```yaml
# experiments/fedavg_cifar10.yaml
exp_name: fedavg_cifar10_reproduction

trainer:
  type: default
  args:
    max_rounds: 500
    local_epochs: 5
    client_fraction: 0.1    # 10% 参与

aggregator:
  type: fedavg

learner:
  type: default
  args:
    batch_size: 50
    lr: 0.01
    optimizer: sgd
    momentum: 0.0           # 论文未使用动量

model:
  type: cifar10_cnn         # 匹配论文模型

datasets:
  - type: cifar10
    split: train
    partition:
      strategy: dirichlet
      num_partitions: 100   # 100 个客户端
      config:
        alpha: 0.5
        seed: 42

  - type: cifar10
    split: test
```

### 运行

```bash
python -m src.runner \
    --config experiments/fedavg_cifar10.yaml \
    --mode parallel \
    --num-clients 100
```

### 验证结果

论文报告：~70% accuracy @ 500 rounds
复现目标：65-75% 范围内

---

## 示例：复现 MOON

### 论文设置

- 基础：FedAvg
- 对比温度：0.5
- 对比权重：1.0
- 其他同 FedAvg

### 配置文件

```yaml
# experiments/moon_cifar10.yaml
exp_name: moon_cifar10_reproduction

extend: fedavg_cifar10.yaml  # 继承基础设置

learner:
  type: moon
  args:
    temperature: 0.5
    mu: 1.0
    batch_size: 50
    lr: 0.01
```

### 对比实验

```bash
# 运行 FedAvg 基线
python -m src.runner --config experiments/fedavg_cifar10.yaml

# 运行 MOON
python -m src.runner --config experiments/moon_cifar10.yaml
```

---

## 示例：复现 SCAFFOLD

### 配置文件

```yaml
# experiments/scaffold_cifar10.yaml
exp_name: scaffold_cifar10_reproduction

extend: fedavg_cifar10.yaml

aggregator:
  type: scaffold

learner:
  type: scaffold  # SCAFFOLD 需要特殊的 Learner
```

---

## 常用实验配置

### CIFAR-10 基础配置

```yaml
# configs/base/cifar10.yaml
model:
  type: cifar10_cnn

datasets:
  - type: cifar10
    split: train
    args:
      data_dir: ./data
      download: true

  - type: cifar10
    split: test
    args:
      data_dir: ./data
```

### Non-IID 配置模板

```yaml
# configs/base/noniid.yaml
partition:
  strategy: dirichlet
  config:
    seed: 42

# 使用时覆盖 alpha
# experiments/alpha_0.1.yaml
extend: noniid.yaml
partition:
  config:
    alpha: 0.1
```

### 多组实验

```bash
#!/bin/bash
# run_alpha_sweep.sh

for alpha in 0.1 0.3 0.5 1.0; do
    python -m src.runner \
        --config base.yaml \
        --override partition.config.alpha=$alpha \
        --override exp_name=fedavg_alpha_$alpha
done
```

---

## 结果记录

### MLflow 追踪

```yaml
tracker:
  backends:
    - type: mlflow
      tracking_uri: ./mlruns
      experiment_name: paper_reproduction
```

```bash
# 查看结果
mlflow ui
```

### 结果对比表

```markdown
| 算法 | 论文结果 | 复现结果 | 差异 |
|------|----------|----------|------|
| FedAvg | 70.2% | 69.8% | -0.4% |
| MOON | 75.1% | 74.7% | -0.4% |
| SCAFFOLD | 73.5% | 73.2% | -0.3% |
```

---

## 常见问题

### Q: 结果与论文差异较大？

检查：
1. 模型结构是否一致
2. 数据预处理是否相同
3. 超参数是否匹配
4. 随机种子是否固定
5. 数据划分是否一致

### Q: 如何确保可复现？

```yaml
# 固定所有随机种子
seed: 42

partition:
  config:
    seed: 42

# 使用确定性算法
torch_deterministic: true
```

### Q: 训练不收敛？

尝试：
1. 降低学习率
2. 增加本地 epoch
3. 检查数据划分是否过于极端
4. 检查梯度是否爆炸

---

## 复现检查清单

- [ ] 模型结构与论文一致
- [ ] 数据集版本正确
- [ ] 数据预处理匹配
- [ ] 客户端数量正确
- [ ] 参与比例正确
- [ ] 本地训练设置正确
- [ ] 聚合算法正确
- [ ] 超参数匹配
- [ ] 随机种子固定
- [ ] 评估指标一致

---

## 贡献复现结果

如果你成功复现了论文，欢迎贡献：

1. 创建配置文件：`experiments/paper_name.yaml`
2. 记录结果：添加到复现结果表
3. 提交 PR

---

## 下一步

- [已复现论文](reproduced-papers.md)
- [内置算法](../01-guides/algorithms.md)
- [配置系统](../01-guides/configuration.md)
