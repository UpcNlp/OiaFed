# 已复现论文

OiaFed 内置算法对应的论文列表。

---

## 纵向联邦学习 (VFL)

### SplitNN

**Split Learning for Health: Distributed Deep Learning without Sharing Raw Data**

- 作者：Gupta & Raskar (MIT Media Lab)
- 年份：2018
- 链接：[arXiv:1812.00564](https://arxiv.org/abs/1812.00564)

```yaml
learner:
  type: vfl.splitnn
  args:
    split_layer: 2
```

**核心思想**：
- 模型垂直分割，不同参与方持有不同层
- 只传输中间激活值，不传输原始数据
- 适用于不同参与方持有相同样本的不同特征

**配置文件**：`configs/papers/vfl/splitnn_mnist.yaml`

---

## 联邦学习基础

### FedAvg

**Communication-Efficient Learning of Deep Networks from Decentralized Data**

- 作者：McMahan et al.
- 会议：AISTATS 2017
- 链接：[arXiv:1602.05629](https://arxiv.org/abs/1602.05629)

```yaml
aggregator:
  type: fedavg
```

---

### FedProx

**Federated Optimization in Heterogeneous Networks**

- 作者：Li et al.
- 会议：MLSys 2020
- 链接：[arXiv:1812.06127](https://arxiv.org/abs/1812.06127)

```yaml
aggregator:
  type: fedprox
  args:
    mu: 0.01
```

---

### SCAFFOLD

**SCAFFOLD: Stochastic Controlled Averaging for Federated Learning**

- 作者：Karimireddy et al.
- 会议：ICML 2020
- 链接：[arXiv:1910.06378](https://arxiv.org/abs/1910.06378)

```yaml
aggregator:
  type: scaffold
```

---

### FedNova

**Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization**

- 作者：Wang et al.
- 会议：NeurIPS 2020
- 链接：[arXiv:2007.07481](https://arxiv.org/abs/2007.07481)

```yaml
aggregator:
  type: fednova
```

---

### FedAdam / FedYogi

**Adaptive Federated Optimization**

- 作者：Reddi et al.
- 会议：ICLR 2021
- 链接：[arXiv:2003.00295](https://arxiv.org/abs/2003.00295)

```yaml
aggregator:
  type: fedadam  # 或 fedyogi
  args:
    lr: 0.01
```

---

### FedBN

**FedBN: Federated Learning on Non-IID Features via Local Batch Normalization**

- 作者：Li et al.
- 会议：ICLR 2021
- 链接：[arXiv:2102.07623](https://arxiv.org/abs/2102.07623)

```yaml
aggregator:
  type: fedbn
```

---

### FedDyn

**Federated Learning Based on Dynamic Regularization**

- 作者：Acar et al.
- 会议：ICLR 2021
- 链接：[arXiv:2111.04263](https://arxiv.org/abs/2111.04263)

```yaml
aggregator:
  type: feddyn
  args:
    alpha: 0.01
```

---

## 个性化联邦学习

### MOON

**Model-Contrastive Federated Learning**

- 作者：Li et al.
- 会议：CVPR 2021
- 链接：[arXiv:2103.16257](https://arxiv.org/abs/2103.16257)

```yaml
learner:
  type: moon
  args:
    temperature: 0.5
    mu: 1.0
```

---

### FedPer

**Federated Learning with Personalization Layers**

- 作者：Arivazhagan et al.
- 会议：arXiv 2019
- 链接：[arXiv:1912.00818](https://arxiv.org/abs/1912.00818)

```yaml
learner:
  type: fedper
  args:
    personal_layers: ["fc", "classifier"]
```

---

### FedRep

**Exploiting Shared Representations for Personalized Federated Learning**

- 作者：Collins et al.
- 会议：ICML 2021
- 链接：[arXiv:2102.07078](https://arxiv.org/abs/2102.07078)

```yaml
learner:
  type: fedrep
  args:
    head_epochs: 5
    body_epochs: 1
```

---

### FedBABU

**FedBABU: Toward Enhanced Representation for Federated Image Classification**

- 作者：Oh et al.
- 会议：ICLR 2022
- 链接：[arXiv:2106.06042](https://arxiv.org/abs/2106.06042)

```yaml
learner:
  type: fedbabu
  args:
    finetune_epochs: 5
```

---

### FedRoD

**On Bridging Generic and Personalized Federated Learning for Image Classification**

- 作者：Chen et al.
- 会议：ICLR 2023
- 链接：[arXiv:2107.00778](https://arxiv.org/abs/2107.00778)

```yaml
learner:
  type: fedrod
```

---

### FedProto

**FedProto: Federated Prototype Learning across Heterogeneous Clients**

- 作者：Tan et al.
- 会议：AAAI 2022
- 链接：[arXiv:2105.00243](https://arxiv.org/abs/2105.00243)

```yaml
aggregator:
  type: fedproto
learner:
  type: fedproto
```

---

### GPFL

**GPFL: Simultaneously Learning Global and Personalized Feature Information for Personalized Federated Learning**

- 作者：Zhang et al.
- 会议：ICCV 2023
- 链接：[arXiv:2308.10279](https://arxiv.org/abs/2308.10279)

```yaml
learner:
  type: gpfl
```

---

## 联邦持续学习

### TARGET

**Federated Class-Incremental Learning**

- 作者：Dong et al.
- 会议：CVPR 2022
- 链接：[arXiv:2203.11473](https://arxiv.org/abs/2203.11473)

```yaml
trainer:
  type: target
learner:
  type: target
  args:
    distill_weight: 1.0
```

---

### FedWEIT

**Federated Continual Learning with Weighted Inter-client Transfer**

- 作者：Yoon et al.
- 会议：ICML 2021
- 链接：[arXiv:2003.03196](https://arxiv.org/abs/2003.03196)

```yaml
learner:
  type: fedweit
  args:
    decompose_ratio: 0.5
```

---

### FedKNOW

**FedKNOW: Federated Continual Learning with Signature Task Knowledge Integration at Edge**

- 作者：未公开
- 会议：待定

```yaml
learner:
  type: fedknow
  args:
    distill_weight: 1.0
```

---

### GLFC

**Federated Learning for Vision-and-Language Grounding Problems**

- 作者：Dong et al.
- 会议：CVPR 2022

```yaml
learner:
  type: glfc
```

---

### LGA

**Learn from Others and Be Yourself in Heterogeneous Federated Learning**

- 作者：未公开

```yaml
learner:
  type: lga
  args:
    adapter_dim: 64
```

---

### FedCPrompt

**Continual Federated Learning via Prompt Learning**

- 作者：未公开

```yaml
learner:
  type: fedcprompt
  args:
    prompt_length: 10
```

---

### FOT (新增)

**Federated Orthogonal Training for Continual Learning**

- 相关论文：Orthogonal Gradient Descent for Continual Learning
- 链接：[arXiv:1910.07104](https://arxiv.org/abs/1910.07104)

```yaml
learner:
  type: cl.fot
  args:
    orthogonal_weight: 1.0
    use_projection: true
    num_tasks: 5
    classes_per_task: 2
```

**核心思想**：
- 将新任务梯度投影到旧任务参数的正交补空间
- 确保更新方向不干扰已学习的知识
- 无需存储历史样本

**配置文件**：`configs/papers/fcl/fot_cifar10.yaml`

---

## 联邦遗忘 (FU)

### FedEraser (新增)

**FedEraser: Enabling Efficient Client-Level Data Removal from Federated Learning Models**

- 作者：Liu et al.
- 会议：IEEE INFOCOM 2022
- 链接：[arXiv:2111.08096](https://arxiv.org/abs/2111.08096)

```yaml
aggregator:
  type: faderaser
  args:
    history_dir: ./fed_history
    calibration_rounds: 10
    unlearn_strategy: recalibrate
```

**核心思想**：
- 高效删除特定客户端数据对模型的影响
- 记录训练历史，遗忘时通过校准重新聚合
- 无需从头重新训练，大幅降低计算开销

**适用场景**：
- GDPR 数据删除请求
- 恶意客户端移除
- 客户端退出联邦

**配置文件**：`configs/papers/fu/faderaser_cifar10.yaml`

---

## 按场景分类

| 场景 | 算法 | 论文 |
|------|------|------|
| **VFL** | SplitNN | MIT'18 |
| **HFL** | FedAvg, FedProx, SCAFFOLD | AISTATS'17, MLSys'20, ICML'20 |
| **PFL** | MOON, FedPer, FedRep | CVPR'21, arXiv'19, ICML'21 |
| **FCL** | TARGET, FedWEIT, FOT | CVPR'22, ICML'21, NeurIPS Workshop |
| **FU** | FedEraser | INFOCOM'22 |

---

## 下一步

- [复现指南](reproduction-guide.md)
- [内置算法](../01-guides/algorithms.md)
