# HFL: 横向联邦学习 (Horizontal Federated Learning)

场景：多个客户端拥有相同特征但不同样本的数据

## 论文列表

### 基础算法
- **fedavg** - FedAvg: Communication-Efficient Learning (AISTATS 2017)
- **fedprox** - FedProx: Federated Optimization in Heterogeneous Networks (MLSys 2020)
- **scaffold** - SCAFFOLD: Stochastic Controlled Averaging (ICML 2020)
- **fednova** - FedNova: Tackling Objective Inconsistency (NeurIPS 2020)

### 自适应优化
- **fedadam** - FedAdam: Adaptive Federated Optimization (ICLR 2021)
- **fedyogi** - FedYogi: Adaptive Federated Optimization with Yogi (ICLR 2021)
- **feddyn** - FedDyn: Federated Learning based on Dynamic Regularizer (ICLR 2021)

### 个性化联邦学习
- **fedbn** - FedBN: Federated Learning via Local Batch Normalization (ICLR 2021)
- **fedper** - FedPer: Federated Learning with Personalization Layers (AISTATS 2020)
- **fedrep** - FedRep: Exploiting Shared Representations (ICML 2021)
- **fedbabu** - FedBABU: Towards Enhanced Representation (ICLR 2022)
- **fedrod** - FedRoD: Robust Disentangled Representation (CVPR 2023)
- **gpfl** - GPFL: Generalized Personalized Federated Learning (NeurIPS 2022)
- **fedcp** - FedCP: Contrastive Personalization (KDD 2023)

### 对比学习 & 知识蒸馏
- **moon** - MOON: Model-Contrastive Federated Learning (CVPR 2021)
- **fedproto** - FedProto: Federated Prototype Learning (AAAI 2022)
- **feddistill** - FedDistill: Knowledge Distillation (NeurIPS 2020)
- **feddbe** - FedDBE: Data-Free Knowledge Distillation (CVPR 2022)

### One-shot 联邦学习基线
- **ofedavg** - 独立本地训练后的统一参数平均
- **ensemble** - 独立客户端模型的直接 logits 集成
- **fafi** - 学习式原型与特征集成
- **fusefl** - 四阶段可扩展分层融合
- **fedcgs** - 全局特征充分统计量与共享协方差分类器
- **coboosting** - 动态教师权重与数据自由服务器蒸馏
