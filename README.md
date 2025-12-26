<div align="center">

# 🌐 OiaFed

**One Framework for All Federation**

*统一的联邦学习框架，支持所有联邦场景*

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7+-ee4c2c.svg)](https://pytorch.org/)

[English](README_EN.md) | 简体中文

[文档](docs/README.md) · [快速开始](#快速开始) · [示例](examples/)

</div>

---

## ✨ 为什么选择 OiaFed？

**OiaFed** 是一个模块化、可扩展的通用联邦学习框架。无论你的研究场景是横向联邦、纵向联邦、联邦持续学习还是个性化联邦，OiaFed 都能满足你的需求。

### 🎯 支持的联邦场景

| 场景 | 描述 | 状态 |
|------|------|------|
| **横向联邦 (HFL)** | 样本划分，特征相同 | ✅ 完整支持 |
| **纵向联邦 (VFL)** | 特征划分，样本相同 | ✅ 支持 |
| **联邦持续学习 (FCL)** | 任务序列学习，避免灾难性遗忘 | ✅ 完整支持 |
| **联邦遗忘 (FU)** | 选择性遗忘特定数据 | ✅ 支持 |
| **个性化联邦 (PFL)** | 客户端个性化模型 | ✅ 完整支持 |
| **多服务器联邦** | 层次化/去中心化拓扑 | ✅ 支持 |
| **异步联邦** | 非同步更新 | ✅ 支持 |

### 🚀 核心优势

```
┌─────────────────────────────────────────────────────────────┐
│                      OiaFed 架构                            │
├─────────────────────────────────────────────────────────────┤
│  📦 联邦框架层                                               │
│  Trainer · Learner · Aggregator · Callback · Tracker        │
├─────────────────────────────────────────────────────────────┤
│  🔌 通信抽象层                                               │
│  Node · Proxy · Transport · Serialization                   │
├─────────────────────────────────────────────────────────────┤
│  🌐 传输后端                                                 │
│  Memory (调试) · gRPC (生产) · 自定义                        │
└─────────────────────────────────────────────────────────────┘
```

- **🔧 高度模块化**：组件可插拔，Registry 注册系统让扩展变得简单
- **🚀 三种运行模式**：Serial（调试）、Parallel（多进程）、Distributed（分布式）
- **📚 26+ 内置论文**：FedAvg、MOON、TARGET、SplitNN 等，一键复现
- **⚙️ 配置驱动**：YAML 配置 + 论文默认参数，实验可复现
- **📈 实验追踪**：原生支持 MLflow、Loguru，完整记录实验过程
- **🔗 通信透明**：Memory/gRPC 无缝切换，上层代码无感知

---

## 📦 安装

### 使用 uv（推荐）

```bash
git clone https://github.com/oiafed/oiafed.git
cd oiafed
uv sync
```

### 使用 pip

```bash
git clone https://github.com/oiafed/oiafed.git
cd oiafed
pip install -e .
```

### 依赖要求

- Python >= 3.12
- PyTorch >= 2.7
- 其他依赖见 `pyproject.toml`

---

## 🚀 快速开始

### 方式一：一键复现论文（推荐）

**最简单的方式**：直接指定论文和客户端数量

```bash
# 运行 FedAvg，10 个客户端
python -m src.cli run --paper fedavg -n 10

# 运行 MOON，5 个客户端，50 轮
python -m src.cli run --paper moon -n 5 --rounds 50

# 运行 SplitNN（纵向联邦），2 个客户端
python -m src.cli run --paper splitnn -n 2

# 运行 TARGET（联邦持续学习），3 个客户端
python -m src.cli run --paper target -n 3
```

**查看可用论文**

```bash
# 列出所有论文
python -m src.cli papers list

# 按类别筛选
python -m src.cli papers list --category HFL   # 横向联邦
python -m src.cli papers list --category VFL   # 纵向联邦
python -m src.cli papers list --category FCL   # 联邦持续学习

# 查看论文详情
python -m src.cli papers show fedavg
python -m src.cli papers show moon --params    # 包含可调参数
```

**覆盖默认参数**

```bash
# 使用 base.yaml 作为基础配置
python -m src.cli run --paper fedavg -n 10 --config configs/base.yaml

# 命令行覆盖参数
python -m src.cli run --paper fedavg -n 10 --rounds 100 --lr 0.01 --batch-size 32

# 预览配置（不运行）
python -m src.cli run --paper fedavg -n 10 --dry-run

# 保存生成的配置
python -m src.cli run --paper fedavg -n 10 --save-config ./my_configs
```

### 方式二：配置文件运行

**1. 创建配置文件** (`my_experiment.yaml`)

```yaml
# 实验配置
exp_name: my_first_fl
node_id: trainer
role: trainer

# 训练器配置
trainer:
  type: default
  args:
    max_rounds: 10
    local_epochs: 5

# 聚合器
aggregator:
  type: fedavg

# 模型
model:
  type: simple_cnn
  args:
    num_classes: 10

# 数据集
datasets:
  - type: mnist
    split: train
    partition:
      strategy: dirichlet
      num_partitions: 5
      config:
        alpha: 0.5
```

**2. 运行实验**

```bash
# 配置文件夹模式
python -m src.cli run --config ./configs/my_experiment/

# 指定运行模式
python -m src.cli run --config ./configs/my_experiment/ --mode parallel
```

**3. 查看结果**

```bash
# MLflow UI
mlflow ui --backend-store-uri ./mlruns

# 日志
cat logs/my_first_fl/trainer.log
```

### 方式三：编程方式

```python
import asyncio
from src.runner import FederationRunner

async def main():
    # 方式1：配置文件
    runner = FederationRunner("my_experiment.yaml")
    result = await runner.run()
    
    # 方式2：配置文件夹
    runner = FederationRunner("configs/experiment/")
    result = await runner.run()

asyncio.run(main())
```

---

## 📚 内置论文

### 横向联邦学习 (HFL)

| 论文 | ID | 年份 | 会议/期刊 | 关键特性 |
|------|-----|------|-----------|----------|
| **FedAvg** | `fedavg` | 2017 | AISTATS | 加权平均，FL 基准 |
| **FedProx** | `fedprox` | 2020 | MLSys | 近端项正则化 |
| **SCAFFOLD** | `scaffold` | 2020 | ICML | 控制变量修正 |
| **FedNova** | `fednova` | 2020 | NeurIPS | 归一化平均 |
| **FedAdam** | `fedadam` | 2021 | ICLR | 自适应服务端优化 |
| **FedYogi** | `fedyogi` | 2021 | ICLR | 自适应服务端优化 |
| **FedBN** | `fedbn` | 2021 | ICLR | 跳过 BN 层聚合 |
| **FedDyn** | `feddyn` | 2021 | ICLR | 动态正则化 |
| **MOON** | `moon` | 2021 | CVPR | 对比学习 |
| **FedPer** | `fedper` | 2019 | NeurIPS-W | 个性化层 |
| **FedRep** | `fedrep` | 2021 | ICML | 表示学习 |
| **FedBABU** | `fedbabu` | 2022 | ICLR | Body 冻结微调 |
| **FedRod** | `fedrod` | 2023 | ICLR | 超网络个性化 |
| **FedProto** | `fedproto` | 2022 | AAAI | 原型聚合 |
| **GPFL** | `gpfl` | 2023 | ICLR | 分组个性化 |
| **FedCP** | `fedcp` | 2023 | KDD | 条件策略 |
| **FedDistill** | `feddistill` | 2022 | NeurIPS | 知识蒸馏 |
| **FedDBE** | `feddbe` | 2023 | CVPR | 域偏移估计 |

### 纵向联邦学习 (VFL)

| 论文 | ID | 年份 | 来源 | 关键特性 |
|------|-----|------|------|----------|
| **SplitNN** | `splitnn` | 2018 | MIT | 模型分割，激活值传输 |

### 联邦持续学习 (FCL)

| 论文 | ID | 年份 | 会议 | 关键特性 |
|------|-----|------|------|----------|
| **TARGET** | `target` | 2023 | CVPR | 任务无关表示学习 |
| **FedWEIT** | `fedweit` | 2021 | NeurIPS | 权重分解 |
| **FedKNOW** | `fedknow` | 2023 | - | 知识蒸馏 |
| **FedCPrompt** | `fed_cprompt` | 2023 | - | Prompt 学习 |
| **GLFC** | `glfc` | 2022 | CVPR | 全局-局部特征 |
| **LGA** | `lga` | 2023 | - | 轻量适配器 |
| **FOT** | `fot` | 2024 | AAAI | 遗忘优化迁移 |

### 联邦遗忘 (FU)

| 论文 | ID | 年份 | 会议 | 关键特性 |
|------|-----|------|------|----------|
| **FadEraser** | `faderaser` | 2024 | INFOCOM | 异步遗忘 |

---

## 🖥️ CLI 命令参考

### run 命令

```bash
# 论文模式
python -m src.cli run --paper <paper_id> -n <num_clients> [OPTIONS]

# 配置模式
python -m src.cli run --config <config_path> [OPTIONS]

# 通用选项
  --paper TEXT          论文 ID（如 fedavg, moon, target）
  -n, --num-clients     客户端数量（论文模式必需）
  --config PATH         配置文件/目录路径
  --mode [serial|parallel]  运行模式（默认: parallel）
  --rounds INT          训练轮数
  --local-epochs INT    本地训练轮数
  --lr FLOAT            学习率
  --batch-size INT      批大小
  --seed INT            随机种子
  --dry-run             仅预览配置，不运行
  --save-config PATH    保存生成的配置到目录
  --log-level TEXT      日志级别（默认: INFO）
```

### papers 命令

```bash
# 列出论文
python -m src.cli papers list [--category HFL|VFL|FCL|FU]

# 查看论文详情
python -m src.cli papers show <paper_id> [--params]

# 生成论文配置模板
python -m src.cli papers init <paper_id> -n <num_clients> -o <output_dir>
```

### 其他命令

```bash
# 查看版本
python -m src.cli version

# 查看帮助
python -m src.cli --help
python -m src.cli run --help
python -m src.cli papers --help
```

---

## ⚙️ 配置系统

### 三层配置优先级

```
┌─────────────────────────────┐
│  CLI 参数（最高优先级）       │  --rounds 50 --lr 0.01
├─────────────────────────────┤
│  配置文件                    │  configs/base.yaml
├─────────────────────────────┤
│  论文默认值（最低优先级）     │  papers/defs/hfl/fedavg.yaml
└─────────────────────────────┘
```

### 基础配置模板 (configs/base.yaml)

```yaml
exp_name: default_exp
data_dir: ./data
output_dir: ./outputs
mode: parallel

logging:
  level: INFO
  console: true

tracker:
  enabled: true
  backends:
    - type: mlflow
      tracking_uri: ./mlruns

network:
  trainer_port: 50051
  learner_base_port: 50052
  auto_find_port: true

seed: 42
```

### 配置继承

```yaml
# experiment.yaml
extend: base.yaml  # 继承基础配置

trainer:
  args:
    max_rounds: 50  # 覆盖特定值
```

### 数据划分

```yaml
datasets:
  - type: cifar10
    split: train
    partition:
      strategy: dirichlet  # iid | dirichlet | label_skew | quantity_skew
      num_partitions: 10
      config:
        alpha: 0.5  # 越小越异构
        seed: 42
```

---

## 🛠️ 扩展开发

### 自定义 Aggregator

```python
from src.core import Aggregator, ClientUpdate
from src.registry import aggregator
from typing import List, Any

@aggregator("my_aggregator", description="My custom aggregator")
class MyAggregator(Aggregator):
    def aggregate(self, updates: List[ClientUpdate], global_model=None) -> Any:
        # 你的聚合逻辑
        total_samples = sum(u.num_samples for u in updates)
        # ...
        return aggregated_weights
```

### 自定义 Learner

```python
from src.core import Learner, TrainResult, EvalResult
from src.registry import learner

@learner("my_learner", description="My custom learner")
class MyLearner(Learner):
    async def train_step(self, batch, batch_idx: int):
        # 单步训练逻辑
        loss = self.compute_loss(batch)
        return {"loss": loss.item()}

    async def evaluate(self, config=None) -> EvalResult:
        # 评估逻辑
        return EvalResult(num_samples=100, metrics={"accuracy": 0.95})
```

### 添加新论文定义

```yaml
# src/papers/defs/hfl/my_paper.yaml
id: my_paper
name: "My Paper: A New FL Algorithm"
category: HFL
venue: "ICML"
year: 2024
url: "https://arxiv.org/abs/xxxx.xxxxx"
description: |
  论文描述...

components:
  learner: fl.my_learner
  aggregator: fedavg
  trainer: default
  model: simple_cnn
  dataset: cifar10

defaults:
  trainer:
    num_rounds: 100
    local_epochs: 5
  learner:
    learning_rate: 0.01
    batch_size: 64
```

---

## 📂 项目结构

```
oiafed/
├── src/
│   ├── core/           # 核心抽象 (Trainer, Learner, Aggregator)
│   ├── comm/           # 通信层 (Node, Transport, gRPC)
│   ├── methods/        # 内置算法实现
│   │   ├── aggregators/    # 聚合器 (FedAvg, FedProx, ...)
│   │   ├── learners/       # 学习器
│   │   │   ├── fl/         # 横向联邦 (MOON, FedPer, ...)
│   │   │   ├── cl/         # 持续学习 (TARGET, FOT, ...)
│   │   │   └── vfl/        # 纵向联邦 (SplitNN, ...)
│   │   ├── models/         # 模型 (CNN, ResNet, ...)
│   │   ├── trainers/       # 训练器
│   │   └── datasets/       # 数据集
│   ├── papers/         # 论文定义系统 ⭐ NEW
│   │   ├── defs/           # 论文 YAML 定义
│   │   │   ├── hfl/        # 横向联邦论文
│   │   │   ├── vfl/        # 纵向联邦论文
│   │   │   ├── fcl/        # 联邦持续学习论文
│   │   │   └── fu/         # 联邦遗忘论文
│   │   ├── loader.py       # 论文加载器
│   │   └── __init__.py     # 论文注册表
│   ├── config/         # 配置系统
│   ├── registry/       # 组件注册系统
│   ├── callback/       # 回调系统
│   ├── tracker/        # 实验追踪
│   ├── proxy/          # 远程代理
│   ├── infra/          # 基础设施 (日志, 检查点)
│   ├── cli.py          # 命令行接口 ⭐ NEW
│   └── runner.py       # 运行入口
├── configs/            # 示例配置
│   └── base.yaml       # 基础配置模板
├── examples/           # 示例代码
├── docs/               # 文档
└── pyproject.toml      # 项目配置
```

---

## 📖 文档

| 文档 | 描述 |
|------|------|
| [快速开始](docs/getting-started/quickstart.md) | 5 分钟入门教程 |
| [核心概念](docs/getting-started/concepts.md) | 框架基本概念 |
| [配置指南](docs/user-guide/configuration.md) | 完整配置说明 |
| [论文系统](docs/user-guide/papers.md) | 论文复现指南 |
| [架构设计](docs/architecture/overview.md) | 系统架构详解 |
| [API 参考](docs/api-reference/core.md) | 完整 API 文档 |
| [算法指南](docs/user-guide/algorithms.md) | 内置算法使用 |
| [扩展开发](docs/development/extending.md) | 自定义组件开发 |

---

## 🤝 贡献

欢迎贡献代码、文档、Issue 和建议！请查看 [贡献指南](CONTRIBUTING.md)。

```bash
# 开发环境设置
git clone https://github.com/oiafed/oiafed.git
cd oiafed
uv sync --dev

# 运行测试
pytest tests/ -v

# 代码格式化
black src/
isort src/
```

---

## 📄 许可证

本项目采用 [MIT 许可证](LICENSE)。

---

<div align="center">

**如果这个项目对你有帮助，请给个 ⭐ Star！**

Made with ❤️ by the OiaFed Team

</div>