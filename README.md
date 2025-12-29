<div align="center">

# 🌐 OiaFed

**One Framework for All Federation**

*统一的联邦学习框架，一套代码适配所有联邦场景*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/oiafed.svg)](https://pypi.org/project/oiafed/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)

[English](README_EN.md) | 简体中文

[官网](https://oiafed.cn) · [文档](https://docs.oiafed.cn) · [快速开始](#-快速开始) · [CLI 命令](#-cli-命令参考)

</div>

---

## ✨ 核心特性

### 🔄 三种运行模式，配置切换

**同一套代码，不同模式无缝切换**——从本地调试到生产部署，只需修改配置：

| 模式 | transport | 说明 | 适用场景 |
|------|-----------|------|----------|
| **Serial（串行）** | `memory` | 单进程顺序执行，方便断点调试 | 算法开发、代码调试 |
| **Parallel（并行）** | `grpc` | 多进程本地模拟，真实并发通信 | 本地测试、性能验证 |
| **Distributed（分布式）** | `grpc` | 多机部署，配置不同 IP 地址 | 生产环境、真实联邦 |

```bash
# 串行调试（单进程，可断点）
python -m oiafed.cli run --paper fedavg -n 5 --mode serial

# 本地并行（多进程，gRPC 通信）
python -m oiafed.cli run --paper fedavg -n 10 --mode parallel

# 分布式部署（多机，需要手动配置各节点 IP）
# 在不同机器上分别运行各节点的配置文件
python -m oiafed.cli run --config configs/distributed/trainer.yaml   # 机器 A
python -m oiafed.cli run --config configs/distributed/learner_0.yaml # 机器 B
python -m oiafed.cli run --config configs/distributed/learner_1.yaml # 机器 C
```

### 🧩 高度模块化，扩展灵活

**插件式架构**——每个组件独立可替换，轻松适配不同联邦场景：

```
┌────────────────────────────────────────────────────────────────┐
│                        OiaFed 架构                              │
├────────────────────────────────────────────────────────────────┤
│  🎯 场景层    HFL │ VFL │ FCL │ PFL │ FU │ Async │ Hierarchical │
├────────────────────────────────────────────────────────────────┤
│  📦 组件层    Trainer · Learner · Aggregator · Callback         │
├────────────────────────────────────────────────────────────────┤
│  🔌 通信层    Node · Proxy · Transport (Memory / gRPC)          │
└────────────────────────────────────────────────────────────────┘
```

- **Learner**：客户端学习算法（FedAvg、MOON、SplitNN...）
- **Aggregator**：聚合策略（加权平均、动量、自适应...）
- **Trainer**：训练流程控制（同步、异步、层次化...）
- **Callback**：生命周期钩子（日志、检查点、早停...）

### 🏗️ 多主体架构，灵活拓扑

**不仅仅是 1 Server + N Clients**——通过配置文件支持复杂的多角色架构：

```yaml
# 分布式配置示例：Trainer 在机器 A，Learners 在机器 B、C
# trainer.yaml (机器 A: 192.168.1.100)
node_id: trainer
role: trainer
listen:
  host: 0.0.0.0
  port: 50051
min_peers: 2
transport:
  mode: grpc

# learner_0.yaml (机器 B: 192.168.1.101)
node_id: learner_0
role: learner
peers:
  - host: 192.168.1.100  # Trainer 地址
    port: 50051
transport:
  mode: grpc
```

支持的架构模式：
- **星形**：1 Trainer + N Learners（最常用）
- **多 Trainer**：多个 Trainer 协调（需自定义 Trainer）
- **层次化**：通过嵌套配置实现 Cloud → Edge → Clients

---

## 🎯 支持的联邦场景

| 场景 | 描述 | 内置算法 |
|------|------|----------|
| **横向联邦 (HFL)** | 样本划分，特征相同 | FedAvg, FedProx, SCAFFOLD, MOON, FedBN... |
| **纵向联邦 (VFL)** | 特征划分，样本相同 | SplitNN |
| **联邦持续学习 (FCL)** | 任务序列，避免遗忘 | TARGET, GLFC, FOT, FedKNOW... |
| **个性化联邦 (PFL)** | 客户端个性化模型 | FedPer, FedRep, FedBABU, FedProto... |
| **联邦遗忘 (FU)** | 选择性遗忘数据 | FadEraser |
| **异步联邦** | 非同步更新 | FedAsync |

---

## 📦 安装

### 使用 pip

```bash
pip install oiafed
```

### 从源码安装

```bash
git clone https://github.com/oiafed/oiafed.git
cd oiafed
pip install -e .

# 包含 MLflow 追踪（可选）
pip install -e ".[mlflow]"

# 开发环境
pip install -e ".[dev]"
```

### 依赖要求

- Python >= 3.10
- PyTorch >= 1.12
- gRPC（自动安装）

---

## 🚀 快速开始

### 30 秒运行第一个实验

```bash
# 安装
pip install oiafed

# 运行 FedAvg，10 个客户端，50 轮
python -m oiafed.cli run --paper fedavg -n 10 --rounds 50
```

### 使用不同算法

```bash
# 横向联邦
python -m oiafed.cli run --paper fedavg -n 10      # FedAvg
python -m oiafed.cli run --paper moon -n 5         # MOON（对比学习）
python -m oiafed.cli run --paper scaffold -n 10    # SCAFFOLD（方差修正）

# 纵向联邦
python -m oiafed.cli run --paper splitnn -n 2      # SplitNN

# 联邦持续学习
python -m oiafed.cli run --paper target -n 5       # TARGET
python -m oiafed.cli run --paper glfc -n 5         # GLFC

# 个性化联邦
python -m oiafed.cli run --paper fedper -n 10      # FedPer
python -m oiafed.cli run --paper fedproto -n 10    # FedProto
```

### 切换运行模式

```bash
# 串行模式（单进程，可断点调试）
python -m oiafed.cli run --paper fedavg -n 5 --mode serial

# 并行模式（默认，本地多进程 + gRPC）
python -m oiafed.cli run --paper fedavg -n 10 --mode parallel

# 分布式模式：手动在不同机器运行各节点配置
# 需要先生成配置文件，然后分发到各机器
python -m oiafed.cli run --paper fedavg -n 3 --save-config ./dist_configs/
# 然后在各机器上分别运行对应的配置文件
```

### 自定义参数

```bash
# 覆盖默认参数
python -m oiafed.cli run --paper fedavg -n 10 \
    --rounds 100 \
    --local-epochs 5 \
    --lr 0.01 \
    --batch-size 32 \
    --seed 42

# 使用基础配置文件
python -m oiafed.cli run --paper fedavg -n 10 --config configs/base.yaml

# 预览配置（不运行）
python -m oiafed.cli run --paper fedavg -n 10 --dry-run

# 保存生成的配置
python -m oiafed.cli run --paper fedavg -n 10 --save-config ./my_configs/
```

---

## 🖥️ CLI 命令参考

### run - 运行实验

```bash
python -m oiafed.cli run [OPTIONS]

# 论文模式（快速复现）
python -m oiafed.cli run --paper <paper_id> -n <num_clients> [OPTIONS]

# 配置模式（自定义实验）
python -m oiafed.cli run --config <config_path> [OPTIONS]
```

**选项说明：**

| 选项 | 短写 | 说明 | 示例 |
|------|------|------|------|
| `--paper` | | 论文 ID | `--paper fedavg` |
| `--num-clients` | `-n` | 客户端数量 | `-n 10` |
| `--config` | `-c` | 配置文件/目录 | `--config configs/exp.yaml` |
| `--mode` | `-m` | 运行模式 | `--mode serial` |
| `--rounds` | | 训练轮数 | `--rounds 100` |
| `--local-epochs` | | 本地训练轮数 | `--local-epochs 5` |
| `--lr` | | 学习率 | `--lr 0.01` |
| `--batch-size` | | 批大小 | `--batch-size 64` |
| `--seed` | | 随机种子 | `--seed 42` |
| `--dry-run` | | 仅预览配置 | `--dry-run` |
| `--save-config` | | 保存配置到目录 | `--save-config ./configs/` |
| `--log-level` | | 日志级别 | `--log-level DEBUG` |

### papers - 论文管理

```bash
# 列出所有论文
python -m oiafed.cli papers list

# 按类别筛选
python -m oiafed.cli papers list --category HFL    # 横向联邦
python -m oiafed.cli papers list --category VFL    # 纵向联邦
python -m oiafed.cli papers list --category FCL    # 联邦持续学习
python -m oiafed.cli papers list --category PFL    # 个性化联邦
python -m oiafed.cli papers list --category FU     # 联邦遗忘

# 查看论文详情
python -m oiafed.cli papers show fedavg
python -m oiafed.cli papers show moon --params     # 包含可调参数

# 生成配置模板
python -m oiafed.cli papers init fedavg -n 10 -o ./my_experiment/
```

### 其他命令

```bash
# 验证配置文件
python -m oiafed.cli validate --config config.yaml

# 列出已注册组件
python -m oiafed.cli list aggregators
python -m oiafed.cli list learners
python -m oiafed.cli list models

# 查看版本
python -m oiafed.cli version

# 帮助
python -m oiafed.cli --help
python -m oiafed.cli run --help
python -m oiafed.cli papers --help
```

---

## ⚙️ 配置系统

### 三层配置优先级

```
CLI 参数（最高）  >  配置文件  >  论文默认值（最低）
```

### 基础配置模板

```yaml
# configs/base.yaml
exp_name: my_experiment
seed: 42

# 运行模式
mode: parallel  # serial | parallel | distributed

# 训练配置
trainer:
  type: default
  args:
    max_rounds: 100
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
  - type: cifar10
    split: train
    partition:
      strategy: dirichlet
      num_partitions: 10
      config:
        alpha: 0.5

# 日志
logging:
  level: INFO
  console: true

# 实验追踪（可选）
tracker:
  enabled: true
  backends:
    - type: mlflow
      tracking_uri: ./mlruns
```

### 数据划分策略

```yaml
partition:
  strategy: dirichlet    # iid | dirichlet | label_skew | quantity_skew
  num_partitions: 10
  config:
    alpha: 0.5           # Dirichlet 参数，越小越异构
    seed: 42
```

---

## 📚 内置论文（26+）

### 横向联邦学习 (HFL)

| 论文 | ID | 会议 | 关键特性 |
|------|-----|------|----------|
| FedAvg | `fedavg` | AISTATS'17 | 加权平均，FL 基准 |
| FedProx | `fedprox` | MLSys'20 | 近端项正则化 |
| SCAFFOLD | `scaffold` | ICML'20 | 控制变量方差修正 |
| FedNova | `fednova` | NeurIPS'20 | 归一化平均 |
| FedBN | `fedbn` | ICLR'21 | 跳过 BN 层聚合 |
| MOON | `moon` | CVPR'21 | 模型对比学习 |
| FedDyn | `feddyn` | ICLR'21 | 动态正则化 |
| FedPer | `fedper` | NeurIPS-W'19 | 个性化层 |
| FedRep | `fedrep` | ICML'21 | 表示学习分离 |
| FedBABU | `fedbabu` | ICLR'22 | Body 冻结微调 |
| FedProto | `fedproto` | AAAI'22 | 原型聚合 |

### 纵向联邦学习 (VFL)

| 论文 | ID | 来源 | 关键特性 |
|------|-----|------|----------|
| SplitNN | `splitnn` | MIT'18 | 模型分割，激活值传输 |

### 联邦持续学习 (FCL)

| 论文 | ID | 会议 | 关键特性 |
|------|-----|------|----------|
| TARGET | `target` | CVPR'23 | 任务无关表示 |
| GLFC | `glfc` | CVPR'22 | 全局-局部特征 |
| FOT | `fot` | AAAI'24 | 遗忘优化迁移 |
| FedKNOW | `fedknow` | - | 知识蒸馏 |

---

## 🛠️ 扩展开发

### 自定义 Learner

```python
from oiafed.core import Learner, TrainResult, EvalResult
from oiafed.registry import learner

@learner("my_learner", description="My custom learner")
class MyLearner(Learner):
    """自定义学习器"""
    
    async def train_step(self, batch, batch_idx: int):
        inputs, labels = batch
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {"loss": loss.item()}

    async def evaluate(self, config=None) -> EvalResult:
        # 评估逻辑
        accuracy = self._compute_accuracy()
        return EvalResult(
            num_samples=len(self.test_loader.dataset),
            metrics={"accuracy": accuracy}
        )
```

### 自定义 Aggregator

```python
from oiafed.core import Aggregator, ClientUpdate
from oiafed.registry import aggregator

@aggregator("my_aggregator", description="My custom aggregator")
class MyAggregator(Aggregator):
    """自定义聚合器"""
    
    def aggregate(self, updates: List[ClientUpdate], global_model=None):
        # 加权平均
        total_samples = sum(u.num_samples for u in updates)
        
        aggregated = {}
        for key in updates[0].weights.keys():
            aggregated[key] = sum(
                u.weights[key] * u.num_samples / total_samples
                for u in updates
            )
        
        return aggregated
```

### 使用自定义组件

```yaml
# config.yaml
learner:
  type: my_learner
  args:
    custom_param: value

aggregator:
  type: my_aggregator
```

---

## 📂 项目结构

```
oiafed/
├── src/
│   ├── core/           # 核心抽象 (Trainer, Learner, Aggregator)
│   ├── comm/           # 通信层 (Node, Transport, gRPC)
│   ├── methods/        # 内置算法
│   │   ├── aggregators/    # 聚合器
│   │   ├── learners/       # 学习器 (fl/, cl/, vfl/)
│   │   ├── trainers/       # 训练器
│   │   ├── models/         # 模型
│   │   └── datasets/       # 数据集
│   ├── papers/         # 论文定义 (YAML)
│   ├── config/         # 配置系统
│   ├── registry/       # 组件注册
│   ├── callback/       # 回调系统
│   ├── tracker/        # 实验追踪
│   ├── cli.py          # 命令行接口
│   └── runner.py       # 运行入口
├── configs/            # 配置模板
├── docs/               # 文档
└── pyproject.toml
```

---

## 📖 文档与资源

| 资源 | 链接 |
|------|------|
| 官方网站 | [https://oiafed.cn](https://oiafed.cn) |
| 完整文档 | [https://docs.oiafed.cn](https://docs.oiafed.cn) |
| API 参考 | [https://docs.oiafed.cn/api](https://docs.oiafed.cn/api) |
| 示例代码 | [examples/](examples/) |
| GitHub | [https://github.com/oiafed/oiafed](https://github.com/oiafed/oiafed) |
| PyPI | [https://pypi.org/project/oiafed](https://pypi.org/project/oiafed) |

---

## 🤝 贡献

欢迎贡献代码、文档和建议！

```bash
# 克隆仓库
git clone https://github.com/oiafed/oiafed.git
cd oiafed

# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest tests/ -v

# 代码格式化
black src/
isort src/
```

详见 [贡献指南](CONTRIBUTING.md)

---

## 📄 许可证

[MIT License](LICENSE)

---

<div align="center">

**如果这个项目对你有帮助，请给个 ⭐ Star！**

Made with ❤️ by OiaFed Team

</div>