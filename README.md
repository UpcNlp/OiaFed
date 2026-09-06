<div align="center">

# 🌐 OiaFed

**One Framework for All Federation**

*统一的联邦学习框架，一套代码适配所有联邦场景*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/oiafed.svg)](https://pypi.org/project/oiafed/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)

[English](README_EN.md) | 简体中文

[快速开始](#-快速开始) · [API 示例](#-api-使用示例)

</div>

---

## ✨ 核心特性

- 🔄 **三种运行模式** - 串行调试、本地并行、分布式部署，配置一键切换
- 🧩 **高度模块化** - Trainer、Learner、Aggregator、Callback 插件式架构
- 📦 **26+ 内置算法** - FedAvg、MOON、SCAFFOLD、SplitNN、TARGET 等主流算法
- 🛡️ **生产级通信** - 基于 gRPC + HTTP/2 原生 keepalive，稳定可靠
- 📊 **实验追踪** - 内置 MLflow 集成，自动记录指标和模型
- ⚡ **Early Stopping** - 智能早停，自动恢复最佳权重

---

## 📦 安装

### 方式一：pip 安装（推荐）

```bash
pip install oiafed
```

### 方式二：从源码安装

```bash
git clone https://github.com/oiafed/oiafed.git
cd oiafed
pip install -e .
```

### 可选依赖

```bash
# MLflow 实验追踪
pip install oiafed[mlflow]

# 开发环境（测试、格式化等）
pip install oiafed[dev]

# 完整安装
pip install oiafed[all]
```

### 系统要求

| 依赖 | 版本要求 |
|------|---------|
| Python | >= 3.10 |
| PyTorch | >= 1.12 |
| gRPC | >= 1.50（自动安装） |

---

## 🚀 快速开始

### 30 秒运行第一个实验

```bash
# 安装
pip install oiafed

# 运行 FedAvg，10 个客户端，50 轮
oiafed run --paper fedavg -n 10 --rounds 50
```

### CLI 命令

```bash
# 运行实验
oiafed run --paper fedavg -n 10 --rounds 100

# 串行模式（可断点调试）
oiafed run --paper fedavg -n 5 --mode serial

# 并行模式（本地多进程）
oiafed run --paper fedavg -n 10 --mode parallel

# 列出所有算法
oiafed papers list

# 查看算法详情
oiafed papers show fedavg --params

# 生成配置模板
oiafed papers init fedavg -n 10 -o ./my_experiment/

# 查看版本
oiafed version
```

### 常用参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--paper` | 论文/算法 ID | `--paper fedavg` |
| `-n, --num-clients` | 客户端数量 | `-n 10` |
| `--rounds` | 训练轮数 | `--rounds 100` |
| `--local-epochs` | 本地训练轮数 | `--local-epochs 5` |
| `--lr` | 学习率 | `--lr 0.01` |
| `--batch-size` | 批大小 | `--batch-size 32` |
| `--mode` | 运行模式 | `--mode serial` |
| `--seed` | 随机种子 | `--seed 42` |
| `--config` | 配置文件 | `--config config.yaml` |

---

## 🐍 API 使用示例

### 基础示例：FedAvg

```python
import asyncio
from oiafed import (
    FederatedSystem,
    DefaultTrainer,
    FedAvgLearner,
    FedAvgAggregator,
    SimpleCNN,
)
from oiafed.methods.datasets import get_cifar10_loaders

async def main():
    # 1. 准备数据（Non-IID 划分）
    train_loaders, test_loader = get_cifar10_loaders(
        num_clients=10,
        batch_size=32,
        partition="dirichlet",
        alpha=0.5
    )
    
    # 2. 创建模型
    model = SimpleCNN(num_classes=10)
    
    # 3. 创建聚合器
    aggregator = FedAvgAggregator()
    
    # 4. 创建联邦系统
    system = FederatedSystem(
        model=model,
        aggregator=aggregator,
        learner_class=FedAvgLearner,
        train_loaders=train_loaders,
        test_loader=test_loader,
        config={
            "max_rounds": 100,
            "local_epochs": 5,
            "lr": 0.01,
        }
    )
    
    # 5. 运行训练
    results = await system.run()
    print(f"Final accuracy: {results['accuracy']:.2%}")

asyncio.run(main())
```

### 使用 Early Stopping

```python
from oiafed.callback import EarlyStopping, CallbackManager

# 创建 Early Stopping 回调
early_stopping = EarlyStopping(
    monitor="loss",             # 监控的指标
    patience=10,                # 容忍轮次
    min_delta=0.001,            # 最小改善量
    mode="min",                 # "min"=越小越好, "max"=越大越好
    restore_best_weights=True,  # 恢复最佳权重
    verbose=True
)

# 添加到 Trainer
callbacks = CallbackManager([early_stopping])
trainer = DefaultTrainer(
    learners=learners,
    aggregator=aggregator,
    model=model,
    callbacks=callbacks,
)
```

### 使用 MLflow 追踪

```python
from oiafed.tracker import Tracker

# 创建追踪器
tracker = Tracker(
    experiment_name="fedavg_cifar10",
    tracking_uri="./mlruns",
    auto_log=True
)

# 传给系统
system = FederatedSystem(
    ...,
    tracker=tracker
)

# 训练完成后查看
# mlflow ui --port 5000
```

### 自定义 Learner

```python
from oiafed.core import Learner, TrainResult
from oiafed.registry import learner

@learner("my_learner", description="My custom learner")
class MyLearner(Learner):
    """自定义学习器"""
    
    async def fit(self, config=None) -> TrainResult:
        """本地训练"""
        self.model.train()
        total_loss = 0
        
        for epoch in range(config.get("local_epochs", 1)):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
        
        return TrainResult(
            weights=self.model.state_dict(),
            num_samples=len(self.train_loader.dataset),
            metrics={"loss": total_loss / len(self.train_loader)}
        )
```

### 自定义 Aggregator

```python
from oiafed.core import Aggregator, ClientUpdate
from oiafed.registry import aggregator

@aggregator("my_aggregator", description="My custom aggregator")
class MyAggregator(Aggregator):
    """自定义聚合器"""
    
    def aggregate(self, updates: list[ClientUpdate], global_model=None):
        """加权平均聚合"""
        total_samples = sum(u.num_samples for u in updates)
        
        aggregated = {}
        for key in updates[0].weights.keys():
            aggregated[key] = sum(
                u.weights[key] * (u.num_samples / total_samples)
                for u in updates
            )
        
        return aggregated
```

---

## ⚙️ 配置文件

### 基础配置模板

```yaml
# config.yaml
exp_name: my_experiment
seed: 42
mode: parallel  # serial | parallel | distributed

# 训练配置
trainer:
  type: default
  args:
    max_rounds: 100

# 学习器
learner:
  type: fedavg
  args:
    local_epochs: 5
    lr: 0.01
    
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
    partition:
      strategy: dirichlet
      alpha: 0.5

# 回调
callbacks:
  - type: early_stopping
    config:
      monitor: loss
      patience: 10
      mode: min
      restore_best_weights: true

# 追踪器
tracker:
  enabled: true
  backends:
    - type: mlflow
      tracking_uri: ./mlruns
```

### 分布式配置

```yaml
# trainer.yaml (服务器)
node_id: trainer
role: trainer
listen:
  host: 0.0.0.0
  port: 50051
min_peers: 2
transport:
  mode: grpc

# learner.yaml (客户端)
node_id: learner_0
role: learner
peers:
  - host: 192.168.1.100
    port: 50051
transport:
  mode: grpc
```

---

## 🎯 支持的算法

### 横向联邦 (HFL)

| 算法 | ID | 论文 | 特性 |
|------|-----|------|------|
| FedAvg | `fedavg` | AISTATS'17 | 加权平均基准 |
| FedProx | `fedprox` | MLSys'20 | 近端项正则化 |
| SCAFFOLD | `scaffold` | ICML'20 | 方差修正 |
| MOON | `moon` | CVPR'21 | 对比学习 |
| FedBN | `fedbn` | ICLR'21 | 跳过 BN 聚合 |
| FedNova | `fednova` | NeurIPS'20 | 归一化平均 |
| FedDyn | `feddyn` | ICLR'21 | 动态正则化 |

### 个性化联邦 (PFL)

| 算法 | ID | 论文 | 特性 |
|------|-----|------|------|
| FedPer | `fedper` | NeurIPS-W'19 | 个性化层 |
| FedRep | `fedrep` | ICML'21 | 表示分离 |
| FedBABU | `fedbabu` | ICLR'22 | Body 冻结 |
| FedProto | `fedproto` | AAAI'22 | 原型聚合 |

### 联邦持续学习 (FCL)

| 算法 | ID | 论文 | 特性 |
|------|-----|------|------|
| TARGET | `target` | CVPR'23 | 任务无关表示 |
| GLFC | `glfc` | CVPR'22 | 全局-局部特征 |
| FOT | `fot` | AAAI'24 | 遗忘优化迁移 |
| FedKNOW | `fedknow` | - | 知识蒸馏 |

### 纵向联邦 (VFL)

| 算法 | ID | 来源 | 特性 |
|------|-----|------|------|
| SplitNN | `splitnn` | MIT'18 | 模型分割 |

---

## 📂 项目结构

```
oiafed/
├── oiafed/
│   ├── core/           # 核心抽象 (Trainer, Learner, Aggregator)
│   ├── comm/           # 通信层 (Node, Transport, gRPC)
│   ├── methods/        # 内置算法
│   │   ├── aggregators/    # 聚合器实现
│   │   ├── learners/       # 学习器实现
│   │   ├── trainers/       # 训练器实现
│   │   ├── models/         # 模型定义
│   │   └── datasets/       # 数据集加载
│   ├── callback/       # 回调系统 (EarlyStopping, Checkpoint...)
│   ├── tracker/        # 实验追踪 (MLflow)
│   ├── config/         # 配置解析
│   ├── registry/       # 组件注册
│   ├── proxy/          # 远程代理
│   ├── infra/          # 基础设施 (日志、工具)
│   ├── cli.py          # 命令行接口
│   └── runner.py       # 运行入口
├── configs/            # 配置模板
├── examples/           # 示例代码
├── docs/               # 文档
└── pyproject.toml      # 项目配置
```

---

## 🔧 高级功能

### Callback 系统

```python
from oiafed.callback import Callback, CallbackManager

class MyCallback(Callback):
    async def on_train_begin(self, trainer, context):
        print("Training started!")
    
    async def on_round_end(self, trainer, round_num, context):
        print(f"Round {round_num} completed")
    
    async def on_train_end(self, trainer, context):
        print("Training finished!")

callbacks = CallbackManager([
    MyCallback(),
    EarlyStopping(monitor="loss", patience=10),
    ModelCheckpoint(save_dir="./checkpoints", save_freq=10),
])
```

### 内置 Callback

| Callback | 说明 |
|----------|------|
| `EarlyStopping` | 早停，支持恢复最佳权重 |
| `ModelCheckpoint` | 定期保存模型检查点 |
| `LoggingCallback` | 训练日志记录 |
| `MLflowCallback` | MLflow 指标记录 |

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
black oiafed/
isort oiafed/
```

详见 [贡献指南](CONTRIBUTING.md)

---

## 📖 文档与资源

| 资源 | 链接 |
|------|------|
| GitHub | [https://github.com/oiafed/oiafed](https://github.com/oiafed/oiafed) |
| PyPI | [https://pypi.org/project/oiafed](https://pypi.org/project/oiafed) |
| 示例代码 | [examples/](examples/) |

---

## 📄 许可证

[MIT License](LICENSE)

---

<div align="center">

**如果这个项目对你有帮助，请给个 ⭐ Star！**

Made with ❤️ by OiaFed Team

</div>
