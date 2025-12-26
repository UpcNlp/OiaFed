<div align="center">

# 🌐 OiaFed

**One Framework for All Federation**

*A Unified Federated Learning Framework for All Scenarios*

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7+-ee4c2c.svg)](https://pytorch.org/)

English | [简体中文](README.md)

[Website](https://oiafed.cn) · [Documentation](docs/README.md) · [Quick Start](#quick-start) · [Examples](examples/)

</div>

---

## ✨ Why OiaFed?

**OiaFed** is a modular, extensible, and universal federated learning framework. Whether your research involves horizontal federated learning, vertical federated learning, federated continual learning, or personalized federated learning, OiaFed has you covered.

### 🎯 Supported Federation Scenarios

| Scenario | Description | Status |
|----------|-------------|--------|
| **Horizontal FL (HFL)** | Sample partitioning, same features | ✅ Full support |
| **Vertical FL (VFL)** | Feature partitioning, same samples | ✅ Supported |
| **Federated Continual Learning (FCL)** | Sequential task learning, avoiding catastrophic forgetting | ✅ Full support |
| **Federated Unlearning (FU)** | Selective data forgetting | ✅ Supported |
| **Personalized FL (PFL)** | Client-specific personalized models | ✅ Full support |
| **Multi-Server Federation** | Hierarchical/decentralized topology | ✅ Supported |
| **Asynchronous Federation** | Non-synchronous updates | ✅ Supported |

### 🚀 Core Advantages

```
┌─────────────────────────────────────────────────────────────┐
│                    OiaFed Architecture                       │
├─────────────────────────────────────────────────────────────┤
│  📦 Federation Framework Layer                               │
│  Trainer · Learner · Aggregator · Callback · Tracker        │
├─────────────────────────────────────────────────────────────┤
│  🔌 Communication Abstraction Layer                          │
│  Node · Proxy · Transport · Serialization                   │
├─────────────────────────────────────────────────────────────┤
│  🌐 Transport Backends                                       │
│  Memory (Debug) · gRPC (Production) · Custom                │
└─────────────────────────────────────────────────────────────┘
```

- **🔧 Highly Modular**: Plug-and-play components with Registry system for easy extension
- **🚀 Three Running Modes**: Serial (debugging), Parallel (multi-process), Distributed
- **📊 20+ Built-in Algorithms**: FedAvg, FedProx, SCAFFOLD, MOON, TARGET, and more
- **⚙️ Configuration-Driven**: YAML configs with inheritance for reproducible experiments
- **📈 Experiment Tracking**: Native MLflow and Loguru integration
- **🔗 Transparent Communication**: Seamless Memory/gRPC switching

---

## 📦 Installation

### Using uv (Recommended)

```bash
git clone https://github.com/oiafed/oiafed.git
cd oiafed
uv sync
```

### Using pip

```bash
git clone https://github.com/oiafed/oiafed.git
cd oiafed
pip install -e .
```

### Requirements

- Python >= 3.12
- PyTorch >= 2.7
- See `pyproject.toml` for all dependencies

---

## 🚀 Quick Start

### Run Your First FL Experiment in 5 Minutes

**1. Create a config file** (`my_experiment.yaml`)

```yaml
# Experiment configuration
exp_name: my_first_fl
node_id: trainer
role: trainer

# Trainer configuration
trainer:
  type: default
  args:
    max_rounds: 10
    local_epochs: 5

# Aggregator
aggregator:
  type: fedavg

# Model
model:
  type: simple_cnn
  args:
    num_classes: 10

# Dataset
datasets:
  - type: mnist
    split: train
    partition:
      strategy: dirichlet
      num_partitions: 5
      config:
        alpha: 0.5
```

**2. Run the experiment**

```bash
# Using CLI (recommended)
oiafed run --config my_experiment.yaml

# Using Python module
python -m oiafed run --config my_experiment.yaml

# Specify running mode
oiafed run --config my_experiment.yaml --mode parallel
```

**3. View results**

```bash
# MLflow UI
mlflow ui --backend-store-uri ./mlruns

# Logs
cat logs/my_first_fl/trainer.log
```

### Programmatic Usage

```python
import asyncio
from oiafed import FederationRunner

async def main():
    # Option 1: Config file
    runner = FederationRunner("my_experiment.yaml")
    result = await runner.run()
    
    # Option 2: Config directory
    runner = FederationRunner("configs/experiment/")
    result = await runner.run()

asyncio.run(main())
```

---

## 📚 Built-in Algorithms

### Federated Learning Aggregators

| Algorithm | Type ID | Paper | Key Feature |
|-----------|---------|-------|-------------|
| **FedAvg** | `fedavg` | [McMahan+ 2017](https://arxiv.org/abs/1602.05629) | Weighted averaging, FL baseline |
| **FedProx** | `fedprox` | [Li+ 2020](https://arxiv.org/abs/1812.06127) | Proximal regularization |
| **SCAFFOLD** | `scaffold` | [Karimireddy+ 2020](https://arxiv.org/abs/1910.06378) | Control variates |
| **FedNova** | `fednova` | [Wang+ 2020](https://arxiv.org/abs/2007.07481) | Normalized averaging |
| **FedAdam** | `fedadam` | [Reddi+ 2021](https://arxiv.org/abs/2003.00295) | Adaptive optimization |
| **FedYogi** | `fedyogi` | [Reddi+ 2021](https://arxiv.org/abs/2003.00295) | Adaptive optimization |
| **MOON** | `moon` | [Li+ 2021](https://arxiv.org/abs/2103.16257) | Contrastive learning |
| **FedBN** | `fedbn` | [Li+ 2021](https://arxiv.org/abs/2102.07623) | Skip BN layers |
| **FedDyn** | `feddyn` | [Acar+ 2021](https://arxiv.org/abs/2111.04263) | Dynamic regularization |

### Personalized FL Learners

| Algorithm | Type ID | Paper | Key Feature |
|-----------|---------|-------|-------------|
| **FedPer** | `fedper` | [Arivazhagan+ 2019](https://arxiv.org/abs/1912.00818) | Personal layers |
| **FedRep** | `fedrep` | [Collins+ 2021](https://arxiv.org/abs/2102.07078) | Representation learning |
| **FedBABU** | `fedbabu` | [Oh+ 2022](https://arxiv.org/abs/2106.06042) | Body freezing |
| **FedRod** | `fedrod` | [Chen+ 2023](https://arxiv.org/abs/2301.00524) | Hypernetwork |
| **FedProto** | `fedproto` | [Tan+ 2022](https://arxiv.org/abs/2105.00243) | Prototype learning |
| **GPFL** | `gpfl` | - | Group personalization |

### Federated Continual Learning Learners

| Algorithm | Type ID | Paper | Key Feature |
|-----------|---------|-------|-------------|
| **TARGET** | `target` | - | Task generators |
| **FedWEIT** | `fedweit` | [Yoon+ 2021](https://arxiv.org/abs/2104.07409) | Weight decomposition |
| **FedKNOW** | `fedknow` | - | Knowledge distillation |
| **FedCPrompt** | `fedcprompt` | - | Prompt learning |
| **GLFC** | `glfc` | - | Global-local features |
| **LGA** | `lga` | - | Lightweight adapter |

---

## ⚙️ Configuration System

### Configuration Inheritance

```yaml
# base.yaml
trainer:
  args:
    max_rounds: 100
    local_epochs: 5

model:
  type: resnet18
```

```yaml
# experiment.yaml
extend: base.yaml  # Inherit from base

trainer:
  args:
    max_rounds: 50  # Override specific values
```

### Data Partitioning

```yaml
datasets:
  - type: cifar10
    split: train
    partition:
      strategy: dirichlet  # iid | dirichlet | label_skew | quantity_skew
      num_partitions: 10
      config:
        alpha: 0.5  # Lower = more heterogeneous
        seed: 42
```

### Experiment Tracking

```yaml
tracker:
  backends:
    - type: mlflow
      tracking_uri: ./mlruns
      experiment_name: my_experiment

    - type: loguru
      level: INFO
      file: ./logs/training.log
```

---

## 🛠️ Extension Development

### Custom Aggregator

```python
from oiafed import Aggregator, register, ClientUpdate
from typing import List, Any

@register("aggregator.my_aggregator")
class MyAggregator(Aggregator):
    def aggregate(self, updates: List[ClientUpdate], global_model=None) -> Any:
        # Your aggregation logic
        total_samples = sum(u.num_samples for u in updates)
        # ...
        return aggregated_weights
```

### Custom Learner

```python
from oiafed import Learner, register, TrainResult, EvalResult

@register("learner.my_learner")
class MyLearner(Learner):
    async def train_step(self, batch, batch_idx: int):
        # Single step training logic
        loss = self.compute_loss(batch)
        return {"loss": loss.item()}

    async def evaluate(self, config=None) -> EvalResult:
        # Evaluation logic
        return EvalResult(num_samples=100, metrics={"accuracy": 0.95})
```

### Using Custom Components

```yaml
learner:
  type: my_learner  # Use your registered type identifier
  args:
    custom_param: value
```

---

## 📂 Project Structure

```
oiafed/
├── src/
│   ├── core/           # Core abstractions (Trainer, Learner, Aggregator)
│   ├── comm/           # Communication layer (Node, Transport, gRPC)
│   ├── methods/        # Built-in algorithm implementations
│   │   ├── aggregators/    # Aggregators (FedAvg, FedProx, ...)
│   │   ├── learners/       # Learners (FL, CL algorithms)
│   │   ├── models/         # Models (CNN, ResNet, ...)
│   │   └── datasets/       # Datasets
│   ├── config/         # Configuration system
│   ├── registry/       # Component registry system
│   ├── callback/       # Callback system
│   ├── tracker/        # Experiment tracking
│   ├── proxy/          # Remote proxy
│   ├── infra/          # Infrastructure (logging, checkpoints)
│   └── runner.py       # Entry point
├── configs/            # Example configurations
├── examples/           # Example code
├── docs/               # Documentation
└── pyproject.toml      # Project configuration
```

---

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [Quick Start](docs/getting-started/quickstart.md) | 5-minute tutorial |
| [Core Concepts](docs/getting-started/concepts.md) | Framework fundamentals |
| [Configuration Guide](docs/user-guide/configuration.md) | Complete config reference |
| [Architecture](docs/architecture/overview.md) | System architecture |
| [API Reference](docs/api-reference/core.md) | Complete API docs |
| [Algorithms Guide](docs/user-guide/algorithms.md) | Built-in algorithms |
| [Extension Development](docs/development/extending.md) | Custom component development |

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md).

```bash
# Development setup
git clone https://github.com/oiafed/oiafed.git
cd oiafed
uv sync --dev

# Run tests
pytest tests/ -v

# Code formatting
black src/
isort src/
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🔗 Links

- **Website**: [https://oiafed.cn](https://oiafed.cn)
- **Documentation**: [https://docs.oiafed.cn](https://docs.oiafed.cn)
- **GitHub**: [https://github.com/oiafed/oiafed](https://github.com/oiafed/oiafed)
- **PyPI**: [https://pypi.org/project/oiafed](https://pypi.org/project/oiafed)

---

<div align="center">

**If you find this project helpful, please give us a ⭐ Star!**

Made with ❤️ by the OiaFed Team

</div>
