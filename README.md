# MOE-FedCL

**A Modular and Extensible Federated Continual Learning Framework**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7+-ee4c2c.svg)](https://pytorch.org/)

A flexible, production-ready framework for federated learning and continual learning research. Built with modularity and extensibility at its core, MOE-FedCL supports multiple learning scenarios, running modes, and 20+ built-in algorithms.

---

## Why MOE-FedCL?

### Core Strengths

- **🎯 4 Learning Scenarios**: Federated Learning (FL), Continual Learning (CL), Hybrid FL+CL, and Custom scenarios
- **🚀 3 Running Modes**: Seamless switching between Serial (debugging), Parallel (single-machine multi-GPU), and Distributed (multi-machine)
- **🔌 20+ Built-in Algorithms**: FedAvg, FedProx, MOON, SCAFFOLD, TARGET, FedWEIT, and more - all ready to use
- **📊 Experiment Tracking**: Native MLflow and Loguru integration for comprehensive experiment management
- **⚙️ Configuration-Driven**: YAML-based configs with inheritance support for reproducible experiments
- **🛠️ Easy to Extend**: Clean Registry system for adding custom algorithms in minutes
- **🔄 Communication Flexibility**: Memory, gRPC, and custom transport layers with transparent switching

### Architecture Overview

MOE-FedCL follows a clean two-layer architecture:

```
┌─────────────────────────────────────────┐
│    Federation Framework Layer           │
│  (Trainer, Learner, Aggregator, etc.)   │
├─────────────────────────────────────────┤
│    Communication Layer (node_comm)      │
│  (Transport, Serialization, Messaging)  │
├─────────────────────────────────────────┤
│    Transport Backends                   │
│  (Memory, gRPC, Custom)                 │
└─────────────────────────────────────────┘
```

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/MOE-FedCL.git
cd MOE-FedCL

# Install dependencies (using uv recommended)
uv sync

# Or with pip
pip install -e .
```

### Your First Federated Learning Experiment (5 minutes)

**1. Create a simple config** (`my_config.yaml`):

```yaml
# Trainer configuration
role: trainer
node_id: trainer
listen:
  host: localhost
  port: 50051

trainer:
  type: default
  args:
    max_rounds: 10
    local_epochs: 5

aggregator:
  type: fedavg

model:
  type: simple_cnn
  args:
    num_classes: 10

datasets:
  - type: mnist
    split: train
```

**2. Run the experiment**:

```bash
# Serial mode (single process, for debugging)
python -m src.core.system --config my_config.yaml --mode serial --num-clients 5

# Parallel mode (multi-process, for production)
python -m src.core.system --config my_config.yaml --mode parallel --num-clients 5
```

**3. Check results**:

Results are automatically tracked in `mlruns/` (MLflow) and `logs/` directories.

---

## Key Features

### 1. Multiple Learning Scenarios

#### Federated Learning (FL)
Support for 12+ FL algorithms:
- **Basic**: FedAvg, FedProx, FedNova
- **Adaptive**: FedAdam, FedYogi, FedAdaGrad
- **Personalized**: FedPer, FedRep, FedBABU, FedRod
- **Advanced**: MOON, SCAFFOLD, FedBN, FedProto, GPFL

#### Continual Learning (CL)
Support for 7 CL methods:
- TARGET, FedWEIT, FedKNOW, FedCPrompt, GLFC, LGA

#### Hybrid Scenarios
Combine FL and CL for realistic scenarios with data heterogeneity and task evolution.

### 2. Flexible Running Modes

| Mode | Use Case | Performance | Setup |
|------|----------|-------------|-------|
| **Serial** | Debugging, quick tests | Single process | Zero config |
| **Parallel** | Single-machine, multi-GPU | High performance | Specify `--num-clients` |
| **Distributed** | Multi-machine deployment | Production scale | Configure `listen` and `connect_to` |

### 3. Configuration System

Powerful YAML-based configuration with inheritance:

```yaml
# base.yaml
trainer:
  args:
    max_rounds: 100

# experiment.yaml
extend: base.yaml  # Inherit from base
trainer:
  args:
    max_rounds: 50  # Override specific values
```

### 4. Experiment Management

Built-in tracking with multiple backends:

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

### 5. Data Partitioning

Support for various non-IID scenarios:

```yaml
partition:
  strategy: dirichlet  # or: iid, label_skew, quantity_skew
  num_partitions: 10
  config:
    alpha: 0.5  # Controls heterogeneity
    seed: 42
```

---

## Built-in Algorithms

### Federated Learning Aggregators

| Algorithm | Paper | Key Feature |
|-----------|-------|-------------|
| FedAvg | [McMahan et al., 2017](https://arxiv.org/abs/1602.05629) | Basic weighted averaging |
| FedProx | [Li et al., 2020](https://arxiv.org/abs/1812.06127) | Proximal term for stability |
| SCAFFOLD | [Karimireddy et al., 2020](https://arxiv.org/abs/1910.06378) | Control variates |
| FedNova | [Wang et al., 2020](https://arxiv.org/abs/2007.07481) | Normalized averaging |
| FedAdam/FedYogi | [Reddi et al., 2021](https://arxiv.org/abs/2003.00295) | Adaptive server optimization |
| MOON | [Li et al., 2021](https://arxiv.org/abs/2103.16257) | Model contrastive learning |
| FedBN | [Li et al., 2021](https://arxiv.org/abs/2102.07623) | Local batch normalization |

### Continual Learning Methods

| Method | Paper | Key Feature |
|--------|-------|-------------|
| TARGET | [Your paper] | Task-specific generators |
| FedWEIT | [Yoon et al., 2021](https://arxiv.org/abs/2104.07409) | Weight decomposition |
| FedKNOW | [Your paper] | Knowledge distillation |
| GLFC | [Your paper] | Global-local feature composition |

[See full algorithm list →](docs/builtin-algorithms.md)

---

## Examples

### Example 1: FedAvg on MNIST (IID)

```yaml
# configs/examples/fedavg_mnist.yaml
extend: configs/presets/base.yaml

datasets:
  - type: mnist
    partition:
      strategy: iid

aggregator:
  type: fedavg
```

Run: `python -m src.core.system --config configs/examples/fedavg_mnist.yaml`

### Example 2: FedProx on CIFAR-10 (Non-IID)

```yaml
# configs/examples/fedprox_cifar10.yaml
datasets:
  - type: cifar10
    partition:
      strategy: dirichlet
      config:
        alpha: 0.5

aggregator:
  type: fedprox
  args:
    mu: 0.01  # Proximal term
```

### Example 3: Custom Learner

```python
# src/methods/learners/my_learner.py
from src.core import Learner, register

@register("learner", "my_custom")
class MyCustomLearner(Learner):
    async def fit(self, config):
        # Your training logic
        return {"loss": 0.1, "accuracy": 0.95}

    async def evaluate(self, config):
        # Your evaluation logic
        return {"accuracy": 0.96}
```

Use in config:
```yaml
learner:
  type: my_custom
```

[More examples →](examples/)

---

## Reproducing Papers

We provide ready-to-use configurations for reproducing 288 experiments across 8 datasets and 9 algorithms:

```bash
# Run a single experiment
python -m src.core.system --config configs/table3_experiments/mnist/fedavg/dir_0.5/trainer.yaml

# Run batch experiments
python scripts/run_experiments.py --config-dir configs/table3_experiments
```

**Available configurations**:
- **Datasets**: MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100, SVHN, EMNIST, FEMNIST, CINIC-10
- **Algorithms**: FedAvg, FedProx, SCAFFOLD, FedNova, FedAdam, FedYogi, MOON, FedBN
- **Non-IID scenarios**: Dirichlet(α=0.5), Quantity skew (levels 1/2/3)

[Reproduction guide →](docs/reproducing-papers.md)

---

## Documentation

### For Users
- [Installation Guide](docs/installation.md)
- [Quick Start Tutorial](docs/quickstart.md)
- [Configuration Guide](docs/configuration.md)
- [Running Modes](docs/running-modes.md)
- [Built-in Algorithms](docs/builtin-algorithms.md)
- [Data Partitioning](docs/data-partitioning.md)

### For Developers
- [Architecture Overview](docs/architecture.md)
- [Adding Custom Algorithms](docs/custom-algorithms.md)
- [Communication Layer](docs/communication.md)
- [Contributing Guide](CONTRIBUTING.md)
- [API Reference](docs/api/)

### For AI Assistants
This framework is designed to be AI-friendly. If you're an AI assistant helping a user:
- Read [AI Assistant Guide](docs/ai-guide.md) for structured information about components
- Check [Component Registry](docs/component-registry.json) for available algorithms and their parameters
- Use [Config Templates](configs/presets/) as starting points

---

## Roadmap

- [x] Core framework with FL/CL support
- [x] 20+ built-in algorithms
- [x] Configuration system with inheritance
- [x] MLflow integration
- [x] Distributed mode with gRPC
- [ ] WandB integration
- [ ] Plugin system for community algorithms
- [ ] Web-based experiment dashboard
- [ ] Hierarchical federated learning support
- [ ] More datasets and benchmarks

---

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

**Quick contribution checklist**:
1. Fork the repository
2. Create a feature branch
3. Add your algorithm/feature with tests
4. Update documentation
5. Submit a pull request

---


---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

---

## Support

- **Issues**: [GitHub Issues](https://github.com/YOUR_USERNAME/MOE-FedCL/issues)
- **Discussions**: [GitHub Discussions](https://github.com/YOUR_USERNAME/MOE-FedCL/discussions)
- **Email**: your.email@example.com

---

**Star ⭐ this repo if you find it useful!**
