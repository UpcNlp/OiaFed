# AI Guide

> **Purpose**: Structured reference for AI assistants (Claude, GPT, etc.) to help users with OiaFed.

---

## Framework Overview

**OiaFed** (One Framework for All Federation) is a unified federated learning framework supporting:

| Scenario | Abbreviation | Description |
|----------|--------------|-------------|
| Horizontal FL | HFL | Same features, different samples |
| Vertical FL | VFL | Different features, same samples |
| Federated Continual Learning | FCL | Learning new tasks over time |
| Personalized FL | PFL | Client-specific models |
| Federated Unlearning | FU | Removing data influence |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Federation Framework                      │
│              Trainer · Learner · Aggregator                  │
├─────────────────────────────────────────────────────────────┤
│                    Node Abstraction                          │
│                Node · Proxy · ProxyCollection                │
├─────────────────────────────────────────────────────────────┤
│                    Communication Layer                       │
│              Transport · Serializer · Interceptor            │
└─────────────────────────────────────────────────────────────┘
```

---

## Component Registry

### Aggregators

| Type ID | Description | Key Parameters |
|---------|-------------|----------------|
| `fedavg` | Weighted averaging | `weighted: bool` |
| `fedprox` | Proximal term | `mu: float (0.001-0.1)` |
| `scaffold` | Control variates | - |
| `fednova` | Normalized averaging | - |
| `fedadam` | Server Adam | `lr, beta1, beta2` |
| `fedyogi` | Server Yogi | `lr, beta1, beta2` |
| `fedbn` | Skip BatchNorm | - |
| `feddyn` | Dynamic regularization | `alpha: float` |
| `fedproto` | Prototype aggregation | - |

### Learners

| Type ID | Description | Key Parameters |
|---------|-------------|----------------|
| `default` | Standard SGD training | `batch_size, lr, optimizer` |
| `moon` | Contrastive learning | `temperature, mu` |
| `fedper` | Personal layers | `personal_layers: List[str]` |
| `fedrep` | Representation learning | `head_epochs, body_epochs` |
| `fedbabu` | Freeze body, finetune head | `finetune_epochs` |
| `fedproto` | Prototype learning | `proto_dim` |
| `target` | Task generator (FCL) | `distill_weight` |
| `fedweit` | Weight decomposition | `decompose_ratio` |

### Models

| Type ID | Description |
|---------|-------------|
| `simple_cnn` | Basic CNN |
| `cifar10_cnn` | CIFAR-10 optimized |
| `mnist_cnn` | MNIST optimized |
| `resnet18/34/50` | ResNet variants |
| `mlp` | Multi-layer perceptron |

### Datasets

| Type ID | Classes |
|---------|---------|
| `mnist` | 10 |
| `fmnist` | 10 |
| `cifar10` | 10 |
| `cifar100` | 100 |

### Partitioners

| Type ID | Parameters |
|---------|------------|
| `iid` | `seed` |
| `dirichlet` | `alpha (0.1-10.0), seed` |
| `label_skew` | `num_labels_per_client, seed` |
| `quantity_skew` | `imbalance_ratio, seed` |

---

## Configuration Template

```yaml
# Basic structure
exp_name: experiment_name
node_id: trainer
role: trainer

trainer:
  type: default
  args:
    max_rounds: 100
    local_epochs: 5
    client_fraction: 1.0

aggregator:
  type: fedavg
  args: {}

learner:
  type: default
  args:
    batch_size: 64
    lr: 0.01
    device: auto

model:
  type: cifar10_cnn
  args:
    num_classes: 10

datasets:
  - type: cifar10
    split: train
    partition:
      strategy: dirichlet
      num_partitions: 10
      config:
        alpha: 0.5

  - type: cifar10
    split: test
```

---

## Running Commands

```bash
# Serial mode (debugging)
python -m src.runner --config config.yaml --mode serial --num-clients 5

# Parallel mode (multi-process)
python -m src.runner --config config.yaml --mode parallel --num-clients 10

# Distributed mode (multi-machine)
python -m src.runner --config trainer.yaml  # on server
python -m src.runner --config learner.yaml  # on clients
```

---

## Algorithm Selection Guide

```
User wants to handle Non-IID data?
├── Yes, moderate Non-IID → FedProx (mu=0.01)
├── Yes, severe Non-IID → SCAFFOLD or MOON
├── Yes, feature distribution shift → FedBN
└── No, IID data → FedAvg

User wants personalization?
├── Yes, simple → FedPer
├── Yes, representation learning → FedRep
└── Yes, prototype-based → FedProto

User has continual learning scenario?
├── Yes, class-incremental → TARGET
├── Yes, task-incremental → FedWEIT
└── Yes, knowledge distillation → FedKNOW
```

---

## Parameter Guidelines

| Parameter | Range | Default | Notes |
|-----------|-------|---------|-------|
| `lr` | 0.001-0.1 | 0.01 | Decrease for Non-IID |
| `batch_size` | 32-256 | 64 | Smaller for limited memory |
| `local_epochs` | 1-10 | 5 | More epochs = more drift |
| `fedprox_mu` | 0.001-0.1 | 0.01 | Increase for severe Non-IID |
| `moon_temperature` | 0.1-1.0 | 0.5 | Lower = sharper contrast |
| `dirichlet_alpha` | 0.1-10.0 | 0.5 | Lower = more heterogeneous |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA OOM | Reduce `batch_size`, use `device: cpu` |
| Slow training | Use `mode: parallel`, reduce `num_clients` |
| Poor accuracy | Check `alpha`, try FedProx/SCAFFOLD |
| Connection timeout | Increase `timeout`, check firewall |
| Import error | Run `pip install -e .` |

---

## Key Files & Directories

```
src/
├── core/           # Trainer, Learner, Aggregator
├── comm/           # Node, Proxy, Transport
├── methods/        # Algorithm implementations
│   ├── aggregators/
│   ├── learners/
│   └── models/
├── infra/          # Tracker, Callback
└── config/         # Configuration loading

configs/            # Example configurations
docs/               # Documentation
```

---

## Common Tasks

### Create a simple FL experiment

```yaml
aggregator: { type: fedavg }
learner: { type: default, args: { lr: 0.01 } }
model: { type: cifar10_cnn }
datasets:
  - type: cifar10
    partition: { strategy: iid, num_partitions: 10 }
```

### Handle Non-IID data

```yaml
aggregator: { type: fedprox, args: { mu: 0.01 } }
partition: { strategy: dirichlet, config: { alpha: 0.3 } }
```

### Add personalization

```yaml
learner:
  type: moon
  args:
    temperature: 0.5
    mu: 1.0
```

### Enable experiment tracking

```yaml
tracker:
  backends:
    - type: mlflow
      experiment_name: my_experiment
```

---

## Best Practices for Assisting Users

1. **Ask about scenario**: HFL? PFL? FCL?
2. **Ask about data**: IID or Non-IID? How heterogeneous?
3. **Start simple**: FedAvg first, then add complexity
4. **Suggest configs**: Provide complete YAML examples
5. **Explain parameters**: Why this value for this scenario
6. **Mention alternatives**: If one algorithm doesn't work

---

*Framework Version: 0.1.0*
