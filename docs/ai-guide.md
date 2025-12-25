# AI Assistant Guide

**For AI Assistants (Claude, GPT, and other LLMs)**

This guide provides structured information about MOE-FedCL to help AI assistants understand the framework and assist users effectively.

---

## Framework Overview

### Identity
- **Name**: MOE-FedCL (Modular and Extensible Federated Continual Learning)
- **Purpose**: Research framework for federated learning and continual learning
- **Architecture**: Two-layer (Communication + Federation)
- **Language**: Python 3.12+
- **Main Dependencies**: PyTorch, gRPC, MLflow, Loguru

### Core Strengths
1. **Modularity**: Clean separation between communication and federation logic
2. **Flexibility**: 3 running modes (serial/parallel/distributed), plug-and-play components
3. **Extensibility**: Registry system for easy addition of new algorithms
4. **Reproducibility**: YAML configs with inheritance, integrated experiment tracking
5. **Scenario Support**: FL, CL, Hybrid FL+CL, Custom scenarios

---

## Component Registry

### 1. Aggregators (Server-side weight combination)

| Name | Type Identifier | Key Parameters | Use Case | Paper |
|------|----------------|----------------|----------|-------|
| **FedAvg** | `fedavg` | `weighted: bool` | Baseline FL | [McMahan+ 2017](https://arxiv.org/abs/1602.05629) |
| **FedProx** | `fedprox` | `mu: float` (0.01) | Heterogeneous clients | [Li+ 2020](https://arxiv.org/abs/1812.06127) |
| **SCAFFOLD** | `scaffold` | - | Reduce client drift | [Karimireddy+ 2020](https://arxiv.org/abs/1910.06378) |
| **FedNova** | `fednova` | - | Normalized averaging | [Wang+ 2020](https://arxiv.org/abs/2007.07481) |
| **FedAdam** | `fedadam` | `lr: float`, `beta1/2: float` | Adaptive optimization | [Reddi+ 2021](https://arxiv.org/abs/2003.00295) |
| **FedYogi** | `fedyogi` | `lr: float`, `beta1/2: float` | Adaptive optimization | [Reddi+ 2021](https://arxiv.org/abs/2003.00295) |
| **FedAdaGrad** | `fedadagrad` | `lr: float` | Adaptive optimization | [Reddi+ 2021](https://arxiv.org/abs/2003.00295) |
| **FedBN** | `fedbn` | - | Skip batch norm layers | [Li+ 2021](https://arxiv.org/abs/2102.07623) |

**Config Example**:
```yaml
aggregator:
  type: fedprox
  args:
    mu: 0.01
```

### 2. Learners (Client-side training logic)

#### Federated Learning Learners

| Name | Type Identifier | Key Parameters | Specialization |
|------|----------------|----------------|----------------|
| **Generic** | `default` or `generic` | `batch_size`, `lr`, `optimizer` | Standard FL |
| **FedPer** | `fedper` | `personal_layers: list` | Personalized layers |
| **FedRep** | `fedrep` | `head_epochs: int` | Representation learning |
| **FedBABU** | `fedbabu` | `finetune_epochs: int` | Body freezing |
| **FedRod** | `fedrod` | `hyper_dim: int` | Hypernet for personalization |
| **FedProto** | `fedproto` | `proto_dim: int` | Prototype-based |
| **GPFL** | `gpfl` | `group_num: int` | Group personalization |
| **MOON** | `moon` | `temperature: float`, `mu: float` | Model contrastive |
| **FedDis** | `feddistill` | `distill_alpha: float` | Knowledge distillation |

#### Continual Learning Learners

| Name | Type Identifier | Key Parameters | Strategy |
|------|----------------|----------------|----------|
| **TARGET** | `target` | `generator_type: str`, `synthesizer_type: str` | Task-specific generators |
| **FedWEIT** | `fedweit` | `weight_decay: float` | Weight decomposition |
| **FedKNOW** | `fedknow` | `distill_weight: float` | Knowledge distillation |
| **FedCPrompt** | `fedcprompt` | `prompt_length: int` | Prompt-based |
| **GLFC** | `glfc` | `local_coef: float`, `global_coef: float` | Feature composition |
| **LGA** | `lga` | `adapter_dim: int` | Lightweight adapter |

**Config Example**:
```yaml
learner:
  type: moon
  args:
    batch_size: 64
    lr: 0.01
    temperature: 0.5
    mu: 1.0
    device: cuda
```

### 3. Models (Neural Networks)

| Name | Type Identifier | Parameters | Use Case |
|------|----------------|------------|----------|
| **Simple CNN** | `simple_cnn` | `num_classes: int` | MNIST, Fashion-MNIST |
| **ResNet-18** | `resnet18` | `num_classes: int`, `pretrained: bool` | CIFAR, ImageNet |
| **ResNet-34** | `resnet34` | `num_classes: int`, `pretrained: bool` | CIFAR, ImageNet |
| **ResNet-50** | `resnet50` | `num_classes: int`, `pretrained: bool` | CIFAR, ImageNet |
| **MLP** | `mlp` | `input_size: int`, `hidden_sizes: list`, `num_classes: int` | Tabular data |
| **LeNet** | `lenet` | `num_classes: int` | MNIST variants |

**Config Example**:
```yaml
model:
  type: resnet18
  args:
    num_classes: 10
    pretrained: false
```

### 4. Datasets

| Name | Type Identifier | Classes | Size | Auto-download |
|------|----------------|---------|------|---------------|
| **MNIST** | `mnist` | 10 | 60K train, 10K test | ✅ |
| **Fashion-MNIST** | `fmnist` | 10 | 60K train, 10K test | ✅ |
| **CIFAR-10** | `cifar10` | 10 | 50K train, 10K test | ✅ |
| **CIFAR-100** | `cifar100` | 100 | 50K train, 10K test | ✅ |
| **SVHN** | `svhn` | 10 | 73K train, 26K test | ✅ |
| **EMNIST** | `emnist` | 62 | 697K train, 116K test | ✅ |
| **FEMNIST** | `femnist` | 62 | Varies by client | ⚠️ Manual |
| **CINIC-10** | `cinic10` | 10 | 270K total | ⚠️ Manual |

**Config Example**:
```yaml
datasets:
  - type: cifar10
    split: train
    args:
      data_dir: ./data
      download: true
```

### 5. Partition Strategies (Data Heterogeneity)

| Strategy | Identifier | Key Parameters | Description |
|----------|-----------|----------------|-------------|
| **IID** | `iid` | `num_partitions: int` | Uniform random split |
| **Dirichlet** | `dirichlet` | `alpha: float`, `num_partitions: int` | Label distribution skew (lower α = more skew) |
| **Label Skew** | `label_skew` | `num_labels_per_client: int` | Each client gets subset of labels |
| **Quantity Skew** | `quantity_skew` | `imbalance_ratio: float` | Unequal dataset sizes |

**Config Example**:
```yaml
partition:
  strategy: dirichlet
  num_partitions: 10
  config:
    alpha: 0.5  # 0.1 = very heterogeneous, 10.0 = nearly IID
    seed: 42
```

---

## Common Patterns and Templates

### Pattern 1: Basic FedAvg Experiment

```yaml
role: trainer
node_id: trainer

listen:
  host: localhost
  port: 50051

trainer:
  type: default
  args:
    max_rounds: 50
    local_epochs: 5
    client_fraction: 1.0

aggregator:
  type: fedavg
  args:
    weighted: true

model:
  type: simple_cnn
  args:
    num_classes: 10

datasets:
  - type: mnist
    split: train
    args:
      data_dir: ./data
      download: true
    partition:
      strategy: iid
      num_partitions: 10
```

### Pattern 2: Non-IID with FedProx

```yaml
aggregator:
  type: fedprox
  args:
    mu: 0.01  # Proximal term (0.001-0.1 typical)

partition:
  strategy: dirichlet
  config:
    alpha: 0.5  # Heterogeneity level
```

### Pattern 3: Configuration Inheritance

```yaml
# base.yaml
trainer:
  args:
    max_rounds: 100
    local_epochs: 5

# experiment.yaml
extend: base.yaml
trainer:
  args:
    max_rounds: 50  # Override only what changes
```

### Pattern 4: Multi-Dataset (Train + Test)

```yaml
datasets:
  - type: mnist
    split: train
    partition:
      strategy: dirichlet
      num_partitions: 10
      config:
        alpha: 0.5
      partition_id: 0  # For specific client

  - type: mnist
    split: test  # For evaluation
```

### Pattern 5: Experiment Tracking

```yaml
tracker:
  backends:
    - type: loguru
      level: INFO
      file: ./logs/experiment.log

    - type: mlflow
      tracking_uri: ./mlruns
      experiment_name: my_experiment
      run_name: fedavg_mnist_iid
```

---

## Parameter Guidelines

### Hyperparameter Ranges

| Parameter | Typical Range | Default | Notes |
|-----------|--------------|---------|-------|
| `learning_rate` | 0.001 - 0.1 | 0.01 | Lower for larger models |
| `batch_size` | 32 - 256 | 64 | Higher for more data/GPU |
| `local_epochs` | 1 - 10 | 5 | Tradeoff: accuracy vs communication |
| `max_rounds` | 50 - 500 | 100 | Dataset dependent |
| `client_fraction` | 0.1 - 1.0 | 1.0 | Fraction to sample per round |
| `dirichlet_alpha` | 0.1 - 10.0 | 0.5 | Lower = more heterogeneous |
| `fedprox_mu` | 0.001 - 0.1 | 0.01 | Higher = more regularization |
| `moon_temperature` | 0.1 - 1.0 | 0.5 | Contrastive learning |

### Resource Recommendations

| Scenario | Clients | GPUs | RAM | Mode |
|----------|---------|------|-----|------|
| Quick test | 5 | 0-1 | 8GB | serial |
| Development | 10 | 1 | 16GB | parallel |
| Small experiment | 20 | 1-2 | 32GB | parallel |
| Large experiment | 50+ | 2-4 | 64GB+ | parallel/distributed |
| Production | 100+ | Multiple machines | - | distributed |

---

## Running Commands

### Serial Mode (Debugging)
```bash
python -m src.core.system --config config.yaml --mode serial --num-clients 5
```

### Parallel Mode (Single Machine)
```bash
python -m src.core.system --config config.yaml --mode parallel --num-clients 10
```

### Distributed Mode (Multi-Machine)

**Server**:
```bash
python -m src.core.system --config server_config.yaml
```

**Each Client**:
```bash
python -m src.core.system --config client_N_config.yaml
```

### Batch Experiments
```bash
# Run multiple experiments from directory
python scripts/run_experiments.py --config-dir configs/table3_experiments

# Run specific dataset/algorithm
python scripts/run_experiments.py \
    --config-dir configs/table3_experiments \
    --dataset mnist \
    --algorithm fedavg
```

---

## Troubleshooting Decision Tree

When a user reports an issue, follow this decision tree:

```
Issue Type?
├─ Installation
│  ├─ PyTorch → Check CUDA version, suggest matching PyTorch wheel
│  ├─ gRPC → Install build tools, reinstall grpcio
│  └─ Other → Check Python version (need 3.12+)
│
├─ Configuration
│  ├─ Syntax Error → Validate YAML syntax
│  ├─ Unknown Type → Check component registry, suggest alternatives
│  └─ Inheritance Issue → Verify extend path, check merge logic
│
├─ Runtime
│  ├─ CUDA OOM → Reduce batch_size, suggest smaller model, or use CPU
│  ├─ Port in use → Change listen.port in config
│  ├─ Connection timeout → Check network, firewall, increase default_timeout
│  └─ Slow training → Suggest parallel mode, check GPU utilization
│
└─ Results
   ├─ Poor accuracy → Check hyperparameters, data partition, increase rounds
   ├─ No convergence → Lower learning rate, try different aggregator
   └─ Client divergence → Use FedProx with mu parameter
```

---

## Key Files and Directories

| Path | Purpose |
|------|---------|
| `src/core/` | Core abstractions (System, Node, Trainer, Learner) |
| `src/comm/` | Communication layer (Node, Transport, Serialization) |
| `src/methods/` | Algorithms (aggregators/, learners/, models/) |
| `src/data/` | Dataset management and partitioning |
| `src/infra/` | Infrastructure (tracker, logging, checkpoint) |
| `src/config/` | Configuration loading and validation |
| `src/callback/` | Callback system |
| `configs/` | Example configurations |
| `configs/table3_experiments/` | 288 paper reproduction configs |
| `configs/presets/` | Pre-configured algorithm presets |
| `examples.bak/` | Example scripts |
| `docs/` | Documentation |

---

## Assisting Users: Best Practices

### When User Wants to...

#### 1. **Run a basic experiment**
→ Point to [Quick Start](getting-started/quickstart.md) or provide a minimal config template

#### 2. **Reproduce a paper**
→ Check if algorithm is in the registry (see Component Registry above)
→ Provide config using correct type identifier
→ Check `configs/table3_experiments/` for existing configs

#### 3. **Use Non-IID data**
→ Suggest `dirichlet` partition with α=0.5 as starting point
→ Explain: lower α = more heterogeneous

#### 4. **Improve accuracy**
→ Check: learning rate, number of rounds, local epochs
→ Try: FedProx (for heterogeneity), MOON (for contrastive), SCAFFOLD (for drift)

#### 5. **Scale to more clients**
→ Suggest parallel mode for same machine
→ Suggest distributed mode for multiple machines
→ Warn about resource requirements

#### 6. **Add custom algorithm**
→ Provide decorator pattern example
→ Point to existing learner code in `src/methods/learners/`
→ Explain registry system

#### 7. **Debug issues**
→ Suggest serial mode for easier debugging
→ Check logs in `logs/` directory
→ Suggest enabling verbose logging

---

## Quick Reference: Configuration Cheatsheet

```yaml
# Minimal trainer config
role: trainer
node_id: trainer
listen: {host: localhost, port: 50051}
trainer: {type: default, args: {max_rounds: 50, local_epochs: 5}}
aggregator: {type: fedavg}
model: {type: simple_cnn, args: {num_classes: 10}}
datasets: [{type: mnist, split: train, args: {download: true}}]

# Minimal learner config
role: learner
node_id: learner_0
listen: {host: localhost, port: 50052}
connect_to: ["trainer@localhost:50051"]
learner: {type: default, args: {batch_size: 64, lr: 0.01}}
model: {type: simple_cnn, args: {num_classes: 10}}
datasets: [{type: mnist, split: train, partition: {strategy: iid, num_partitions: 10, partition_id: 0}}]
```

---

## Algorithm Selection Guide

Help users choose the right algorithm:

| User Need | Recommended Algorithm | Why |
|-----------|----------------------|-----|
| Baseline / Getting started | FedAvg | Simple, well-understood |
| Heterogeneous clients | FedProx | Proximal term handles drift |
| Non-IID data | SCAFFOLD or MOON | Reduce client drift |
| Personalization | FedPer, FedRep | Local layers for each client |
| Limited communication | FedNova | Better with variable local steps |
| Continual learning | TARGET, GLFC | Designed for task sequences |
| New to FL research | FedAvg → FedProx → MOON | Progressive complexity |

---

## Helpful Responses Templates

### When user asks "How do I..."

**Template**:
```
To [user's goal], you can:

1. [Step-by-step instructions]
2. [Config example]
3. [Command to run]

Here's a complete example:
[Full working code/config]

Related documentation: [link]
```

### When user reports error

**Template**:
```
This error typically occurs because [reason].

To fix:
1. [Solution step 1]
2. [Solution step 2]

Try this config:
[Fixed config snippet]

If the issue persists, check [relevant doc link].
```

---

## Version and Compatibility

- **Python**: 3.12+ required
- **PyTorch**: 2.7+ recommended (check CUDA compatibility)
- **gRPC**: Included in dependencies
- **MLflow**: Included, no extra setup needed

---

**Last Updated**: 2025-12-25

*This guide is specifically designed for AI assistants. For human users, see the main [Documentation](README.md).*
