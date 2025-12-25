# Core Concepts

This guide introduces the fundamental concepts and components of MOE-FedCL.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Key Components](#key-components)
- [Running Modes](#running-modes)
- [Configuration System](#configuration-system)
- [Data Flow](#data-flow)
- [Learning Scenarios](#learning-scenarios)

---

## Architecture Overview

MOE-FedCL follows a **two-layer architecture**:

```
┌──────────────────────────────────────────────┐
│     Application Layer (Federation)           │
│  ┌──────────┐  ┌──────────┐  ┌────────────┐ │
│  │ Trainer  │  │ Learner  │  │ Aggregator │ │
│  └──────────┘  └──────────┘  └────────────┘ │
├──────────────────────────────────────────────┤
│     Communication Layer (node_comm)          │
│  ┌──────┐  ┌─────────────┐  ┌─────────────┐ │
│  │ Node │  │ Serializer  │  │ Interceptor │ │
│  └──────┘  └─────────────┘  └─────────────┘ │
├──────────────────────────────────────────────┤
│     Transport Layer                          │
│  ┌────────┐  ┌──────┐  ┌─────────────┐     │
│  │ Memory │  │ gRPC │  │ Custom...   │     │
│  └────────┘  └──────┘  └─────────────┘     │
└──────────────────────────────────────────────┘
```

### Design Principles

1. **Separation of Concerns**: Communication layer is independent of federation logic
2. **Modularity**: Components are loosely coupled and easily replaceable
3. **Extensibility**: Add new algorithms, datasets, or transport modes without modifying core
4. **Configuration-Driven**: Behavior controlled via YAML, not code changes

---

## Key Components

### 1. FederatedSystem

The top-level container that manages the entire federated learning system.

**Responsibilities**:
- Initialize and manage nodes (trainer/learner)
- Set up communication channels
- Manage callbacks and trackers
- Coordinate the training lifecycle

**Usage**:
```python
from src.core import FederatedSystem
from src.config import load_config

config = load_config("config.yaml")
system = FederatedSystem(config)
await system.run()
```

### 2. Trainer

The server-side component that orchestrates federated training.

**Responsibilities**:
- Broadcast global model to learners
- Collect updates from learners
- Trigger aggregation
- Evaluate global model
- Track training progress

**Key Methods**:
- `broadcast_weights()` - Send model to clients
- `collect_results()` - Gather client updates
- `aggregate()` - Combine updates
- `evaluate()` - Test global model

**Example**:
```yaml
trainer:
  type: default
  args:
    max_rounds: 100
    local_epochs: 5
    client_fraction: 0.8
    min_available_clients: 5
```

### 3. Learner

The client-side component that trains on local data.

**Responsibilities**:
- Receive global model from trainer
- Train on local dataset
- Send updates back to trainer
- Optionally evaluate locally

**Key Methods**:
- `fit()` - Train on local data
- `evaluate()` - Evaluate local model
- `get_weights()` - Return model parameters
- `set_weights()` - Load new parameters

**Example**:
```yaml
learner:
  type: default
  args:
    batch_size: 64
    lr: 0.01
    optimizer: adam
```

### 4. Aggregator

Combines updates from multiple learners into a new global model.

**Responsibilities**:
- Aggregate model weights/gradients
- Apply server-side optimization (if needed)
- Handle heterogeneous client updates

**Available Aggregators**:
- **FedAvg**: Weighted averaging
- **FedProx**: With proximal term
- **SCAFFOLD**: Control variates
- **FedAdam/FedYogi**: Adaptive optimization
- **FedBN**: Skip batch norm layers

**Example**:
```yaml
aggregator:
  type: fedavg
  args:
    weighted: true  # Weight by dataset size
```

### 5. Model

Neural network definition.

**Responsibilities**:
- Define network architecture
- Provide parameter access
- Handle forward pass

**Built-in Models**:
- `simple_cnn` - Basic CNN (MNIST/Fashion-MNIST)
- `resnet18/34/50` - ResNet variants (CIFAR/ImageNet)
- `mlp` - Multi-layer perceptron

**Example**:
```yaml
model:
  type: resnet18
  args:
    num_classes: 10
    pretrained: false
```

### 6. Dataset & Partitioner

Manages data loading and partitioning.

**Responsibilities**:
- Load datasets (MNIST, CIFAR, etc.)
- Partition data across clients
- Create data loaders

**Partition Strategies**:
- **IID**: Random uniform split
- **Dirichlet**: Label distribution skew (α parameter)
- **Label Skew**: Each client gets subset of labels
- **Quantity Skew**: Unequal dataset sizes

**Example**:
```yaml
datasets:
  - type: cifar10
    split: train
    partition:
      strategy: dirichlet
      num_partitions: 10
      config:
        alpha: 0.5  # Lower α = more heterogeneous
```

### 7. Communication Node

Low-level abstraction for message passing.

**Responsibilities**:
- Send/receive messages
- Manage connections
- Handle serialization
- Apply interceptors (logging, auth, etc.)

**Transparently handles**:
- Multiple transport modes (memory, gRPC)
- Retry logic
- Timeout handling
- Error propagation

---

## Running Modes

MOE-FedCL supports three running modes:

### 1. Serial Mode

**What it is**: Single-process simulation where all nodes run in the same process.

**When to use**:
- Debugging and development
- Quick experiments
- Limited resources

**Pros**:
- Fast to start
- Easy to debug
- No network overhead

**Cons**:
- No true parallelism
- Limited scalability

**Example**:
```bash
python -m src.core.system --config config.yaml --mode serial --num-clients 5
```

### 2. Parallel Mode

**What it is**: Multi-process execution where each node runs in a separate process on the same machine.

**When to use**:
- Single machine with multiple GPUs
- Production-quality experiments
- Performance critical

**Pros**:
- True parallelism
- Realistic resource usage
- Multi-GPU support

**Cons**:
- Slower startup
- More resource intensive

**Example**:
```bash
python -m src.core.system --config config.yaml --mode parallel --num-clients 10
```

### 3. Distributed Mode

**What it is**: True distributed deployment across multiple machines.

**When to use**:
- Large-scale experiments
- Multiple machines/clusters
- Real-world deployment

**Pros**:
- Unlimited scalability
- Real network conditions
- Production deployment

**Cons**:
- Complex setup
- Network dependencies
- Harder to debug

**Example**:
```bash
# On server machine
python -m src.core.system --config server_config.yaml

# On each client machine
python -m src.core.system --config client_config.yaml
```

---

## Configuration System

MOE-FedCL uses a powerful YAML-based configuration system with **inheritance**.

### Basic Structure

```yaml
# Required fields
node_id: unique_identifier
role: trainer  # or learner

# Communication
listen:
  host: localhost
  port: 50051

connect_to:
  - trainer@localhost:50051  # For learners

# Components
trainer: { }     # Trainer config (if role=trainer)
learner: { }     # Learner config (if role=learner)
aggregator: { }  # Aggregator config
model: { }       # Model config
datasets: [ ]    # Dataset configs

# Infrastructure
tracker: { }     # Experiment tracking
callbacks: [ ]   # Event callbacks
```

### Configuration Inheritance

Extend base configurations to avoid repetition:

```yaml
# base.yaml
trainer:
  args:
    max_rounds: 100
    local_epochs: 5

model:
  type: simple_cnn

# experiment1.yaml
extend: base.yaml
trainer:
  args:
    max_rounds: 50  # Override only what changes
```

**Inheritance rules**:
- Child configs override parent values
- Lists are replaced, not merged
- Deep merge for nested dictionaries

### Environment Variables

Use environment variables for sensitive data:

```yaml
tracker:
  backends:
    - type: mlflow
      tracking_uri: ${MLFLOW_TRACKING_URI}
```

---

## Data Flow

Understanding how data flows through the system:

### Training Round Flow

```
1. [Trainer] Broadcast global model weights
      ↓
2. [Learner 1..N] Receive weights
      ↓
3. [Learner 1..N] Train on local data
      ↓
4. [Learner 1..N] Send updates (weights/gradients)
      ↓
5. [Trainer] Collect all updates
      ↓
6. [Aggregator] Aggregate updates
      ↓
7. [Trainer] Update global model
      ↓
8. [Trainer] Evaluate global model
      ↓
9. [Tracker] Log metrics
      ↓
   Repeat for next round
```

### Message Types

- **Broadcast**: Trainer → All Learners (model weights)
- **Request-Response**: Trainer asks specific Learner to train
- **Notify**: One-way message (heartbeat, status)

---

## Learning Scenarios

MOE-FedCL supports multiple learning scenarios:

### 1. Federated Learning (FL)

Standard federated learning where clients collaboratively train a global model.

**Characteristics**:
- Fixed task across all rounds
- Data heterogeneity (Non-IID)
- Privacy-preserving

**Use cases**: Medical imaging, mobile keyboards, IoT

**Example**:
```yaml
trainer:
  type: default  # Standard FL trainer

aggregator:
  type: fedavg  # Or fedprox, scaffold, etc.
```

### 2. Continual Learning (CL)

Learning new tasks sequentially without forgetting previous tasks.

**Characteristics**:
- Multiple tasks over time
- Catastrophic forgetting prevention
- Knowledge transfer

**Use cases**: Lifelong learning, incremental classes

**Example**:
```yaml
trainer:
  type: continual

learner:
  type: target  # Or fedweit, fedknow, etc.
```

### 3. Hybrid FL + CL

Federated continual learning: distributed clients learning tasks sequentially.

**Characteristics**:
- Combines FL data heterogeneity with CL task evolution
- Most realistic scenario
- Most challenging

**Example**:
```yaml
trainer:
  type: continual

aggregator:
  type: fedavg

learner:
  type: glfc  # Global-local feature composition
```

### 4. Custom Scenarios

Extend for your own scenarios using the registry system.

---

## Registry System

MOE-FedCL uses a **registry pattern** for extensibility.

### How it Works

Components register themselves with decorators:

```python
from src.core import Learner, register

@register("learner", "my_custom")
class MyCustomLearner(Learner):
    def fit(self, config):
        # Your implementation
        pass
```

Then use in config:
```yaml
learner:
  type: my_custom
```

### Registered Component Types

- `trainer` - Training orchestrators
- `learner` - Client-side learners
- `aggregator` - Aggregation strategies
- `model` - Neural network architectures
- `dataset` - Data loaders
- `callback` - Event hooks
- `tracker` - Experiment loggers

---

## Event Callbacks

Callbacks provide hooks into the training lifecycle:

### Callback Lifecycle

```python
on_system_start()          # System initialization
  ↓
on_train_begin()           # Before training
  ↓
  for each round:
    on_round_begin()       # Before round
      ↓
    ... training ...
      ↓
    on_round_end()         # After round, before aggregation
      ↓
    on_aggregate_end()     # After aggregation
  ↓
on_train_end()             # After all rounds
  ↓
on_system_stop()           # System cleanup
```

### Built-in Callbacks

- **CheckpointCallback**: Save models periodically
- **EarlyStoppingCallback**: Stop on convergence
- **LoggingCallback**: Custom logging
- **MLflowCallback**: Automatic MLflow tracking

---

## Next Steps

Now that you understand the core concepts:

1. **Deep Dive**: [Configuration Guide](../user-guide/configuration.md) for mastering configs
2. **Practice**: [Tutorials](../tutorials/) for hands-on learning
3. **Extend**: [Custom Algorithm Tutorial](../tutorials/custom-algorithm.md) to add your own methods
4. **API**: [API Reference](../api-reference/) for detailed documentation

---

## Further Reading

- [Architecture Deep Dive](../architecture/overview.md)
- [Communication Layer Details](../architecture/communication.md)
- [Registry System Design](../architecture/registry.md)
