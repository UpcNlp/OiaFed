# Quick Start Tutorial

Get started with MOE-FedCL in 5 minutes! This tutorial will walk you through running your first federated learning experiment.

## Prerequisites

- MOE-FedCL installed (see [Installation Guide](installation.md))
- Basic understanding of federated learning concepts

---

## Your First Experiment: FedAvg on MNIST

Let's run a simple FedAvg experiment on MNIST with 5 clients.

### Step 1: Create Configuration

Create a file `quickstart_config.yaml`:

```yaml
# quickstart_config.yaml
role: trainer
node_id: trainer

listen:
  host: localhost
  port: 50051

trainer:
  type: default
  args:
    max_rounds: 5           # Run 5 federated rounds
    local_epochs: 3         # Each client trains for 3 epochs per round
    client_fraction: 1.0    # Use all available clients
    min_available_clients: 2
    fit_config:
      epochs: 3
      evaluate_after_fit: true
    eval_interval: 1        # Evaluate every round

aggregator:
  type: fedavg             # Use FedAvg aggregation
  args:
    weighted: true         # Weight by dataset size

model:
  type: simple_cnn         # Simple CNN for MNIST
  args:
    num_classes: 10

datasets:
  - type: mnist
    split: train
    args:
      data_dir: ./data
      download: true       # Auto-download if needed
    partition:
      strategy: iid        # IID partition for simplicity
      num_partitions: 5
      config:
        seed: 42

tracker:
  backends:
    - type: loguru
      level: INFO
      file: ./logs/quickstart.log

    - type: mlflow
      tracking_uri: ./mlruns
      experiment_name: quickstart
```

### Step 2: Run the Experiment

```bash
# Serial mode (single process, good for debugging)
python -m src.core.system --config quickstart_config.yaml --mode serial --num-clients 5
```

You should see output like:

```
[INFO] Initializing FederatedSystem...
[INFO] Starting trainer node...
[INFO] Round 1/5 starting...
[INFO] Collecting results from 5 clients...
[INFO] Aggregating weights...
[INFO] Round 1/5 completed - Accuracy: 0.87
...
[INFO] Training completed!
```

### Step 3: View Results

#### Option A: Check Logs

```bash
cat logs/quickstart.log
```

#### Option B: Use MLflow UI

```bash
# Start MLflow UI
mlflow ui --port 5000

# Open http://localhost:5000 in your browser
# You'll see your experiment with metrics and parameters
```

---

## Understanding What Happened

Let's break down what just happened:

### 1. System Initialization

MOE-FedCL created:
- 1 **Trainer** node (server)
- 5 **Learner** nodes (clients)
- Each client got an IID partition of MNIST

### 2. Training Loop

For each of 5 rounds:
1. **Broadcast**: Trainer sends current model to all clients
2. **Local Training**: Each client trains for 3 epochs on its local data
3. **Upload**: Clients send updated weights back to trainer
4. **Aggregation**: Trainer aggregates weights using FedAvg
5. **Evaluation**: New global model is evaluated

### 3. Results Tracking

Metrics were logged to:
- `logs/quickstart.log` - Detailed text logs
- `mlruns/` - MLflow experiment tracking

---

## Experiment Variations

### Variation 1: Non-IID Data

Change the partition strategy to Dirichlet:

```yaml
partition:
  strategy: dirichlet
  num_partitions: 5
  config:
    alpha: 0.5  # Lower = more heterogeneous
    seed: 42
```

### Variation 2: Different Algorithm

Try FedProx instead of FedAvg:

```yaml
aggregator:
  type: fedprox
  args:
    mu: 0.01  # Proximal term
```

### Variation 3: More Rounds

Increase training rounds:

```yaml
trainer:
  args:
    max_rounds: 20  # More rounds
    eval_interval: 2  # Evaluate every 2 rounds
```

### Variation 4: Parallel Mode

Use multiple processes for better performance:

```bash
python -m src.core.system --config quickstart_config.yaml --mode parallel --num-clients 5
```

---

## Next Steps

### Explore More Datasets

Try CIFAR-10 instead of MNIST:

```yaml
model:
  type: resnet18
  args:
    num_classes: 10

datasets:
  - type: cifar10
    split: train
    args:
      data_dir: ./data
      download: true
```

### Try Other Algorithms

Available algorithms:
- `fedavg` - Basic averaging
- `fedprox` - With proximal term
- `scaffold` - Control variates
- `moon` - Contrastive learning
- `fedbn` - Local batch normalization

See [Built-in Algorithms](../user-guide/builtin-algorithms.md) for full list.

### Distributed Mode

To run across multiple machines, see [Distributed Tutorial](../tutorials/distributed-setup.md).

---

## Common Patterns

### Pattern 1: Using Configuration Inheritance

Create a base config and extend it:

```yaml
# base.yaml
trainer:
  args:
    max_rounds: 10
    local_epochs: 5

model:
  type: simple_cnn

# experiment.yaml
extend: base.yaml
trainer:
  args:
    max_rounds: 20  # Override
```

### Pattern 2: Multiple Datasets

Train on multiple datasets:

```yaml
datasets:
  - type: mnist
    split: train
  - type: mnist
    split: test  # For evaluation
```

### Pattern 3: Custom Callbacks

Add custom logging or checkpointing:

```yaml
callbacks:
  - type: checkpoint
    args:
      save_dir: ./checkpoints
      save_every: 5  # Save every 5 rounds
```

---

## Troubleshooting

### Issue: "Port already in use"

**Solution**: Change the port in config:
```yaml
listen:
  port: 50052  # Use a different port
```

### Issue: "CUDA out of memory"

**Solution**: Reduce batch size or use CPU:
```yaml
learner:
  args:
    batch_size: 32  # Smaller batch size
    device: cpu     # Or use CPU
```

### Issue: "Dataset not found"

**Solution**: Ensure download is enabled:
```yaml
datasets:
  - type: mnist
    args:
      download: true  # Auto-download
```

---

## What's Next?

Now that you've run your first experiment:

1. 📚 **Learn Core Concepts**: [Core Concepts Guide](concepts.md)
2. ⚙️ **Master Configuration**: [Configuration Guide](../user-guide/configuration.md)
3. 🎓 **Deep Dive Tutorials**: [Tutorials](../tutorials/)
4. 🔧 **Add Custom Algorithms**: [Custom Algorithm Tutorial](../tutorials/custom-algorithm.md)

---

## Example Code

If you prefer Python code over YAML:

```python
import asyncio
from src.core import FederatedSystem
from src.config import load_config

async def main():
    # Load config
    config = load_config("quickstart_config.yaml")

    # Create system
    system = FederatedSystem(config)

    # Initialize
    await system.initialize()

    # Run training
    await system.run()

    # Cleanup
    await system.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

---

**Congratulations! 🎉** You've completed your first federated learning experiment with MOE-FedCL!
