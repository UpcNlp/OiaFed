# MOE-FedCL Examples

This directory contains example scripts and configurations to help you get started with MOE-FedCL.

## Quick Start Examples

### 1. Simple FedAvg (`simple_fedavg.py`)

The simplest possible example - FedAvg on MNIST with 5 clients.

```bash
python examples/simple_fedavg.py
```

**What it demonstrates**:
- Creating a config programmatically
- Running federated learning in serial mode
- Basic trainer/learner setup

**Expected output**: ~97% accuracy after 5 rounds (takes ~2 minutes)

---

## Example Configurations

Pre-configured YAML files for common scenarios:

### `configs/examples/` (if you create them)

```
configs/examples/
├── fedavg_mnist_iid.yaml       # Baseline: IID data
├── fedavg_mnist_noniid.yaml    # Non-IID with Dirichlet
├── fedprox_cifar10.yaml        # FedProx on CIFAR-10
├── moon_heterogeneous.yaml     # MOON with strong heterogeneity
└── custom_learner.yaml         # Using a custom learner
```

---

## Running Examples

### From Command Line

```bash
# Using a YAML config
python -m src.core.system --config examples/config.yaml --mode serial --num-clients 5

# Parallel mode (better performance)
python -m src.core.system --config examples/config.yaml --mode parallel --num-clients 10
```

### From Python Code

```python
import asyncio
from src.core import FederatedSystem
from src.config import load_config

async def main():
    config = load_config("examples/config.yaml")
    system = FederatedSystem(config)
    await system.run()

asyncio.run(main())
```

---

## Example Progression

We recommend trying examples in this order:

1. **`simple_fedavg.py`** - Understand basics
2. **FedAvg with IID** - Standard setup
3. **FedAvg with Non-IID** - Real-world scenario
4. **FedProx** - Handling heterogeneity
5. **MOON** - Advanced contrastive learning
6. **Custom algorithm** - Extend the framework

---

## Modifying Examples

All examples are designed to be easily modified. Common modifications:

### Change the Dataset

```python
"datasets": [
    {
        "type": "cifar10",  # Change from mnist
        "split": "train",
        # ... rest stays same
    }
]
```

### Change the Algorithm

```python
"aggregator": {
    "type": "fedprox",  # Change from fedavg
    "args": {"mu": 0.01}
}
```

### Use Non-IID Data

```python
"partition": {
    "strategy": "dirichlet",  # Change from iid
    "config": {
        "alpha": 0.5  # Lower = more heterogeneous
    }
}
```

### Add More Clients

```python
"partition": {
    "num_partitions": 20  # Increase from 5
}
```

---

## Troubleshooting

### Example runs slowly
→ Switch to parallel mode: `--mode parallel`

### CUDA out of memory
→ Reduce batch size in config:
```python
"learner": {"args": {"batch_size": 32}}
```

### Port already in use
→ Change port in config:
```python
"listen": {"port": 50052}  # Use different port
```

---

## Advanced Examples

For more advanced usage, see:

- **`examples.bak/`** - Archived examples with more features
- **`configs/table3_experiments/`** - 288 paper reproduction configs
- **`configs/presets/`** - Algorithm presets

---

## Creating Your Own Example

Template for a custom example:

```python
import asyncio
from src.core import FederatedSystem

async def my_experiment():
    config = {
        "role": "trainer",
        "node_id": "trainer",
        # ... your config
    }

    system = FederatedSystem.from_dict(config)
    await system.initialize()
    await system.run()
    await system.stop()

if __name__ == "__main__":
    asyncio.run(my_experiment())
```

---

## Getting Help

- **Documentation**: [docs/README.md](../docs/README.md)
- **Quick Start**: [docs/getting-started/quickstart.md](../docs/getting-started/quickstart.md)
- **Issues**: [GitHub Issues](https://github.com/YOUR_USERNAME/MOE-FedCL/issues)

---

Happy experimenting! 🚀
