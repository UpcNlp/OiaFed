"""
Simple FedAvg on MNIST Example

This is the simplest possible federated learning example using MOE-FedCL.
Run this to verify your installation and understand the basic workflow.

Usage:
    python examples/simple_fedavg.py

Requirements:
    - MOE-FedCL installed
    - PyTorch installed
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.core import FederatedSystem
from src.config import load_config


async def main():
    """Run a simple FedAvg experiment on MNIST."""

    print("=" * 60)
    print("MOE-FedCL: Simple FedAvg on MNIST Example")
    print("=" * 60)
    print()

    # Create inline config
    config_dict = {
        "role": "trainer",
        "node_id": "trainer",
        "listen": {"host": "localhost", "port": 50051},
        "trainer": {
            "type": "default",
            "args": {
                "max_rounds": 5,  # Just 5 rounds for quick demo
                "local_epochs": 3,
                "client_fraction": 1.0,
                "min_available_clients": 2,
                "fit_config": {"epochs": 3, "evaluate_after_fit": True},
                "eval_interval": 1,
            },
        },
        "aggregator": {"type": "fedavg", "args": {"weighted": True}},
        "model": {"type": "simple_cnn", "args": {"num_classes": 10}},
        "datasets": [
            {
                "type": "mnist",
                "split": "train",
                "args": {"data_dir": "./data", "download": True},
                "partition": {
                    "strategy": "iid",
                    "num_partitions": 5,
                    "config": {"seed": 42},
                },
            }
        ],
        "tracker": {
            "backends": [
                {"type": "loguru", "level": "INFO", "file": "./logs/simple_example.log"}
            ]
        },
    }

    print("Configuration:")
    print(f"  - Algorithm: FedAvg")
    print(f"  - Dataset: MNIST (IID partition)")
    print(f"  - Clients: 5")
    print(f"  - Rounds: 5")
    print(f"  - Mode: Serial (single process)")
    print()

    print("Starting federated learning...")
    print()

    try:
        # Create system from config dict
        system = FederatedSystem.from_dict(config_dict)

        # Initialize
        await system.initialize()

        # Run training
        await system.run()

        print()
        print("=" * 60)
        print("Training completed successfully! 🎉")
        print("=" * 60)
        print()
        print("Next steps:")
        print("  1. Check logs at: ./logs/simple_example.log")
        print("  2. View results in MLflow: mlflow ui --port 5000")
        print("  3. Try modifying the config above to experiment")
        print("  4. Read the documentation: docs/README.md")
        print()

    except Exception as e:
        print()
        print("=" * 60)
        print(f"Error occurred: {e}")
        print("=" * 60)
        print()
        print("Troubleshooting:")
        print("  - Ensure MOE-FedCL is installed: pip install -e .")
        print("  - Check CUDA is available if using GPU")
        print("  - See docs/getting-started/installation.md for help")
        raise

    finally:
        # Cleanup
        await system.stop()


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())
