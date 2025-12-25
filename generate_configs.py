#!/usr/bin/env python
"""
Configuration generator for paper implementations

Usage:
    python generate_configs.py fedper --num-learners 4 --dataset cifar10
    python generate_configs.py moon --num-learners 8 --dataset cifar100
    python generate_configs.py --list  # List registered methods
"""

import argparse
from pathlib import Path
import yaml
import sys

# Add src to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import registry to check registered components
from src.registry import registry
# Import models to ensure they are registered
import src.methods.models


def list_registered_methods():
    """List all registered methods from the registry"""
    print("Registered Learners:")
    for name in sorted(registry.list('learner.')):
        print(f"  - {name.replace('learner.', '')}")

    print("\nRegistered Models:")
    for name in sorted(registry.list('model.')):
        print(f"  - {name.replace('model.', '')}")

    print("\nRegistered Datasets:")
    for name in sorted(registry.list('dataset.')):
        print(f"  - {name.replace('dataset.', '')}")


def generate_paper_config(
    paper: str,
    num_learners: int = 4,
    dataset: str = "cifar10",
    num_rounds: int = 100,
    local_epochs: int = 3,  # 默认3个epoch
    mlflow_uri: str = "http://172.19.138.200:5000",
    mlflow_user: str = "dongshou",
    mlflow_pass: str = "admin123admin",
):
    """Generate config files for a paper implementation"""

    config_dir = Path(f"configs/papers/{paper}")
    config_dir.mkdir(parents=True, exist_ok=True)

    # Determine num_classes based on dataset
    num_classes_map = {
        "mnist": 10,
        "cifar10": 10,
        "cifar100": 100,
        "tinyimagenet": 200,
    }
    num_classes = num_classes_map.get(dataset, 10)

    # Base config
    base_config = {
        "transport": {
            "mode": "grpc",
            "grpc": {
                "max_workers": 10,
                "max_message_length": 104857600,
                "dual_thread": {"enabled": True},
                "heartbeat": {
                    "enabled": True,
                    "interval": 5.0,
                    "timeout": 30.0,
                    "check_interval": 10.0,
                },
            },
        },
        "serialization": {"default": "pickle"},
        "logging": {
            "level": "DEBUG",
            "log_dir": "./logs",
            "exp_name": f"{paper}_{dataset}",
            "console": True,
            "console_level": "INFO",
            "diagnose": False,
            "rotation": "10 MB",
            "retention": "30 days",
            "compression": "zip",
        },
        "tracker": {
            "enabled": True,
            "tracking_dir": "tracking",
            "backends": [
                {
                    "type": "mlflow",
                    "args": {
                        "tracking_uri": mlflow_uri,
                        "experiment_name": f"{paper.upper()}-{dataset.upper()}",
                        "username": mlflow_user,
                        "password": mlflow_pass,
                    },
                }
            ],
        },
        "callbacks": [
            {
                "type": "logging",
                "args": {"log_epoch": True, "log_round": True, "log_fit": False},
            }
        ],
        "default_timeout": 60.0,
    }

    # Write base.yaml
    with open(config_dir / "base.yaml", "w") as f:
        yaml.dump(base_config, f, default_flow_style=False, sort_keys=False)

    # Trainer config
    trainer_config = {
        "extend": "base.yaml",
        "node_id": "trainer",
        "role": "trainer",
        "listen": {"host": "localhost", "port": 50051},
        "trainer": {
            "type": "default",  # Use default trainer or specify custom trainer type
            "args": {
                "num_rounds": num_rounds,
                "min_clients": num_learners,
                "sample_fraction": 1.0,
                "min_sample_size": 1,
                "min_num_clients": 1,
                "local_epochs": local_epochs,
                "fit_config": {
                    "epochs": local_epochs,
                    "evaluate_after_fit": True,  # 训练后评估
                },
                "eval_interval": 1,  # 每轮都评估
                "evaluate_after_aggregation": False,
            },
        },
        "aggregator": {"type": "fedavg", "args": {"weighted": True}},
        "model": {"type": "simple_cnn", "args": {"num_classes": num_classes}},
    }

    with open(config_dir / "trainer.yaml", "w") as f:
        yaml.dump(trainer_config, f, default_flow_style=False, sort_keys=False)

    # Learner configs
    for i in range(num_learners):
        learner_config = {
            "extend": "base.yaml",
            "node_id": f"learner_{i}",
            "role": "learner",
            "listen": {"host": "localhost", "port": 50059 + i},
            "connect_to": ["trainer@localhost:50051"],
            "learner": {
                "type": "default",  # Use default learner or specify custom learner type
                "args": {
                    "batch_size": 32,
                    "lr": 0.01,
                    "device": f"cuda:{i % 2}",  # Alternate between cuda:0 and cuda:1
                    "local_epochs": local_epochs,  # 使用参数中的 local_epochs
                },
            },
            "datasets": [  # Changed from dataset to datasets (list format)
                {
                    "type": dataset,
                    "split": "train",
                    "args": {
                        "data_dir": "./data",
                        "download": True,
                    },
                    "partition": {
                        "strategy": "dirichlet",
                        "num_partitions": num_learners,
                        "config": {
                            "alpha": 0.5,
                            "seed": 42,
                        },
                        "partition_id": i,
                    },
                }
            ],
            "model": {
                "type": "simple_cnn",
                "args": {"num_classes": num_classes},
            },
        }

        with open(config_dir / f"learner_{i}.yaml", "w") as f:
            yaml.dump(learner_config, f, default_flow_style=False, sort_keys=False)

    print(f"✅ Generated configs for {paper} in {config_dir}")
    print(f"   - 1 trainer, {num_learners} learners")
    print(f"   - Dataset: {dataset}")
    print(f"   - Rounds: {num_rounds}")
    print(f"   - Local epochs: {local_epochs}")
    print(f"\nTo run the experiment:")
    print(f"   python run_paper.py {paper}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate configuration files for paper implementations"
    )
    parser.add_argument("paper", nargs="?", help="Paper name (e.g., fedper, moon)")
    parser.add_argument(
        "--num-learners", type=int, default=4, help="Number of learners (default: 4)"
    )
    parser.add_argument(
        "--dataset", default="cifar10", help="Dataset name (default: cifar10)"
    )
    parser.add_argument(
        "--num-rounds", type=int, default=100, help="Number of rounds (default: 100)"
    )
    parser.add_argument(
        "--local-epochs", type=int, default=3, help="Local epochs per round (default: 3)"
    )
    parser.add_argument(
        "--mlflow-uri",
        default="http://172.19.138.200:5000",
        help="MLflow tracking URI",
    )
    parser.add_argument(
        "--mlflow-user", default="dongshou", help="MLflow username"
    )
    parser.add_argument(
        "--mlflow-pass", default="admin123admin", help="MLflow password"
    )
    parser.add_argument(
        "--list", "-l", action="store_true", help="List registered methods"
    )

    args = parser.parse_args()

    if args.list:
        list_registered_methods()
        sys.exit(0)

    if not args.paper:
        parser.print_help()
        print("\nUse --list to see available methods")
        sys.exit(1)

    generate_paper_config(
        args.paper,
        args.num_learners,
        args.dataset,
        args.num_rounds,
        args.local_epochs,
        args.mlflow_uri,
        args.mlflow_user,
        args.mlflow_pass,
    )
