#!/usr/bin/env python3
"""
FedCL 默认配置生成器

当找不到配置文件时，提供默认配置
"""

from pathlib import Path
from typing import Dict, Any
from .config_manager import DictConfig


def get_default_server_config() -> Dict[str, Any]:
    """获取默认服务器配置"""
    return {
        "server": {
            "id": "default_fedcl_server",
            "name": "Default FedCL Server",
            "type": "improved",
            "version": "1.0"
        },
        "communication": {
            "host": "localhost",
            "port": 8080,
            "max_workers": 5,
            "timeout": 60.0,
            "heartbeat_interval": 30
        },
        "federated_config": {
            "rounds": 3,
            "min_clients": 1,
            "max_clients": 3,
            "client_selection": "all"
        },
        "aggregator": {
            "class": "fedavg",
            "type": "fedavg",
            "strategy": "weighted_average",
            "aggregation_params": {
                "weight_by_samples": True,
                "min_participation": 0.8
            }
        },
        "global_model": {
            "type": "SimpleMLP",
            "input_size": 784,
            "hidden_sizes": [256, 128],
            "num_classes": 10,
            "dropout_rate": 0.2,
            "activation": "relu"
        },
        "evaluation": {
            "frequency": 1,
            "metrics": ["accuracy", "loss"],
            "test_data": {
                "dataset": "MNIST",
                "path": "data/MNIST",
                "split": "test",
                "batch_size": 64
            }
        },
        "system": {
            "device": "cpu",
            "random_seed": 42,
            "num_threads": 4
        },
        "logging": {
            "level": "INFO",
            "save_logs": True,
            "log_frequency": 1,
            "log_dir": "logs/default_server"
        },
        "experiment": {
            "name": "default_fedcl_experiment",
            "description": "Default MNIST federated learning experiment",
            "output_dir": "experiments/default_experiment"
        }
    }


def get_default_client_config(client_id: int = 1) -> Dict[str, Any]:
    """获取默认客户端配置"""
    return {
        "client": {
            "id": f"default_client_{client_id}",
            "name": f"Default Client {client_id}",
            "type": "multi_learner",
            "version": "1.0"
        },
        "communication": {
            "server_host": "localhost",
            "server_port": 8080,
            "timeout": 30.0,
            "heartbeat_interval": 15,
            "mode": "pseudo_federation"
        },
        "learners": {
            "default_learner": {
                "class": "default",
                "enabled": True,
                "scheduler": None,  # 明确设置为None，避免自动生成scheduler_id
                "model": {
                    "type": "SimpleMLP",
                    "input_size": 784,
                    "hidden_sizes": [256, 128],  # 保持与服务器一致
                    "num_classes": 10,
                    "dropout_rate": 0.2,  # 保持与服务器一致
                    "activation": "relu"
                },
                "optimizer": {
                    "type": "SGD",
                    "lr": 0.01,
                    "momentum": 0.9,
                    "weight_decay": 5e-4
                },
                "loss_function": {
                    "type": "CrossEntropyLoss"
                },
                "training": {
                    "local_epochs": 3,
                    "batch_size": 32
                }
            }
        },
        "dataloaders": {
            "default": {
                "type": "default",  # 使用默认类型而不是MNISTDataLoader
                "dataset": {
                    "name": "MNIST",
                    "path": "data/MNIST",
                    "split": "train",
                    "download": True,
                    "transform": "basic"
                },
                "federated_config": {
                    "client_id": client_id,
                    "num_clients": 3,
                    "distribution": "iid",
                    "samples_per_client": 1000
                },
                "loader_params": {
                    "batch_size": 32,
                    "shuffle": True,
                    "num_workers": 0
                }
            }
        },
        "system": {
            "device": "cpu",
            "random_seed": 42,
            "num_threads": 2
        },
        "logging": {
            "level": "INFO",
            "save_logs": True,
            "log_frequency": 1,
            "log_dir": f"logs/default_client_{client_id}"
        }
    }


def create_default_config_directory(config_dir: Path, num_clients: int = 3) -> None:
    """创建默认配置目录"""
    config_dir = Path(config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建服务器配置
    server_config_path = config_dir / "server_config.yaml"
    if not server_config_path.exists():
        import yaml
        with open(server_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(get_default_server_config(), f, default_flow_style=False, allow_unicode=True)
    
    # 创建客户端配置目录
    client_dir = config_dir / "client"
    client_dir.mkdir(exist_ok=True)
    
    # 创建客户端配置文件
    for i in range(1, num_clients + 1):
        client_config_path = client_dir / f"client_{i}_config.yaml"
        if not client_config_path.exists():
            with open(client_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(get_default_client_config(i), f, default_flow_style=False, allow_unicode=True)


def get_default_single_config() -> DictConfig:
    """获取单文件模式的默认配置"""
    base_config = get_default_server_config()
    
    # 添加客户端信息
    base_config["clients"] = [
        get_default_client_config(i) for i in range(1, 4)
    ]
    
    return DictConfig(base_config)


def get_fallback_config_for_path(config_path: Path) -> DictConfig:
    """根据路径获取备用配置"""
    if config_path.is_dir() or config_path.name.endswith('_configs') or 'demo' in str(config_path):
        # 目录模式 - 创建默认配置目录
        create_default_config_directory(config_path)
        
        # 返回基本的目录配置
        return DictConfig({
            "client_count": 3,
            "config_mode": "directory",
            "_config_files": {
                "client": [
                    str(config_path / "client" / f"client_{i}_config.yaml") 
                    for i in range(1, 4)
                ]
            },
            "experiment": {
                "name": "default_experiment",
                "description": "Auto-generated default experiment"
            }
        })
    else:
        # 单文件模式
        return get_default_single_config()
