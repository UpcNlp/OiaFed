#!/usr/bin/env python3
"""
增强的实验配置记录工具
在每个实验开始时，保存完整的配置信息到日志目录
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


class ExperimentConfigLogger:
    """实验配置记录器"""

    @staticmethod
    def save_experiment_config(log_dir: Path, config: Dict[str, Any]):
        """
        保存实验配置到日志目录

        Args:
            log_dir: 实验日志目录 (e.g., logs/exp_20251117-10-00-00)
            config: 实验配置字典
        """
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # 提取关键配置信息
        experiment_config = {
            # 基本信息
            'experiment_id': log_dir.name,
            'timestamp': datetime.now().isoformat(),

            # 数据集配置
            'dataset': {
                'name': config.get('dataset', 'UNKNOWN'),
                'data_dir': str(config.get('data_dir', '')),
            },

            # 数据划分配置
            'data_partition': {
                'noniid_type': config.get('noniid_type', 'iid'),
                'alpha': config.get('alpha'),
                'num_clients': config.get('num_clients'),
                'samples_per_client': config.get('samples_per_client'),
            },

            # 联邦学习配置
            'federated_learning': {
                'algorithm': config.get('algorithm', 'FedAvg'),
                'aggregator': config.get('aggregator', 'FedAvgAggregator'),
                'num_rounds': config.get('num_rounds'),
                'clients_per_round': config.get('clients_per_round'),
                'local_epochs': config.get('local_epochs'),
            },

            # 模型配置
            'model': {
                'name': config.get('model_name'),
                'architecture': config.get('model_config', {}).get('name'),
                'params': config.get('model_config', {}).get('params', {}),
            },

            # 训练配置
            'training': {
                'learning_rate': config.get('learning_rate'),
                'batch_size': config.get('batch_size'),
                'optimizer': config.get('optimizer', 'SGD'),
                'loss_function': config.get('loss_fn', 'CrossEntropyLoss'),
            },

            # 通信配置
            'communication': {
                'mode': config.get('comm_mode', 'ProcessAndNetwork'),
                'backend': config.get('backend'),
            },

            # 其他配置
            'misc': {
                'seed': config.get('seed'),
                'device': config.get('device', 'cuda'),
                'early_stopping': config.get('early_stopping', True),
                'patience': config.get('patience', 5),
            }
        }

        # 只保存一个结构化的JSON配置文件
        config_file = log_dir / "experiment_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(experiment_config, f, indent=2, ensure_ascii=False)

        print(f"✓ 配置已保存到: {config_file}")

        return experiment_config


# 使用示例
if __name__ == "__main__":
    # 示例配置
    example_config = {
        'dataset': 'MNIST',
        'data_dir': 'data/',
        'noniid_type': 'label_skew',
        'alpha': 0.5,
        'num_clients': 100,
        'samples_per_client': 600,
        'algorithm': 'FedProx',
        'aggregator': 'FedProxAggregator',
        'num_rounds': 100,
        'clients_per_round': 10,
        'local_epochs': 5,
        'model_name': 'MNIST_LeNet',
        'model_config': {
            'name': 'MNIST_LeNet',
            'params': {'num_classes': 10}
        },
        'learning_rate': 0.01,
        'batch_size': 32,
        'optimizer': 'SGD',
        'seed': 42,
        'device': 'cuda:0',
    }

    # 保存配置
    log_dir = Path("logs/exp_example")
    ExperimentConfigLogger.save_experiment_config(log_dir, example_config)

    print("\n示例配置已生成，查看:")
    print(f"  cat {log_dir}/config_summary.txt")
    print(f"  cat {log_dir}/experiment_config.json")
