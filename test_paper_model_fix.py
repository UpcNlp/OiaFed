#!/usr/bin/env python
"""
测试论文标准模型修复
验证CIFAR10现在使用正确的PaperCNN模型而不是MNIST_LeNet
"""

import asyncio
import sys
import os
from pathlib import Path

# 设置环境
os.environ['FEDCL_RECORDER_BACKEND'] = 'mlflow'
os.environ['FEDCL_CONSOLE_LOG_LEVEL'] = 'INFO'

root = Path(__file__).parent
sys.path.insert(0, str(root))

from fedcl.experiment import BatchExperimentRunner

# 论文配置
PAPER_CONFIG = {
    'num_clients': 10,
    'max_rounds': 3,  # 快速测试用3轮
    'batch_size': 64,
    'local_epochs': 1,  # 快速测试用1个epoch
    'learning_rate': 0.01,
    'momentum': 0.9,
}


def get_model_config_for_dataset(dataset: str):
    """根据数据集返回论文标准模型配置"""
    if dataset in ['MNIST', 'FMNIST']:
        return {'name': 'MNIST_PaperCNN', 'params': {'num_classes': 10}}
    elif dataset in ['CIFAR10', 'SVHN', 'CINIC10']:
        return {'name': 'CIFAR10_PaperCNN', 'params': {'num_classes': 10}}
    elif dataset == 'FedISIC2019':
        return {'name': 'FedISIC2019_PaperCNN', 'params': {'num_classes': 8}}
    elif dataset == 'Adult':
        return {'name': 'Adult_PaperMLP', 'params': {'num_classes': 2}}
    elif dataset == 'FCUBE':
        return {'name': 'FCUBE_PaperMLP', 'params': {'num_classes': 2}}
    raise ValueError(f"Unknown dataset: {dataset}")


def create_test_config(dataset: str, partition_config: dict):
    """创建测试配置"""
    model_config = get_model_config_for_dataset(dataset)
    exp_name = f"{dataset}_{partition_config['name']}_FedAvg_test"

    return {
        'name': exp_name,
        'overrides': {
            'dataset': {'name': dataset},
            'partition': partition_config.copy(),
            'algorithm': 'FedAvg',

            # 全局模型配置
            'global_model': model_config,
            'local_model': model_config,

            'training': {
                'num_clients': PAPER_CONFIG['num_clients'],
                'max_rounds': PAPER_CONFIG['max_rounds'],
                'batch_size': PAPER_CONFIG['batch_size'],
                'local_epochs': PAPER_CONFIG['local_epochs'],
                'learning_rate': PAPER_CONFIG['learning_rate'],
                'momentum': PAPER_CONFIG['momentum'],

                'global_model': model_config,

                # 客户端训练数据集配置
                'dataset': {
                    'name': dataset,
                    'params': {'root': './data', 'train': True, 'download': True},
                    'partition': partition_config.copy()
                },

                # 客户端Learner模型配置（必须与服务器一致！）
                'learner': {
                    'params': {
                        'model': model_config
                    }
                },

                'trainer': {
                    'params': {
                        'test_dataset': {
                            'name': dataset,
                            'params': {'root': './data', 'train': False, 'download': True},
                            'batch_size': 1000,
                        }
                    }
                }
            },

            'test_dataset': {
                'name': dataset,
                'params': {'root': './data', 'train': False, 'download': True},
                'batch_size': 1000,
            },
        }
    }


async def main():
    print("=" * 80)
    print("测试论文标准模型修复")
    print("验证各数据集使用正确的模型架构")
    print("=" * 80)

    # 测试配置
    test_cases = [
        # CIFAR10 - 应使用 CIFAR10_PaperCNN
        ('CIFAR10', {'type': 'iid', 'name': 'IID'}),
        ('CIFAR10', {'type': 'pathological', 'classes_per_client': 1, 'alpha': 0.5, 'name': '#C=1'}),
    ]

    experiments = [create_test_config(ds, part) for ds, part in test_cases]

    print(f"\n测试 {len(experiments)} 个配置:")
    for exp in experiments:
        model_name = exp['overrides']['global_model']['name']
        print(f"  - {exp['name']}: 模型={model_name}")

    # 设置MLflow
    import mlflow
    mlflow_uri = f"file:{Path('experiments/test_model_fix_mlruns').absolute()}"
    mlflow.set_tracking_uri(mlflow_uri)
    print(f"\nMLflow URI: {mlflow_uri}")

    # 运行实验
    runner = BatchExperimentRunner(
        base_config="configs/distributed/experiments/table3/",
        experiment_variants=experiments
    )

    print("\n开始运行测试...")
    results = await runner.run_all(parallel=False)

    # 分析结果
    print("\n" + "=" * 80)
    print("测试结果")
    print("=" * 80)

    for result in results:
        name = result.get('experiment', 'Unknown')
        status = result.get('status', 'Unknown')
        metrics = result.get('metrics', {})

        acc = metrics.get('accuracy', 'N/A')
        loss = metrics.get('loss', 'N/A')

        if isinstance(acc, float):
            acc = f"{acc:.4f}"
        if isinstance(loss, float):
            loss = f"{loss:.4f}"

        print(f"\n{name}:")
        print(f"  状态: {status}")
        print(f"  准确率: {acc}")
        print(f"  损失: {loss}")

        # 检查准确率是否合理
        if status == 'success' and isinstance(metrics.get('accuracy'), float):
            acc_val = metrics['accuracy']
            # CIFAR10 IID 应该在 20-80% 之间（短训练）
            # #C=1 应该在 10-30% 之间
            if 'IID' in name:
                if 0.15 < acc_val < 0.85:
                    print(f"  ✓ 准确率合理（IID预期范围）")
                else:
                    print(f"  ✗ 准确率异常！IID应在15-85%之间")
            elif '#C=1' in name:
                if acc_val < 0.50:
                    print(f"  ✓ 准确率合理（#C=1预期较低）")
                else:
                    print(f"  ✗ 准确率异常！#C=1应该较低")

    print("\n" + "=" * 80)
    print("如果CIFAR10准确率不再是99%，则修复成功！")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
