#!/usr/bin/env python3
"""
快速验证批量实验配置
只检查配置生成,不实际运行训练
"""

import sys
from pathlib import Path

# 添加项目路径
root = Path(__file__).parent
sys.path.insert(0, str(root))

from examples.reproduce_table3_experiments import (
    create_experiment_config,
    PAPER_CONFIG
)


def main():
    """验证配置生成"""

    print("=" * 100)
    print("批量实验配置验证")
    print("=" * 100)

    # 测试数据集列表
    test_datasets = ['MNIST', 'CIFAR10', 'FMNIST']

    # 测试Non-IID设置
    noniid_configs = [
        {'type': 'iid', 'name': 'IID'},
        {'type': 'dirichlet', 'alpha': 0.5, 'name': 'pk~Dir(0.5)'},
        {'type': 'pathological', 'classes_per_client': 1, 'alpha': 0.5, 'name': '#C=1'},
    ]

    # 生成实验配置
    experiments = []
    for dataset in test_datasets:
        for noniid_config in noniid_configs:
            exp = create_experiment_config(dataset, 'FedAvg', noniid_config, PAPER_CONFIG)
            experiments.append(exp)

    print(f"\n生成了 {len(experiments)} 个测试实验配置\n")

    # 1. 检查PAPER_CONFIG
    print("=" * 100)
    print("1. PAPER_CONFIG 检查")
    print("=" * 100)
    print(f"Early Stopping配置:")
    print(f"  early_stopping: {PAPER_CONFIG.get('early_stopping', 'NOT SET')}")
    print(f"  patience: {PAPER_CONFIG.get('patience', 'NOT SET')}")
    print(f"  min_delta: {PAPER_CONFIG.get('min_delta', 'NOT SET')}")
    print(f"  monitor: {PAPER_CONFIG.get('monitor', 'NOT SET')}")

    # 2. 检查生成的实验配置
    print("\n" + "=" * 100)
    print("2. 实验配置检查 (样本: MNIST_IID_FedAvg)")
    print("=" * 100)

    sample_exp = experiments[0]
    print(f"\n实验名称: {sample_exp['name']}")

    # 检查trainer params
    trainer_params = sample_exp['overrides']['training']['trainer']['params']
    print(f"\nTrainer Early Stopping配置:")
    print(f"  early_stopping: {trainer_params.get('early_stopping', 'NOT SET')}")
    print(f"  patience: {trainer_params.get('patience', 'NOT SET')}")
    print(f"  min_delta: {trainer_params.get('min_delta', 'NOT SET')}")
    print(f"  monitor: {trainer_params.get('monitor', 'NOT SET')}")

    # 3. 检查所有实验的数据集配置
    print("\n" + "=" * 100)
    print("3. 数据集配置一致性检查")
    print("=" * 100)

    all_consistent = True
    for exp in experiments:
        exp_name = exp['name']
        dataset_name = exp['overrides']['dataset']['name']
        partition_info = exp['overrides']['partition']

        # 检查训练数据集配置
        train_dataset = exp['overrides']['training']['dataset']['name']

        # 检查测试数据集配置 (服务器端)
        test_dataset_server = exp['overrides']['training']['trainer']['params']['test_dataset']['name']

        # 检查测试数据集配置 (客户端)
        test_dataset_client = exp['overrides']['test_dataset']['name']

        # 验证一致性
        is_consistent = (dataset_name == train_dataset == test_dataset_server == test_dataset_client)

        status = "✓" if is_consistent else "✗"
        print(f"\n{status} {exp_name}")
        print(f"   Dataset: {dataset_name}")
        print(f"   Partition: {partition_info.get('type')} ({partition_info.get('name')})")
        print(f"   Train: {train_dataset}, Test(server): {test_dataset_server}, Test(client): {test_dataset_client}")

        if not is_consistent:
            all_consistent = False
            print(f"   ⚠️  数据集配置不一致!")

    # 4. 总结
    print("\n" + "=" * 100)
    print("4. 配置验证总结")
    print("=" * 100)

    print(f"\n✓ PAPER_CONFIG包含Early Stopping配置: {PAPER_CONFIG.get('early_stopping', False)}")
    print(f"✓ 实验配置包含Early Stopping参数: {trainer_params.get('early_stopping', False)}")
    print(f"{'✓' if all_consistent else '✗'} 所有实验的数据集配置一致性: {all_consistent}")

    print("\n" + "=" * 100)
    print("配置验证完成!")
    print("=" * 100)

    # 5. 显示一个完整的实验配置示例
    print("\n" + "=" * 100)
    print("5. 完整配置示例 (MNIST_IID_FedAvg)")
    print("=" * 100)

    import json
    print(json.dumps(sample_exp, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
