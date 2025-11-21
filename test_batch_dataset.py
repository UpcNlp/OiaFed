#!/usr/bin/env python3
"""
测试批量实验脚本的数据集配置应用
验证不同数据集和Non-IID设置能否正确工作
"""

import asyncio
import sys
import os
from pathlib import Path

# 设置环境变量
os.environ['FEDCL_RECORDER_BACKEND'] = 'mlflow'

# 添加项目路径
root = Path(__file__).parent
sys.path.insert(0, str(root))

from examples.reproduce_table3_experiments import (
    create_experiment_config,
    PAPER_CONFIG,
    run_table3_experiments
)


async def main():
    """测试多个数据集和Non-IID设置"""

    print("=" * 100)
    print("批量实验数据集配置测试 (带Early Stopping)")
    print("=" * 100)
    print("\n测试配置:")
    print("  数据集: MNIST, CIFAR10, FMNIST")
    print("  算法: FedAvg")
    print("  划分方式: IID, pk~Dir(0.5)")
    print("  最大轮数: 50 (带Early Stopping)")
    print("  Early Stopping: patience=5, min_delta=0.001")
    print("  重复次数: 1")
    print("\n目标: 验证数据集配置能否正确应用到不同数据集和Non-IID设置")
    print("=" * 100)

    # 创建测试实验
    experiments = []

    # 测试数据集列表
    test_datasets = ['MNIST', 'CIFAR10', 'FMNIST']

    # 测试Non-IID设置
    noniid_configs = [
        {'type': 'iid', 'name': 'IID'},
        {'type': 'dirichlet', 'alpha': 0.5, 'name': 'pk~Dir(0.5)'},
    ]

    # 生成实验配置
    for dataset in test_datasets:
        for noniid_config in noniid_configs:
            exp = create_experiment_config(dataset, 'FedAvg', noniid_config, PAPER_CONFIG)
            experiments.append(exp)

    print(f"\n生成了 {len(experiments)} 个测试实验:")
    for i, exp in enumerate(experiments, 1):
        print(f"  {i}. {exp['name']}")

    # 验证Early Stopping配置
    print("\n检查Early Stopping配置:")
    sample_exp = experiments[0]
    trainer_params = sample_exp['overrides']['training']['trainer']['params']
    print(f"  early_stopping: {trainer_params.get('early_stopping', 'NOT SET')}")
    print(f"  patience: {trainer_params.get('patience', 'NOT SET')}")
    print(f"  min_delta: {trainer_params.get('min_delta', 'NOT SET')}")
    print(f"  monitor: {trainer_params.get('monitor', 'NOT SET')}")

    # 验证数据集配置
    print("\n检查数据集配置:")
    for exp in experiments:
        exp_name = exp['name']
        dataset_name = exp['overrides']['dataset']['name']
        partition_type = exp['overrides']['partition']['type']

        # 检查训练数据集配置
        train_dataset = exp['overrides']['training']['dataset']['name']

        # 检查测试数据集配置 (服务器端)
        test_dataset_server = exp['overrides']['training']['trainer']['params']['test_dataset']['name']

        # 检查测试数据集配置 (客户端)
        test_dataset_client = exp['overrides']['test_dataset']['name']

        print(f"  {exp_name}:")
        print(f"    Dataset: {dataset_name}")
        print(f"    Partition: {partition_type}")
        print(f"    Train dataset: {train_dataset}")
        print(f"    Test dataset (server): {test_dataset_server}")
        print(f"    Test dataset (client): {test_dataset_client}")

        # 验证一致性
        if dataset_name == train_dataset == test_dataset_server == test_dataset_client:
            print(f"    ✓ 数据集配置一致")
        else:
            print(f"    ✗ 数据集配置不一致!")

    # 设置MLflow
    import mlflow
    mlflow_uri = f"file:{Path('experiments/test_batch_dataset_mlruns').absolute()}"
    mlflow.set_tracking_uri(mlflow_uri)
    print(f"\nMLflow tracking URI: {mlflow_uri}")

    # 顺序运行测试（不并行，方便观察）
    print("\n" + "=" * 100)
    print("开始运行测试实验...")
    print("=" * 100)

    await run_table3_experiments(
        experiments,
        parallel=False,  # 顺序执行
        max_parallel=1
    )

    print("\n" + "=" * 100)
    print("✓ 测试完成！")
    print("=" * 100)
    print("\n请检查结果:")
    print("  1. MLflow UI:")
    print("     mlflow ui --backend-store-uri experiments/test_batch_dataset_mlruns")
    print("     访问: http://localhost:5000")
    print("\n  2. 查看实验日志:")
    print("     logs/exp_*/")
    print("\n  3. 验证每个实验:")
    print("     - Early stopping是否触发?")
    print("     - 数据集是否正确加载?")
    print("     - Non-IID划分是否正确?")


if __name__ == "__main__":
    asyncio.run(main())
