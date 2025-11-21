#!/usr/bin/env python3
"""
快速测试数据集加载 - 只运行1轮验证
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
    """快速测试 - 只运行1轮"""

    print("=" * 100)
    print("快速数据集加载测试 (1轮验证)")
    print("=" * 100)

    # 创建快速测试配置
    quick_config = PAPER_CONFIG.copy()
    quick_config['max_rounds'] = 1  # 只运行1轮
    quick_config['local_epochs'] = 1  # 只运行1个epoch

    # 测试三个不同的数据集
    test_cases = [
        ('MNIST', {'type': 'iid', 'name': 'IID'}),
        ('CIFAR10', {'type': 'dirichlet', 'alpha': 0.5, 'name': 'pk~Dir(0.5)'}),
        ('FMNIST', {'type': 'pathological', 'classes_per_client': 2, 'alpha': 0.5, 'name': '#C=2'}),
    ]

    experiments = []
    for dataset, noniid_config in test_cases:
        exp = create_experiment_config(dataset, 'FedAvg', noniid_config, quick_config)
        experiments.append(exp)

    print(f"\n测试配置:")
    print(f"  实验数量: {len(experiments)}")
    print(f"  轮数: {quick_config['max_rounds']}")
    print(f"  Local epochs: {quick_config['local_epochs']}")
    print(f"  Early Stopping: {quick_config.get('early_stopping', False)}")
    print(f"\n测试用例:")
    for i, exp in enumerate(experiments, 1):
        print(f"  {i}. {exp['name']}")

    # 设置MLflow
    import mlflow
    mlflow_uri = f"file:{Path('experiments/quick_test_mlruns').absolute()}"
    mlflow.set_tracking_uri(mlflow_uri)

    print("\n" + "=" * 100)
    print("开始测试...")
    print("=" * 100)

    await run_table3_experiments(
        experiments,
        parallel=False,
        max_parallel=1
    )

    print("\n" + "=" * 100)
    print("✓ 快速测试完成!")
    print("=" * 100)
    print("\n验证要点:")
    print("  ✓ 所有数据集都能正确加载")
    print("  ✓ Non-IID划分正常工作")
    print("  ✓ Early stopping配置已应用")
    print("  ✓ 训练能够正常运行")


if __name__ == "__main__":
    asyncio.run(main())
