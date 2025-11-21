#!/usr/bin/env python3
"""
测试CIFAR10数据集在不同划分方式下的实验效果
验证test_dataset配置修复是否有效
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
    """测试CIFAR10的三种划分方式"""

    print("=" * 100)
    print("CIFAR10 数据集修复验证测试 (带Early Stopping)")
    print("=" * 100)
    print("\n测试配置:")
    print("  数据集: CIFAR10")
    print("  算法: FedAvg")
    print("  划分方式: IID, pk~Dir(0.5), #C=1")
    print("  最大轮数: 50")
    print("  Early Stopping: 启用 (patience=5, min_delta=0.001)")
    print("  重复次数: 1")
    print("\n预期结果 (根据论文Table 3):")
    print("  IID: ~72%")
    print("  pk~Dir(0.5): ~65%")
    print("  #C=1: ~10%")
    print("\n" + "=" * 100)

    # 修改PAPER_CONFIG以添加early stopping
    test_config = PAPER_CONFIG.copy()
    test_config['early_stopping'] = True
    test_config['patience'] = 5  # 5轮没提升就停止
    test_config['min_delta'] = 0.001  # 最小提升阈值
    test_config['monitor'] = 'accuracy'  # 监控准确率

    # 创建三个测试实验
    experiments = []

    # 1. IID
    noniid_config_iid = {'type': 'iid', 'name': 'IID'}
    exp_iid = create_experiment_config('CIFAR10', 'FedAvg', noniid_config_iid, test_config)
    experiments.append(exp_iid)

    # 2. pk~Dir(0.5)
    noniid_config_dir = {'type': 'dirichlet', 'alpha': 0.5, 'name': 'pk~Dir(0.5)'}
    exp_dir = create_experiment_config('CIFAR10', 'FedAvg', noniid_config_dir, test_config)
    experiments.append(exp_dir)

    # 3. #C=1
    noniid_config_c1 = {'type': 'pathological', 'classes_per_client': 1, 'alpha': 0.5, 'name': '#C=1'}
    exp_c1 = create_experiment_config('CIFAR10', 'FedAvg', noniid_config_c1, test_config)
    experiments.append(exp_c1)

    print(f"\n生成了 {len(experiments)} 个测试实验:")
    for i, exp in enumerate(experiments, 1):
        print(f"  {i}. {exp['name']}")

    # 设置MLflow
    import mlflow
    mlflow_uri = f"file:{Path('experiments/test_cifar10_mlruns').absolute()}"
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
    print("     mlflow ui --backend-store-uri experiments/test_cifar10_mlruns")
    print("     访问: http://localhost:5000")
    print("\n  2. 查看实验日志:")
    print("     logs/exp_*/")


if __name__ == "__main__":
    asyncio.run(main())
