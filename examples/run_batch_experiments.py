"""
批量运行实验示例
examples/run_batch_experiments.py

演示如何批量运行多组对比实验
"""

import asyncio
import sys
from pathlib import Path

# 添加项目路径
root = Path(__file__).parent.parent
sys.path.insert(0, str(root))

from fedcl.experiment.batch_runner import (
    BatchExperimentRunner,
    create_algorithm_comparison_experiments,
    create_grid_search_experiments
)


async def main():
    """主函数"""
    print("="*80)
    print("批量实验运行器")
    print("="*80)

    # 方式1: 算法对比实验
    print("\n创建算法对比实验...")
    experiments = create_algorithm_comparison_experiments(
        base_name="mnist_iid_comparison",
        algorithms=['fedavg', 'fedprox', 'scaffold'],
        common_config={
            'learning_rate': 0.01,
            'batch_size': 32,
            'max_rounds': 10
        }
    )

    print(f"创建了 {len(experiments)} 组实验:")
    for exp in experiments:
        print(f"  - {exp['name']}")

    # 运行实验（串行）
    runner = BatchExperimentRunner(
        base_config="configs/distributed/experiments/iid/",
        experiment_variants=experiments
    )

    print("\n开始运行实验...")
    results = await runner.run_all(parallel=False)

    # 输出结果摘要
    print("\n"+"="*80)
    print("实验结果摘要")
    print("="*80)

    for result in results:
        status = "✓" if result['status'] == 'success' else "✗"
        print(f"{status} {result['name']}")
        if result['status'] == 'success':
            print(f"  准确率: {result.get('accuracy', 'N/A'):.4f}")
            print(f"  损失: {result.get('loss', 'N/A'):.4f}")
            print(f"  轮数: {result.get('rounds', 'N/A')}")
            print(f"  耗时: {result.get('duration', 0):.2f}s")
        else:
            print(f"  错误: {result.get('error', 'Unknown')}")


async def grid_search_example():
    """网格搜索示例"""
    print("="*80)
    print("网格搜索实验")
    print("="*80)

    # 创建网格搜索实验
    experiments = create_grid_search_experiments(
        base_name="mnist_grid_search",
        param_grid={
            'learning_rate': [0.01, 0.001],
            'batch_size': [32, 64],
            'local_epochs': [1, 3]
        }
    )

    print(f"\n创建了 {len(experiments)} 组网格搜索实验")

    runner = BatchExperimentRunner(
        base_config="configs/distributed/experiments/iid/",
        experiment_variants=experiments
    )

    # 并行运行（最多3个同时）
    results = await runner.run_all(parallel=True, max_parallel=3)

    # 找出最佳配置
    successful = [r for r in results if r['status'] == 'success']
    if successful:
        best = max(successful, key=lambda x: x.get('accuracy', 0))
        print(f"\n最佳配置: {best['name']}")
        print(f"  准确率: {best['accuracy']:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='批量运行实验')
    parser.add_argument('--mode', choices=['comparison', 'grid_search'],
                       default='comparison',
                       help='实验模式')

    args = parser.parse_args()

    if args.mode == 'comparison':
        asyncio.run(main())
    else:
        asyncio.run(grid_search_example())
