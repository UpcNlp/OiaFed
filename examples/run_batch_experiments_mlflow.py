"""
使用MLflow运行批量实验
examples/run_batch_experiments_mlflow.py

演示如何使用 MLflow 后端进行实验跟踪和可视化

使用方法:
    # 1. 安装 MLflow
    pip install mlflow

    # 2. 运行批量实验
    python examples/run_batch_experiments_mlflow.py --mode comparison

    # 3. 查看MLflow UI
    mlflow ui --backend-store-uri experiments/mlruns
    # 打开浏览器访问: http://localhost:5000
"""

import asyncio
import sys
import os
from pathlib import Path

# 在导入 fedcl 之前设置环境变量，切换到 MLflow 后端
os.environ['FEDCL_RECORDER_BACKEND'] = 'mlflow'

# 添加项目路径
root = Path(__file__).parent.parent
sys.path.insert(0, str(root))

from fedcl.experiment import (
    BatchExperimentRunner,
    create_algorithm_comparison_experiments,
    create_grid_search_experiments,
    Recorder  # 会自动使用 MLflow 后端
)


async def algorithm_comparison():
    """算法对比实验 - 使用MLflow"""
    print("=" * 80)
    print("算法对比实验 - MLflow 版本")
    print("=" * 80)

    # 创建算法对比实验配置
    experiments = create_algorithm_comparison_experiments(
        base_name="mnist_iid_mlflow",
        algorithms=['fedavg', 'fedprox', 'scaffold'],
        common_config={
            'learning_rate': 0.01,
            'batch_size': 32,
            'max_rounds': 3  # 快速演示，仅3轮
        }
    )

    print(f"\n创建了 {len(experiments)} 组实验:")
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
    print_results_summary(results)

    print("\n" + "=" * 80)
    print("✓ 实验完成！")
    print("=" * 80)
    print("\n查看MLflow UI:")
    print("  cd /home/nlp/ct/projects/MOE-FedCL")
    print("  mlflow ui --backend-store-uri experiments/mlruns")
    print("  然后访问: http://localhost:5000")
    print("\n在UI中你可以:")
    print("  - 对比不同算法的准确率曲线")
    print("  - 查看每轮训练的详细指标")
    print("  - 筛选和排序实验结果")
    print("  - 下载实验数据和模型")


async def grid_search():
    """网格搜索实验 - 使用MLflow"""
    print("=" * 80)
    print("网格搜索实验 - MLflow 版本")
    print("=" * 80)

    # 创建网格搜索实验
    experiments = create_grid_search_experiments(
        base_name="mnist_grid_search_mlflow",
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

    # 并行运行（最多2个同时）
    print("\n开始并行运行实验（最多2个同时）...")
    results = await runner.run_all(parallel=True, max_parallel=2)

    # 找出最佳配置
    successful = [r for r in results if r['status'] == 'success']
    if successful:
        best = max(successful, key=lambda x: x.get('accuracy', 0))
        print(f"\n🏆 最佳配置: {best['name']}")
        print(f"   准确率: {best['accuracy']:.4f}")

    print_results_summary(results)

    print("\n查看MLflow UI进行深入分析:")
    print("  mlflow ui --backend-store-uri experiments/mlruns")


def print_results_summary(results):
    """打印结果摘要"""
    print("\n" + "=" * 80)
    print("实验结果摘要")
    print("=" * 80)

    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']

    print(f"\n总计: {len(results)} 组实验")
    print(f"成功: {len(successful)} 组")
    print(f"失败: {len(failed)} 组")

    if successful:
        print("\n成功的实验:")
        print(f"{'实验名称':<50} {'准确率':<10} {'轮数':<8} {'耗时(s)':<10}")
        print("-" * 80)
        for r in successful:
            name = r['name'][:48]
            acc = r.get('accuracy', 0)
            rounds = r.get('rounds', 'N/A')
            duration = r.get('duration', 0)
            print(f"{name:<50} {acc:.4f}    {rounds:<8} {duration:.2f}")

        # 找出最佳
        best = max(successful, key=lambda x: x.get('accuracy', 0))
        print(f"\n🏆 最佳结果: {best['name']}")
        print(f"   准确率: {best['accuracy']:.4f}")

    if failed:
        print("\n失败的实验:")
        for r in failed:
            print(f"✗ {r['name']}: {r.get('error', 'Unknown error')}")


async def quick_demo():
    """快速演示 - 单个实验"""
    print("=" * 80)
    print("MLflow 快速演示 - 单个实验")
    print("=" * 80)

    experiments = [{
        'name': 'mlflow_demo_fedavg',
        'overrides': {'trainer.name': 'fedavg', 'learning_rate': 0.01}
    }]

    runner = BatchExperimentRunner(
        base_config="configs/distributed/experiments/iid/",
        experiment_variants=experiments
    )

    print("\n运行单个实验进行演示...")
    results = await runner.run_all(parallel=False)

    print_results_summary(results)

    print("\n提示: 运行 'mlflow ui --backend-store-uri experiments/mlruns' 查看详细结果")


async def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description='使用MLflow运行批量实验',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

1. 算法对比:
   python examples/run_batch_experiments_mlflow.py --mode comparison

2. 网格搜索:
   python examples/run_batch_experiments_mlflow.py --mode grid_search

3. 快速演示:
   python examples/run_batch_experiments_mlflow.py --mode demo

查看结果:
   mlflow ui --backend-store-uri experiments/mlruns
   访问: http://localhost:5000
        """
    )

    parser.add_argument('--mode',
                       choices=['comparison', 'grid_search', 'demo'],
                       default='demo',
                       help='实验模式')

    args = parser.parse_args()

    # 检查 MLflow 是否安装
    try:
        import mlflow
        print(f"✓ MLflow 版本: {mlflow.__version__}")
    except ImportError:
        print("✗ MLflow 未安装!")
        print("  请运行: pip install mlflow")
        sys.exit(1)

    # 运行对应的实验
    if args.mode == 'comparison':
        await algorithm_comparison()
    elif args.mode == 'grid_search':
        await grid_search()
    else:
        await quick_demo()


if __name__ == "__main__":
    asyncio.run(main())
