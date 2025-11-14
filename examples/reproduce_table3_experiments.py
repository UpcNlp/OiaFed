"""
复现论文 TABLE III 的批量实验脚本
examples/reproduce_table3_experiments.py

论文: Fundamentals and Experimental Analysis of Federated Learning Algorithms:
      A Comparative Study on Non-IID Data Silos (TPAMI 2025)

使用方法:
    # 运行所有实验（需要很长时间）
    python examples/reproduce_table3_experiments.py --mode all

    # 运行单个数据集的实验
    python examples/reproduce_table3_experiments.py --mode single --dataset mnist

    # 运行特定Non-IID类型的实验
    python examples/reproduce_table3_experiments.py --mode noniid --noniid_type label_skew

    # 查看结果
    mlflow ui --backend-store-uri experiments/table3_mlruns
"""

import asyncio
import argparse
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
from itertools import product

# 在导入前设置MLflow后端
os.environ['FEDCL_RECORDER_BACKEND'] = 'mlflow'

# 添加项目路径
root = Path(__file__).parent.parent
sys.path.insert(0, str(root))

from fedcl.experiment import BatchExperimentRunner


# ============================================================================
# 论文TABLE III配置
# ============================================================================

# 9个算法
ALGORITHMS = [
    'FedAvg',
    'FedProx',
    'SCAFFOLD',
    'FedNova',
    'FedAdagrad',
    'FedYogi',
    'FedAdam',
    'MOON',
    'FedBN',
]

# 8个数据集
DATASETS = [
    'MNIST',
    'FMNIST',
    'SVHN',
    'CINIC10',
    'CIFAR10',
    'FedISIC2019',
    'Adult',
    'FCUBE',
]

# Non-IID设置（按照论文TABLE III的结构）
NONIID_SETTINGS = {
    # 标签分布倾斜 (Label distribution skew)
    'label_skew': [
        {'type': 'dirichlet', 'alpha': 0.5, 'name': 'pk~Dir(0.5)'},
        {'type': 'pathological', 'classes_per_client': 1, 'alpha': 0.5, 'name': '#C=1'},
        {'type': 'pathological', 'classes_per_client': 2, 'alpha': 0.5, 'name': '#C=2'},
        {'type': 'pathological', 'classes_per_client': 3, 'alpha': 0.5, 'name': '#C=3'},
    ],

    # 特征分布倾斜 (Feature distribution skew)
    'feature_skew': [
        {'type': 'feature_skew', 'noise_std': 0.1, 'name': 'x~Gau(0.1)'},
    ],

    # 数量倾斜 (Quantity skew)
    'quantity_skew': [
        {'type': 'quantity_skew', 'alpha': 0.5, 'name': 'q~Dir(0.5)'},
    ],

    # IID基线
    'iid': [
        {'type': 'iid', 'name': 'IID'},
    ],
}

# 论文超参数设置（按照论文第VI节）
PAPER_CONFIG = {
    # 训练参数
    'num_clients': 10,
    'max_rounds': 50,  # 论文中使用50轮通信
    'batch_size': 64,
    'local_epochs': 10,  # 论文中使用10个本地epoch
    'learning_rate': 0.01,
    'momentum': 0.9,

    # 算法特定参数
    'fedprox_mu': 0.01,  # FedProx的proximal term
    'moon_mu': 10,  # MOON的对比学习权重
    'moon_temperature': 0.5,  # MOON的温度参数
    'fedadam_beta1': 0.9,  # 自适应优化器的动量参数
    'fedadam_beta2': 0.99,
    'fedyogi_beta1': 0.9,
    'fedyogi_beta2': 0.99,
}


# ============================================================================
# 实验配置生成函数
# ============================================================================

def create_experiment_config(dataset: str, algorithm: str, noniid_config: Dict[str, Any],
                            base_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    创建单个实验的配置

    Args:
        dataset: 数据集名称
        algorithm: 算法名称
        noniid_config: Non-IID配置
        base_config: 基础配置

    Returns:
        实验配置字典
    """
    # 实验名称
    exp_name = f"{dataset}_{noniid_config['name']}_{algorithm}"

    # 基础配置
    config = {
        'name': exp_name,
        'overrides': {
            # 数据集配置
            'dataset': {'name': dataset},
            'partition': noniid_config.copy(),

            # 算法配置
            'algorithm': algorithm,

            # 训练配置
            'training': {
                'num_clients': base_config['num_clients'],
                'max_rounds': base_config['max_rounds'],
                'batch_size': base_config['batch_size'],
                'local_epochs': base_config['local_epochs'],
                'learning_rate': base_config['learning_rate'],
                'momentum': base_config['momentum'],
            },
        }
    }

    # 添加算法特定参数
    if algorithm == 'FedProx':
        config['overrides']['fedprox_mu'] = base_config['fedprox_mu']
    elif algorithm == 'MOON':
        config['overrides']['moon_mu'] = base_config['moon_mu']
        config['overrides']['moon_temperature'] = base_config['moon_temperature']
    elif algorithm in ['FedAdagrad', 'FedAdam']:
        config['overrides']['beta1'] = base_config['fedadam_beta1']
        config['overrides']['beta2'] = base_config['fedadam_beta2']
    elif algorithm == 'FedYogi':
        config['overrides']['beta1'] = base_config['fedyogi_beta1']
        config['overrides']['beta2'] = base_config['fedyogi_beta2']

    return config


def generate_all_experiments(datasets: List[str] = None,
                            algorithms: List[str] = None,
                            noniid_types: List[str] = None) -> List[Dict[str, Any]]:
    """
    生成所有实验配置

    Args:
        datasets: 要运行的数据集列表（None表示全部）
        algorithms: 要运行的算法列表（None表示全部）
        noniid_types: 要运行的Non-IID类型列表（None表示全部）

    Returns:
        实验配置列表
    """
    # 默认使用全部
    if datasets is None:
        datasets = DATASETS
    if algorithms is None:
        algorithms = ALGORITHMS
    if noniid_types is None:
        noniid_types = list(NONIID_SETTINGS.keys())

    # 生成所有实验组合
    experiments = []

    for dataset in datasets:
        for noniid_type in noniid_types:
            for noniid_config in NONIID_SETTINGS[noniid_type]:
                # 特殊处理：FCUBE只测试feature_skew（论文中标注为synthetic）
                if dataset == 'FCUBE' and noniid_type != 'feature_skew':
                    continue

                for algorithm in algorithms:
                    exp_config = create_experiment_config(
                        dataset, algorithm, noniid_config, PAPER_CONFIG
                    )
                    experiments.append(exp_config)

    return experiments


# ============================================================================
# 实验运行函数
# ============================================================================

async def run_table3_experiments(experiments: List[Dict[str, Any]],
                                 parallel: bool = True,
                                 max_parallel: int = 2):
    """
    运行TABLE III的批量实验

    Args:
        experiments: 实验配置列表
        parallel: 是否并行运行
        max_parallel: 最大并行数
    """
    print("=" * 100)
    print("复现论文 TABLE III 实验")
    print("Paper: Fundamentals and Experimental Analysis of Federated Learning Algorithms (TPAMI 2025)")
    print("=" * 100)

    print(f"\n总实验数量: {len(experiments)}")
    print(f"并行模式: {'是' if parallel else '否'}")
    if parallel:
        print(f"最大并行数: {max_parallel}")

    # 统计实验配置
    datasets_used = set()
    algorithms_used = set()
    for exp in experiments:
        datasets_used.add(exp['overrides']['dataset']['name'])
        algorithms_used.add(exp['overrides']['algorithm'])

    print(f"\n数据集 ({len(datasets_used)}): {', '.join(sorted(datasets_used))}")
    print(f"算法 ({len(algorithms_used)}): {', '.join(sorted(algorithms_used))}")

    # 设置MLflow tracking URI
    import mlflow
    mlflow_uri = f"file:{Path('experiments/table3_mlruns').absolute()}"
    mlflow.set_tracking_uri(mlflow_uri)
    print(f"\nMLflow tracking URI: {mlflow_uri}")

    # 创建批量实验运行器
    runner = BatchExperimentRunner(
        base_config="configs/distributed/experiments/table3/",  # 使用TABLE III实验配置
        experiment_variants=experiments
    )

    # 运行实验
    print("\n" + "=" * 100)
    print("开始运行实验...")
    print("=" * 100)

    results = await runner.run_all(parallel=parallel, max_parallel=max_parallel)

    # 输出结果摘要
    print_results_summary(results)

    # 保存结果到CSV（便于后续分析）
    save_results_to_csv(results, "experiments/table3_results.csv")

    print("\n" + "=" * 100)
    print("✓ 实验完成！")
    print("=" * 100)
    print("\n查看结果:")
    print("  1. MLflow UI:")
    print("     mlflow ui --backend-store-uri experiments/table3_mlruns")
    print("     访问: http://localhost:5000")
    print("\n  2. CSV结果文件:")
    print("     experiments/table3_results.csv")


def print_results_summary(results: List[Dict[str, Any]]):
    """打印结果摘要"""
    print("\n" + "=" * 100)
    print("实验结果摘要")
    print("=" * 100)

    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']

    print(f"\n总计: {len(results)} 组实验")
    print(f"成功: {len(successful)} 组 ({len(successful)/len(results)*100:.1f}%)")
    print(f"失败: {len(failed)} 组 ({len(failed)/len(results)*100:.1f}%)")

    if successful:
        # 按数据集和算法分组统计最佳结果
        print("\n" + "-" * 100)
        print("各数据集最佳结果:")
        print("-" * 100)

        datasets = set(r['name'].split('_')[0] for r in successful)
        for dataset in sorted(datasets):
            dataset_results = [r for r in successful if r['name'].startswith(dataset)]
            if dataset_results:
                best = max(dataset_results, key=lambda x: x.get('accuracy', 0))
                print(f"{dataset:15s}: {best.get('accuracy', 0):.4f} ({best['name']})")

    if failed:
        print(f"\n失败的实验 ({len(failed)}):")
        for r in failed[:10]:  # 只显示前10个失败的实验
            print(f"  ✗ {r['name']}: {r.get('error', 'Unknown error')[:80]}")
        if len(failed) > 10:
            print(f"  ... 还有 {len(failed)-10} 个失败的实验")


def save_results_to_csv(results: List[Dict[str, Any]], filepath: str):
    """保存结果到CSV文件"""
    import csv

    # 确保目录存在
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    # 写入CSV
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)

        # 表头
        writer.writerow(['Dataset', 'NonIID_Type', 'Algorithm', 'Accuracy', 'Loss',
                        'Rounds', 'Duration_sec', 'Status', 'Error'])

        # 数据行
        for r in results:
            # 从name中解析数据集、Non-IID类型和算法
            parts = r['name'].split('_')
            dataset = parts[0]
            algorithm = parts[-1]
            noniid_type = '_'.join(parts[1:-1])

            writer.writerow([
                dataset,
                noniid_type,
                algorithm,
                r.get('accuracy', ''),
                r.get('loss', ''),
                r.get('rounds', ''),
                r.get('duration', ''),
                r['status'],
                r.get('error', '')
            ])

    print(f"\n✓ 结果已保存到: {filepath}")


# ============================================================================
# 命令行接口
# ============================================================================

async def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='复现论文TABLE III实验',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

1. 运行所有实验:
   python examples/reproduce_table3_experiments.py --mode all

2. 运行单个数据集:
   python examples/reproduce_table3_experiments.py --mode single --dataset MNIST

3. 运行特定Non-IID类型:
   python examples/reproduce_table3_experiments.py --mode noniid --noniid_type label_skew

4. 快速测试（仅IID + MNIST + FedAvg）:
   python examples/reproduce_table3_experiments.py --mode test

查看结果:
   mlflow ui --backend-store-uri experiments/table3_mlruns
        """
    )

    parser.add_argument('--mode',
                       choices=['all', 'single', 'noniid', 'test'],
                       default='test',
                       help='运行模式')
    parser.add_argument('--dataset',
                       choices=DATASETS,
                       help='单个数据集名称（mode=single时使用）')
    parser.add_argument('--noniid_type',
                       choices=list(NONIID_SETTINGS.keys()),
                       help='Non-IID类型（mode=noniid时使用）')
    parser.add_argument('--parallel',
                       action='store_true',
                       default=False,
                       help='是否并行运行')
    parser.add_argument('--max_parallel',
                       type=int,
                       default=2,
                       help='最大并行数')

    args = parser.parse_args()

    # 检查MLflow是否安装
    try:
        import mlflow
        print(f"✓ MLflow 版本: {mlflow.__version__}")
    except ImportError:
        print("✗ MLflow 未安装!")
        print("  请运行: pip install mlflow")
        sys.exit(1)

    # 根据模式生成实验
    if args.mode == 'all':
        print("模式: 运行所有实验 (数据集 x Non-IID x 算法)")
        experiments = generate_all_experiments()

    elif args.mode == 'single':
        if not args.dataset:
            print("错误: --dataset 参数是必须的")
            sys.exit(1)
        print(f"模式: 单个数据集 ({args.dataset})")
        experiments = generate_all_experiments(datasets=[args.dataset])

    elif args.mode == 'noniid':
        if not args.noniid_type:
            print("错误: --noniid_type 参数是必须的")
            sys.exit(1)
        print(f"模式: 特定Non-IID类型 ({args.noniid_type})")
        experiments = generate_all_experiments(noniid_types=[args.noniid_type])

    else:  # test
        print("模式: 快速测试 (MNIST + IID + 3个算法)")
        experiments = generate_all_experiments(
            datasets=['MNIST'],
            algorithms=['FedAvg', 'FedProx', 'SCAFFOLD'],
            noniid_types=['iid']
        )

    # 运行实验
    await run_table3_experiments(
        experiments,
        parallel=args.parallel,
        max_parallel=args.max_parallel
    )


if __name__ == "__main__":
    asyncio.run(main())
