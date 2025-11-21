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

# 智能调度器（个人需求工具）

from examples.smart_batch_runner import SmartBatchRunner, ExperimentConfig
SMART_RUNNER_AVAILABLE = True

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

    # Early Stopping配置
    'early_stopping': True,
    'patience': 5,  # 5轮没提升就停止
    'min_delta': 0.001,  # 最小提升阈值0.1%
    'monitor': 'accuracy',  # 监控准确率

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

def get_model_config_for_dataset(dataset: str) -> Dict[str, Any]:
    """
    根据数据集返回论文标准模型配置

    论文 TPAMI 2025 Section VI:
    - 图像数据集: CNN (6,16 channels; 120,84 FC units)
    - 表格数据集: MLP (32,16,8 hidden units)
    """
    # 图像数据集 - 使用PaperCNN
    if dataset in ['MNIST', 'FMNIST']:
        return {
            'name': 'MNIST_PaperCNN',
            'params': {'num_classes': 10}
        }
    elif dataset in ['CIFAR10', 'SVHN', 'CINIC10']:
        return {
            'name': 'CIFAR10_PaperCNN',
            'params': {'num_classes': 10}
        }
    elif dataset == 'FedISIC2019':
        return {
            'name': 'FedISIC2019_PaperCNN',
            'params': {'num_classes': 8}
        }
    # 表格数据集 - 使用PaperMLP
    elif dataset == 'Adult':
        return {
            'name': 'Adult_PaperMLP',
            'params': {'num_classes': 2}
        }
    elif dataset == 'FCUBE':
        return {
            'name': 'FCUBE_PaperMLP',
            'params': {'num_classes': 2}
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


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

    # 获取论文标准模型配置
    model_config = get_model_config_for_dataset(dataset)

    # 基础配置
    config = {
        'name': exp_name,
        'overrides': {
            # 数据集配置
            'dataset': {'name': dataset},
            'partition': noniid_config.copy(),

            # 算法配置
            'algorithm': algorithm,

            # 全局模型配置（论文标准模型）
            'global_model': model_config,

            # 本地模型配置（与全局模型一致）
            'local_model': model_config,

            # 训练配置
            'training': {
                'num_clients': base_config['num_clients'],
                'max_rounds': base_config['max_rounds'],
                'batch_size': base_config['batch_size'],
                'local_epochs': base_config['local_epochs'],
                'learning_rate': base_config['learning_rate'],
                'momentum': base_config['momentum'],

                # 全局模型配置（嵌套位置 - 服务器端）
                'global_model': model_config,

                # 客户端训练数据集配置
                'dataset': {
                    'name': dataset,
                    'params': {
                        'root': './data',
                        'train': True,
                        'download': True,
                    },
                    'partition': noniid_config.copy()
                },

                # 客户端Learner模型配置（必须与服务器一致！）
                'learner': {
                    'params': {
                        'model': model_config
                    }
                },

                # 训练器配置（包含服务器端test_dataset）
                'trainer': {
                    'params': {
                        # Early stopping配置
                        'early_stopping': base_config.get('early_stopping', True),
                        'patience': base_config.get('patience', 5),
                        'min_delta': base_config.get('min_delta', 0.001),
                        'monitor': base_config.get('monitor', 'accuracy'),

                        # 服务器端测试数据集配置（必须与训练数据集一致！）
                        'test_dataset': {
                            'name': dataset,  # 使用与训练相同的数据集
                            'params': {
                                'root': './data',
                                'train': False,  # 使用测试集
                                'download': True,
                            },
                            'batch_size': 1000,
                        }
                    }
                }
            },

            # 客户端测试数据集配置（用于客户端本地评估）
            'test_dataset': {
                'name': dataset,  # 使用与训练相同的数据集
                'params': {
                    'root': './data',
                    'train': False,  # 使用测试集
                    'download': True,
                },
                'batch_size': 1000,
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

    # 保存结果到CSV（便于后续分析）- 传入experiments配置
    save_results_to_csv(results, "experiments/table3_results.csv", experiments)

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


def save_results_to_csv(results: List[Dict[str, Any]], filepath: str, experiments: List[Dict[str, Any]] = None):
    """
    保存结果到CSV文件（增强版）

    Args:
        results: 实验结果列表
        filepath: CSV文件路径
        experiments: 实验配置列表（可选，用于提取详细配置信息）
    """
    import csv
    from datetime import datetime

    # 确保目录存在
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    # 创建实验配置查找字典
    exp_config_map = {}
    if experiments:
        for exp in experiments:
            exp_config_map[exp['name']] = exp

    # 写入CSV
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # 增强的表头
        writer.writerow([
            'Experiment_Name',
            'Dataset',
            'Model',
            'NonIID_Type',
            'Partition_Detail',
            'Algorithm',
            'Num_Clients',
            'Max_Rounds',
            'Completed_Rounds',
            'Local_Epochs',
            'Batch_Size',
            'Learning_Rate',
            'Final_Accuracy',
            'Final_Loss',
            'Duration_sec',
            'Early_Stopping',
            'Patience',
            'Status',
            'Error'
        ])

        # 数据行
        for r in results:
            exp_name = r['name']

            # 从name中解析基本信息
            parts = exp_name.split('_')
            dataset = parts[0]
            algorithm = parts[-1]
            noniid_type = '_'.join(parts[1:-1])

            # 从实验配置中提取详细信息
            if exp_name in exp_config_map:
                exp_config = exp_config_map[exp_name]
                overrides = exp_config.get('overrides', {})
                training_config = overrides.get('training', {})

                # 模型信息
                model_config = overrides.get('global_model', {})
                model_name = model_config.get('name', '')

                # 数据划分详细信息
                partition_config = overrides.get('partition', {})
                partition_detail = partition_config.get('type', '')
                if partition_config.get('type') == 'dirichlet':
                    partition_detail = f"Dirichlet(α={partition_config.get('alpha', '')})"
                elif partition_config.get('type') == 'pathological':
                    partition_detail = f"Pathological(#C={partition_config.get('classes_per_client', '')})"
                elif partition_config.get('type') == 'feature_skew':
                    partition_detail = f"FeatureSkew(σ={partition_config.get('noise_std', '')})"
                elif partition_config.get('type') == 'quantity_skew':
                    partition_detail = f"QuantitySkew(α={partition_config.get('alpha', '')})"

                # 训练超参数
                num_clients = training_config.get('num_clients', '')
                max_rounds = training_config.get('max_rounds', '')
                local_epochs = training_config.get('local_epochs', '')
                batch_size = training_config.get('batch_size', '')
                learning_rate = training_config.get('learning_rate', '')

                # Early stopping配置
                trainer_params = training_config.get('trainer', {}).get('params', {})
                early_stopping = trainer_params.get('early_stopping', '')
                patience = trainer_params.get('patience', '')
            else:
                # 如果没有配置信息，使用默认值
                model_name = ''
                partition_detail = noniid_type
                num_clients = ''
                max_rounds = ''
                local_epochs = ''
                batch_size = ''
                learning_rate = ''
                early_stopping = ''
                patience = ''

            writer.writerow([
                exp_name,
                dataset,
                model_name,
                noniid_type,
                partition_detail,
                algorithm,
                num_clients,
                max_rounds,
                r.get('rounds', ''),  # 实际完成的轮数
                local_epochs,
                batch_size,
                learning_rate,
                f"{r.get('accuracy', ''):.4f}" if isinstance(r.get('accuracy'), (int, float)) else r.get('accuracy', ''),
                f"{r.get('loss', ''):.4f}" if isinstance(r.get('loss'), (int, float)) else r.get('loss', ''),
                f"{r.get('duration', ''):.2f}" if isinstance(r.get('duration'), (int, float)) else r.get('duration', ''),
                early_stopping,
                patience,
                r['status'],
                r.get('error', '')
            ])

    print(f"\n✓ 结果已保存到: {filepath}")
    print(f"  包含 {len(results)} 条实验记录")
    print(f"  字段: 实验名称, 数据集, 模型, 划分方式, 算法, 超参数, 准确率, 损失等")


async def run_table3_experiments_with_smart_runner(
    experiments: List[Dict[str, Any]],
    repetitions: int = 3,
    enable_gpu_scheduling: bool = True,
    db_path: str = "experiments/experiment_tracker.db",
    concurrent: bool = False,
    multiprocess: bool = False,
    max_concurrent: int = 5
):
    """
    使用智能调度器运行TABLE III实验

    Args:
        experiments: 实验配置列表
        repetitions: 每个实验的重复次数
        enable_gpu_scheduling: 是否启用GPU显存调度
        db_path: SQLite数据库路径
        concurrent: 是否并发执行（AsyncIO模式）
        multiprocess: 是否使用多进程并发（推荐用于批量实验）
        max_concurrent: 最大并发实验数（默认5）
    """
    if not SMART_RUNNER_AVAILABLE:
        print("✗ SmartBatchRunner 不可用!")
        print("  请确保以下文件存在:")
        print("  - examples/smart_batch_runner.py")
        print("  - examples/experiment_tracker.py")
        print("  - examples/gpu_monitor.py")
        sys.exit(1)

    print("=" * 100)
    print("使用智能调度器运行 TABLE III 实验")
    print("Paper: Fundamentals and Experimental Analysis of Federated Learning Algorithms (TPAMI 2025)")
    print("=" * 100)

    # 确定执行模式
    if multiprocess:
        exec_mode = "multiprocess"
        mode_desc = "多进程并发（真实进程隔离）"
    elif concurrent:
        exec_mode = "concurrent"
        mode_desc = "AsyncIO并发"
    else:
        exec_mode = "sequential"
        mode_desc = "顺序执行"

    print(f"\n实验配置:")
    print(f"  总实验数: {len(experiments)}")
    print(f"  每个实验重复: {repetitions} 次")
    print(f"  执行模式: {mode_desc}")
    if multiprocess or concurrent:
        print(f"  最大并发数: {max_concurrent}")
    print(f"  GPU显存调度: {'启用' if enable_gpu_scheduling else '禁用'}")
    print(f"  数据库路径: {db_path}")

    # 转换实验配置为 ExperimentConfig 格式
    exp_configs = []
    for exp in experiments:
        # 从实验配置中提取信息
        dataset = exp['overrides']['dataset']['name']
        algorithm = exp['overrides']['algorithm']
        partition_config = exp['overrides']['partition']
        noniid_type = partition_config.get('name', partition_config.get('type', 'unknown'))

        exp_config = ExperimentConfig(
            name=exp['name'],
            dataset=dataset,
            algorithm=algorithm,
            noniid_type=noniid_type,
            config=exp
        )
        exp_configs.append(exp_config)

    # 设置MLflow tracking URI
    import mlflow
    mlflow_uri = f"file:{Path('experiments/table3_mlruns').absolute()}"
    mlflow.set_tracking_uri(mlflow_uri)
    print(f"\nMLflow tracking URI: {mlflow_uri}")

    # 创建智能批量调度器
    runner = SmartBatchRunner(
        config_base_dir="configs/distributed/experiments/table3/",
        experiments=exp_configs,
        max_repetitions=repetitions,
        db_path=db_path,
        log_dir="logs/smart_batch",
        enable_gpu_scheduling=enable_gpu_scheduling,
        max_concurrent_experiments=max_concurrent
    )

    # 运行实验
    print("\n" + "=" * 100)
    print("开始运行实验...")
    print("=" * 100)

    results = await runner.run_all_experiments(mode=exec_mode)

    # 输出结果摘要
    print("\n" + "=" * 100)
    print("实验完成！")
    print("=" * 100)
    print(f"\n总实验运行数: {len(results)}")
    print(f"成功: {runner.completed_experiments}")
    print(f"失败: {runner.failed_experiments}")
    print(f"跳过（已完成）: {runner.skipped_experiments}")

    print("\n查看结果:")
    print("  1. MLflow UI:")
    print("     mlflow ui --backend-store-uri experiments/table3_mlruns")
    print("     访问: http://localhost:5000")
    print(f"\n  2. 日志文件目录:")
    print(f"     {runner.log_dir}")
    print(f"\n  3. SQLite数据库:")
    print(f"     {db_path}")


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

    # 智能调度器选项（个人需求工具）
    parser.add_argument('--use-smart-runner',
                       action='store_true',
                       default=False,
                       help='使用智能调度器（GPU显存感知 + SQLite断点续跑）')
    parser.add_argument('--repetitions',
                       type=int,
                       default=3,
                       help='每个实验的重复次数（仅smart-runner）')
    parser.add_argument('--enable-gpu-scheduling',
                       action='store_true',
                       default=True,
                       help='启用GPU显存调度（仅smart-runner）')
    parser.add_argument('--db-path',
                       type=str,
                       default='experiments/experiment_tracker.db',
                       help='SQLite数据库路径（仅smart-runner）')
    parser.add_argument('--concurrent',
                       action='store_true',
                       default=False,
                       help='启用并发模式（仅smart-runner）')
    parser.add_argument('--multiprocess',
                       action='store_true',
                       default=False,
                       help='启用多进程并发模式（仅smart-runner，推荐用于批量实验）')
    parser.add_argument('--max-concurrent',
                       type=int,
                       default=5,
                       help='最大并发实验数（仅smart-runner，默认5）')
    parser.add_argument('--quiet',
                       action='store_true',
                       default=False,
                       help='简洁模式：控制台只显示ERROR和进度信息（推荐用于大规模批量实验）')

    args = parser.parse_args()

    # 如果启用quiet模式或多进程模式，设置环境变量
    if args.quiet or (args.use_smart_runner and args.multiprocess):
        os.environ['FEDCL_CONSOLE_LOG_LEVEL'] = 'ERROR'
        print("✓ 启用简洁模式：控制台只显示 ERROR 和进度信息")
        print("  详细日志已保存到文件：logs/exp_*/")

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
    if args.use_smart_runner:
        # 使用智能调度器（GPU显存感知 + SQLite断点续跑）
        await run_table3_experiments_with_smart_runner(
            experiments,
            repetitions=args.repetitions,
            enable_gpu_scheduling=args.enable_gpu_scheduling,
            db_path=args.db_path,
            concurrent=args.concurrent,
            multiprocess=args.multiprocess,
            max_concurrent=args.max_concurrent
        )
    else:
        # 使用默认批量运行器
        await run_table3_experiments(
            experiments,
            parallel=args.parallel,
            max_parallel=args.max_parallel
        )


if __name__ == "__main__":
    asyncio.run(main())
