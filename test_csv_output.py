#!/usr/bin/env python3
"""
测试增强版CSV输出
使用模拟数据验证CSV格式
"""

import sys
from pathlib import Path

# 添加项目路径
root = Path(__file__).parent
sys.path.insert(0, str(root))

from examples.reproduce_table3_experiments import (
    create_experiment_config,
    save_results_to_csv,
    PAPER_CONFIG
)


def main():
    """测试CSV输出"""

    print("=" * 100)
    print("测试增强版CSV输出")
    print("=" * 100)

    # 创建几个测试实验配置
    experiments = []

    # 1. MNIST IID
    exp1 = create_experiment_config('MNIST', 'FedAvg', {'type': 'iid', 'name': 'IID'}, PAPER_CONFIG)
    experiments.append(exp1)

    # 2. CIFAR10 Dirichlet
    exp2 = create_experiment_config('CIFAR10', 'FedProx', {'type': 'dirichlet', 'alpha': 0.5, 'name': 'pk~Dir(0.5)'}, PAPER_CONFIG)
    experiments.append(exp2)

    # 3. FMNIST Pathological
    exp3 = create_experiment_config('FMNIST', 'SCAFFOLD', {'type': 'pathological', 'classes_per_client': 2, 'alpha': 0.5, 'name': '#C=2'}, PAPER_CONFIG)
    experiments.append(exp3)

    print(f"\n创建了 {len(experiments)} 个测试实验配置\n")

    # 创建模拟结果
    results = [
        {
            'name': 'MNIST_IID_FedAvg',
            'accuracy': 0.9823,
            'loss': 0.0543,
            'rounds': 12,
            'duration': 234.56,
            'status': 'success'
        },
        {
            'name': 'CIFAR10_pk~Dir(0.5)_FedProx',
            'accuracy': 0.7234,
            'loss': 0.8765,
            'rounds': 25,
            'duration': 1234.78,
            'status': 'success'
        },
        {
            'name': 'FMNIST_#C=2_SCAFFOLD',
            'accuracy': 0.8567,
            'loss': 0.4321,
            'rounds': 18,
            'duration': 567.89,
            'status': 'success'
        }
    ]

    # 保存到CSV
    output_file = "experiments/test_csv_output.csv"
    print(f"保存结果到: {output_file}\n")

    save_results_to_csv(results, output_file, experiments)

    # 读取并显示CSV内容
    print("\n" + "=" * 100)
    print("CSV文件内容预览:")
    print("=" * 100)

    with open(output_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            print(f"{i+1:3d}: {line.rstrip()}")

    print("\n" + "=" * 100)
    print("CSV字段说明:")
    print("=" * 100)
    print("""
    1. Experiment_Name      : 实验名称
    2. Dataset              : 数据集名称 (MNIST, CIFAR10, etc.)
    3. Model                : 模型名称 (MNIST_PaperCNN, CIFAR10_PaperCNN, etc.)
    4. NonIID_Type          : Non-IID类型简称 (IID, pk~Dir(0.5), #C=2, etc.)
    5. Partition_Detail     : 划分方式详细信息 (Dirichlet(α=0.5), Pathological(#C=2), etc.)
    6. Algorithm            : 联邦学习算法 (FedAvg, FedProx, SCAFFOLD, etc.)
    7. Num_Clients          : 客户端数量
    8. Max_Rounds           : 最大训练轮数
    9. Completed_Rounds     : 实际完成轮数 (可能因Early Stopping提前结束)
    10. Local_Epochs        : 本地训练轮数
    11. Batch_Size          : 批次大小
    12. Learning_Rate       : 学习率
    13. Final_Accuracy      : 最终准确率
    14. Final_Loss          : 最终损失
    15. Duration_sec        : 运行时长(秒)
    16. Early_Stopping      : 是否启用Early Stopping
    17. Patience            : Early Stopping耐心值
    18. Status              : 实验状态 (success/failed)
    19. Error               : 错误信息(如果失败)
    """)

    print("\n" + "=" * 100)
    print("✓ CSV输出测试完成!")
    print("=" * 100)
    print(f"\n可以使用Excel/Pandas等工具打开: {output_file}")
    print("\n示例分析:")
    print("  import pandas as pd")
    print(f"  df = pd.read_csv('{output_file}')")
    print("  print(df[['Dataset', 'Algorithm', 'Final_Accuracy']].sort_values('Final_Accuracy', ascending=False))")


if __name__ == "__main__":
    main()
