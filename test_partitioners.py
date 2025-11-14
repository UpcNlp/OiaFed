"""
测试三种Non-IID数据划分器

测试 Measuring the Effects of Non-Identical Data Distribution 论文中的三种划分方式：
1. Pathological Label Skew (病理性标签倾斜) - 最严重
2. Feature Distribution Skew (特征分布倾斜) - 中等
3. Quantity Skew (数量倾斜) - 最温和
"""

import sys
import numpy as np
import torch
from torchvision import datasets, transforms

# 添加项目路径
sys.path.insert(0, '/home/nlp/ct/projects/MOE-FedCL')

from fedcl.methods.datasets.partition import (
    create_partitioner,
    PathologicalLabelSkewPartitioner,
    FeatureSkewPartitioner,
    QuantitySkewPartitioner
)


def test_pathological_label_skew():
    """测试病理性标签倾斜 - 最严重的Non-IID"""
    print("\n" + "="*80)
    print("测试1: Pathological Label Skew (病理性标签倾斜)")
    print("="*80)
    print("论文结果: CIFAR-10上FedAvg准确率从72.59% (IID) 降至 9.64% (#C=1)")
    print("-"*80)

    # 加载MNIST数据集 (简化测试)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # 测试不同的#C值
    for classes_per_client in [1, 2, 3, None]:
        print(f"\n--- #C = {classes_per_client if classes_per_client else 'unlimited'} ---")

        partitioner = create_partitioner('pathological', seed=42)
        client_indices = partitioner.partition(
            dataset,
            num_clients=10,
            alpha=0.5,  # 论文设置
            classes_per_client=classes_per_client
        )

        # 验证结果
        total_samples = sum(len(indices) for indices in client_indices.values())
        print(f"\n总样本数: {total_samples} / {len(dataset)}")


def test_feature_distribution_skew():
    """测试特征分布倾斜 - 中等影响"""
    print("\n" + "="*80)
    print("测试2: Feature Distribution Skew (特征分布倾斜)")
    print("="*80)
    print("论文结果: CIFAR-10上FedAvg准确率降至64.02%")
    print("数学表示: x̃ ~ Gau(0, 0.1)")
    print("-"*80)

    # 加载MNIST数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # 创建特征倾斜划分
    partitioner = create_partitioner('feature_skew', seed=42)

    # 获取第一个客户端的数据 (带噪声)
    noisy_dataset = partitioner.get_client_partition(
        dataset,
        client_id=0,
        num_clients=10,
        noise_std=0.1,
        different_noise_per_client=True
    )

    # 验证噪声效果
    print(f"\n验证噪声添加:")
    original_data, label = dataset[0]
    noisy_data, noisy_label = noisy_dataset[0]

    print(f"  原始数据范围: [{original_data.min():.3f}, {original_data.max():.3f}]")
    print(f"  噪声数据范围: [{noisy_data.min():.3f}, {noisy_data.max():.3f}]")
    print(f"  噪声标准差: {noisy_dataset.noise_std:.3f}")
    print(f"  标签是否相同: {label == noisy_label}")


def test_quantity_skew():
    """测试数量倾斜 - 最温和的Non-IID"""
    print("\n" + "="*80)
    print("测试3: Quantity Skew (数量倾斜)")
    print("="*80)
    print("论文结果: FM-NIST上FedAvg准确率88.80% vs IID的89.27%")
    print("数学表示: q ~ Dir(0.5)")
    print("-"*80)

    # 加载MNIST数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # 测试不同的alpha值
    for alpha in [0.1, 0.5, 1.0]:
        print(f"\n--- α = {alpha} ---")

        partitioner = create_partitioner('quantity_skew', seed=42)
        client_indices = partitioner.partition(
            dataset,
            num_clients=10,
            alpha=alpha,
            min_samples=100
        )

        # 验证结果
        sample_counts = [len(indices) for indices in client_indices.values()]
        print(f"数据量差异系数 (CV): {np.std(sample_counts) / np.mean(sample_counts):.2f}")


def compare_all_partitioners():
    """对比所有划分器的效果"""
    print("\n" + "="*80)
    print("对比所有Non-IID划分器")
    print("="*80)

    # 加载MNIST数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    num_clients = 10

    strategies = [
        ('IID', 'iid', {}),
        ('Pathological (#C=2)', 'pathological', {'alpha': 0.5, 'classes_per_client': 2}),
        ('Pathological (#C=1)', 'pathological', {'alpha': 0.5, 'classes_per_client': 1}),
        ('Quantity Skew', 'quantity_skew', {'alpha': 0.5}),
        ('Dirichlet (α=0.5)', 'dirichlet', {'alpha': 0.5}),
    ]

    print("\n划分器性能对比（预期影响从温和到严重）:")
    print("-"*80)

    for name, strategy, params in strategies:
        partitioner = create_partitioner(strategy, seed=42)

        if strategy == 'iid':
            client_indices = partitioner.partition(dataset, num_clients)
        else:
            client_indices = partitioner.partition(dataset, num_clients, **params)

        # 统计每个客户端的类别分布
        client_label_counts = []
        for indices in client_indices.values():
            labels = [dataset[idx][1] for idx in indices[:min(len(indices), 1000)]]
            unique_labels = len(set(labels))
            client_label_counts.append(unique_labels)

        avg_classes = np.mean(client_label_counts)
        sample_counts = [len(indices) for indices in client_indices.values()]

        print(f"\n{name}:")
        print(f"  平均类别数: {avg_classes:.2f}")
        print(f"  样本数CV: {np.std(sample_counts) / np.mean(sample_counts):.3f}")


if __name__ == '__main__':
    print("="*80)
    print("Non-IID数据划分器测试")
    print("基于论文: Measuring the Effects of Non-Identical Data Distribution")
    print("="*80)

    # 运行所有测试
    test_pathological_label_skew()
    test_feature_distribution_skew()
    test_quantity_skew()
    compare_all_partitioners()

    print("\n" + "="*80)
    print("✅ 所有测试完成!")
    print("="*80)
