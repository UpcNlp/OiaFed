# examples/dataset_manager_example.py
"""
DatasetManager使用示例

展示如何使用DatasetManager进行数据集管理、缓存、验证和任务序列创建。
"""

import torch
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from fedcl.config.config_manager import DictConfig
from fedcl.data.dataset_manager import DatasetManager
from fedcl.data.dataset import Dataset
from fedcl.data.task_generator import TaskGenerator
from fedcl.data.split_strategy import IIDSplitStrategy


def create_sample_dataset(name: str, num_samples: int = 1000, num_classes: int = 10) -> Dataset:
    """创建示例数据集"""
    data = torch.randn(num_samples, 32, 32, 3)
    targets = torch.randint(0, num_classes, (num_samples,))
    return Dataset(name, data, targets)


def main():
    """主函数：演示DatasetManager的使用"""
    
    print("=== DatasetManager使用示例 ===\n")
    
    # 1. 创建配置
    config = DictConfig({
        'datasets': {
            'cifar10': {
                'type': 'torchvision',
                'name': 'CIFAR10',
                'root': './data',
                'download': False  # 设为False避免实际下载
            },
            'custom_dataset': {
                'type': 'custom',
                'data_path': './custom_data'
            }
        },
        'cache': {
            'enable': True,
            'max_size': '100MB',
            'strategy': 'LRU'
        },
        'task_generation': {
            'num_tasks': 5,
            'classes_per_task': 2,
            'type': 'class_incremental',
            'random_seed': 42
        }
    })
    
    # 2. 创建任务生成器和数据集管理器
    split_strategy = IIDSplitStrategy(config)
    task_generator = TaskGenerator(config, split_strategy)
    dataset_manager = DatasetManager(config, task_generator)
    
    print("✅ DatasetManager初始化完成")
    
    # 3. 创建和注册示例数据集
    print("\n📁 创建和注册数据集...")
    datasets = []
    for i in range(3):
        dataset = create_sample_dataset(f"sample_dataset_{i}", 500, 10)
        datasets.append(dataset)
        dataset_manager.register_dataset(f"sample_{i}", dataset)
        print(f"   - 注册数据集: sample_{i} (大小: {len(dataset)})")
    
    # 4. 获取数据集统计信息
    print("\n📊 数据集统计信息:")
    for i in range(3):
        stats = dataset_manager.get_dataset_statistics(f"sample_{i}")
        print(f"   - sample_{i}:")
        print(f"     * 样本数量: {stats['size']}")
        print(f"     * 类别数量: {stats['num_classes']}")
        print(f"     * 内存使用: {stats['memory_usage_mb']:.2f} MB")
        print(f"     * 数据形状: {stats['data_shape']}")
    
    # 5. 测试缓存功能
    print("\n💾 缓存功能测试:")
    dataset_manager.cache_dataset("cached_test", datasets[0])
    
    # 获取缓存统计
    cache_stats = dataset_manager.cache.get_stats()
    print(f"   - 缓存大小: {cache_stats['size']}")
    print(f"   - 内存使用: {cache_stats['memory_usage_mb']:.2f} MB")
    print(f"   - 缓存利用率: {cache_stats['utilization']:.2%}")
    
    # 6. 数据集验证
    print("\n✅ 数据集验证:")
    for i in range(3):
        validation_result = dataset_manager.validate_dataset(datasets[i])
        status = "通过" if validation_result.is_valid else "失败"
        print(f"   - sample_{i}: {status}")
        if validation_result.warnings:
            for warning in validation_result.warnings:
                print(f"     警告: {warning}")
    
    # 7. 创建任务序列
    print("\n🎯 创建任务序列:")
    try:
        tasks = dataset_manager.create_task_sequence("sample_0", 5)
        print(f"   - 创建了 {len(tasks)} 个任务")
        for i, task in enumerate(tasks):
            print(f"     * 任务 {i+1}: 类别 {task.classes}")
    except Exception as e:
        print(f"   - 任务创建失败: {e}")
    
    # 8. 列出可用数据集
    print("\n📋 可用数据集列表:")
    available_datasets = dataset_manager.list_available_datasets()
    for dataset_name in available_datasets:
        print(f"   - {dataset_name}")
    
    # 9. 管理器统计信息
    print("\n📈 DatasetManager统计:")
    manager_stats = dataset_manager.get_manager_statistics()
    print(f"   - 已注册数据集: {manager_stats['registered_datasets']}")
    print(f"   - 配置数据集: {manager_stats['configured_datasets']}")
    print(f"   - 缓存命中: {manager_stats['cache_hits']}")
    print(f"   - 缓存未命中: {manager_stats['cache_misses']}")
    print(f"   - 已加载数据集: {manager_stats['datasets_loaded']}")
    
    # 10. 清理缓存
    print("\n🧹 清理缓存...")
    dataset_manager.clear_cache()
    print("   - 缓存已清理")
    
    print("\n🎉 DatasetManager示例完成！")


if __name__ == "__main__":
    main()
