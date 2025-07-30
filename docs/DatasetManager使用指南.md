# DatasetManager使用指南

## 概述

`DatasetManager` 是 FedCL 框架中的核心组件，负责统一管理数据集的加载、缓存、验证和任务序列创建。它提供了智能缓存、多格式数据集支持、并发安全访问等高级功能。

## 主要功能

### 1. 数据集管理
- 支持多种数据集格式：torchvision、自定义数据集、文件数据集
- 统一的数据集注册和检索接口
- 数据集元数据管理

### 2. 智能缓存系统
- LRU缓存策略，自动内存管理
- 可配置的缓存大小和策略
- 缓存统计和监控

### 3. 数据集验证
- 完整性检查（数据与标签大小匹配）
- 格式验证（数据类型检查）
- 分布验证（类别平衡性检查）
- 自定义验证规则

### 4. 任务序列创建
- 支持持续学习任务生成
- 类增量、域增量、任务增量学习
- 与TaskGenerator集成

### 5. 并发安全
- 线程安全的数据集注册和访问
- 原子操作保证数据一致性

## 基本使用

### 初始化

```python
from fedcl.config.config_manager import DictConfig
from fedcl.data.dataset_manager import DatasetManager
from fedcl.data.task_generator import TaskGenerator
from fedcl.data.split_strategy import IIDSplitStrategy

# 创建配置
config = DictConfig({
    'datasets': {
        'cifar10': {
            'type': 'torchvision',
            'name': 'CIFAR10',
            'root': './data',
            'download': True
        }
    },
    'cache': {
        'enable': True,
        'max_size': '2GB',
        'strategy': 'LRU'
    }
})

# 初始化组件
split_strategy = IIDSplitStrategy(config)
task_generator = TaskGenerator(config, split_strategy)
dataset_manager = DatasetManager(config, task_generator)
```

### 数据集操作

```python
# 加载数据集
dataset = dataset_manager.load_dataset('cifar10', config.datasets.cifar10)

# 注册数据集
custom_dataset = create_custom_dataset()
dataset_manager.register_dataset('custom', custom_dataset)

# 获取数据集统计
stats = dataset_manager.get_dataset_statistics('cifar10')
print(f"数据集大小: {stats['size']}")
print(f"类别数量: {stats['num_classes']}")

# 验证数据集
result = dataset_manager.validate_dataset(dataset)
if result.is_valid:
    print("数据集验证通过")
```

### 缓存管理

```python
# 手动缓存数据集
dataset_manager.cache_dataset('my_dataset', dataset)

# 获取缓存统计
cache_stats = dataset_manager.cache.get_stats()
print(f"缓存使用率: {cache_stats['utilization']:.2%}")

# 清理缓存
dataset_manager.clear_cache()  # 清理所有
dataset_manager.clear_cache('specific_dataset')  # 清理特定数据集
```

### 任务序列创建

```python
# 创建持续学习任务序列
tasks = dataset_manager.create_task_sequence('cifar10', num_tasks=5)

for i, task in enumerate(tasks):
    print(f"任务 {i}: 类别 {task.classes}")
```

## 配置选项

### 数据集配置

```yaml
datasets:
  cifar10:
    type: "torchvision"
    name: "CIFAR10"
    root: "./data"
    download: true
    transforms:
      train:
        - name: "random_horizontal_flip"
          params: {p: 0.5}
        - name: "normalize" 
          params:
            mean: [0.4914, 0.4822, 0.4465]
            std: [0.2023, 0.1994, 0.2010]
  
  custom_dataset:
    type: "custom"
    data_path: "./custom_data"
    loader_class: "CustomDatasetLoader"
    
  file_dataset:
    type: "file"
    file_path: "./data/dataset.pkl"
```

### 缓存配置

```yaml
cache:
  enable: true
  max_size: "2GB"      # 最大缓存大小
  strategy: "LRU"      # 缓存策略
  persist: true        # 是否持久化
```

### 验证配置

```yaml
validation:
  enable: true
  strict_mode: false   # 严格模式
  checks:
    integrity: true    # 完整性检查
    format: true       # 格式检查  
    size: true         # 大小检查
    labels: true       # 标签检查
    distribution: true # 分布检查
```

## 支持的数据集类型

### 1. Torchvision数据集
自动支持所有torchvision.datasets中的数据集：
- CIFAR10/CIFAR100
- MNIST/FashionMNIST
- ImageNet
- 等等

### 2. 自定义数据集
通过指定数据路径和加载器类支持自定义数据集格式。

### 3. 文件数据集
支持从pickle文件加载预处理的数据集。

## 性能优化

### 缓存策略
- **LRU (Least Recently Used)**: 默认策略，适合大多数场景
- 自动内存管理，防止内存溢出
- 智能预取和延迟加载

### 并发优化
- 线程安全的数据访问
- 读写锁保护关键资源
- 原子操作确保数据一致性

### 内存优化
- 智能内存使用监控
- 大数据集分块处理
- 内存映射支持

## 错误处理

```python
from fedcl.data.dataset_manager import (
    DatasetManagerError,
    DatasetNotFoundError, 
    DatasetCacheError,
    DatasetValidationError
)

try:
    dataset = dataset_manager.load_dataset('unknown', config)
except DatasetNotFoundError as e:
    print(f"数据集未找到: {e}")
except DatasetValidationError as e:
    print(f"数据集验证失败: {e}")
```

## 监控和调试

### 统计信息
```python
# 获取管理器统计
stats = dataset_manager.get_manager_statistics()
print(f"已注册数据集: {stats['registered_datasets']}")
print(f"缓存命中率: {stats['cache_hits']/(stats['cache_hits']+stats['cache_misses']):.2%}")

# 获取缓存统计
cache_stats = dataset_manager.cache.get_stats()
print(f"缓存大小: {cache_stats['size']}")
print(f"内存使用: {cache_stats['memory_usage_mb']:.2f} MB")
```

### 日志记录
DatasetManager使用loguru进行日志记录，支持以下级别：
- `INFO`: 基本操作信息
- `DEBUG`: 详细调试信息
- `WARNING`: 警告信息
- `ERROR`: 错误信息

## 最佳实践

### 1. 配置管理
- 使用YAML配置文件管理数据集设置
- 为不同环境维护不同的配置文件
- 验证配置完整性

### 2. 缓存使用
- 根据可用内存调整缓存大小
- 监控缓存命中率
- 定期清理不需要的缓存

### 3. 性能优化
- 使用合适的batch size
- 预加载常用数据集
- 避免频繁的数据集切换

### 4. 错误处理
- 实现适当的异常处理
- 验证数据集完整性
- 记录操作日志

## 示例项目

查看 `examples/dataset_manager_example.py` 获取完整的使用示例。

## API参考

详细的API文档请参考各个类的docstring：
- `DatasetManager`: 主要管理器类
- `DatasetValidationManager`: 数据集验证管理器
- `DatasetCache`: 缓存管理器
