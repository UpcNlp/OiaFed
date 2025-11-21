# 批量实验CSV输出使用指南

## 功能概述

批量实验脚本已增强,运行结束后会自动生成包含详细信息的CSV文件,便于分析和比较实验结果。

## CSV文件字段说明

生成的CSV文件包含以下19个字段:

| 字段 | 说明 | 示例 |
|------|------|------|
| Experiment_Name | 实验名称 | MNIST_IID_FedAvg |
| Dataset | 数据集名称 | MNIST, CIFAR10, FMNIST |
| Model | 模型名称 | MNIST_PaperCNN, CIFAR10_PaperCNN |
| NonIID_Type | Non-IID类型简称 | IID, pk~Dir(0.5), #C=1 |
| Partition_Detail | 划分方式详细信息 | Dirichlet(α=0.5), Pathological(#C=2) |
| Algorithm | 联邦学习算法 | FedAvg, FedProx, SCAFFOLD |
| Num_Clients | 客户端数量 | 10 |
| Max_Rounds | 最大训练轮数 | 50 |
| Completed_Rounds | 实际完成轮数 | 12, 25, 50 |
| Local_Epochs | 本地训练轮数 | 10 |
| Batch_Size | 批次大小 | 64 |
| Learning_Rate | 学习率 | 0.01 |
| Final_Accuracy | 最终准确率 | 0.9823 |
| Final_Loss | 最终损失 | 0.0543 |
| Duration_sec | 运行时长(秒) | 234.56 |
| Early_Stopping | 是否启用Early Stopping | True/False |
| Patience | Early Stopping耐心值 | 5 |
| Status | 实验状态 | success/failed |
| Error | 错误信息 | (如果失败) |

## 使用方法

### 1. 运行批量实验

```bash
# 运行所有实验
python examples/reproduce_table3_experiments.py --mode all

# 运行单个数据集
python examples/reproduce_table3_experiments.py --mode single --dataset MNIST

# 快速测试
python examples/reproduce_table3_experiments.py --mode test
```

实验完成后会自动生成CSV文件:
- 默认位置: `experiments/table3_results.csv`

### 2. 分析结果

使用提供的分析脚本:

```bash
# 分析默认CSV文件
python analyze_results.py

# 分析指定CSV文件
python analyze_results.py experiments/table3_results.csv
```

分析脚本会生成:
1. **控制台报告**: 包含8个分析维度的详细统计
2. **Markdown报告**: `experiments/table3_results_report.md`

### 3. 使用Pandas进行自定义分析

```python
import pandas as pd

# 读取CSV
df = pd.read_csv('experiments/table3_results.csv')

# 示例1: 查看所有成功实验的准确率分布
success_df = df[df['Status'] == 'success']
print(success_df[['Dataset', 'Algorithm', 'Final_Accuracy']].sort_values('Final_Accuracy', ascending=False))

# 示例2: 对比不同算法在同一数据集上的表现
mnist_df = success_df[success_df['Dataset'] == 'MNIST']
print(mnist_df.groupby('Algorithm')['Final_Accuracy'].mean().sort_values(ascending=False))

# 示例3: 分析Early Stopping效果
early_stopped = success_df[success_df['Completed_Rounds'] < success_df['Max_Rounds']]
print(f"Early Stopping触发率: {len(early_stopped)/len(success_df)*100:.1f}%")
print(f"平均节省轮数: {(early_stopped['Max_Rounds'] - early_stopped['Completed_Rounds']).mean():.1f}")

# 示例4: Non-IID难度排名
noniid_difficulty = success_df.groupby('NonIID_Type')['Final_Accuracy'].mean().sort_values()
print("\nNon-IID难度排名 (准确率越低越难):")
print(noniid_difficulty)

# 示例5: 找出每个数据集的最佳算法组合
for dataset in success_df['Dataset'].unique():
    dataset_df = success_df[success_df['Dataset'] == dataset]
    best = dataset_df.loc[dataset_df['Final_Accuracy'].idxmax()]
    print(f"\n{dataset}最佳: {best['Algorithm']} + {best['NonIID_Type']} = {best['Final_Accuracy']:.4f}")
```

### 4. 使用Excel分析

直接用Excel打开CSV文件:
1. 打开 `experiments/table3_results.csv`
2. 使用Excel的数据透视表功能
3. 创建图表和可视化

推荐分析视角:
- 按数据集分组,对比不同算法
- 按算法分组,对比在不同数据集上的表现
- 按Non-IID类型,分析数据异质性对性能的影响
- Early Stopping效果分析

## 分析报告示例

运行 `python analyze_results.py` 后会显示:

```
====================================================================================================
实验结果分析报告
====================================================================================================

1. 基本统计
   - 总实验数: 270
   - 成功: 265 (98.1%)
   - 失败: 5 (1.9%)
   - 平均准确率: 0.7234
   - Early Stopping节省: 35.2%

2. 按数据集分组 - Top准确率
   - MNIST: 0.9823 (MNIST_IID_FedAvg)
   - CIFAR10: 0.7234 (CIFAR10_IID_FedAvg)
   - FMNIST: 0.8567 (FMNIST_pk~Dir(0.5)_SCAFFOLD)
   ...

3. 按算法分组 - 性能对比
   算法平均准确率排名:
   1. SCAFFOLD: 0.7543
   2. FedAvg: 0.7234
   3. FedProx: 0.7123
   ...

4. 按Non-IID类型分组 - 难度分析
   Non-IID难度排名 (从难到易):
   1. #C=1: 0.3456 (最难)
   2. pk~Dir(0.5): 0.6789
   3. IID: 0.8234 (最易)
   ...

5. Early Stopping效果分析
   - 提前停止: 180 (67.9%)
   - 平均节省轮数: 23.5
   - 总轮数节省: 35.2%

6. Top 10 最佳实验
   1. 0.9823 - MNIST_IID_FedAvg
   2. 0.8567 - FMNIST_#C=2_SCAFFOLD
   ...
```

## 文件输出位置

默认输出文件:
- **CSV结果**: `experiments/table3_results.csv`
- **Markdown报告**: `experiments/table3_results_report.md`
- **实验日志**: `logs/exp_*/`
- **MLflow记录**: `experiments/table3_mlruns/`

## 快速测试

使用测试脚本验证CSV输出功能:

```bash
# 1. 生成测试CSV
python test_csv_output.py

# 2. 分析测试CSV
python analyze_results.py experiments/test_csv_output.csv

# 3. 查看Markdown报告
cat experiments/test_csv_output_report.md
```

## 常见问题

### Q: CSV文件在哪里?
A: 默认在 `experiments/table3_results.csv`,可以在调用 `save_results_to_csv()` 时指定路径。

### Q: 如何修改CSV输出字段?
A: 修改 `examples/reproduce_table3_experiments.py` 中的 `save_results_to_csv()` 函数。

### Q: Completed_Rounds < Max_Rounds 意味着什么?
A: 表示Early Stopping被触发,训练在达到最大轮数前就收敛了。

### Q: 如何处理失败的实验?
A: 查看CSV中的Error字段,或查看对应的实验日志 `logs/exp_*/train/server.log`。

### Q: 可以生成图表吗?
A: 可以! 使用pandas + matplotlib或直接在Excel中创建图表:

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('experiments/table3_results.csv')
success_df = df[df['Status'] == 'success']

# 算法性能对比
algo_perf = success_df.groupby('Algorithm')['Final_Accuracy'].mean().sort_values()
algo_perf.plot(kind='barh', figsize=(10, 6))
plt.xlabel('Average Accuracy')
plt.title('Algorithm Performance Comparison')
plt.tight_layout()
plt.savefig('algorithm_comparison.png')
```

## 高级用法

### 合并多次实验结果

```python
import pandas as pd
import glob

# 读取所有CSV文件
csv_files = glob.glob('experiments/*_results.csv')
dfs = [pd.read_csv(f) for f in csv_files]

# 合并
merged_df = pd.concat(dfs, ignore_index=True)

# 保存合并结果
merged_df.to_csv('experiments/all_results.csv', index=False)

# 分析
print(f"总实验数: {len(merged_df)}")
print(merged_df.groupby('Algorithm')['Final_Accuracy'].describe())
```

### 统计显著性检验

```python
from scipy import stats

# 对比两个算法
fedavg_acc = success_df[success_df['Algorithm'] == 'FedAvg']['Final_Accuracy']
fedprox_acc = success_df[success_df['Algorithm'] == 'FedProx']['Final_Accuracy']

t_stat, p_value = stats.ttest_ind(fedavg_acc, fedprox_acc)
print(f"T-test结果: t={t_stat:.4f}, p={p_value:.4f}")
```

## 相关文件

- `examples/reproduce_table3_experiments.py` - 主批量实验脚本
- `analyze_results.py` - 结果分析脚本
- `test_csv_output.py` - CSV输出测试脚本
- `verify_batch_config.py` - 配置验证脚本
