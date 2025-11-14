# 论文TABLE III实验复现指南

## 实验配置检查结果

### ✅ 已实现的算法（9个全部齐全）

| 算法 | 实现文件 | 状态 |
|------|---------|------|
| FedAvg | `fedcl/methods/aggregators/fedavg.py` | ✅ |
| FedProx | `fedcl/methods/aggregators/fedprox.py` | ✅ |
| SCAFFOLD | `fedcl/methods/aggregators/scaffold.py` | ✅ |
| FedNova | `fedcl/methods/aggregators/fednova.py` | ✅ |
| FedAdagrad | `fedcl/methods/aggregators/fedadam.py` | ✅ |
| FedYogi | `fedcl/methods/aggregators/fedyogi.py` | ✅ |
| FedAdam | `fedcl/methods/aggregators/fedadam.py` | ✅ |
| MOON | `fedcl/methods/learners/moon.py` | ✅ |
| FedBN | `fedcl/methods/aggregators/fedbn.py` | ✅ |

### ✅ 已实现的数据集（8个全部齐全）

| 数据集 | 实现文件 | 状态 |
|--------|---------|------|
| MNIST | `fedcl/methods/datasets/mnist.py` | ✅ |
| FMNIST | `fedcl/methods/datasets/fmnist.py` | ✅ |
| SVHN | `fedcl/methods/datasets/svhn.py` | ✅ |
| CINIC10 | `fedcl/methods/datasets/cinic10.py` | ✅ |
| CIFAR10 | `fedcl/methods/datasets/cifar10.py` | ✅ |
| FedISIC2019 | `fedcl/methods/datasets/fedisic2019.py` | ✅ |
| Adult | `fedcl/methods/datasets/adult.py` | ✅ |
| FCUBE | `fedcl/methods/datasets/fcube.py` | ✅ **新实现** |

### ✅ 已实现的Non-IID划分策略（全部齐全）

| 类别 | 划分策略 | 论文表示 | 实现 |
|------|---------|---------|------|
| Label distribution skew | Dirichlet分布 | pk ~ Dir(0.5) | ✅ |
| | Pathological (#C=1) | #C = 1 | ✅ |
| | Pathological (#C=2) | #C = 2 | ✅ |
| | Pathological (#C=3) | #C = 3 | ✅ |
| Feature distribution skew | Gaussian噪声 | x̃ ~ Gau(0.1) | ✅ |
| | Synthetic (FCUBE) | synthetic | ✅ |
| Quantity skew | Dirichlet分布 | q ~ Dir(0.5) | ✅ |
| IID | 均匀划分 | IID | ✅ |

---

## 使用指南

### 1. 环境准备

确保已安装MLflow（用于实验跟踪）：

```bash
pip install mlflow
```

### 2. 运行实验

#### 2.1 快速测试（推荐首次使用）

运行小规模测试实验（MNIST + IID + 3个算法）：

```bash
python examples/reproduce_table3_experiments.py --mode test
```

#### 2.2 运行单个数据集的所有实验

```bash
# 例如：只运行MNIST的所有实验
python examples/reproduce_table3_experiments.py --mode single --dataset MNIST

# 支持的数据集：MNIST, FMNIST, SVHN, CINIC10, CIFAR10, FedISIC2019, Adult, FCUBE
```

#### 2.3 运行特定Non-IID类型的实验

```bash
# 只运行标签分布倾斜的实验
python examples/reproduce_table3_experiments.py --mode noniid --noniid_type label_skew

# 支持的类型：label_skew, feature_skew, quantity_skew, iid
```

#### 2.4 运行全部实验（需要很长时间）

```bash
# 串行运行（更稳定）
python examples/reproduce_table3_experiments.py --mode all

# 并行运行（更快，但需要更多资源）
python examples/reproduce_table3_experiments.py --mode all --parallel --max_parallel 2
```

#### 2.5 实验规模估算

- **测试模式**: 3个实验 (~30分钟)
- **单个数据集**: ~60-70个实验 (~10-20小时)
- **全部实验**: 8个数据集 x ~60个配置 = ~500个实验 (~100-200小时)

**建议分阶段运行**，先运行几个数据集，检查结果无误后再运行全部。

### 3. 查看实验结果

#### 3.1 使用MLflow UI（推荐）

```bash
# 启动MLflow UI
mlflow ui --backend-store-uri experiments/table3_mlruns

# 然后在浏览器访问: http://localhost:5000
```

在MLflow UI中，你可以：
- 对比不同算法的性能曲线
- 按数据集、Non-IID类型、算法筛选实验
- 导出实验结果
- 下载模型和日志

#### 3.2 查看CSV结果文件

实验完成后，结果会自动保存到：

```bash
experiments/table3_results.csv
```

CSV文件包含字段：
- Dataset: 数据集名称
- NonIID_Type: Non-IID类型
- Algorithm: 算法名称
- Accuracy: 最终测试准确率
- Loss: 最终损失值
- Rounds: 训练轮数
- Duration_sec: 运行时间（秒）
- Status: 实验状态（success/failed）
- Error: 错误信息（如果失败）

### 4. 论文中的实验参数

批量脚本使用了论文中描述的标准参数：

```python
{
    'num_clients': 10,          # 客户端数量
    'max_rounds': 50,           # 通信轮数
    'batch_size': 64,           # 批大小
    'local_epochs': 10,         # 本地训练轮数
    'learning_rate': 0.01,      # 学习率
    'momentum': 0.9,            # 动量

    # 算法特定参数
    'fedprox_mu': 0.01,         # FedProx proximal term
    'moon_mu': 10,              # MOON对比学习权重
    'moon_temperature': 0.5,    # MOON温度参数
    'fedadam_beta1': 0.9,       # Adam动量
    'fedadam_beta2': 0.99,
}
```

如需修改参数，请编辑 `examples/reproduce_table3_experiments.py` 中的 `PAPER_CONFIG`。

### 5. 故障排查

#### 5.1 内存不足

如果遇到内存问题，建议：
- 降低 `--max_parallel` 参数（如设为1）
- 分批运行不同数据集
- 使用更小的数据集（如MNIST, FMNIST）

#### 5.2 实验失败

查看具体失败原因：
```bash
# 检查MLflow UI中的错误日志
# 或查看CSV文件中的Error列
```

常见问题：
- **数据集下载失败**: 检查网络连接，或手动下载数据集到 `./data` 目录
- **CUDA内存不足**: 减小batch_size或使用CPU
- **配置文件缺失**: 确保 `configs/` 目录完整

### 6. 自定义实验

如需运行自定义实验，可以修改脚本中的配置：

```python
# 修改数据集列表
DATASETS = ['MNIST', 'CIFAR10']

# 修改算法列表
ALGORITHMS = ['FedAvg', 'FedProx', 'MOON']

# 修改Non-IID设置
NONIID_SETTINGS = {
    'label_skew': [
        {'type': 'dirichlet', 'alpha': 0.5, 'name': 'pk~Dir(0.5)'},
    ],
}

# 修改训练参数
PAPER_CONFIG = {
    'max_rounds': 20,  # 减少轮数以加快实验
    'batch_size': 128,
    ...
}
```

---

## 预期结果

### 论文TABLE III的关键发现

1. **Label distribution skew** (pk ~ Dir(0.5), #C=k)
   - **最严重的Non-IID场景**
   - 病理性标签倾斜（#C=1）: FedAvg在CIFAR-10上准确率从72.59%降至9.64%
   - FedAdagrad和FedYogi表现最佳

2. **Feature distribution skew** (x̃ ~ Gau(0.1))
   - **中等影响**
   - **FedBN表现最佳**（通过保留客户端特定的BN统计量）
   - CIFAR-10: FedAvg准确率降至64.02%

3. **Quantity skew** (q ~ Dir(0.5))
   - **最温和的Non-IID形式**
   - FM-NIST: FedAvg 88.80% vs IID 89.27%（影响很小）

4. **算法选择建议**:
   - Label skew严重时 → FedAdagrad / FedYogi
   - Feature skew → FedBN
   - 轻度heterogeneity → FedProx / MOON
   - IID baseline → FedAvg

---

## 文件说明

- `examples/reproduce_table3_experiments.py` - 批量实验运行脚本
- `fedcl/methods/datasets/fcube.py` - FCUBE合成数据集实现
- `fedcl/experiment/mlflow_recorder.py` - MLflow实验记录器
- `experiments/table3_mlruns/` - MLflow实验结果存储目录
- `experiments/table3_results.csv` - CSV格式的实验结果

---

## 引用

如果你使用这个实验复现代码，请引用原论文：

```bibtex
@article{nechba2025fundamentals,
  title={Fundamentals and Experimental Analysis of Federated Learning Algorithms:
         A Comparative Study on Non-IID Data Silos},
  author={Nechba, Mohammed and El Afia, Abdellatif and Abdulrazak, Bessam},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2025}
}
```

---

## 问题反馈

如有问题，请通过以下方式反馈：
- GitHub Issues: [项目链接]
- Email: [联系邮箱]
