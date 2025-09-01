# DDDR (Diffusion-Driven Data Replay) 使用指南

## 概述

本指南介绍如何使用基于FedCL框架实现的DDDR联邦持续学习方法。

## 环境准备

### 1. 安装依赖

```bash
# 安装DDDR项目的依赖
pip install -r DDDR-master/requirements.txt

# 安装CLIP
pip install git+https://github.com/openai/CLIP.git

# 安装FedCL框架依赖
pip install torch torchvision omegaconf tqdm einops
```

### 2. 下载预训练模型

```bash
# 创建目录
mkdir -p models/ldm/text2img-large

# 下载预训练模型
wget -O models/ldm/text2img-large/model.ckpt \
  https://ommer-lab.com/files/latent-diffusion/nitro/txt2img-f8-large/model.ckpt

# 复制配置文件
cp DDDR-master/ldm/ldm_dddr.yaml ldm/ldm_dddr.yaml
```

### 3. 准备数据集

支持的数据集：
- CIFAR-100
- CIFAR-10  
- Tiny ImageNet
- ImageNet
- MNIST
- Fashion-MNIST

## 基本使用

### 1. 命令行运行

```bash
# 基本运行
python example_dddr_federation.py

# 自定义参数
python example_dddr_federation.py \
  --dataset cifar100 \
  --tasks 5 \
  --num_users 5 \
  --com_round 100 \
  --local_ep 5 \
  --local_bs 128 \
  --w_kd 10.0 \
  --w_scl 1.0 \
  --exp_name my_experiment
```

### 2. 参数说明

#### 通用参数
- `--exp_name`: 实验名称
- `--save_dir`: 保存目录
- `--seed`: 随机种子
- `--dataset`: 数据集名称
- `--tasks`: 任务数量
- `--num_users`: 客户端数量
- `--com_round`: 通信轮数
- `--local_ep`: 本地训练轮数
- `--local_bs`: 本地批处理大小

#### DDDR特定参数
- `--w_kd`: 知识蒸馏损失权重
- `--w_ce_pre`: 合成数据交叉熵损失权重
- `--w_scl`: 监督对比学习损失权重
- `--com_round_gen`: 生成器通信轮数
- `--g_local_train_steps`: 生成器本地训练步数
- `--pre_size`: 每类历史合成数据大小
- `--cur_size`: 每类当前合成数据大小
- `--n_iter`: 生成合成数据迭代次数

#### 扩散模型参数
- `--config`: 扩散模型配置文件
- `--ldm_ckpt`: LDM检查点路径
- `--g_local_bs`: 生成器本地批处理大小
- `--syn_image_path`: 预生成合成数据路径

#### FedCL框架参数
- `--mode`: 执行模式 (local/pseudo/federation)
- `--device`: 设备 (cuda/cpu)
- `--log_level`: 日志级别

## 高级使用

### 1. 使用预生成合成数据

```bash
python example_dddr_federation.py \
  --syn_image_path /path/to/synthetic/images \
  --dataset cifar100
```

### 2. 调整DDDR参数

```bash
# 增加知识蒸馏权重
python example_dddr_federation.py \
  --w_kd 15.0 \
  --w_ce_pre 0.8 \
  --w_scl 1.5

# 调整生成器参数
python example_dddr_federation.py \
  --com_round_gen 20 \
  --g_local_train_steps 100 \
  --pre_size 300 \
  --cur_size 100
```

### 3. 不同执行模式

```bash
# 本地模拟模式
python example_dddr_federation.py --mode local

# 伪联邦模式 (单机多进程)
python example_dddr_federation.py --mode pseudo

# 真实联邦模式 (多机网络)
python example_dddr_federation.py --mode federation
```

### 4. 不同数据集

```bash
# CIFAR-100
python example_dddr_federation.py --dataset cifar100

# Tiny ImageNet
python example_dddr_federation.py --dataset tiny_imagenet

# MNIST
python example_dddr_federation.py --dataset mnist
```

## 输出结果

### 1. 训练日志

程序运行时会显示详细的训练日志：

```
============================================================
DDDR (Diffusion-Driven Data Replay) 联邦学习
基于FedCL框架实现
============================================================
实验名称: beta_0.5_tasks_5_seed_2024_sigma_0_20241201_143022
保存目录: outputs/dddr/cifar100/beta_0.5_tasks_5_seed_2024_sigma_0_20241201_143022

检查DDDR依赖...
✓ LDM模块已安装
✓ CLIP模块已安装
✓ 所有依赖检查通过

训练配置:
  数据集: cifar100
  任务数: 5
  客户端数: 5
  通信轮数: 100
  本地轮数: 5
  批处理大小: 128
  执行模式: pseudo

DDDR配置:
  知识蒸馏权重: 10.0
  合成数据权重: 0.5
  对比学习权重: 1.0
  生成器通信轮数: 10
  生成器训练步数: 50
  历史数据大小: 200
  当前数据大小: 50

开始DDDR联邦训练...
任务 0: 类别 0-19
开始联邦类别反演训练
任务 0, 轮次 1/100
训练完成 - 轮次: 0, 任务: 0, 损失: 2.3456, 样本数: 1280
...
```

### 2. 保存文件

训练完成后会生成以下文件：

```
outputs/dddr/cifar100/experiment_name/
├── training_result.json          # 训练结果
├── syn_imgs/                     # 生成的合成图像
│   ├── task_0/
│   ├── task_1/
│   └── ...
└── logs/                         # 日志文件
```

### 3. 结果分析

`training_result.json` 包含：

```json
{
  "config": {
    "num_clients": 5,
    "num_tasks": 5,
    "com_rounds": 100,
    ...
  },
  "result": {
    "training_history": [...],
    "evaluation_history": [...],
    "final_accuracy": 0.8234,
    "total_tasks": 5,
    "total_rounds": 100
  },
  "training_time": 3600.5
}
```

## 故障排除

### 1. 依赖问题

**错误**: `ModuleNotFoundError: No module named 'ldm'`
**解决**: 
```bash
pip install -r DDDR-master/requirements.txt
```

**错误**: `ModuleNotFoundError: No module named 'clip'`
**解决**:
```bash
pip install git+https://github.com/openai/CLIP.git
```

### 2. 模型文件问题

**错误**: `LDM配置文件不存在`
**解决**:
```bash
cp DDDR-master/ldm/ldm_dddr.yaml ldm/ldm_dddr.yaml
```

**错误**: `LDM检查点不存在`
**解决**:
```bash
mkdir -p models/ldm/text2img-large
wget -O models/ldm/text2img-large/model.ckpt \
  https://ommer-lab.com/files/latent-diffusion/nitro/txt2img-f8-large/model.ckpt
```

### 3. 内存不足

**错误**: `CUDA out of memory`
**解决**:
```bash
# 减少批处理大小
python example_dddr_federation.py --local_bs 64 --g_local_bs 6

# 使用CPU
python example_dddr_federation.py --device cpu
```

### 4. 数据管理器问题

**错误**: `需要实现数据管理器`
**解决**: 当前版本需要实现数据管理器，可以参考FedCL框架的数据管理器接口。

## 性能优化

### 1. 计算资源

- **GPU**: 推荐使用至少8GB显存的GPU
- **内存**: 推荐至少16GB系统内存
- **存储**: 合成图像需要大量存储空间

### 2. 参数调优

- **生成器训练**: 增加 `g_local_train_steps` 提高生成质量
- **合成数据**: 增加 `pre_size` 和 `cur_size` 提高防遗忘效果
- **知识蒸馏**: 调整 `w_kd` 平衡新旧知识

### 3. 分布式训练

对于大规模实验，可以使用多GPU或多机训练：

```bash
# 多GPU训练
CUDA_VISIBLE_DEVICES=0,1,2,3 python example_dddr_federation.py

# 多机训练 (需要配置网络通信)
python example_dddr_federation.py --mode federation
```

## 扩展开发

### 1. 添加新数据集

在 `fedcl/models/transforms.py` 中添加新的数据变换：

```python
def get_train_transform(dataset_name: str):
    transforms_dict = {
        # 现有数据集...
        "new_dataset": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
    }
```

### 2. 自定义损失函数

在 `fedcl/models/losses.py` 中添加新的损失函数：

```python
class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, outputs, targets):
        # 自定义损失计算
        return loss
```

### 3. 修改网络架构

在 `fedcl/models/networks.py` 中修改网络架构：

```python
class CustomIncrementalNet(IncrementalNet):
    def __init__(self, config, pretrained=False):
        super().__init__(config, pretrained)
        # 自定义修改
```

## 总结

DDDR联邦持续学习方法通过扩散模型生成历史数据来缓解灾难性遗忘问题。基于FedCL框架的实现提供了：

- **完整的联邦训练流程**
- **灵活的配置参数**
- **多种执行模式**
- **详细的日志记录**
- **易于扩展的架构**

通过本指南，您可以快速上手DDDR方法，并根据需要进行定制和优化。
