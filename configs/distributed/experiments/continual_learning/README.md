# 持续学习实验配置使用指南

本目录包含联邦持续学习（Federated Continual Learning）实验的配置文件。

## 文件结构

```
configs/distributed/experiments/continual_learning/
├── README.md          # 本文件
├── server.yaml        # 服务器配置
├── client_0.yaml      # 客户端0配置
├── client_1.yaml      # 客户端1配置
└── client_2.yaml      # 客户端2配置
```

## 快速开始

### 1. 运行持续学习实验

```bash
cd /home/nlp/ct/projects/MOE-FedCL

# 使用run_federated_learning.py运行
PYTHONPATH=. python examples/run_federated_learning.py \
    --config configs/distributed/experiments/continual_learning/
```

### 2. 后台运行（推荐）

```bash
nohup PYTHONPATH=. python examples/run_federated_learning.py \
    --config configs/distributed/experiments/continual_learning/ \
    > /tmp/continual_learning_test.log 2>&1 &

# 查看日志
tail -f /tmp/continual_learning_test.log
```

### 3. 启用实验记录

```bash
PYTHONPATH=. python examples/run_federated_learning.py \
    --config configs/distributed/experiments/continual_learning/ \
    --sacred \
    --exp_name target_mnist_test
```

## 配置说明

### 当前配置

- **方法**: TARGET (ICCV 2023)
- **数据集**: MNIST
- **任务数**: 5个任务，每个任务2个类别
- **场景**: Class-Incremental Learning (CIL)
- **客户端数**: 3个
- **训练轮数**: 每个任务20轮

### TARGET方法参数

```yaml
learner:
  name: "TARGET"
  params:
    # 持续学习基本参数
    num_tasks: 5                    # 任务总数
    classes_per_task: 2             # 每个任务的类别数
    scenario: "class_incremental"   # 场景类型

    # 知识蒸馏参数
    use_distillation: true          # 启用知识蒸馏
    distill_temperature: 2.0        # 蒸馏温度
    distill_weight: 1.0             # 蒸馏损失权重

    # TARGET特定参数
    generator_lr: 0.001             # 生成器学习率
    num_pseudo_samples: 100         # 伪样本数量
```

## 切换到其他持续学习方法

### 使用FedKNOW方法

修改客户端配置文件中的learner部分：

```yaml
learner:
  name: "FedKNOW"
  params:
    num_tasks: 5
    classes_per_task: 2
    scenario: "class_incremental"

    # FedKNOW特定参数
    signature_ratio: 0.1           # 签名样本比例
    knowledge_weight: 0.5          # 知识蒸馏权重
    integration_method: "weighted_sum"  # ensemble, attention
```

### 使用GLFC方法

```yaml
learner:
  name: "GLFC"
  params:
    num_tasks: 5
    classes_per_task: 2
    scenario: "class_incremental"

    # GLFC特定参数
    gradient_compensation_weight: 0.5
    semantic_distill_weight: 1.0
    relation_temperature: 4.0
```

### 使用FedWeIT方法

```yaml
learner:
  name: "FedWeIT"
  params:
    num_tasks: 5
    classes_per_task: 2
    scenario: "class_incremental"

    # FedWeIT特定参数
    use_attention: true
    feature_dim: 512
    transfer_weight: 0.5
```

### 使用Fed-CPrompt方法

```yaml
learner:
  name: "FedCPrompt"
  params:
    num_tasks: 5
    classes_per_task: 2
    scenario: "class_incremental"

    # Fed-CPrompt特定参数
    prompt_length: 10
    embed_dim: 768
    contrastive_temperature: 0.07
    contrastive_weight: 0.5
```

### 使用LGA方法

```yaml
learner:
  name: "LGA"
  params:
    num_tasks: 5
    classes_per_task: 2
    scenario: "class_incremental"

    # LGA特定参数
    accumulation_steps: 4
    importance_method: "fisher"  # fisher, gradient_magnitude
    protection_threshold: 0.1
    update_ratio: 0.8
```

## 其他持续学习场景

### Task-Incremental Learning (TIL)

```yaml
learner:
  params:
    scenario: "task_incremental"
    num_tasks: 5
```

### Domain-Incremental Learning (DIL)

```yaml
learner:
  params:
    scenario: "domain_incremental"
    num_tasks: 5
```

## 支持的数据集

当前配置使用MNIST，可以替换为其他数据集：

```yaml
dataset:
  name: "CIFAR10"  # 或 FMNIST, SVHN, CIFAR100等
  params:
    root: "./data"
    train: true
    download: true
```

## 监控实验

### 查看实验日志

```bash
# 查看最新实验的服务器日志
tail -f logs/exp_*/train/server.log

# 查看客户端日志
tail -f logs/exp_*/train/client_0.log
```

### 查看保存的配置

```bash
# 查看最新实验的配置
cat logs/exp_$(ls -t logs/ | grep exp_ | head -1)/experiment_config.json | jq '.'
```

## 评估持续学习性能

持续学习的评估指标包括：

- **Average Accuracy (AA)**: 所有任务的平均准确率
- **Forgetting Measure (FM)**: 遗忘程度（越小越好）
- **Backward Transfer (BWT)**: 新任务对旧任务的影响
- **Forward Transfer (FWT)**: 旧任务对新任务的帮助

这些指标会在训练过程中自动计算并记录到日志中。

## 故障排除

### 问题1: 找不到模型/数据集

确保配置文件中的模型名称和数据集名称正确注册：

```bash
# 查看已注册的组件
PYTHONPATH=. python -c "from fedcl.api.registry import ComponentRegistry; ..."
```

### 问题2: 训练不收敛

持续学习任务较难，可以尝试：
- 增加local_epochs
- 调整learning_rate
- 增加distill_weight
- 使用不同的方法

### 问题3: 内存不足

- 减少batch_size
- 减少num_pseudo_samples (TARGET)
- 减少buffer_size (如果使用replay buffer)

## 参考论文

1. **TARGET**: Zhang et al., "TARGET: Federated Class-Continual Learning via Exemplar-Free Distillation", ICCV 2023
2. **FedKNOW**: Li et al., "FedKNOW: Federated Continual Learning with Signature Task Knowledge Integration", INFOCOM 2023
3. **GLFC**: Dong et al., "Federated Class-Incremental Learning", CVPR 2022
4. **FedWeIT**: Yoon et al., "Federated Continual Learning with Weighted Inter-client Transfer", ICML 2021
5. **Fed-CPrompt**: Qi et al., "Fed-CPrompt: Contrastive Prompt for Rehearsal-Free Federated Continual Learning", CVPR 2023
6. **LGA**: Layerwise Gradient Accumulation for Federated Continual Learning, TPAMI 2023

## 更多信息

- 完整文档：请参考项目根目录的README.md
- 问题反馈：https://github.com/your-repo/issues
