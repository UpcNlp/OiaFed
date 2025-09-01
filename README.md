# FedCL: 透明联邦持续学习框架

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

FedCL (Federated Continual Learning) 是一个全新的透明联邦学习框架，旨在让真联邦和伪联邦对用户完全透明，专注于算法逻辑而非分布式细节。

## 🎯 核心理念

**分布式联邦写代码过程和集中式一样，底层自动处理权重、梯度、特征获取等。**

## ✨ 主要特性

- 🚀 **一行代码启动**: `fedcl.train(dataset="mnist", num_clients=3, rounds=10)`
- 🔄 **透明执行模式**: 自动检测和适配本地/伪联邦/真联邦模式
- 🧩 **模块化设计**: 学习器、聚合器、评估器、训练器组件化
- 🎨 **装饰器驱动**: `@fedcl.learner`, `@fedcl.aggregator` 等简化组件注册
- ⚙️ **配置驱动**: YAML配置文件管理实验参数
- 🔧 **生产就绪**: 支持多种部署方式和错误处理
- 📊 **内置算法**: FedAvg、FedProx、SCAFFOLD等主流联邦学习算法

## 🏗️ 架构设计

```
┌─────────────────────────────────────┐
│              API Layer              │  ← 用户接口层
├─────────────────────────────────────┤
│           Transparent Layer         │  ← 透明代理层
├─────────────────────────────────────┤
│           Automation Layer          │  ← 自动化层
├─────────────────────────────────────┤
│           Execution Layer           │  ← 执行层
├─────────────────────────────────────┤
│           Comm Layer                │  ← 通信层
├─────────────────────────────────────┤
│           Methods Layer             │  ← 算法层
├─────────────────────────────────────┤
│           Registry Layer            │  ← 注册层
└─────────────────────────────────────┘
```

## 📦 安装

### 环境要求

- Python 3.8+
- PyTorch 1.8+
- CUDA (可选，用于GPU加速)

### 安装依赖

```bash
# 使用pip安装
pip install torch torchvision loguru omegaconf

# 或使用conda安装
conda install pytorch torchvision -c pytorch
pip install loguru omegaconf
```

### 克隆项目

```bash
git clone https://github.com/your-username/Moe-Fedcl.git
cd Moe-Fedcl
```

## 🚀 快速开始

### 1. 一行代码启动

```python
import fedcl

# 最简单的使用方式
result = fedcl.train(
    dataset="mnist",
    num_clients=3,
    rounds=10
)
print(f"最终准确率: {result.accuracy:.4f}")
```

### 2. 自定义模型

```python
import torch.nn as nn
from fedcl.methods.learners import DefaultLearner

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平输入
        return self.network(x)
    
    def forward_with_loss(self, x, target):
        output = self.forward(x)
        loss = self.criterion(output, target)
        return output, loss

# 创建学习器
config = {
    "model": {"instance": MyModel()},
    "optimizer": {"type": "adam", "learning_rate": 0.01},
    "local_epochs": 2
}
learner = DefaultLearner("client_0", config)
```

### 3. 使用StandardFederationTrainer

```python
from fedcl.methods.trainers import StandardFederationTrainer

# 配置训练器
config = {
    "num_clients": 3,
    "local_epochs": 2,
    "learning_rate": 0.01,
    "batch_size": 32,
    "aggregator": "fedavg",
    "learner": "default"
}

# 创建训练器并开始训练
trainer = StandardFederationTrainer(config)
result = await trainer.train()
print(f"训练完成，最终准确率: {result.accuracy:.4f}")
```

### 4. 自定义聚合器

```python
from fedcl.api import aggregator
from fedcl.methods.aggregators import AbstractAggregator

@aggregator
class MyAggregator(AbstractAggregator):
    def aggregate(self, client_results):
        # 实现自定义聚合逻辑
        aggregated_weights = {}
        total_samples = sum(r["num_samples"] for r in client_results)
        
        for key in client_results[0]["model_weights"].keys():
            aggregated_weights[key] = sum(
                r["model_weights"][key] * r["num_samples"] / total_samples
                for r in client_results
            )
        
        return {
            "aggregated_weights": aggregated_weights,
            "num_clients": len(client_results)
        }
```

## 📚 文档

- [项目设计文档](docs/项目设计文档.md) - 详细的设计思路和架构说明
- [快速入门指南](docs/快速入门指南.md) - 快速上手教程
- [API参考文档](docs/API参考文档.md) - 完整的API文档
- [数据集加载机制分析](docs/数据集加载机制分析.md) - 数据管理详解

## 🧪 示例

### MNIST联邦学习示例

```python
#!/usr/bin/env python3
"""
完整的MNIST联邦学习示例
"""

import torch.nn as nn
from fedcl.methods.learners import DefaultLearner
from fedcl.methods.trainers import StandardFederationTrainer

# 1. 定义模型
class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x)
    
    def forward_with_loss(self, x, target):
        output = self.forward(x)
        loss = self.criterion(output, target)
        return output, loss

# 2. 配置训练器
config = {
    "num_clients": 3,
    "local_epochs": 2,
    "learning_rate": 0.01,
    "batch_size": 32,
    "aggregator": "fedavg",
    "learner": "default"
}

# 3. 创建训练器并开始训练
trainer = StandardFederationTrainer(config)
result = await trainer.train()

print(f"🎉 训练完成！")
print(f"   最终准确率: {result.accuracy:.4f}")
print(f"   训练轮数: {result.rounds}")
print(f"   客户端数量: {result.num_clients}")
```

运行示例：
```bash
python example_mnist_federation.py
```

## 🔧 配置选项

### 基础配置

```python
config = {
    # 数据集配置
    "dataset": "mnist",
    "data_path": "./data",
    "batch_size": 32,
    
    # 联邦学习配置
    "num_clients": 3,
    "rounds": 10,
    "local_epochs": 2,
    "client_selection_ratio": 1.0,
    
    # 组件配置
    "learner": "default",
    "aggregator": "fedavg",
    "evaluator": "prototype",
    
    # 模型配置
    "model": {
        "type": "mlp",
        "input_dim": 784,
        "hidden_dims": [128, 64],
        "output_dim": 10
    },
    
    # 优化器配置
    "optimizer": {
        "type": "adam",
        "learning_rate": 0.01
    }
}
```

### 高级配置

```python
config = {
    # 执行模式配置
    "execution": {
        "mode": "auto",  # auto, local, pseudo, distributed
        "num_workers": 4,
        "timeout": 300
    },
    
    # 通信配置
    "communication": {
        "transport": "auto",  # auto, memory, process, network
        "host": "localhost",
        "port": 8080
    },
    
    # 数据分区配置
    "data_partition": {
        "type": "iid",  # iid, non_iid_label, non_iid_quantity
        "alpha": 0.5  # 用于non_iid_label的Dirichlet分布参数
    }
}
```

## 🎯 执行模式

### 本地模拟模式
- **特点**: 单机多进程模拟联邦学习
- **适用场景**: 算法验证、快速原型
- **优势**: 开发效率高，调试方便

### 伪联邦模式
- **特点**: 单机多进程，真实网络通信
- **适用场景**: 通信协议测试、性能基准
- **优势**: 真实通信，单机部署

### 真联邦模式
- **特点**: 多机分布式，真实网络通信
- **适用场景**: 生产环境、大规模部署
- **优势**: 真实分布式，可扩展性强

## 🔍 组件管理

### 查看可用组件

```python
# 列出所有已注册的组件
components = fedcl.list_components()
print("可用组件:", components)

# 获取组件详细信息
info = fedcl.get_component_info("fedavg")
print("FedAvg聚合器信息:", info)
```

### 内置组件

- **学习器**: `default`, `contrastive`, `personalized_client`, `meta`
- **聚合器**: `fedavg`, `fedprox`, `scaffold`, `fednova`, `fedadam`, `fedyogi`, `feddyn`
- **评估器**: `prototype`, `fairness`
- **训练器**: `standard_federation`, `personalized_federation`

## 🛠️ 开发指南

### 自定义组件

```python
from fedcl.api import learner
from fedcl.execution.base_learner import AbstractLearner

@learner
class CustomLearner(AbstractLearner):
    def __init__(self, client_id: str, config: Dict[str, Any]):
        super().__init__(client_id, config)
        # 初始化代码
    
    async def train_epoch(self, **kwargs):
        # 训练逻辑
        return {"model_weights": weights, "loss": loss}
    
    async def evaluate(self, **kwargs):
        # 评估逻辑
        return {"accuracy": acc, "loss": loss}
    
    def get_model_weights(self):
        return self.model.state_dict()
    
    def set_model_weights(self, weights):
        self.model.load_state_dict(weights)
```

### 最佳实践

1. **模型设计**: 实现 `forward_with_loss` 方法，支持内置损失计算
2. **学习器设计**: 继承 `AbstractLearner`，实现所有抽象方法
3. **配置管理**: 使用YAML文件管理配置，分离开发和生产配置
4. **错误处理**: 实现适当的异常处理，使用日志记录关键信息
5. **性能优化**: 选择合适的执行模式，优化数据传输

## 🐛 故障排除

### 常见问题

1. **组件未注册**
   ```
   ValueError: 学习器 'my_learner' 未注册
   ```
   **解决方案**: 确保使用 `@fedcl.learner` 装饰器注册组件

2. **配置错误**
   ```
   ValueError: 配置项 'dataset' 缺失
   ```
   **解决方案**: 检查配置文件，确保所有必需项都存在

3. **模型权重不匹配**
   ```
   RuntimeError: size mismatch
   ```
   **解决方案**: 确保所有客户端的模型结构一致

### 调试技巧

```python
# 启用调试日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 使用本地模式快速调试
config = {"execution": {"mode": "local"}}

# 检查组件注册状态
print(fedcl.list_components())
```

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🤝 贡献

欢迎贡献代码！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解贡献指南。

### 贡献方式

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📞 联系我们

- 项目主页: [https://github.com/your-username/Moe-Fedcl](https://github.com/your-username/Moe-Fedcl)
- 问题反馈: [Issues](https://github.com/your-username/Moe-Fedcl/issues)
- 讨论区: [Discussions](https://github.com/your-username/Moe-Fedcl/discussions)

## 🙏 致谢

感谢所有为这个项目做出贡献的开发者和研究人员！

---

**FedCL** - 让联邦学习变得简单透明！ 🚀
