# FedCL: 联邦持续学习框架

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/framework-PyTorch-red.svg)](https://pytorch.org/)

## 📖 简介

**FedCL (Federated Continual Learning)** 是一个功能强大的联邦持续学习框架，专为学术研究和产业应用设计。框架提供简洁的装饰器API和灵活的配置系统，支持从简单原型到复杂算法的渐进式开发。

### ✨ 核心特性

- 🎯 **装饰器驱动**: 通过 `@fedcl.loss`、`@fedcl.hook`、`@fedcl.model` 等装饰器简化自定义组件开发
- 🔧 **配置化管理**: 支持YAML配置文件驱动的实验管理，易于复现和扩展
- 🌐 **联邦学习**: 支持真联邦和伪联邦两种模式，满足不同场景需求
- 📊 **多学习器**: 支持多学习器协同训练，提供灵活的学习策略
- 🔍 **完整日志**: 详细的日志系统，支持DEBUG级别的训练过程追踪
- 🚀 **快速原型**: 提供 `quick_experiment()` API，5分钟内完成第一个实验

## 🛠️ 安装

### 环境要求

- Python >= 3.12
- PyTorch >= 2.7.1
- CUDA (可选，用于GPU加速)

### 安装方式

```bash
# 克隆项目
git clone https://github.com/UPC518/MOE-FedCL.git
cd MOE-FedCL

# 安装依赖 (推荐使用uv)
uv install

# 或使用pip
pip install -e .
```

## 🚀 快速开始

### 1. 5分钟快速体验

```python
import fedcl

# 零配置快速实验
results = fedcl.quick_experiment(
    method="fedavg", 
    dataset="mnist", 
    num_clients=3,
    num_rounds=3
)
print(f"平均准确率: {results.avg_accuracy:.2f}")
```

### 2. 基于配置文件的完整实验

#### 创建实验配置 `experiment_config.yaml`

```yaml
# 实验基本信息
experiment:
  name: "mnist_federated_demo"
  description: "MNIST联邦学习演示"
  version: "1.0"
  log_level: "DEBUG"

# 数据配置
dataset:
  name: "MNIST"
  path: "data/MNIST"
  split_strategy: "federated"
  split_config:
    num_clients: 3
    distribution: "iid"
    test_split: 0.2

# 联邦学习配置
federation:
  num_rounds: 3
  min_clients: 2
  max_clients: 3
  aggregation_strategy: "fedavg"

# 模型配置
model:
  type: "SimpleMLP"
  input_size: 784
  hidden_sizes: [256, 128]
  num_classes: 10

# 训练配置
training:
  local_epochs: 3
  batch_size: 32
  optimizer:
    type: "SGD"
    lr: 0.01
    momentum: 0.9
```

#### 运行实验

```python
from fedcl import FedCLExperiment

# 创建并运行实验
experiment = FedCLExperiment("configs/experiment_config.yaml")
results = experiment.run()

# 查看结果
print(f"实验完成！准确率: {results.final_accuracy:.3f}")
```

### 3. 自定义组件开发

#### 自定义损失函数

```python
import fedcl
import torch.nn.functional as F

@fedcl.loss("weighted_cross_entropy")
def weighted_cross_entropy(predictions, targets, context):
    """带权重的交叉熵损失"""
    weights = context.get_state("class_weights", None)
    return F.cross_entropy(predictions, targets, weight=weights)
```

#### 自定义训练钩子

```python
@fedcl.hook("before_epoch", priority=100)
class DataAugmentationHook:
    """数据增强钩子"""
    def execute(self, context, **kwargs):
        # 在每个epoch开始前进行数据增强
        dataloader = kwargs.get('dataloader')
        # 实现数据增强逻辑
        return {"augmented_dataloader": enhanced_dataloader}
```

#### 自定义辅助模型

```python
@fedcl.model("knowledge_distillation_teacher")
class TeacherModel:
    """知识蒸馏教师模型"""
    def __init__(self, config=None, context=None):
        self.model = self._load_pretrained_model()
        
    def get_soft_targets(self, inputs, temperature=4.0):
        """获取软标签"""
        with torch.no_grad():
            outputs = self.model(inputs)
            return F.softmax(outputs / temperature, dim=1)
```

## 📁 项目结构

```
FedCL/
├── fedcl/                    # 核心框架代码
│   ├── core/                 # 核心基类和组件
│   ├── federation/           # 联邦学习核心
│   ├── communication/        # 通信系统
│   ├── data/                 # 数据处理
│   ├── training/             # 训练引擎
│   ├── utils/                # 工具函数
│   └── __init__.py          # 主要API入口
├── configs/                  # 配置文件示例
│   └── mnist_federated_demo/ # MNIST演示配置
├── tests/                    # 测试代码
│   └── configs/             # 测试配置
├── examples/                 # 使用示例
├── docs/                     # 详细文档
└── logs/                     # 实验日志输出
```

## 🔧 配置系统

FedCL使用分层配置系统，支持多种配置文件：

### 主实验配置
- `experiment_config.yaml` - 实验主配置
- `server_config.yaml` - 服务端配置  
- `client_*_config.yaml` - 客户端配置

### 配置示例结构

```yaml
# 完整配置示例
experiment:
  name: "my_experiment"
  log_level: "DEBUG"
  
dataset:
  name: "MNIST"
  path: "data/MNIST"
  split_config:
    num_clients: 3
    distribution: "iid"
    
federation:
  num_rounds: 10
  aggregation_strategy: "fedavg"
  
model:
  type: "SimpleMLP"
  input_size: 784
  hidden_sizes: [256, 128]
  
training:
  local_epochs: 3
  batch_size: 32
  optimizer:
    type: "SGD"
    lr: 0.01
```

## 📊 支持的算法和数据集

### 联邦学习算法
- **FedAvg**: 联邦平均算法
- **FedProx**: 带正则化的联邦学习
- **SCAFFOLD**: 控制变量方法
- **自定义算法**: 通过装饰器轻松扩展

### 数据集
- **MNIST**: 手写数字识别
- **CIFAR-10/100**: 图像分类
- **自定义数据集**: 支持PyTorch Dataset格式

### 模型架构
- **SimpleMLP**: 多层感知机
- **ResNet**: 残差网络
- **自定义模型**: 通过注册系统扩展

## 🔍 日志和调试

### 日志级别配置

```yaml
experiment:
  log_level: "DEBUG"  # INFO, DEBUG, WARNING, ERROR
```

### 日志输出结构

```
logs/
└── experiment_20250804_160024/
    ├── main_experiment.log     # 主实验日志
    ├── server.log             # 服务端日志
    └── clients/               # 客户端日志
        ├── test_client_1.log
        ├── test_client_2.log
        └── test_client_3.log
```

### 调试工具

```bash
# 使用内置调试脚本
./scripts/debug_tools.sh

# 查看实验运行状态
python -m fedcl.debug.experiment_monitor
```

## 🧪 运行测试

```bash
# 运行所有测试
pytest

# 运行联邦学习集成测试
pytest tests/integration/test_federation_framework.py

# 运行MNIST真实数据测试
pytest tests/test_real_mnist_federation.py -v
```

## 📈 实验结果示例

运行MNIST联邦学习实验后，可以看到类似的训练日志：

```
2025-08-04 16:00:24 | INFO | 联邦学习开始，总轮次: 3
2025-08-04 16:00:24 | INFO | 客户端[test_client_1] | 开始执行训练阶段: default_training
2025-08-04 16:00:25 | INFO | 客户端[test_client_1] | Epoch 1 完成，损失: 0.6983
2025-08-04 16:00:25 | INFO | 客户端[test_client_1] | Epoch 2 完成，损失: 0.6634
2025-08-04 16:00:25 | INFO | 客户端[test_client_1] | Epoch 3 完成，损失: 0.6302
2025-08-04 16:00:26 | INFO | 服务端 | Round 1 聚合完成，全局模型已更新
```

## 🛡️ 安全特性

- **通信加密**: 支持TLS加密通信
- **身份验证**: 客户端-服务端身份验证
- **差分隐私**: 可选的差分隐私保护
- **安全聚合**: 防止模型逆向工程

## 🤝 贡献指南

我们欢迎社区贡献！请遵循以下步骤：

1. Fork 项目仓库
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

### 开发环境设置

```bash
# 安装开发依赖
uv install --group dev

# 运行代码格式化
black fedcl/
isort fedcl/

# 运行类型检查
mypy fedcl/
```

## 📚 进阶教程

### 1. 多学习器协同训练

```yaml
# client_config.yaml
learners:
  main_learner:
    class: "default"
    model:
      type: "SimpleMLP"
    priority: 0
    
  auxiliary_learner:
    class: "ewc"  # Experience Weighted Clustering
    model:
      type: "SimpleMLP" 
    priority: 1
```

### 2. 自定义聚合策略

```python
@fedcl.aggregator("weighted_fedavg")
class WeightedFedAvg(BaseAggregator):
    def aggregate(self, client_updates, client_weights=None):
        """基于数据量加权的联邦平均"""
        # 实现加权聚合逻辑
        return aggregated_model
```

### 3. 分布式部署

```yaml
# server_config.yaml
communication:
  host: "0.0.0.0"
  port: 8080
  ssl_enabled: true
  ssl_cert: "./certs/server.crt"
  ssl_key: "./certs/server.key"
```

## 🐛 故障排除

### 常见问题

1. **内存不足**
   ```yaml
   training:
     batch_size: 16  # 减小批次大小
   ```

2. **通信超时**
   ```yaml
   communication:
     timeout: 120.0  # 增加超时时间
   ```

3. **依赖冲突**
   ```bash
   uv install --force-reinstall
   ```

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- PyTorch团队提供的深度学习框架
- 联邦学习社区的开源贡献
- 所有为项目做出贡献的开发者

## 📞 联系我们

- **GitHub Issues**: [提交问题](https://github.com/UPC518/MOE-FedCL/issues)
- **文档**: [完整文档](docs/)
- **邮箱**: fedcl-team@example.com

---

**🌟 如果FedCL对您的研究有帮助，请给我们一个星标！**

---

## 📖 引用

如果您在研究中使用FedCL，请引用：

```bibtex
@misc{fedcl2025,
  title={FedCL: A Federated Continual Learning Framework},
  author={FedCL Development Team},
  year={2025},
  url={https://github.com/UPC518/MOE-FedCL}
}
```
