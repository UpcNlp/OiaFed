# FedCL 联邦学习框架使用手册

## 目录
1. [概述](#概述)
2. [系统架构](#系统架构)
3. [环境准备](#环境准备)
4. [快速开始](#快速开始)
5. [配置文件详解](#配置文件详解)
6. [核心组件使用](#核心组件使用)
7. [实验运行流程](#实验运行流程)
8. [日志监控与调试](#日志监控与调试)
9. [测试与验证](#测试与验证)
10. [高级功能](#高级功能)
11. [故障排除](#故障排除)
12. [最佳实践](#最佳实践)

---

## 概述

FedCL（Federated Learning Framework）是一个功能完整的联邦学习框架，支持多种聚合算法、灵活的配置管理、自适应通信机制和完善的实验管理系统。

### 核心特性
- ✅ **多种聚合算法**: FedAvg、FedProx等
- ✅ **自适应通信**: 自动选择本地/网络通信模式
- ✅ **灵活配置**: YAML配置文件驱动的实验设置
- ✅ **完整日志**: 详细的训练和聚合过程日志
- ✅ **实验管理**: 自动化实验运行和结果保存
- ✅ **多客户端支持**: 支持任意数量的联邦客户端
- ✅ **模型检查点**: 自动模型保存和恢复
- ✅ **测试框架**: 完整的单元测试和集成测试

---

## 系统架构

### 核心组件

```
FedCL Framework
├── 实验管理 (FedCLExperiment)
│   ├── 配置管理 (ConfigManager)
│   ├── 结果保存 (ExperimentResults)
│   └── 生命周期管理
├── 联邦协调器 (Federation Coordinators)
│   ├── 联邦服务器 (ImprovedFederatedServer)
│   ├── 联邦客户端 (MultiLearnerClient)
│   └── 状态管理 (StateManager)
├── 聚合器 (Aggregators)
│   ├── FedAvg聚合器
│   ├── FedProx聚合器
│   └── 自定义聚合器
├── 通信管理 (Communication)
│   ├── 自适应通信管理器
│   ├── 本地通信
│   └── 网络通信
├── 学习器 (Learners)
│   ├── 默认学习器
│   ├── 模型管理
│   └── 训练引擎
└── 工具组件 (Utils)
    ├── 日志管理
    ├── 配置解析
    └── 组件注册
```

### 数据流程

```
配置加载 → 实验初始化 → 服务器启动 → 客户端连接 → 
联邦训练轮次 → 模型聚合 → 全局模型分发 → 结果保存
```

---

## 环境准备

### 系统要求
- Python 3.8+
- PyTorch 1.9+
- 8GB+ RAM (推荐)
- 5GB+ 磁盘空间

### 安装依赖

```bash
# 克隆仓库
git clone <repository-url>
cd Moe-Fedcl

# 安装依赖（使用uv推荐）
uv install

# 或使用pip
pip install -r requirements.txt
```

### 数据准备

```bash
# MNIST数据集会自动下载到data/MNIST目录
# 首次运行时会自动下载，请确保网络连接正常
python -c "import torchvision.datasets as datasets; datasets.MNIST('data/MNIST', download=True)"
```

---

## 快速开始

### 1. 运行预配置的MNIST演示

```bash
# 使用预配置的MNIST联邦学习演示
cd configs/mnist_federated_demo
python -m fedcl.experiment.experiment .
```

### 2. 运行集成测试验证系统

```bash
# 运行完整的集成测试
pytest tests/test_real_mnist_federation.py -v

# 跳过长时间运行的测试
pytest tests/test_real_mnist_federation.py -m "not slow" -v
```

### 3. 使用Python脚本运行实验

```python
from fedcl.experiment.experiment import FedCLExperiment

# 创建实验实例
experiment = FedCLExperiment("configs/mnist_federated_demo")

# 运行实验
results = experiment.run()

print(f"实验完成，运行时间: {results.experiment_duration}")
print(f"联邦轮数: {results.total_rounds}")
print(f"参与客户端: {results.total_clients}")
```

---

## 配置文件详解

### 实验配置结构

```
configs/mnist_federated_demo/
├── experiment_config.yaml     # 主实验配置
├── server_config.yaml         # 服务器配置
├── client_1_config.yaml       # 客户端1配置
├── client_2_config.yaml       # 客户端2配置
├── client_3_config.yaml       # 客户端3配置
├── client_4_config.yaml       # 客户端4配置
└── data_split_config.yaml     # 数据分割配置
```

### 主实验配置 (experiment_config.yaml)

```yaml
experiment:
  name: "mnist_federated_unified_demo"
  description: "基于统一初始化流程的MNIST联邦学习演示"
  
# 统一初始化配置
unified_initialization:
  enabled: true
  config_dir: "configs/mnist_federated_demo"
  scan_order: ["data_split", "federation_server", "auxiliary_model", "client"]
  
# 实验参数
parameters:
  dataset: "MNIST"
  num_clients: 4
  num_rounds: 20
  local_epochs: 3
  batch_size: 32
  learning_rate: 0.01
  aggregation_algorithm: "fedavg"
  data_distribution: "iid"
```

### 服务器配置 (server_config.yaml)

```yaml
server:
  server_id: "mnist_test_server"
  host: "localhost"
  port: 8080
  
# 联邦学习配置
federation:
  num_rounds: 3
  min_updates_per_round: 2
  round_timeout: 120.0
  aggregation_strategy: "fedavg"
  
# 聚合器配置
aggregators:
  fedavg:
    type: "fedavg"
    weighted_average: true
    
# 模型管理
model_management:
  checkpoint_dir: "checkpoints"
  save_interval: 5
```

### 客户端配置 (client_1_config.yaml)

```yaml
client:
  client_id: "test_client_1"
  host: "localhost"
  port: 8081
  
# 多学习器配置
multi_learner:
  learners:
    default_learner:
      class: "default"
      model:
        type: "SimpleMLP"
        input_size: 784
        hidden_sizes: [256, 128]
        num_classes: 10
      optimizer:
        type: "SGD"
        lr: 0.01
        momentum: 0.9
      dataloader: "mnist_data"
      
# 数据加载器配置
dataloaders:
  mnist_data:
    type: "StandardDataLoader"
    dataset: "MNIST"
    data_dir: "data/MNIST"
    batch_size: 32
    split_config: "data_split_config.yaml"
    client_split_id: 0
```

---

## 核心组件使用

### 1. 实验管理器

```python
from fedcl.experiment.experiment import FedCLExperiment

# 从配置目录创建实验
experiment = FedCLExperiment("configs/mnist_federated_demo")

# 从单个配置文件创建实验  
experiment = FedCLExperiment("experiment_config.yaml")

# 运行实验
results = experiment.run()

# 获取实验信息
print(f"实验ID: {experiment.experiment_id}")
print(f"配置模式: {experiment.config_mode}")
print(f"输出目录: {experiment.output_dir}")
```

### 2. 联邦服务器

```python
from fedcl.federation.coordinators.federated_server import ImprovedFederatedServer
from fedcl.config.config_manager import DictConfig

# 创建服务器配置
server_config = DictConfig({
    "server": {
        "server_id": "test_server",
        "host": "localhost",
        "port": 8080
    },
    "federation": {
        "num_rounds": 5,
        "min_updates_per_round": 2,
        "round_timeout": 120.0,
        "aggregation_strategy": "fedavg"
    }
})

# 创建服务器实例
server = ImprovedFederatedServer.create_from_config(server_config)

# 启动联邦学习
results = server.start_federation()
```

### 3. 联邦客户端

```python
from fedcl.federation.coordinators.federated_client import MultiLearnerClient
from fedcl.config.config_manager import DictConfig

# 创建客户端配置
client_config = DictConfig({
    "client": {
        "client_id": "test_client",
        "host": "localhost", 
        "port": 8081
    },
    "multi_learner": {
        "learners": {
            "default_learner": {
                "class": "default",
                "model": {"type": "SimpleMLP", "input_size": 784}
            }
        }
    }
})

# 创建客户端实例
client = MultiLearnerClient.create_from_config(client_config)

# 开始客户端运行
client.start()
```

### 4. 聚合器

```python
from fedcl.implementations.aggregators.fedavg_aggregator import FedAvgAggregator

# 创建FedAvg聚合器
aggregator = FedAvgAggregator(weighted_average=True)

# 聚合模型更新
client_updates = [
    {"client_id": "client1", "num_samples": 1000, "model_update": model1},
    {"client_id": "client2", "num_samples": 1500, "model_update": model2}
]

# 执行聚合
aggregated_model = aggregator.aggregate(client_updates)
```

---

## 实验运行流程

### 标准实验运行

```python
# 1. 导入必要模块
from fedcl.experiment.experiment import FedCLExperiment
import logging

# 2. 设置日志级别
logging.basicConfig(level=logging.INFO)

# 3. 创建实验
experiment = FedCLExperiment("configs/mnist_federated_demo")

# 4. 运行实验
try:
    results = experiment.run()
    print("实验成功完成！")
    print(f"总轮数: {results.total_rounds}")
    print(f"参与客户端: {results.total_clients}")
    print(f"运行时间: {results.experiment_duration}")
except Exception as e:
    print(f"实验运行失败: {e}")
```

### 实验生命周期

1. **初始化阶段**
   - 配置文件加载和验证
   - 组件注册和创建
   - 输出目录准备

2. **启动阶段**
   - 服务器启动
   - 客户端连接
   - 通信建立

3. **训练阶段**
   - 联邦训练轮次
   - 客户端本地训练
   - 模型聚合

4. **完成阶段**
   - 结果保存
   - 资源清理
   - 实验总结

---

## 日志监控与调试

### 日志文件结构

```
logs/experiment_YYYYMMDD_HHMMSS/
├── federated_training.log      # 主要训练日志
├── client_test_client_1.log    # 客户端1日志
├── client_test_client_2.log    # 客户端2日志
├── client_test_client_3.log    # 客户端3日志
└── error.log                   # 错误日志
```

### 关键日志信息

#### 1. 服务器启动日志

```log
2025-08-04 11:43:05.864 | INFO | ImprovedFederatedServer initialized: mnist_test_server
2025-08-04 11:43:05.864 | INFO | Round config - timeout: 120.0s, min_updates: 2
2025-08-04 11:43:05.864 | INFO | Aggregator: FedAvgAggregator
```

#### 2. 客户端连接日志

```log
2025-08-04 11:43:05.910 | INFO | MultiLearnerFederatedClient初始化完成: test_client_3
2025-08-04 11:43:05.910 | INFO | Learners: ['default_learner']
```

#### 3. 聚合过程日志

```log
2025-08-04 11:43:06.392 | INFO | Starting aggregation with 3 updates
2025-08-04 11:43:08.033 | INFO | Round 1 completed successfully
```

#### 4. 实验完成日志

```log
2025-08-04 11:43:18.102 | INFO | Federation completed: 3 rounds
2025-08-04 11:43:18.106 | SUCCESS | Experiment completed in 12.29s
```

### 日志监控工具

```python
# 实时监控日志
import time
from pathlib import Path

def monitor_experiment_logs(log_dir):
    """监控实验日志"""
    log_file = Path(log_dir) / "federated_training.log"
    
    if not log_file.exists():
        print(f"等待日志文件创建: {log_file}")
        return
        
    with open(log_file, 'r') as f:
        # 移动到文件末尾
        f.seek(0, 2)
        
        while True:
            line = f.readline()
            if line:
                print(line.strip())
            else:
                time.sleep(0.1)

# 使用示例
# monitor_experiment_logs("logs/experiment_20250804_114305")
```

---

## 测试与验证

### 运行测试套件

```bash
# 运行所有测试
pytest tests/ -v

# 运行集成测试
pytest tests/test_real_mnist_federation.py -v

# 运行特定测试类
pytest tests/test_real_mnist_federation.py::TestSingleClientTraining -v

# 跳过慢速测试
pytest tests/test_real_mnist_federation.py -m "not slow" -v
```

### 单客户端验证

```python
# 运行单客户端测试验证基础功能
from fedcl.federation.coordinators.federated_client import MultiLearnerClient
from fedcl.config.config_manager import DictConfig

# 创建单客户端测试配置
config = DictConfig({
    "client": {"client_id": "test_client"},
    "multi_learner": {
        "learners": {
            "default_learner": {
                "class": "default",
                "model": {"type": "SimpleMLP", "input_size": 784}
            }
        }
    }
})

# 测试客户端创建
client = MultiLearnerClient.create_from_config(config)
print(f"客户端创建成功: {client.client_id}")
```

### 系统验证清单

- [ ] **配置验证**: 配置文件格式正确，路径有效
- [ ] **数据验证**: MNIST数据集完整，可正常加载
- [ ] **组件验证**: 服务器、客户端、聚合器正常初始化
- [ ] **通信验证**: 客户端与服务器通信正常
- [ ] **训练验证**: 本地训练正常执行，模型参数更新
- [ ] **聚合验证**: 聚合流程完整，无死锁现象
- [ ] **结果验证**: 实验结果正确保存，格式符合预期

---

## 高级功能

### 1. 自定义聚合器

```python
from fedcl.core.base_aggregator import BaseAggregator
import torch

class CustomAggregator(BaseAggregator):
    """自定义聚合器示例"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.custom_param = kwargs.get('custom_param', 1.0)
    
    def aggregate(self, client_updates):
        """实现自定义聚合逻辑"""
        if not client_updates:
            return None
            
        # 提取模型参数
        models = []
        weights = []
        
        for update in client_updates:
            if isinstance(update, dict):
                models.append(update['model_update'])
                weights.append(update.get('num_samples', 1))
            else:
                models.append(update)
                weights.append(1)
        
        # 自定义加权平均
        total_weight = sum(weights)
        aggregated_params = {}
        
        for key in models[0].keys():
            weighted_sum = sum(
                w * model[key] for w, model in zip(weights, models)
            )
            aggregated_params[key] = weighted_sum / total_weight * self.custom_param
            
        return aggregated_params
```

### 2. 自定义数据分割

```yaml
# data_split_config.yaml
data_split:
  strategy: "dirichlet"  # 非IID分布
  alpha: 0.5
  num_clients: 4
  
client_data_splits:
  client_0:
    train_indices: [0, 1000]
    test_indices: [0, 200]
  client_1:
    train_indices: [1000, 2000] 
    test_indices: [200, 400]
```

### 3. 动态客户端管理

```python
from fedcl.federation.managers.client_manager import ClientManager

# 创建客户端管理器
client_manager = ClientManager(selection_strategy="random")

# 添加客户端
client_manager.register_client("client_1", {"capability": "high"})
client_manager.register_client("client_2", {"capability": "medium"})

# 选择参与训练的客户端
selected_clients = client_manager.select_clients(
    num_clients=2,
    criteria={"capability": "high"}
)
```

### 4. 模型检查点管理

```python
from fedcl.federation.managers.model_manager import ModelManager

# 创建模型管理器
model_manager = ModelManager(checkpoint_dir="checkpoints")

# 保存模型检查点
model_manager.save_checkpoint(
    model_state=model.state_dict(),
    round_num=5,
    metadata={"accuracy": 0.95}
)

# 加载模型检查点
checkpoint = model_manager.load_checkpoint(round_num=5)
model.load_state_dict(checkpoint['model_state'])
```

---

## 故障排除

### 常见问题与解决方案

#### 1. 聚合流程卡死

**现象**: 日志显示"Starting aggregation"后无后续输出

**原因**: 锁嵌套导致死锁

**解决方案**:
```python
# 检查聚合器日志
grep "Starting aggregation" logs/*/federated_training.log

# 确认聚合流程状态
grep "Round.*completed" logs/*/federated_training.log
```

#### 2. 客户端连接失败

**现象**: 客户端无法连接到服务器

**排查步骤**:
1. 检查端口是否被占用
2. 确认服务器已启动
3. 验证网络配置

```bash
# 检查端口占用
lsof -i :8080

# 测试连接
telnet localhost 8080
```

#### 3. 数据加载错误

**现象**: MNIST数据集加载失败

**解决方案**:
```bash
# 重新下载数据集
rm -rf data/MNIST
python -c "import torchvision.datasets as datasets; datasets.MNIST('data/MNIST', download=True)"
```

#### 4. 内存不足

**现象**: OOM错误

**优化策略**:
- 减小批次大小
- 减少客户端数量
- 使用梯度压缩

```yaml
# 调整配置
parameters:
  batch_size: 16  # 减小批次大小
  num_clients: 2  # 减少客户端数量
```

#### 5. 配置文件错误

**现象**: 配置解析失败

**检查清单**:
- YAML格式正确
- 文件路径有效
- 必要字段完整

```bash
# 验证YAML格式
python -c "import yaml; yaml.safe_load(open('config.yaml'))"
```

### 调试模式

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 使用调试配置
experiment = FedCLExperiment("configs/debug_config.yaml")
```

---

## 最佳实践

### 1. 配置管理

- **版本控制**: 将配置文件纳入版本控制
- **环境分离**: 不同环境使用不同配置
- **参数验证**: 运行前验证关键参数

```yaml
# 推荐的配置结构
experiment:
  name: "descriptive_experiment_name"
  version: "1.0.0"
  environment: "development"  # development/testing/production
```

### 2. 实验管理

- **命名规范**: 使用有意义的实验名称
- **结果备份**: 定期备份重要实验结果
- **文档记录**: 记录实验目的和关键发现

```python
# 实验管理示例
experiment_name = f"mnist_fedavg_{num_clients}clients_{num_rounds}rounds"
experiment = FedCLExperiment(config_dir, experiment_name=experiment_name)
```

### 3. 性能优化

- **批次大小**: 根据可用内存调整
- **并行度**: 合理设置客户端数量
- **检查点**: 定期保存模型状态

```yaml
# 性能优化配置
parameters:
  batch_size: 32
  num_workers: 4
  prefetch_factor: 2
  
model_management:
  save_interval: 5
  max_checkpoints: 10
```

### 4. 监控与日志

- **日志级别**: 生产环境使用INFO，调试使用DEBUG
- **日志轮转**: 防止日志文件过大
- **关键指标**: 监控准确率、损失、通信开销

```python
# 日志配置
import logging
from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler(
    'federated_training.log',
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
logging.basicConfig(handlers=[handler], level=logging.INFO)
```

### 5. 安全考虑

- **通信加密**: 生产环境启用TLS
- **访问控制**: 限制客户端访问
- **数据隐私**: 确保本地数据不泄露

```yaml
# 安全配置示例
security:
  enable_tls: true
  cert_file: "server.crt"
  key_file: "server.key"
  client_auth_required: true
```

### 6. 扩展性设计

- **模块化**: 保持组件独立性
- **插件化**: 支持自定义扩展
- **配置驱动**: 通过配置控制行为

```python
# 扩展示例
from fedcl.registry.component_registry import ComponentRegistry

# 注册自定义组件
ComponentRegistry.register_aggregator("custom", CustomAggregator)
ComponentRegistry.register_learner("advanced", AdvancedLearner)
```

---

## 总结

FedCL框架提供了完整的联邦学习解决方案，具备以下优势：

### ✅ 验证通过的功能
- **聚合流程**: 无死锁，自动化多轮训练
- **组件注册**: 通过配置钩子正确注册聚合器
- **日志系统**: 完整可追踪的INFO级别日志
- **实验管理**: 自动结果保存和实验生命周期管理
- **测试覆盖**: 完善的单元测试和集成测试

### 🎯 适用场景
- **研究实验**: 快速搭建联邦学习实验
- **教学演示**: MNIST等经典数据集演示
- **算法开发**: 自定义聚合算法验证
- **性能评估**: 不同配置下的性能对比

### 📈 发展方向
- **更多数据集**: 支持CIFAR、ImageNet等
- **高级聚合**: FedProx、SCAFFOLD等算法
- **分布式部署**: 真实网络环境支持
- **可视化界面**: Web界面监控和控制

通过本使用手册，您可以快速上手FedCL框架，构建自己的联邦学习实验。如有问题，请参考故障排除章节或查看详细的测试用例文档。

---

*最后更新: 2025年8月4日*  
*版本: 1.0.0*  
*作者: FedCL开发团队*
