# FedCL: 透明联邦学习框架

## 项目概述

FedCL (Federated Continual Learning) 是一个现代化的联邦学习框架，专注于透明化、易用性和生产环境部署。该框架支持多种联邦学习算法，包括标准的FedAvg以及先进的DDDR (Decentralized Diffusion-based Rehearsal) 等持续学习方法。

## 🚀 核心特性

### 1. 透明化设计
- **一行代码启动**：`fedcl.train()` 即可启动联邦学习
- **装饰器注册**：使用 `@fedcl.learner` 和 `@fedcl.trainer` 装饰器轻松注册组件
- **自动配置管理**：基于OmegaConf的智能配置系统

### 2. 多模式执行
- **伪联邦模式 (Pseudo Federation)**：单机多进程模拟联邦学习
- **真实联邦模式 (True Federation)**：分布式多节点联邦学习
- **自动模式 (Auto Mode)**：根据配置自动选择执行模式

### 3. 灵活的通信架构
- **透明通信**：支持进程内和进程间通信
- **Learner代理**：透明的客户端学习器代理机制
- **消息队列**：基于multiprocessing.Manager的可靠消息传递

### 4. 结构化日志系统
- **实验隔离**：按时间戳组织日志目录
- **分离式日志**：服务端和客户端日志独立存储
- **可配置格式**：支持自定义日志路径和格式

## 📁 项目结构

```
MOE-FedCL/
├── fedcl/                          # 核心框架代码
│   ├── api/                        # API接口层
│   │   ├── trainer.py              # 统一训练接口
│   │   └── decorators.py           # 组件注册装饰器
│   ├── fl/                         # 联邦学习核心
│   │   └── abstract_trainer.py    # 抽象训练器基类
│   ├── methods/                    # 具体算法实现
│   │   ├── trainers/              # 训练器实现
│   │   │   ├── standard_federation_trainer.py  # 标准FedAvg
│   │   │   └── dddr_federation_trainer.py      # DDDR训练器
│   │   └── learners/              # 学习器实现
│   │       └── dddr.py            # DDDR学习器
│   ├── execution/                  # 执行引擎
│   │   └── base_learner.py        # 基础学习器
│   ├── comm/                       # 通信模块
│   │   └── transparent_communication.py  # 透明通信
│   ├── models/                     # 模型定义
│   │   └── ldm/                    # Latent Diffusion Model
│   └── registry/                   # 组件注册表
├── config/                         # 配置文件
│   └── ldm_dddr.yaml              # LDM配置
├── logs/                           # 日志目录
│   └── experiment_YYYYMMDD-HH-MM-SS/
│       ├── server.log             # 服务端日志
│       └── clients/               # 客户端日志
│           ├── client_0.log
│           └── client_1.log
└── example_dddr_federation.py     # DDDR联邦学习示例
```

## 🛠️ 安装与配置

### 环境要求
- Python 3.8+
- PyTorch 1.8+
- CUDA (可选，用于GPU加速)

### 依赖安装
```bash
# 基础依赖
pip install torch torchvision
pip install omegaconf loguru tqdm

# 可选依赖
pip install transformers  # 用于BERT tokenizer
```

### 项目设置
```bash
# 克隆项目
git clone <repository-url>
cd MOE-FedCL

# 设置Python路径
export PYTHONPATH=/path/to/MOE-FedCL:$PYTHONPATH
```

## 📖 使用指南

### 1. 快速开始

#### 标准联邦学习
```python
from fedcl import train

# 一行代码启动联邦学习
result = train(
    trainer_type="standard",
    dataset="mnist",
    num_clients=3,
    num_rounds=10
)
```

#### DDDR联邦学习
```python
from fedcl import train

# 启动DDDR联邦学习
result = train(
    trainer_type="dddr",
    dataset="cifar10",
    num_clients=5,
    num_rounds=20,
    ldm_config="config/ldm_dddr.yaml"
)
```

### 2. 自定义学习器

```python
from fedcl import learner
import torch.nn as nn

@learner("custom")
class CustomLearner:
    def __init__(self, client_id, config):
        self.client_id = client_id
        self.config = config
        self.model = nn.Linear(784, 10)
    
    def train(self, data, global_weights=None):
        # 实现训练逻辑
        pass
    
    def evaluate(self, data):
        # 实现评估逻辑
        pass
```

### 3. 自定义训练器

```python
from fedcl import trainer
from fedcl.fl import AbstractFederationTrainer

@trainer("custom")
class CustomTrainer(AbstractFederationTrainer):
    def train(self, num_rounds, **kwargs):
        # 实现联邦训练逻辑
        pass
    
    def evaluate(self, test_data=None, **kwargs):
        # 实现联邦评估逻辑
        pass
    
    def _init_learner_proxies(self):
        # 初始化学习器代理
        pass
```

### 4. 配置管理

```python
# 基础配置
config = {
    "execution_mode": "pseudo_federation",  # 执行模式
    "trainer_type": "dddr",                  # 训练器类型
    "dataset": "cifar10",                   # 数据集
    "num_clients": 5,                       # 客户端数量
    "num_rounds": 20,                       # 训练轮数
    "federation": {                         # 联邦学习配置
        "client_selection": "random",
        "participation_rate": 1.0
    },
    "training": {                           # 训练配置
        "local_epochs": 1,
        "batch_size": 32,
        "learning_rate": 0.01
    },
    "logging": {                            # 日志配置
        "level": "INFO",
        "server_log_path": "logs/experiment_{date}/server.log",
        "client_log_path": "logs/experiment_{date}/clients/{client_id}.log"
    }
}
```

## 🔧 核心组件详解

### 1. 透明通信系统

FedCL采用透明通信设计，支持多种通信模式：

```python
from fedcl.comm import TransparentCommunication, CommunicationMode

# 进程内通信
comm = TransparentCommunication("node_id", mode=CommunicationMode.THREAD)

# 进程间通信
comm = TransparentCommunication("node_id", mode=CommunicationMode.PROCESS)
```

### 2. Learner代理机制

Learner代理提供透明的远程调用接口：

```python
# 获取学习器代理
learner_proxy = trainer.get_learner_proxy("client_0")

# 透明调用远程方法
result = learner_proxy.train(data, global_weights)
```

### 3. 组件注册系统

基于装饰器的组件注册机制：

```python
from fedcl.registry import get_trainer, get_learner

# 注册组件
@trainer("my_trainer")
class MyTrainer:
    pass

@learner("my_learner") 
class MyLearner:
    pass

# 获取组件
trainer_cls = get_trainer("my_trainer")
learner_cls = get_learner("my_learner")
```

## 🎯 DDDR算法实现

### 算法概述
DDDR (Decentralized Diffusion-based Rehearsal) 是一种基于扩散模型的持续学习方法，通过类反演和图像生成来缓解灾难性遗忘。

### 核心组件

#### 1. 类反演 (Class Inversion)
```python
# 在DDDRFederationTrainer中实现
def _federated_class_inversion(self, task_data):
    # 联邦类反演过程
    # 1. 收集所有客户端的类嵌入
    # 2. 聚合类嵌入
    # 3. 生成文本嵌入
    pass
```

#### 2. 图像生成 (Image Generation)
```python
def _synthesis_images(self, inv_text_embeds):
    # 基于反演的文本嵌入生成图像
    # 1. 使用Latent Diffusion Model
    # 2. 生成合成图像
    # 3. 更新模型参数
    pass
```

### 配置示例
```yaml
# config/ldm_dddr.yaml
model:
  base_learning_rate: 0.0001
  target: ldm.models.diffusion.ddpm.LatentDiffusion
  params:
    linear_start: 0.00085
    linear_end: 0.0120
    num_timesteps_cond: 1
    log_every_t: 200
    timesteps: 1000
    first_stage_config:
      target: ldm.models.autoencoder.AutoencoderKL
      params:
        monitor: val/rec_loss
        embed_dim: 4
        ddconfig:
          double_z: true
          z_channels: 4
          resolution: 256
          in_channels: 3
          out_ch: 3
          ch: 128
          ch_mult: [1, 2, 4, 4]
          num_res_blocks: 2
          attn_resolutions: []
          dropout: 0.0
    cond_stage_config:
      target: ldm.modules.encoders.modules.FrozenCLIPEmbedder
    personalization_config:
      target: ldm.modules.embedding_manager.EmbeddingManager
      params:
        placeholder_strings: ["<placeholder>"]
        initializer_words: ["*"]
        num_vectors_per_placeholder: 1
```

## 📊 日志系统

### 日志结构
```
logs/
└── experiment_20250901-16-02-32/    # 实验时间戳
    ├── server.log                   # 服务端日志
    └── clients/                     # 客户端日志目录
        ├── client_0.log            # 客户端0日志
        ├── client_1.log            # 客户端1日志
        └── ...
```

### 日志配置
```python
logging_config = {
    "level": "INFO",                    # 日志级别
    "server_log_path": "logs/experiment_{date}/server.log",
    "client_log_path": "logs/experiment_{date}/clients/{client_id}.log",
    "date": "20250901-16-02-32"        # 固定时间戳
}
```

### 日志内容示例
```
# 服务端日志
2025-09-01 16:02:32.389 | INFO | TransparentCommunication started
2025-09-01 16:02:32.390 | DEBUG | registered handler for register
2025-09-01 16:02:32.390 | INFO | 🟢 Server communication started
2025-09-01 16:02:33.435 | INFO | ✅ 客户端注册成功并创建代理: client_0

# 客户端日志
2025-09-01 16:02:33.432 | INFO | TransparentCommunication started
2025-09-01 16:02:33.434 | DEBUG | sent message register -> server
2025-09-01 16:02:33.435 | INFO | 📨 已向服务端发送注册消息
```

## 🔍 调试与监控

### 1. 通信调试
```python
# 启用调试模式
config["logging"]["level"] = "DEBUG"

# 查看通信消息
# 日志中会显示详细的消息传递信息
```

### 2. 性能监控
```python
# 监控训练进度
# 日志中会显示每轮的训练状态和指标
```

### 3. 错误处理
```python
# 异常会自动记录到日志中
# 包括堆栈跟踪和错误上下文
```

## 🚀 部署指南

### 1. 单机部署
```bash
# 伪联邦模式（推荐用于开发和测试）
python example_dddr_federation.py
```

### 2. 分布式部署
```bash
# 真实联邦模式（生产环境）
# 需要配置多台机器和网络通信
```

### 3. Docker部署
```dockerfile
# Dockerfile示例
FROM python:3.8-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "example_dddr_federation.py"]
```

## 🤝 贡献指南

### 开发环境设置
1. Fork项目
2. 创建功能分支
3. 实现功能
4. 添加测试
5. 提交Pull Request

### 代码规范
- 使用Python类型注解
- 遵循PEP 8代码风格
- 添加详细的文档字符串
- 编写单元测试

## 📄 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 提交Issue
- 发送邮件
- 参与讨论

---

**FedCL - 让联邦学习更简单、更透明、更强大！** 🚀
