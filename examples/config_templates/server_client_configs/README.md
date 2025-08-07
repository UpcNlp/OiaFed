# ====================================
# FedCL 分布式联邦学习配置说明
# ====================================

## 📋 配置文件架构说明

### 配置文件区别对比

#### 1. **集中式配置** (`experiment_config.yaml`)
```
单一配置文件
├── 实验设置
├── 数据配置  
├── 联邦学习配置
├── 模型配置
├── 训练配置
├── 评估配置
├── Hook配置
└── 日志配置
```

**特点:**
- ✅ 配置集中，管理简单
- ✅ 适合快速原型和单机测试
- ✅ 所有配置在一个文件中
- ❌ 无法支持异构客户端
- ❌ 不适合真实分布式环境
- ❌ 客户端配置无法个性化

**适用场景:**
- 算法验证和原型开发
- 单机模拟联邦学习
- 快速实验和调试

#### 2. **分布式配置** (`server_config.yaml` + `client/`)
```
服务端-客户端分离
├── server_config.yaml          # 服务端配置
│   ├── 服务端设置
│   ├── 联邦学习管理
│   ├── 全局模型配置
│   ├── 聚合策略
│   └── 客户端管理
└── client/                     # 客户端配置文件夹
    ├── config_1.yaml          # 客户端1配置
    ├── config_2.yaml          # 客户端2配置  
    ├── config_3.yaml          # 客户端3配置
    └── config_template.yaml   # 客户端配置模板
        ├── 客户端特定设置
        ├── 本地数据配置
        ├── Learner选择
        ├── 本地训练配置
    └── 个性化Hook
```

**特点:**
- ✅ 支持异构客户端配置
- ✅ 真实分布式部署
- ✅ 客户端个性化设置
- ✅ 灵活的Learner组合
- ❌ 配置文件较多
- ❌ 管理复杂度较高

**适用场景:**
- 真实联邦学习部署
- 异构客户端环境
- 多种Learner协作实验
- 大规模分布式训练

## 🏗️ 分布式配置架构

### 服务端配置 (`server_config.yaml`)

**核心功能:**
- **客户端管理**: 注册、选择、监控
- **全局模型**: 初始化、更新、分发
- **聚合策略**: FedAvg、FedProx、自定义聚合
- **轮次管理**: 超时、重试、协调
- **全局评估**: 服务端测试数据评估

**关键配置项:**
```yaml
# 联邦学习核心配置
federation:
  num_rounds: 10                    # 总轮次
  client_selection_strategy: "random" # 客户端选择策略
  aggregation_strategy: "fedavg"    # 聚合策略
  
# 客户端管理
client_management:
  registration_timeout: 300          # 注册超时
  heartbeat_interval: 30            # 心跳间隔
  max_idle_time: 600                # 最大空闲时间
```

### 客户端配置 (`client/config_*.yaml`)

**核心功能:**
- **本地训练**: Learner选择、本地优化
- **数据管理**: 本地数据分片、预处理
- **模型更新**: 本地模型训练、参数上传
- **个性化设置**: 不同Learner、Hook组合

**支持的异构配置:**
```yaml
# 客户端1: 基础配置 (client/config_1.yaml)
learners:
  default_learner:
    class: "default"
    optimizer: { type: "SGD", lr: 0.01 }
    
# 客户端2: EWC持续学习 (client/config_2.yaml)
learners:
  ewc_learner:
    class: "ewc"
    ewc_config: { lambda_ewc: 0.4 }
    
# 客户端3: 多Learner协作 (client/config_3.yaml)
learners:
  replay_learner: { class: "replay" }
  mas_learner: { class: "mas" }
multi_learner:
  coordination: { strategy: "ensemble" }
```

## 🚀 使用指南

### 1. 选择配置架构

**使用集中式配置的情况:**
```bash
# 快速实验
python main.py --config experiment_config.yaml

# 算法验证
python experiments/algorithm_test.py --config simple_config.yaml
```

**使用分布式配置的情况:**
```bash
# 启动服务端
python server.py --config server_config.yaml

# 启动客户端1
python client.py --config client_1_config.yaml

# 启动客户端2  
python client.py --config client_2_config.yaml

# 启动客户端3
python client.py --config client_3_config.yaml
```

### 2. 配置文件定制

#### 创建新客户端配置
```bash
# 复制模板配置
cp client/config_template.yaml client/config_4.yaml

# 修改关键配置
vim client/config_4.yaml
```

**需要修改的关键项:**
```yaml
client:
  id: "fedcl_client_4"                # 更新客户端ID
  name: "FedCL Client 4"              # 更新客户端名称

federated_config:
  client_id: 4                        # 更新客户端编号

logging:
  log_dir: "logs/client_4"            # 更新日志目录
  formatters:
    default: "[%(asctime)s][CLIENT-4][%(name)s][%(levelname)s] %(message)s"
```

#### 定制Learner组合
```yaml
# 纯EWC配置
learners:
  ewc_learner:
    enabled: true
  default_learner:
    enabled: false
    
# 多Learner集成
learners:
  ewc_learner: { enabled: true, weight: 0.4 }
  replay_learner: { enabled: true, weight: 0.4 }
  mas_learner: { enabled: true, weight: 0.2 }
```

### 3. 实验场景示例

#### 场景1: 同构客户端基础联邦学习
```yaml
# 所有客户端使用相同配置
# server_config.yaml
federation:
  aggregation_strategy: "fedavg"
  
# client_*_config.yaml (所有客户端相同)
learners:
  default_learner:
    class: "default"
    optimizer: { type: "SGD", lr: 0.01 }
```

#### 场景2: 异构客户端持续学习
```yaml
# client_1: 默认learner
learners:
  default_learner: { class: "default" }

# client_2: EWC learner  
learners:
  ewc_learner: { class: "ewc", lambda_ewc: 0.4 }

# client_3: Replay learner
learners:
  replay_learner: { class: "replay", buffer_size: 500 }
```

#### 场景3: 多Learner协作实验
```yaml
# client_3_config.yaml
learners:
  replay_learner: { enabled: true }
  mas_learner: { enabled: true }
  
multi_learner:
  coordination:
    strategy: "ensemble"
    ensemble_config:
      voting_strategy: "weighted"
      learner_weights:
        replay_learner: 0.7
        mas_learner: 0.3
```

## 🔧 高级配置

### 1. 数据分布配置

#### IID数据分布
```yaml
federated_config:
  distribution: "iid"
  samples_per_client: 1000
```

#### 非IID数据分布
```yaml
federated_config:
  distribution: "non_iid"
  non_iid_config:
    alpha: 0.5                      # Dirichlet分布参数
    min_samples_per_class: 10       # 每类最少样本
```

#### 高度非IID（极端情况）
```yaml
federated_config:
  distribution: "non_iid"
  non_iid_config:
    alpha: 0.1                      # 更小的alpha，更不均衡
    min_samples_per_class: 5
```

### 2. 性能优化配置

#### 通信优化
```yaml
communication:
  timeout: 120.0                    # 增加超时时间
  
performance:
  model_compression: true           # 启用模型压缩
  gradient_compression: true        # 启用梯度压缩
```

#### 资源优化
```yaml
system:
  num_threads: 4                    # 增加线程数
  memory_limit: "8GB"               # 设置内存限制
  
performance:
  mixed_precision: true             # 启用混合精度
  gradient_checkpointing: true      # 启用梯度检查点
```

### 3. 安全与隐私配置

#### 差分隐私
```yaml
privacy:
  differential_privacy:
    enabled: true
    epsilon: 1.0                    # 隐私预算
    delta: 1e-5
```

#### 安全聚合
```yaml
privacy:
  secure_aggregation:
    enabled: true
```

## 📊 配置对比总结

| 特性 | 集中式配置 | 分布式配置 |
|------|------------|------------|
| **文件数量** | 1个 | server + client/目录 |
| **管理复杂度** | 低 | 中等 |
| **异构支持** | ❌ | ✅ |
| **真实部署** | ❌ | ✅ |
| **快速原型** | ✅ | ❌ |
| **多Learner** | 有限 | 完全支持 |
| **个性化配置** | ❌ | ✅ |
| **扩展性** | 低 | 高 |

## 🎯 选择建议

### 使用集中式配置，当:
- 🔬 **算法研究**: 验证新的持续学习算法
- 🚀 **快速原型**: 快速测试想法和概念
- 🧪 **单机实验**: 在单台机器上模拟联邦学习
- 📚 **学习使用**: 初次接触框架，学习基本用法

### 使用分布式配置，当:
- 🌐 **真实部署**: 在真实分布式环境中部署
- 🔄 **异构环境**: 不同客户端有不同的硬件/软件环境
- 🤝 **多Learner协作**: 研究多种持续学习方法的协作
- 📈 **大规模实验**: 大量客户端的联邦学习实验
- 🎯 **个性化需求**: 每个客户端需要不同的配置

---

**💡 建议**: 先从集中式配置开始学习和验证算法，然后迁移到分布式配置进行真实环境部署和高级实验。
