# 核心概念

理解 OiaFed 的核心抽象和设计理念。

---

## 框架定位

OiaFed 是一个**统一的联邦学习框架**，目标是：

- **一个框架**支持所有联邦场景（HFL/VFL/FCL/PFL/FU）
- **模块化**设计，组件可插拔
- **配置驱动**，无需修改代码即可切换算法
- **通信透明**，Memory 和 gRPC 无缝切换

---

## 核心组件

### 三层架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Federation Framework                      │
│              Trainer · Learner · Aggregator                  │
├─────────────────────────────────────────────────────────────┤
│                    Node Abstraction                          │
│                Node · Proxy · ProxyCollection                │
├─────────────────────────────────────────────────────────────┤
│                    Communication Layer                       │
│              Transport · Serializer · Interceptor            │
└─────────────────────────────────────────────────────────────┘
```

### Trainer（训练器）

服务端角色，负责：
- 编排训练流程
- 选择参与客户端
- 调用聚合器
- 广播全局模型

```python
class Trainer:
    async def run(self):
        for round in range(max_rounds):
            clients = self.select_clients()
            updates = await self.collect_updates(clients)
            new_weights = self.aggregator.aggregate(updates)
            await self.broadcast_weights(new_weights)
```

### Learner（学习器）

客户端角色，负责：
- 本地数据训练
- 模型评估
- 上传更新

```python
class Learner:
    async def fit(self, weights, config):
        self.load_weights(weights)
        for epoch in range(local_epochs):
            await self.train_epoch()
        return self.get_update()
```

### Aggregator（聚合器）

服务端组件，负责：
- 聚合客户端更新
- 实现各种聚合算法

```python
class Aggregator:
    def aggregate(self, updates, global_model):
        # FedAvg: 加权平均
        # FedProx: 带近端项
        # SCAFFOLD: 控制变量
        return aggregated_weights
```

---

## 通信模型

### Node（节点）

每个参与方是一个 Node：

```python
node = Node(node_id="trainer", role="trainer")
await node.start()
```

### Proxy（代理）

节点间通过 Proxy 通信：

```python
# Trainer 获取 Learner 的代理
learner_proxy = node.get_proxy("learner_0")

# 远程调用
result = await learner_proxy.call("fit", weights=weights)
```

### Transport（传输层）

- **MemoryTransport**: 进程内通信，用于 Serial 模式
- **gRPCTransport**: 跨进程/跨机器通信

```yaml
# 自动选择
transport:
  mode: auto  # serial→memory, parallel/distributed→grpc
```

---

## 注册系统

所有组件通过 Registry 注册和获取：

```python
from oiafed import register

@register("aggregator.my_algo")
class MyAggregator(Aggregator):
    pass
```

```yaml
# 配置中引用
aggregator:
  type: my_algo
```

---

## 配置驱动

一切通过 YAML 配置：

```yaml
# 切换算法只需改配置
aggregator:
  type: fedavg  # → fedprox → scaffold

# 切换模型
model:
  type: resnet18  # → simple_cnn → mlp

# 切换数据划分
partition:
  strategy: iid  # → dirichlet → label_skew
```

### 配置继承

```yaml
# experiment.yaml
extend: base.yaml  # 继承基础配置

trainer:
  args:
    max_rounds: 200  # 只覆盖需要改的
```

---

## 运行模式

| 模式 | 进程 | 通信 | 用途 |
|------|------|------|------|
| Serial | 单进程 | Memory | 调试 |
| Parallel | 多进程 | Memory/gRPC | 单机训练 |
| Distributed | 多机器 | gRPC | 生产部署 |

---

## 生命周期

```
initialize() → run() → finalize()
     ↓          ↓
  setup()   train_round() × N
               ↓
         select_clients()
               ↓
         collect_updates()
               ↓
           aggregate()
               ↓
         broadcast_weights()
               ↓
           evaluate()
```

### Callback 钩子

```python
class MyCallback(Callback):
    async def on_round_start(self, round_num): ...
    async def on_round_end(self, round_num, result): ...
    async def on_train_end(self, final_result): ...
```

---

## 术语表

| 术语 | 含义 |
|------|------|
| Round | 一轮联邦训练（服务端视角） |
| Epoch | 本地数据遍历一次（客户端视角） |
| Update | 客户端训练后的模型更新 |
| Aggregation | 服务端聚合多个更新 |
| Partition | 数据划分到各客户端 |
| Non-IID | 非独立同分布（数据异构） |

---

## 下一步

- [配置系统](../01-guides/configuration.md) - 深入配置
- [架构总览](../03-architecture/overview.md) - 系统设计
- [快速入门](quickstart.md) - 动手实践
