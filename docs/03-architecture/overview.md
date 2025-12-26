# 架构总览

OiaFed 的整体系统架构设计。

---

## 设计原则

1. **模块化** - 组件解耦，可独立替换
2. **配置驱动** - YAML 配置，无需改代码
3. **通信透明** - Memory/gRPC 无缝切换
4. **可扩展** - Registry 注册机制

---

## 分层架构

```
┌─────────────────────────────────────────────────────────────┐
│                   Application Layer                          │
│            FederationRunner · CLI · Config                   │
├─────────────────────────────────────────────────────────────┤
│                Federation Framework Layer                     │
│          Trainer · Learner · Aggregator · Callback           │
├─────────────────────────────────────────────────────────────┤
│                  Node Abstraction Layer                       │
│              Node · Proxy · ProxyCollection                  │
├─────────────────────────────────────────────────────────────┤
│                  Communication Layer                          │
│           Transport · Serializer · Interceptor               │
├─────────────────────────────────────────────────────────────┤
│                Transport Backend Layer                        │
│             MemoryTransport · gRPCTransport                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 核心组件

### Trainer

服务端训练编排。

```
┌─────────────────────────────────────────┐
│               Trainer                    │
├─────────────────────────────────────────┤
│  - select_clients()                     │
│  - broadcast_weights()                  │
│  - collect_updates()                    │
│  - aggregator.aggregate()               │
│  - evaluate()                           │
└─────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────┐
│            Aggregator                    │
│  FedAvg · FedProx · SCAFFOLD · ...      │
└─────────────────────────────────────────┘
```

### Learner

客户端本地训练。

```
┌─────────────────────────────────────────┐
│               Learner                    │
├─────────────────────────────────────────┤
│  - fit(weights, config)                 │
│  - train_epoch()                        │
│  - train_step()                         │
│  - evaluate()                           │
│  - get_weights() / set_weights()        │
└─────────────────────────────────────────┘
```

### Node

通信端点。

```
┌─────────────────────────────────────────┐
│                 Node                     │
├─────────────────────────────────────────┤
│  - register_handler(method, handler)    │
│  - get_proxy(target_id) → Proxy         │
│  - start() / stop()                     │
└─────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────┐
│               Transport                  │
│       Memory │ gRPC │ Custom            │
└─────────────────────────────────────────┘
```

---

## 数据流

### 训练流程

```
Round N:

1. Trainer 选择客户端
   select_clients() → [client_0, client_1, ...]

2. 广播全局模型
   broadcast_weights(global_weights)
      │
      ├──→ Learner_0.set_weights()
      ├──→ Learner_1.set_weights()
      └──→ ...

3. 本地训练
   Learner.fit(config)
      │
      ├── train_epoch() × local_epochs
      │      └── train_step() × batches
      └── return TrainResult

4. 收集更新
   collect_updates() → [update_0, update_1, ...]

5. 聚合
   aggregator.aggregate(updates) → new_weights

6. 评估
   evaluate() → metrics

7. 下一轮
   Round N+1 ...
```

### 消息流

```
Trainer                          Learner
   │                                │
   │──── set_weights ──────────────►│
   │                                │
   │──── fit(config) ──────────────►│
   │                                │
   │◄─── TrainResult ───────────────│
   │                                │
   │──── evaluate ─────────────────►│
   │                                │
   │◄─── EvalResult ────────────────│
```

---

## 运行模式

### Serial

```
┌─────────────────────────────────────────┐
│               单进程                     │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐ │
│  │ Trainer │  │Learner 0│  │Learner 1│ │
│  └────┬────┘  └────┬────┘  └────┬────┘ │
│       └───────┬────┴───────────┬┘      │
│               │                        │
│         MemoryTransport                │
└─────────────────────────────────────────┘
```

### Parallel

```
┌───────────┐  ┌───────────┐  ┌───────────┐
│  进程 1   │  │  进程 2   │  │  进程 3   │
│  Trainer  │  │ Learner 0 │  │ Learner 1 │
└─────┬─────┘  └─────┬─────┘  └─────┬─────┘
      └──────────────┼──────────────┘
              gRPC / IPC
```

### Distributed

```
┌───────────┐      ┌───────────┐      ┌───────────┐
│  机器 A   │ gRPC │  机器 B   │ gRPC │  机器 C   │
│  Trainer  │◄────►│ Learner 0 │◄────►│ Learner 1 │
└───────────┘      └───────────┘      └───────────┘
```

---

## 组件交互

### FederatedSystem

```python
class FederatedSystem:
    """系统容器"""
    
    def __init__(self, config):
        self.node = Node(...)
        self.trainer = Trainer(...)      # 或 Learner
        self.aggregator = Aggregator(...)
        self.tracker = Tracker(...)
        self.callbacks = [...]
    
    async def run(self):
        await self.node.start()
        result = await self.trainer.run()
        await self.node.stop()
        return result
```

### 组件创建流程

```
Config
  │
  ▼
Registry.get("aggregator.fedavg")
  │
  ▼
FedAvgAggregator(config.args)
  │
  ▼
Trainer(aggregator=aggregator, ...)
  │
  ▼
FederatedSystem(trainer=trainer, ...)
```

---

## 目录结构

```
src/
├── core/           # 核心抽象
│   ├── trainer.py
│   ├── learner.py
│   ├── aggregator.py
│   └── types.py
├── comm/           # 通信层
│   ├── node.py
│   ├── transport/
│   └── serialization/
├── methods/        # 算法实现
│   ├── aggregators/
│   ├── learners/
│   └── models/
├── infra/          # 基础设施
│   ├── tracker.py
│   └── checkpoint.py
├── callback/       # 回调系统
├── registry/       # 注册系统
└── config/         # 配置管理
```

---

## 下一步

- [通信层设计](communication.md)
- [Callback 机制](callback-system.md)
- [注册系统](registry-system.md)
