# Federation Framework

一个灵活、可扩展的联邦学习框架，构建于 `node_comm` 通信层之上。

## 特性

- **可扩展架构**: 通过注册系统支持自定义 Trainer、Learner、Aggregator、Model
- **多种运行模式**: 
  - `serial`: 单进程模拟，适合调试
  - `parallel`: 多进程并行，适合单机多卡
  - `distributed`: 分布式部署，适合跨机器训练
- **灵活的数据划分**: 支持 IID、Dirichlet、Label-based 等划分策略
- **完善的基础设施**: 
  - Tracker: 支持 MLflow、WandB、TensorBoard
  - Checkpoint: 模型保存和恢复
  - Timer: 性能分析
- **内置组件**: FedAvg、FedProx、异步聚合等

## 安装

```bash
# 依赖
pip install pyyaml numpy

# 可选依赖
pip install torch  # PyTorch 支持
pip install mlflow wandb  # 实验追踪
```

## 快速开始

### 1. 配置文件

```yaml
# config.yaml
role: server
mode: serial

components:
  trainer: federated.trainer.default
  learner: federated.learner.default
  aggregator: federated.aggregator.fedavg
  model: model.torch.mlp

trainer_config:
  max_rounds: 100
  min_clients: 2

data:
  enabled: true
  dataset: mnist
  partitioner: iid
```

### 2. 启动训练

```bash
# 串行模式（调试）
python -m federation.launch --config config.yaml --mode serial --num_clients 5

# 并行模式（单机多进程）
python -m federation.launch --config config.yaml --mode parallel --num_clients 5

# 分布式模式（跨机器）
# 服务端
python -m federation.launch --config server_config.yaml --mode server

# 客户端
python -m federation.launch --config client_config.yaml --mode client
```

### 3. 编程方式使用

```python
import asyncio
from federation import (
    ServerNode, ClientNode,
    load_config_from_dict,
)
from federation.builtin import (
    FedAvgAggregator, DefaultTrainer, DefaultLearner, MLPModel
)

async def main():
    # 创建组件
    model = MLPModel(input_size=784, output_size=10)
    aggregator = FedAvgAggregator()
    
    # 创建服务端
    server = ServerNode({
        "node_id": "server",
        "transport": {"mode": "memory"},
    })
    
    # 创建训练器
    trainer = DefaultTrainer(
        server=server,
        model=model,
        aggregator=aggregator,
        config={"max_rounds": 10, "min_clients": 2},
    )
    
    # 创建客户端
    clients = []
    for i in range(3):
        learner = DefaultLearner(model=MLPModel())
        client = ClientNode({"node_id": f"client_{i}"})
        client.bind(learner)
        clients.append(client)
    
    # 启动
    await server.start()
    for client in clients:
        await client.start()
        await client.connect_and_register("server")
    
    # 训练
    await trainer.run()
    
    # 清理
    for client in clients:
        await client.stop()
    await server.stop()

asyncio.run(main())
```

## 自定义组件

### 自定义 Aggregator

```python
from federation import Aggregator, register, ClientUpdate

@register("custom.aggregator.my_agg")
class MyAggregator(Aggregator):
    def aggregate(self, updates, global_model=None):
        # 自定义聚合逻辑
        total = sum(u.num_samples for u in updates)
        result = None
        for u in updates:
            weight = u.num_samples / total
            if result is None:
                result = [w * weight for w in u.weights]
            else:
                for i, w in enumerate(u.weights):
                    result[i] += w * weight
        return result
```

### 自定义 Trainer

```python
from federation import Trainer, register

@register("custom.trainer.my_trainer")
class MyTrainer(Trainer):
    async def run(self):
        await self.wait_for_clients(2)
        await self.broadcast_weights(self.model.get_weights())
        
        for round_num in range(100):
            clients = self.get_connected_clients()
            results = await self.collect_results(clients, "fit", {"epochs": 5})
            
            updates = [ClientUpdate.from_result(c.client_id, r) 
                       for c, r in zip(clients, results)]
            
            new_weights = self.aggregator.aggregate(updates)
            self.model.set_weights(new_weights)
            await self.broadcast_weights(new_weights)
```

### 自定义 Learner

```python
from federation import Learner, register, TrainResult, EvalResult

@register("custom.learner.my_learner")
class MyLearner(Learner):
    def fit(self, config=None):
        # 自定义训练逻辑
        for epoch in range(config.get("epochs", 5)):
            for batch_x, batch_y in self.data.get_train_loader():
                # 训练一个 batch
                pass
        
        return TrainResult(
            weights=self.get_weights(),
            num_samples=self.data.get_num_samples(),
            metrics={"loss": 0.1},
        )
    
    def evaluate(self, config=None):
        # 自定义评估逻辑
        return EvalResult(
            num_samples=100,
            metrics={"accuracy": 0.95},
        )
```

## 目录结构

```
federation/
├── core/               # 核心抽象
│   ├── types.py       # TrainResult, EvalResult, ClientUpdate
│   ├── model.py       # Model 抽象基类
│   ├── learner.py     # Learner 抽象基类
│   ├── trainer.py     # Trainer 抽象基类
│   └── aggregator.py  # Aggregator 抽象基类
├── node/               # 节点实现
│   ├── server.py      # ServerNode
│   ├── client.py      # ClientNode
│   └── proxy.py       # ClientProxy
├── registry/           # 组件注册系统
│   └── registry.py    # Registry, @register
├── infra/              # 基础设施
│   ├── tracker.py     # MLflow/WandB/TensorBoard
│   ├── checkpoint.py  # 检查点管理
│   ├── logger.py      # 日志
│   └── timer.py       # 计时器
├── data/               # 数据管理
│   ├── provider.py    # DataProvider
│   ├── partitioner.py # IID/Dirichlet/Label 划分
│   └── manager.py     # DataManager
├── builtin/            # 内置组件
│   ├── aggregators/   # FedAvg, FedProx, Median
│   ├── trainers/      # Default, Async
│   ├── learners/      # Default, Sklearn
│   └── models/        # MLP, CNN
├── config.py           # 配置加载
├── launch.py           # 启动器
└── examples/           # 示例
```

## 配置参考

完整配置选项请参考 `examples/config.yaml`。

## 测试

```bash
python -m federation.tests.test_basic
```

## 与 node_comm 的关系

`federation` 框架构建于 `node_comm` 通信层之上：

- `node_comm`: 提供底层节点通信能力（消息传递、连接管理、序列化）
- `federation`: 提供联邦学习抽象（Trainer、Learner、Aggregator）

```
┌─────────────────────────────────────┐
│         Federation Framework        │
│  (Trainer, Learner, Aggregator)     │
├─────────────────────────────────────┤
│      ServerNode / ClientNode        │
│        (Federation Protocol)        │
├─────────────────────────────────────┤
│           node_comm.Node            │
│     (Message Transport Layer)       │
├─────────────────────────────────────┤
│    Memory Transport / gRPC / ...    │
└─────────────────────────────────────┘
```

## License

MIT
