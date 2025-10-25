# MOE-FedCL 联邦通信系统

一个现代化的联邦学习通信框架，支持 Memory/Process/Network 三种通信模式。

## 🚀 核心特性

### 1. 统一入口 - 最简单的启动方式

**一行代码启动完整的联邦学习系统**：

```python
from fedcl import run_federated_learning, BaseTrainer, BaseLearner

# 一行代码启动！
result = await run_federated_learning(
    trainer_class=MyTrainer,
    learner_class=MyLearner,
    global_model={"weights": [0.1, 0.2, 0.3]},
    server_config_path="configs/server_demo.yaml",
    client_config_path="configs/client_demo_1.yaml",
    num_clients=5,
    max_rounds=10
)
```

### 2. 三种通信模式

- **Memory 模式**：进程内通信，适合开发和调试
- **Process 模式**：多进程 + HTTP 通信，适合本地测试
- **Network 模式**：分布式 + HTTP 通信，适合生产环境

### 3. 五层架构设计

```
Layer 0: FederationCoordinator       # 联邦学习协调器
Layer 1: BaseTrainer / Server        # 训练器和服务端
Layer 2: LearnerProxy / Stub         # 客户端代理和存根
Layer 3: ConnectionManager           # 连接管理
Layer 4: CommunicationManager        # 通信管理
Layer 5: TransportBase               # 传输层
```

### 4. 配置驱动

- **YAML 配置文件**：集中管理所有参数
- **类型安全**：完整的类型提示和验证
- **灵活配置**：支持文件、对象和默认配置

### 5. 多层次 API

```
高层: FederatedLearning          # 统一入口（推荐）
中层: ServerAPI, ClientAPI        # 组件 API
底层: FederationServer, Client    # 底层组件
```

## 📦 安装

```bash
# 克隆项目
git clone <repository-url>
cd MOE-FedCL

# 安装依赖
pip install -r requirements.txt

# 设置环境变量
export PYTHONPATH=/path/to/MOE-FedCL:$PYTHONPATH
```

## 🎯 快速开始

### 方式 1: 统一入口（最推荐）

```python
import asyncio
from fedcl import FederatedLearning, BaseTrainer, BaseLearner

class MyTrainer(BaseTrainer):
    async def train_round(self, round_num, client_ids):
        # 实现训练逻辑
        pass

    async def aggregate_models(self, client_results):
        # 实现聚合逻辑
        pass

class MyLearner(BaseLearner):
    async def train(self, training_params):
        # 实现本地训练
        pass

    async def evaluate(self, evaluation_params):
        # 实现本地评估
        pass

async def main():
    # 使用上下文管理器自动管理资源
    async with FederatedLearning(
        trainer_class=MyTrainer,
        learner_class=MyLearner,
        global_model={"weights": [0.1, 0.2, 0.3]},
        server_config_path="configs/server_demo.yaml",
        client_config_path="configs/client_demo_1.yaml",
        num_clients=5
    ) as fl:
        result = await fl.run(max_rounds=10)
        print(f"训练完成！准确率: {result.final_accuracy:.4f}")

if __name__ == "__main__":
    asyncio.run(main())
```

### 方式 2: 高层 API

```python
from fedcl import ServerAPI, MultiClientAPI

async def main():
    # 启动服务端
    async with ServerAPI(
        trainer_class=MyTrainer,
        global_model={"weights": [0.1, 0.2, 0.3]},
        config_path="configs/server_demo.yaml"
    ) as server:

        # 启动多个客户端
        async with MultiClientAPI(
            learner_class=MyLearner,
            num_clients=5,
            config_path="configs/client_demo_1.yaml"
        ) as clients:

            # 运行训练
            await server.run_training(num_rounds=10)

if __name__ == "__main__":
    asyncio.run(main())
```

### 方式 3: 底层组件（完全控制）

```python
from fedcl.federation import FederationServer, FederationClient
from fedcl.federation.coordinator import FederationCoordinator

async def main():
    # 手动创建和管理所有组件
    server = FederationServer(config)
    await server.initialize_with_trainer(MyTrainer, global_model)
    await server.start_server()

    clients = []
    for i in range(5):
        client = FederationClient(config, f"client_{i}")
        await client.initialize_with_learner(MyLearner)
        await client.start_client()
        clients.append(client)

    coordinator = FederationCoordinator(server, federation_config)
    result = await coordinator.start_federation()

    # 清理
    for client in clients:
        await client.stop_client()
    await server.stop_server()

if __name__ == "__main__":
    asyncio.run(main())
```

## 📖 配置示例

### 服务端配置 (`configs/server_demo.yaml`)

```yaml
mode: process                    # 通信模式: memory, process, network
server_host: "127.0.0.1"
server_port: 8000

transport:
  timeout: 30.0
  retry_attempts: 3

communication:
  heartbeat_interval: 30.0
  heartbeat_timeout: 90.0
  max_clients: 100

federation:
  max_rounds: 100
  min_clients: 2
  client_selection: "all"
```

### 客户端配置 (`configs/client_demo.yaml`)

```yaml
mode: process
server_host: "127.0.0.1"
server_port: 8000
client_host: "127.0.0.1"
client_port: 0                   # 0 表示自动分配端口

stub:
  auto_register: true
  registration_retry_attempts: 3
  request_timeout: 120.0
```

## 📚 文档

详细文档请查看：

- **[统一入口使用指南](docs/统一入口使用指南.md)** - FederatedLearning 类完整指南
- **[API 使用指南](docs/API使用指南.md)** - 高层 API 使用说明
- **[配置系统指南](docs/配置系统使用指南.md)** - 配置文件详解
- **[新架构使用指南](docs/新架构使用指南.md)** - 底层架构说明
- **[架构设计文档](docs/MOE-FedCL联邦通信系统架构设计.md)** - 完整架构设计

## 🔧 核心组件

### 1. FederatedLearning（统一入口）

整合服务端、客户端和协调器的一站式解决方案：

```python
from fedcl import FederatedLearning

fl = FederatedLearning(
    trainer_class=MyTrainer,
    learner_class=MyLearner,
    global_model=initial_model,
    server_config_path="configs/server.yaml",
    client_config_path="configs/client.yaml",
    num_clients=5
)

# 初始化所有组件
await fl.initialize()

# 运行训练
result = await fl.run(max_rounds=10)

# 清理资源
await fl.cleanup()
```

### 2. FederationCoordinator（协调器）

协调整个联邦学习训练流程：

```python
from fedcl.federation.coordinator import FederationCoordinator, FederationConfig

coordinator = FederationCoordinator(
    federation_server=server,
    federation_config=FederationConfig(
        max_rounds=10,
        min_clients=2,
        client_selection="all"
    )
)

result = await coordinator.start_federation()
```

### 3. FederationServer（服务端）

管理全局模型和客户端：

```python
from fedcl.federation.server import FederationServer

server = FederationServer(config)
await server.initialize_with_trainer(
    trainer_class=MyTrainer,
    global_model=initial_model
)
await server.start_server()
```

### 4. FederationClient（客户端）

执行本地训练和评估：

```python
from fedcl.federation.client import FederationClient

client = FederationClient(config, client_id="client_1")
await client.initialize_with_learner(MyLearner)
await client.start_client()
```

### 5. BaseTrainer（训练器基类）

用户需要继承实现的服务端训练器：

```python
from fedcl import BaseTrainer

class MyTrainer(BaseTrainer):
    async def train_round(self, round_num: int, client_ids: List[str]) -> RoundResult:
        """实现单轮训练逻辑"""
        # 1. 向客户端分发任务
        # 2. 收集训练结果
        # 3. 聚合模型
        pass

    async def aggregate_models(self, client_results: Dict) -> ModelData:
        """实现模型聚合逻辑"""
        pass
```

### 6. BaseLearner（学习器基类）

用户需要继承实现的客户端学习器：

```python
from fedcl import BaseLearner

class MyLearner(BaseLearner):
    async def train(self, training_params: Dict) -> TrainingResult:
        """实现本地训练逻辑"""
        pass

    async def evaluate(self, evaluation_params: Dict) -> EvaluationResult:
        """实现本地评估逻辑"""
        pass
```

## 🎨 示例代码

### 完整示例

查看 `examples/` 目录下的示例：

- **[unified_entry_demo.py](examples/unified_entry_demo.py)** - 统一入口示例
- **[api_usage_demo.py](examples/api_usage_demo.py)** - 高层 API 示例
- **[config_usage_demo.py](examples/config_usage_demo.py)** - 配置系统示例
- **[minimal_memory_demo.py](examples/minimal_memory_demo.py)** - 内存模式示例
- **[mnist_process_demo.py](examples/mnist_process_demo.py)** - 进程模式示例

### Memory 模式示例

```python
# 单进程内模拟联邦学习
config = {"mode": "memory"}

async with FederatedLearning(
    trainer_class=MyTrainer,
    learner_class=MyLearner,
    global_model=model,
    num_clients=3
) as fl:
    result = await fl.run(max_rounds=5)
```

### Process 模式示例

```python
# 多进程 + HTTP 通信
server_config = "configs/server_demo.yaml"  # mode: process
client_config = "configs/client_demo_1.yaml"

async with FederatedLearning(
    trainer_class=MyTrainer,
    learner_class=MyLearner,
    global_model=model,
    server_config_path=server_config,
    client_config_path=client_config,
    num_clients=5
) as fl:
    result = await fl.run(max_rounds=10)
```

### Network 模式示例

```python
# 分布式部署
# 服务端脚本
async with ServerAPI(
    trainer_class=MyTrainer,
    global_model=model,
    config_path="configs/server_network.yaml"  # mode: network
) as server:
    await server.run_training(num_rounds=10)

# 客户端脚本（运行在不同机器）
async with ClientAPI(
    learner_class=MyLearner,
    config_path="configs/client_network.yaml"
) as client:
    await client.wait_for_tasks()
```

## 🏗️ 架构特点

### 1. 客户端地址注册

客户端在注册时会告知服务器自己的 IP 地址和端口，服务器可以主动向客户端发送请求：

```python
# 客户端注册时包含地址信息
registration_request = RegistrationRequest(
    client_id="client_1",
    metadata={
        "client_address": {
            "host": "192.168.1.100",
            "port": 8001,
            "url": "http://192.168.1.100:8001"
        }
    }
)

# 服务器缓存客户端地址
transport.register_client_address("client_1", address_info)

# 服务器向客户端发送请求
response = await transport.send_request("client_1", request_data)
```

### 2. 异步通信

所有通信操作都是异步的，提高系统性能：

```python
# 并发训练多个客户端
tasks = []
for client_id in selected_clients:
    task = learner_proxy.train(training_params)
    tasks.append(task)

results = await asyncio.gather(*tasks)
```

### 3. 自动重试和超时

内置重试机制和超时控制：

```yaml
transport:
  timeout: 30.0           # 请求超时时间
  retry_attempts: 3       # 重试次数
  retry_delay: 1.0        # 重试延迟
```

### 4. 心跳机制

自动检测客户端健康状态：

```yaml
communication:
  heartbeat_interval: 30.0    # 心跳间隔
  heartbeat_timeout: 90.0     # 心跳超时
```

## 🔍 调试和监控

### 日志系统

自动设置结构化日志：

```python
from fedcl.utils.auto_logger import setup_auto_logging, get_sys_logger

# 设置日志
setup_auto_logging(level="DEBUG")

# 获取日志器
logger = get_sys_logger()
logger.info("系统启动")
```

### 调试模式

```python
# 启用详细日志
fl = FederatedLearning(
    ...,
    auto_setup_logging=True
)

# 查看通信细节
# 日志会显示每次请求和响应
```

## 🚀 部署指南

### 本地开发

```bash
# 运行示例
python examples/unified_entry_demo.py
```

### 生产环境

```bash
# 服务端（单独运行）
python scripts/run_server.py --config configs/server_production.yaml

# 客户端（多台机器）
python scripts/run_client.py --config configs/client_production.yaml
```

### Docker 部署

```dockerfile
FROM python:3.8-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# 服务端
CMD ["python", "scripts/run_server.py"]

# 或客户端
# CMD ["python", "scripts/run_client.py"]
```

## 🤝 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 代码规范

- 使用 Python 类型注解
- 遵循 PEP 8 代码风格
- 添加详细的文档字符串
- 编写单元测试

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 📞 联系方式

如有问题或建议，欢迎：

- 提交 [Issue](https://github.com/your-repo/issues)
- 发起 [Discussion](https://github.com/your-repo/discussions)
- 发送邮件至 your-email@example.com

## 🌟 致谢

感谢所有贡献者对本项目的支持！

---

**MOE-FedCL - 让联邦学习更简单！** 🚀
