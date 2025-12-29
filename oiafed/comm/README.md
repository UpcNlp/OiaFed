# Node 通信层

轻量级、可扩展的节点通信框架，支持对等通信、多种传输模式无感切换。

## 特性

- **对等通信**：任意节点可以调用任意节点的方法
- **多种传输模式**：Memory（调试）、gRPC（生产）无感切换
- **可扩展性**：通过 Interceptor 机制支持日志、认证、监控等横切关注点
- **类型安全**：统一的消息格式和错误处理
- **事件系统**：支持连接/断开事件处理

## 安装

```bash
pip install grpcio grpcio-tools protobuf PyYAML
```

## 快速开始

### Memory 模式（用于测试和调试）

```python
import asyncio
from node_comm import Node, NodeConfig, TransportConfig

async def main():
    # 创建配置
    config1 = NodeConfig(
        node_id="coordinator",
        transport=TransportConfig(mode="memory"),
    )
    config2 = NodeConfig(
        node_id="participant",
        transport=TransportConfig(mode="memory"),
    )
    
    # 创建节点
    coordinator = Node("coordinator", config1)
    participant = Node("participant", config2)
    
    # 注册方法
    async def train(payload, ctx):
        epochs = payload.get("epochs", 1)
        return {"status": "completed", "loss": 0.01}
    
    participant.register("train", train)
    
    # 启动节点
    await coordinator.start()
    await participant.start()
    
    # 建立连接
    await coordinator.connect("participant")
    
    # 调用远程方法
    result = await coordinator.call("participant", "train", {"epochs": 10})
    print(f"Result: {result}")
    
    # 停止节点
    await coordinator.stop()
    await participant.stop()

asyncio.run(main())
```

### gRPC 模式（用于生产环境）

```python
import asyncio
from node_comm import Node, NodeConfig, TransportConfig, GrpcTransportConfig

async def main():
    # 节点 1 配置
    config1 = NodeConfig(
        node_id="node_1",
        transport=TransportConfig(
            mode="grpc",
            grpc=GrpcTransportConfig(host="0.0.0.0", port=50051),
        ),
    )
    
    # 节点 2 配置
    config2 = NodeConfig(
        node_id="node_2",
        transport=TransportConfig(
            mode="grpc",
            grpc=GrpcTransportConfig(host="0.0.0.0", port=50052),
        ),
    )
    
    node1 = Node("node_1", config1)
    node2 = Node("node_2", config2)
    
    async def echo(payload, ctx):
        return {"echo": payload}
    
    node2.register("echo", echo)
    
    await node1.start()
    await node2.start()
    
    # gRPC 模式需要指定地址
    await node1.connect("node_2", "127.0.0.1:50052")
    
    result = await node1.call("node_2", "echo", {"msg": "hello"})
    print(f"Result: {result}")
    
    await node1.stop()
    await node2.stop()

asyncio.run(main())
```

### 使用上下文管理器

```python
async with Node("my_node", config) as node:
    node.register("method", handler)
    await node.connect("other_node")
    result = await node.call("other_node", "method", payload)
```

### 事件处理

```python
node = Node("my_node", config)

# 注册事件处理器
node.on("connect", lambda nid, addr: print(f"Connected to {nid}"))
node.on("disconnect", lambda nid, reason: print(f"Disconnected from {nid}: {reason}"))

# 链式调用
node.on("connect", handler1).on("disconnect", handler2)

# 移除处理器
node.off("connect", handler1)  # 移除特定处理器
node.off("connect")            # 移除所有 connect 处理器
```

### 广播

```python
# 广播到所有已连接节点
await coordinator.broadcast(
    method="config_update",
    payload={"learning_rate": 0.001},
    exclude=["coordinator"],  # 排除自己
)
```

### 通知（不等待响应）

```python
await node.notify(
    target="coordinator",
    method="_heartbeat",
    payload={"status": "alive"},
)
```

### 使用 Pickle 序列化器

```python
from node_comm import MethodOptions

# 服务端注册时指定序列化器
node.register("process", handler, MethodOptions(serializer="pickle"))

# 客户端调用时指定序列化器
result = await node.call("target", "process", complex_data, serializer="pickle")
```

### 从 YAML 加载配置

```python
from node_comm import load_config, Node

config = load_config("config.yaml")
node = Node(config.node_id, config)
```

## 配置说明

参见 `config.example.yaml` 获取完整配置示例。

### 核心配置项

| 配置 | 说明 | 默认值 |
|------|------|--------|
| `node_id` | 节点唯一标识 | 必填 |
| `debug` | 调试模式（错误响应包含堆栈） | `false` |
| `default_timeout` | 默认调用超时（秒） | `30.0` |
| `transport.mode` | 传输模式：`memory` 或 `grpc` | `memory` |
| `heartbeat.enabled` | 是否启用心跳 | `false` |
| `heartbeat.interval` | 心跳间隔（秒） | `30.0` |
| `heartbeat.timeout` | 心跳超时（秒） | `90.0` |

## 错误处理

```python
from node_comm import (
    NodeNotConnectedError,
    CallTimeoutError,
    RemoteExecutionError,
)

try:
    result = await node.call("target", "method", payload)
except NodeNotConnectedError as e:
    print(f"Node {e.node_id} not connected")
except CallTimeoutError as e:
    print(f"Call to {e.target}.{e.method} timed out")
except RemoteExecutionError as e:
    print(f"Remote error [{e.code}]: {e}")
```

## 目录结构

```
node_comm/
├── __init__.py           # 公共 API
├── node.py               # Node 核心类
├── message.py            # 消息数据结构
├── config.py             # 配置类定义
├── exceptions.py         # 异常定义
├── transport/            # 传输层
│   ├── base.py           # Transport 抽象基类
│   ├── memory.py         # MemoryTransport
│   ├── grpc_transport.py # GrpcTransport
│   └── factory.py        # Transport 工厂
├── serialization/        # 序列化
│   ├── base.py           # Serializer 抽象基类
│   ├── json_serializer.py
│   ├── pickle_serializer.py
│   └── registry.py       # SerializerRegistry
├── interceptor/          # 拦截器
│   ├── base.py           # Interceptor 抽象基类
│   └── logging.py        # LoggingInterceptor
├── proto/                # gRPC Proto
│   ├── node_service.proto
│   ├── node_service_pb2.py
│   └── node_service_pb2_grpc.py
└── utils/                # 工具
    └── logging.py
```

## 扩展

### 自定义序列化器

```python
from node_comm import Serializer

class MySerializer(Serializer):
    @property
    def name(self) -> str:
        return "my_serializer"
    
    def serialize(self, obj) -> bytes:
        # 实现序列化逻辑
        pass
    
    def deserialize(self, data: bytes):
        # 实现反序列化逻辑
        pass

# 注册
node._serializer_registry.register(MySerializer())
```

### 自定义拦截器

```python
from node_comm import Interceptor

class MyInterceptor(Interceptor):
    async def intercept_inbound(self, message, context, next_handler):
        # 入站处理
        return await next_handler(message)
    
    async def intercept_outbound(self, message, context, next_handler):
        # 出站处理
        return await next_handler(message)

# 添加到拦截器链
node._interceptor_chain.add(MyInterceptor())
```

## 运行测试

```bash
# 运行所有测试
python -m pytest node_comm/tests/ -v

# 运行 Memory 模式测试
python node_comm/tests/test_node.py

# 运行 gRPC 模式测试
python node_comm/tests/test_grpc.py
```

## 许可证

MIT
