# 通信层设计

OiaFed 通信层的架构和实现。

---

## 设计目标

1. **透明性** - 业务代码无需关心传输细节
2. **可切换** - Memory/gRPC 一键切换
3. **可扩展** - 支持自定义 Transport
4. **可靠性** - 重试、超时、心跳

---

## 架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Node Abstraction                          │
│                  Node · Proxy · Collection                   │
├─────────────────────────────────────────────────────────────┤
│                    Message Layer                             │
│                 Message · Serializer                         │
├─────────────────────────────────────────────────────────────┤
│                   Interceptor Chain                          │
│              Logging · Auth · Compression                    │
├─────────────────────────────────────────────────────────────┤
│                   Transport Layer                            │
│              MemoryTransport · gRPCTransport                │
└─────────────────────────────────────────────────────────────┘
```

---

## Node

通信端点。

```python
class Node:
    """
    每个参与方的通信端点
    
    职责:
    - 管理本地处理器
    - 创建远程代理
    - 管理 Transport 生命周期
    """
    
    def __init__(self, node_id, role, transport=None):
        self.node_id = node_id
        self.role = role
        self.transport = transport or create_transport()
        self.handlers = {}
        self.proxies = {}
```

### 处理器注册

```python
node = Node("trainer", "trainer")

@node.register_handler("ping")
async def handle_ping(data):
    return {"pong": True}

# 或
node.register_handler("fit", learner.fit)
```

### 远程调用

```python
# 获取代理
proxy = node.get_proxy("learner_0")

# 调用
result = await proxy.call("fit", weights=weights, config=config)
```

---

## Proxy

远程调用代理。

```python
class Proxy:
    """
    封装远程节点调用
    
    屏蔽传输细节，提供类似本地调用的接口
    """
    
    async def call(self, method, **kwargs):
        message = Message(
            method=method,
            source=self.local_id,
            target=self.target_id,
            payload=kwargs,
        )
        response = await self.transport.send(self.target_id, message)
        return response.payload
```

### ProxyCollection

批量操作。

```python
class ProxyCollection:
    """
    管理多个 Proxy
    
    提供广播、收集等批量操作
    """
    
    async def broadcast(self, method, **kwargs):
        """向所有节点广播"""
        tasks = [
            proxy.call(method, **kwargs)
            for proxy in self.proxies.values()
        ]
        results = await asyncio.gather(*tasks)
        return dict(zip(self.proxies.keys(), results))
    
    async def collect(self, method, args_map):
        """向不同节点发送不同参数"""
        tasks = [
            self.proxies[nid].call(method, **args)
            for nid, args in args_map.items()
        ]
        return await asyncio.gather(*tasks)
```

---

## Transport

传输层抽象。

### 接口

```python
class Transport(ABC):
    @abstractmethod
    async def send(self, target, message) -> Message:
        """发送消息并等待响应"""
    
    @abstractmethod
    async def start(self) -> None:
        """启动传输层"""
    
    @abstractmethod
    async def stop(self) -> None:
        """停止传输层"""
```

### MemoryTransport

进程内通信。

```python
class MemoryTransport(Transport):
    """
    基于内存队列的传输
    
    特点:
    - 零拷贝（共享内存）
    - 无序列化开销
    - 仅限单进程
    """
    
    # 全局节点注册表
    _nodes: Dict[str, "MemoryTransport"] = {}
    
    async def send(self, target, message):
        target_transport = self._nodes[target]
        handler = target_transport.get_handler(message.method)
        result = await handler(message.payload)
        return Message(payload=result)
```

### gRPCTransport

跨进程/跨机器通信。

```python
class gRPCTransport(Transport):
    """
    基于 gRPC 的传输
    
    特点:
    - 支持跨进程/跨机器
    - 自动重连
    - 流式传输
    """
    
    def __init__(self, host, port, **options):
        self.server = grpc.aio.server()
        self.channels = {}  # 到其他节点的连接
    
    async def start(self):
        # 启动 gRPC 服务器
        self.server.add_insecure_port(f"{self.host}:{self.port}")
        await self.server.start()
    
    async def send(self, target, message):
        channel = self._get_channel(target)
        stub = NodeServiceStub(channel)
        response = await stub.Call(message.to_proto())
        return Message.from_proto(response)
```

---

## Serializer

序列化器。

### Pickle（默认）

```python
class PickleSerializer(Serializer):
    """
    Pickle 序列化
    
    优点: 支持任意 Python 对象
    缺点: 不跨语言，安全性
    """
    
    def serialize(self, obj):
        return pickle.dumps(obj)
    
    def deserialize(self, data):
        return pickle.loads(data)
```

### JSON

```python
class JsonSerializer(Serializer):
    """
    JSON 序列化
    
    优点: 跨语言，可读
    缺点: 不支持复杂对象
    """
```

---

## Interceptor

拦截器链。

```
Message → [Logging] → [Auth] → [Compress] → Transport
                                               │
Message ← [Logging] ← [Auth] ← [Decompress] ←─┘
```

### LoggingInterceptor

```python
class LoggingInterceptor(Interceptor):
    async def intercept_send(self, message, context):
        logger.debug(f"Sending: {message.method} → {message.target}")
        return message
    
    async def intercept_receive(self, message, context):
        logger.debug(f"Received: {message.method} from {message.source}")
        return message
```

### AuthInterceptor

```python
class AuthInterceptor(Interceptor):
    def __init__(self, token):
        self.token = token
    
    async def intercept_send(self, message, context):
        message.metadata["auth_token"] = self.token
        return message
    
    async def intercept_receive(self, message, context):
        if message.metadata.get("auth_token") != self.token:
            raise AuthError("Invalid token")
        return message
```

---

## 重试机制

```python
class RetryPolicy:
    def __init__(
        self,
        max_attempts: int = 3,
        backoff_strategy: str = "exponential",
        initial_delay: float = 1.0,
        max_delay: float = 30.0,
    ):
        ...
    
    async def execute(self, func, *args, **kwargs):
        for attempt in range(self.max_attempts):
            try:
                return await func(*args, **kwargs)
            except RetriableError:
                delay = self._compute_delay(attempt)
                await asyncio.sleep(delay)
        raise MaxRetriesExceeded()
```

### 配置

```yaml
retry:
  max_attempts: 3
  backoff_strategy: exponential  # constant | linear | exponential
  initial_delay: 1.0
  max_delay: 30.0
```

---

## 心跳检测

```python
class HeartbeatManager:
    def __init__(self, interval=30, timeout=90):
        self.interval = interval
        self.timeout = timeout
    
    async def start(self):
        while True:
            for node_id, proxy in self.proxies.items():
                try:
                    await asyncio.wait_for(
                        proxy.call("ping"),
                        timeout=self.timeout
                    )
                except asyncio.TimeoutError:
                    self._handle_node_failure(node_id)
            await asyncio.sleep(self.interval)
```

---

## 配置参考

```yaml
# 节点配置
node:
  id: trainer
  role: trainer

# 监听
listen:
  host: 0.0.0.0
  port: 50051

# 连接
connect_to:
  - learner_0@192.168.1.2:50052

# 传输
transport:
  mode: auto           # auto | memory | grpc
  serializer: pickle
  interceptors:
    - type: logging
    - type: auth
      token: ${AUTH_TOKEN}

# 超时
timeout:
  connect: 10
  request: 60

# 重试
retry:
  max_attempts: 3
  backoff_strategy: exponential

# 心跳
heartbeat:
  enabled: true
  interval: 30
  timeout: 90
```

---

## 下一步

- [架构总览](overview.md)
- [通信 API](../02-api-reference/comm-api.md)
