# 通信 API

Node、Proxy、Transport 等通信组件的 API 参考。

---

## Node

节点，每个参与方的通信端点。

### 类定义

```python
class Node:
    def __init__(
        self,
        node_id: str,
        role: str,                    # "trainer" | "learner"
        transport: Optional[Transport] = None,
        config: Optional[Dict] = None,
    )
```

### 核心方法

#### `start()` / `stop()`

```python
async def start(self) -> None:
    """启动节点，开始监听"""

async def stop(self) -> None:
    """停止节点"""
```

#### `register_handler()`

```python
def register_handler(
    self,
    method: str,
    handler: Callable,
) -> None:
    """
    注册方法处理器
    
    Args:
        method: 方法名
        handler: 异步处理函数
    """
```

#### `get_proxy()`

```python
def get_proxy(self, target_id: str) -> Proxy:
    """
    获取目标节点的代理
    
    Args:
        target_id: 目标节点 ID
    
    Returns:
        Proxy: 远程调用代理
    """
```

### 使用示例

```python
# 创建节点
node = Node(node_id="trainer", role="trainer")

# 注册处理器
@node.register_handler("ping")
async def handle_ping(data):
    return {"pong": True}

# 启动
await node.start()

# 远程调用
proxy = node.get_proxy("learner_0")
result = await proxy.call("fit", weights=weights)

# 停止
await node.stop()
```

---

## Proxy

代理，用于远程调用。

### 类定义

```python
class Proxy:
    def __init__(
        self,
        target_id: str,
        transport: Transport,
        timeout: float = 30.0,
    )
```

### 核心方法

#### `call()`

```python
async def call(
    self,
    method: str,
    **kwargs,
) -> Any:
    """
    远程方法调用
    
    Args:
        method: 方法名
        **kwargs: 方法参数
    
    Returns:
        方法返回值
    """
```

#### `send()`

```python
async def send(
    self,
    method: str,
    **kwargs,
) -> None:
    """
    单向发送（不等待响应）
    """
```

### 使用示例

```python
proxy = node.get_proxy("learner_0")

# 同步调用
result = await proxy.call("fit", weights=weights, config=config)

# 单向发送
await proxy.send("update_config", config=new_config)
```

---

## ProxyCollection

代理集合，批量操作多个节点。

### 类定义

```python
class ProxyCollection:
    def __init__(self, proxies: Dict[str, Proxy])
```

### 核心方法

#### `broadcast()`

```python
async def broadcast(
    self,
    method: str,
    **kwargs,
) -> Dict[str, Any]:
    """
    广播到所有节点
    
    Returns:
        {node_id: result} 字典
    """
```

#### `collect()`

```python
async def collect(
    self,
    method: str,
    args_map: Dict[str, Dict],
) -> Dict[str, Any]:
    """
    向不同节点发送不同参数
    
    Args:
        args_map: {node_id: kwargs}
    
    Returns:
        {node_id: result}
    """
```

### 使用示例

```python
learners = ProxyCollection({
    "learner_0": proxy0,
    "learner_1": proxy1,
})

# 广播
results = await learners.broadcast("set_weights", weights=weights)

# 收集
updates = await learners.collect("fit", {
    "learner_0": {"config": config0},
    "learner_1": {"config": config1},
})
```

---

## Transport

传输层抽象基类。

### 接口定义

```python
class Transport(ABC):
    @abstractmethod
    async def send(
        self,
        target: str,
        message: Message,
    ) -> Message:
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
    内存传输，用于 Serial 模式
    
    所有节点共享同一个消息队列
    """
```

### gRPCTransport

跨进程/跨机器通信。

```python
class gRPCTransport(Transport):
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 50051,
        **options,
    )
```

---

## Serializer

序列化器。

### 接口定义

```python
class Serializer(ABC):
    @abstractmethod
    def serialize(self, obj: Any) -> bytes:
        """序列化对象"""
    
    @abstractmethod
    def deserialize(self, data: bytes) -> Any:
        """反序列化"""
```

### 内置序列化器

```python
# Pickle（默认）
from oiafed.cnm import PickleSerializer
serializer = PickleSerializer()

# JSON
from oiafed.cnm import JsonSerializer
serializer = JsonSerializer()
```

---

## Interceptor

拦截器，用于消息处理管道。

### 接口定义

```python
class Interceptor(ABC):
    @abstractmethod
    async def intercept_send(
        self,
        message: Message,
        context: Dict,
    ) -> Message:
        """发送前拦截"""
    
    @abstractmethod
    async def intercept_receive(
        self,
        message: Message,
        context: Dict,
    ) -> Message:
        """接收后拦截"""
```

### 内置拦截器

```python
# 日志拦截器
from oiafed.cnm import LoggingInterceptor
interceptor = LoggingInterceptor(level="DEBUG")

# 认证拦截器
from oiafed.cnm import AuthInterceptor
interceptor = AuthInterceptor(token="secret")
```

---

## Message

消息类型。

```python
@dataclass
class Message:
    msg_id: str
    msg_type: str           # "request" | "response"
    source: str             # 源节点 ID
    target: str             # 目标节点 ID
    method: str             # 方法名
    payload: Dict[str, Any] # 数据负载
    timestamp: float
    metadata: Dict[str, Any]
```

---

## 配置参考

```yaml
# 节点配置
node:
  id: trainer
  role: trainer

# 监听配置
listen:
  host: 0.0.0.0
  port: 50051

# 连接配置
connect_to:
  - learner_0@192.168.1.2:50052
  - learner_1@192.168.1.3:50053

# 传输配置
transport:
  mode: grpc              # memory | grpc
  serializer: pickle      # pickle | json
  interceptors:
    - type: logging
    - type: auth
      token: ${AUTH_TOKEN}

# 重试配置
retry:
  max_attempts: 3
  backoff_strategy: exponential
  initial_delay: 1.0
  max_delay: 30.0

# 超时配置
timeout:
  default: 30
  connect: 10
  request: 60
```

---

## 下一步

- [核心 API](core-api.md)
- [通信层设计](../03-architecture/communication.md)
