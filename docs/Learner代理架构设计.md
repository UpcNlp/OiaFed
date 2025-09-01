# Learner代理架构设计 - 远程函数执行器

## 核心概念重新定义

### Learner代理 = 远程函数执行器

Learner代理本质上是一个**远程函数执行器**，它的核心作用是：

1. **透明地执行远程learner方法**
2. **自动处理参数传递和结果返回**  
3. **支持本地/多进程/多机三种执行模式**
4. **对上层trainer完全透明**

```python
# 用户视角：调用看起来像本地函数
result = await learner_proxy.train_epoch(client_id="client_1", epochs=5)

# 实际执行：可能在本地内存、独立进程、或远程机器上
# 用户完全无感知具体在哪里执行
```

## 架构设计详解

### 1. 远程函数执行器的核心接口

```python
class LearnerProxy:
    """
    Learner代理 - 远程函数执行器
    
    核心功能：
    1. 透明地调用客户端learner的任意方法
    2. 自动处理参数序列化/反序列化
    3. 自动路由到正确的执行环境
    4. 统一的异步接口
    """
    
    async def execute_remote_method(
        self, 
        client_id: str, 
        method_name: str, 
        *args, 
        **kwargs
    ) -> Any:
        """
        核心方法：执行远程learner方法
        
        Args:
            client_id: 目标客户端ID  
            method_name: 要执行的learner方法名
            *args, **kwargs: 方法参数
            
        Returns:
            方法执行结果
        """
        # 自动检测执行模式并路由
        executor = self._get_executor_for_client(client_id)
        return await executor.execute_method(client_id, method_name, *args, **kwargs)
    
    # 便捷方法 - 对常用learner方法的封装
    async def train_epoch(self, client_id: str, **kwargs) -> Dict[str, Any]:
        """执行训练 - 实际调用远程learner.train_epoch()"""
        return await self.execute_remote_method(client_id, "train_epoch", **kwargs)
        
    async def evaluate(self, client_id: str, **kwargs) -> Dict[str, Any]:
        """执行评估 - 实际调用远程learner.evaluate()"""
        return await self.execute_remote_method(client_id, "evaluate", **kwargs)
        
    async def get_model_weights(self, client_id: str) -> Dict[str, Any]:
        """获取模型权重 - 实际调用远程learner.get_model_weights()"""
        return await self.execute_remote_method(client_id, "get_model_weights")
        
    async def set_model_weights(self, client_id: str, weights: Dict[str, Any]):
        """设置模型权重 - 实际调用远程learner.set_model_weights()"""
        return await self.execute_remote_method(client_id, "set_model_weights", weights)
    
    # 支持任意自定义方法调用
    async def custom_method(self, client_id: str, method_name: str, **kwargs):
        """支持调用learner的任意自定义方法"""
        return await self.execute_remote_method(client_id, method_name, **kwargs)
```

### 2. 三种执行模式的实现

#### 2.1 本地执行模式 (Local)
```python
class LocalExecutor:
    """本地执行器 - 在当前进程内存中执行"""
    
    def __init__(self):
        # 在内存中维护所有客户端的learner实例
        self.learners = {}  # client_id -> learner_instance
        
    async def execute_method(self, client_id: str, method_name: str, *args, **kwargs):
        """在本地内存中执行learner方法"""
        learner = self.learners.get(client_id)
        if not learner:
            learner = self._create_learner(client_id)
            self.learners[client_id] = learner
            
        # 获取方法并执行
        method = getattr(learner, method_name)
        
        # 如果是异步方法，直接await
        if asyncio.iscoroutinefunction(method):
            return await method(*args, **kwargs)
        else:
            # 如果是同步方法，包装为异步
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: method(*args, **kwargs))
```

#### 2.2 伪联邦执行模式 (Pseudo)
```python
class PseudoExecutor:
    """伪联邦执行器 - 在独立进程中执行"""
    
    def __init__(self, num_processes: int):
        self.process_pool = ProcessPoolExecutor(max_workers=num_processes)
        self.client_processes = {}  # client_id -> process_id
        
    async def execute_method(self, client_id: str, method_name: str, *args, **kwargs):
        """在独立进程中执行learner方法"""
        
        # 序列化参数
        serialized_args = self._serialize_args(args, kwargs)
        
        # 在进程中执行
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.process_pool,
            self._execute_in_process,
            client_id, method_name, serialized_args
        )
        
        # 反序列化结果
        return self._deserialize_result(result)
        
    def _execute_in_process(self, client_id: str, method_name: str, serialized_args):
        """在独立进程中执行的函数"""
        # 在进程中创建learner实例
        learner = self._create_learner_in_process(client_id)
        
        # 反序列化参数
        args, kwargs = self._deserialize_args(serialized_args)
        
        # 执行方法
        method = getattr(learner, method_name)
        result = method(*args, **kwargs) if not asyncio.iscoroutinefunction(method) \
                else asyncio.run(method(*args, **kwargs))
        
        # 序列化结果返回
        return self._serialize_result(result)
```

#### 2.3 分布式执行模式 (Distributed) 
```python
class DistributedExecutor:
    """分布式执行器 - 在远程机器上执行"""
    
    def __init__(self, client_endpoints: Dict[str, str]):
        self.client_endpoints = client_endpoints  # client_id -> endpoint_url
        self.session = aiohttp.ClientSession()
        
    async def execute_method(self, client_id: str, method_name: str, *args, **kwargs):
        """通过网络调用远程learner方法"""
        endpoint = self.client_endpoints[client_id]
        
        # 构造请求数据
        request_data = {
            'method': method_name,
            'args': self._serialize_args(args),
            'kwargs': self._serialize_args(kwargs)
        }
        
        # 发送HTTP请求到远程客户端
        async with self.session.post(
            f"{endpoint}/execute_learner_method",
            json=request_data,
            timeout=aiohttp.ClientTimeout(total=300)  # 5分钟超时
        ) as response:
            if response.status == 200:
                result_data = await response.json()
                return self._deserialize_result(result_data['result'])
            else:
                error_info = await response.text()
                raise RuntimeError(f"Remote execution failed: {error_info}")
```

### 3. 自动模式检测与路由

```python
class ExecutorRouter:
    """执行器路由器 - 自动选择合适的执行器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._executors = {}
        self._init_executors()
        
    def _init_executors(self):
        """根据配置初始化执行器"""
        mode = self.config.get('execution_mode', 'auto')
        
        if mode == 'auto':
            mode = self._detect_mode()
            
        if mode == 'local':
            self._executors['default'] = LocalExecutor()
        elif mode == 'pseudo':
            self._executors['default'] = PseudoExecutor(
                self.config.get('num_processes', 4)
            )
        elif mode == 'distributed':
            self._executors['default'] = DistributedExecutor(
                self.config['client_endpoints']
            )
            
    def _detect_mode(self) -> str:
        """自动检测执行模式"""
        # 检查是否有网络配置
        if 'client_endpoints' in self.config:
            return 'distributed'
            
        # 检查是否启用多进程
        if self.config.get('use_multiprocessing', False):
            return 'pseudo'
            
        # 默认本地模式
        return 'local'
        
    def get_executor(self, client_id: str) -> AbstractExecutor:
        """为指定客户端获取执行器"""
        # 目前使用统一执行器，未来可以支持混合模式
        return self._executors['default']
```

### 4. 参数序列化与传递

```python
class ParameterSerializer:
    """参数序列化器 - 处理复杂参数的序列化"""
    
    @staticmethod
    def serialize(obj: Any) -> bytes:
        """序列化对象为字节流"""
        if isinstance(obj, torch.Tensor):
            # PyTorch张量特殊处理
            return pickle.dumps({
                'type': 'torch.Tensor',
                'data': obj.cpu().numpy(),
                'dtype': str(obj.dtype),
                'device': str(obj.device)
            })
        elif isinstance(obj, np.ndarray):
            # NumPy数组处理
            return pickle.dumps({
                'type': 'numpy.ndarray', 
                'data': obj,
                'dtype': str(obj.dtype)
            })
        else:
            # 其他对象使用pickle
            return pickle.dumps(obj)
    
    @staticmethod 
    def deserialize(data: bytes) -> Any:
        """从字节流反序列化对象"""
        obj = pickle.loads(data)
        
        if isinstance(obj, dict):
            if obj.get('type') == 'torch.Tensor':
                tensor = torch.from_numpy(obj['data'])
                tensor = tensor.to(dtype=getattr(torch, obj['dtype'].split('.')[-1]))
                return tensor
            elif obj.get('type') == 'numpy.ndarray':
                return obj['data'].astype(obj['dtype'])
                
        return obj
```

## 使用示例

### 1. 对用户完全透明的调用

```python
# 在trainer中使用learner代理
@fedcl.trainer("my_fedavg")
class MyFedAvgTrainer(AbstractFederationTrainer):
    async def train(self):
        for round_num in range(self.config.num_rounds):
            # 选择客户端
            selected_clients = self.select_clients(round_num)
            
            # 并行训练 - 完全透明，用户不知道是本地/进程/网络执行
            training_tasks = []
            for client_id in selected_clients:
                task = self.learner_proxy.train_epoch(
                    client_id=client_id,
                    round_num=round_num,
                    local_epochs=5,
                    learning_rate=0.01
                )
                training_tasks.append(task)
            
            # 等待所有客户端完成训练
            results = await asyncio.gather(*training_tasks)
            
            # 聚合权重
            weights_list = [result['model_weights'] for result in results]
            global_weights = await self.aggregate_weights(weights_list)
            
            # 更新所有客户端模型
            update_tasks = []
            for client_id in selected_clients:
                task = self.learner_proxy.set_model_weights(
                    client_id=client_id,
                    weights=global_weights
                )
                update_tasks.append(task)
                
            await asyncio.gather(*update_tasks)
            
            # 评估 
            if round_num % 5 == 0:
                eval_tasks = [
                    self.learner_proxy.evaluate(client_id=client_id)
                    for client_id in selected_clients
                ]
                eval_results = await asyncio.gather(*eval_tasks)
                
                avg_accuracy = sum(r['accuracy'] for r in eval_results) / len(eval_results)
                logger.info(f"Round {round_num} - Average Accuracy: {avg_accuracy:.4f}")
```

### 2. 支持自定义learner方法调用

```python
# 用户可以调用learner的任意自定义方法
class MyCustomTrainer(AbstractFederationTrainer):
    async def train(self):
        # 调用自定义的预处理方法
        await self.learner_proxy.custom_method("client_1", "preprocess_data", batch_size=32)
        
        # 调用自定义的特殊训练方法
        result = await self.learner_proxy.custom_method(
            "client_1", 
            "train_with_distillation",
            teacher_weights=teacher_model_weights,
            alpha=0.7
        )
        
        # 调用自定义的后处理方法
        await self.learner_proxy.custom_method("client_1", "post_process_results", result)
```

### 3. 配置文件驱动的模式切换

```yaml
# 本地开发配置
execution_mode: "local"
num_clients: 3

# 本地测试配置 
execution_mode: "pseudo"
num_processes: 4
use_multiprocessing: true

# 生产部署配置
execution_mode: "distributed"
client_endpoints:
  client_1: "http://192.168.1.101:8080"
  client_2: "http://192.168.1.102:8080"
  client_3: "http://192.168.1.103:8080"
```

## 关键优势

### 1. 完全透明
- 用户调用`learner_proxy.train_epoch()`，不知道也不需要知道是在本地、进程还是网络执行
- 代码在三种模式下完全一致

### 2. 参数传递自动化
- 自动处理PyTorch张量、NumPy数组等复杂对象的序列化
- 支持任意复杂的参数传递

### 3. 异步并发
- 所有调用都是异步的，支持高效的并发执行
- 自动适配同步/异步learner方法

### 4. 高度可扩展
- 支持调用learner的任意方法，不限于预定义接口
- 可以轻松扩展新的执行模式

### 5. 配置驱动
- 通过配置文件控制执行模式，无需修改代码
- 支持自动模式检测

这种设计真正实现了项目规范中的"零联邦概念暴露"和"用户只需关心联邦学习算法逻辑"的目标！