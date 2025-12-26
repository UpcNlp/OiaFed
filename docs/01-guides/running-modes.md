# 运行模式

OiaFed 支持三种运行模式，适应从调试到生产的不同场景。

---

## 模式对比

| 模式 | 进程 | 通信 | GPU | 用途 |
|------|------|------|-----|------|
| **Serial** | 单进程 | Memory | 共享 | 开发调试 |
| **Parallel** | 多进程 | Memory/gRPC | 独立 | 单机训练 |
| **Distributed** | 多机器 | gRPC | 独立 | 生产部署 |

```
Serial:     [Trainer + Learners] 单进程
Parallel:   [Trainer] ←→ [Learner1] ←→ [Learner2] 多进程
Distributed: [机器A] ←gRPC→ [机器B] ←gRPC→ [机器C] 多机器
```

---

## Serial 模式

**特点**：所有节点在单进程运行，便于调试。

### 使用

```bash
python -m src.runner --config config.yaml --mode serial --num-clients 5
```

### 优点

- ✅ 可设断点调试
- ✅ 启动快
- ✅ 无进程开销

### 缺点

- ❌ 单线程，无法利用多核
- ❌ 共享 GPU 内存

### 调试技巧

```python
# 在代码中设断点
import pdb; pdb.set_trace()
```

---

## Parallel 模式

**特点**：每个节点独立进程，单机多核/多卡。

### 使用

```bash
python -m src.runner --config config.yaml --mode parallel --num-clients 10
```

### GPU 分配

```yaml
# 环境变量方式
learner:
  args:
    device: cuda:${GPU_ID}

# 或在配置中指定
# learner_0.yaml
learner:
  args:
    device: cuda:0

# learner_1.yaml
learner:
  args:
    device: cuda:1
```

### 优点

- ✅ 利用多核 CPU
- ✅ 支持多 GPU
- ✅ 进程隔离

### 缺点

- ❌ 进程启动开销
- ❌ 调试困难

---

## Distributed 模式

**特点**：跨机器部署，真实分布式环境。

### Trainer 配置 (192.168.1.100)

```yaml
# trainer.yaml
node_id: trainer
role: trainer

listen:
  host: 0.0.0.0
  port: 50051

transport:
  mode: grpc
```

### Learner 配置 (192.168.1.101)

```yaml
# learner_0.yaml
node_id: learner_0
role: learner

listen:
  host: 0.0.0.0
  port: 50052

connect_to:
  - trainer@192.168.1.100:50051

transport:
  mode: grpc
```

### 启动

```bash
# 机器 A
python -m src.runner --config trainer.yaml

# 机器 B
python -m src.runner --config learner_0.yaml

# 机器 C
python -m src.runner --config learner_1.yaml
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'
services:
  trainer:
    image: oiafed:latest
    command: python -m src.runner --config /configs/trainer.yaml
    ports:
      - "50051:50051"

  learner_0:
    image: oiafed:latest
    command: python -m src.runner --config /configs/learner_0.yaml
    depends_on:
      - trainer
```

---

## 选择指南

| 场景 | 推荐模式 |
|------|----------|
| 开发新算法 | Serial |
| 调试 bug | Serial |
| 单机多卡 | Parallel |
| 性能测试 | Parallel |
| 跨机房部署 | Distributed |
| 大规模实验 | Distributed |

### 开发流程

```
Serial 开发 → Serial 测试 → Parallel 验证 → Distributed 部署
```

---

## 故障排除

### 端口占用

```bash
lsof -i :50051
kill -9 <PID>
```

### 连接超时

```yaml
retry:
  max_attempts: 5
  initial_delay: 2.0

default_timeout: 120
```

### GPU OOM

```yaml
learner:
  args:
    batch_size: 32  # 减小
    device: cpu     # 或用 CPU
```

---

## 下一步

- [分布式部署](distributed-setup.md) - 详细部署指南
- [配置系统](configuration.md) - 配置详解
