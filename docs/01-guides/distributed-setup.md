# 分布式部署

跨机器部署 OiaFed 的完整指南。

---

## 部署架构

```
┌─────────────┐     gRPC      ┌─────────────┐
│   Trainer   │◄─────────────►│  Learner 0  │
│ 192.168.1.1 │               │ 192.168.1.2 │
└──────┬──────┘               └─────────────┘
       │
       │ gRPC
       ▼
┌─────────────┐               ┌─────────────┐
│  Learner 1  │               │  Learner N  │
│ 192.168.1.3 │               │ 192.168.1.x │
└─────────────┘               └─────────────┘
```

---

## 配置文件

### Trainer 配置

```yaml
# trainer.yaml
node_id: trainer
role: trainer

listen:
  host: 0.0.0.0
  port: 50051

transport:
  mode: grpc

trainer:
  type: default
  args:
    max_rounds: 100

aggregator:
  type: fedavg

model:
  type: cifar10_cnn
```

### Learner 配置

```yaml
# learner_0.yaml
node_id: learner_0
role: learner

listen:
  host: 0.0.0.0
  port: 50052

connect_to:
  - trainer@192.168.1.1:50051

transport:
  mode: grpc

learner:
  type: default
  args:
    batch_size: 64
    device: cuda:0

datasets:
  - type: cifar10
    partition:
      partition_id: 0
```

---

## 启动流程

### 1. 先启动 Trainer

```bash
# 在 192.168.1.1 上
python -m src.runner --config trainer.yaml
```

### 2. 再启动 Learners

```bash
# 在 192.168.1.2 上
python -m src.runner --config learner_0.yaml

# 在 192.168.1.3 上
python -m src.runner --config learner_1.yaml
```

### 批量启动脚本

```bash
#!/bin/bash
# start_cluster.sh

TRAINER_HOST="192.168.1.1"
LEARNER_HOSTS=("192.168.1.2" "192.168.1.3" "192.168.1.4")

# 启动 Trainer
ssh user@$TRAINER_HOST "cd /path/to/oiafed && python -m src.runner --config trainer.yaml" &

sleep 5  # 等待 Trainer 就绪

# 启动 Learners
for i in "${!LEARNER_HOSTS[@]}"; do
    ssh user@${LEARNER_HOSTS[$i]} \
        "cd /path/to/oiafed && python -m src.runner --config learner_$i.yaml" &
done

wait
```

---

## Docker 部署

### Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY . .
RUN pip install -e .

ENTRYPOINT ["python", "-m", "src.runner"]
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  trainer:
    build: .
    command: --config /configs/trainer.yaml
    volumes:
      - ./configs:/configs
      - ./data:/data
    ports:
      - "50051:50051"
    networks:
      - fl_net

  learner_0:
    build: .
    command: --config /configs/learner_0.yaml
    volumes:
      - ./configs:/configs
      - ./data:/data
    depends_on:
      - trainer
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    networks:
      - fl_net

networks:
  fl_net:
    driver: bridge
```

```bash
docker-compose up -d
docker-compose logs -f
```

---

## Kubernetes 部署

### Trainer Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fl-trainer
spec:
  replicas: 1
  selector:
    matchLabels:
      app: fl-trainer
  template:
    spec:
      containers:
      - name: trainer
        image: oiafed:latest
        command: ["python", "-m", "src.runner", "--config", "/configs/trainer.yaml"]
        ports:
        - containerPort: 50051
        volumeMounts:
        - name: configs
          mountPath: /configs
      volumes:
      - name: configs
        configMap:
          name: fl-configs
---
apiVersion: v1
kind: Service
metadata:
  name: fl-trainer
spec:
  selector:
    app: fl-trainer
  ports:
  - port: 50051
```

### Learner StatefulSet

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: fl-learner
spec:
  replicas: 10
  serviceName: fl-learner
  selector:
    matchLabels:
      app: fl-learner
  template:
    spec:
      containers:
      - name: learner
        image: oiafed:latest
        command: ["python", "-m", "src.runner", "--config", "/configs/learner.yaml"]
        env:
        - name: NODE_ID
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        resources:
          limits:
            nvidia.com/gpu: 1
```

---

## 网络配置

### 防火墙

```bash
# 开放 gRPC 端口
sudo ufw allow 50051/tcp
sudo ufw allow 50052:50100/tcp
```

### 超时设置

```yaml
transport:
  mode: grpc
  grpc:
    timeout: 300        # 请求超时
    keepalive: 60       # 心跳间隔

retry:
  max_attempts: 5
  backoff_strategy: exponential
  initial_delay: 2.0
```

---

## 故障处理

### 节点掉线

```yaml
# 启用心跳检测
heartbeat:
  enabled: true
  interval: 30
  timeout: 90
```

### 自动重连

```yaml
retry:
  max_attempts: 10
  backoff_strategy: exponential
  max_delay: 60
```

### 日志收集

```yaml
tracker:
  backends:
    - type: loguru
      log_dir: /var/log/oiafed
      rotation: "100 MB"
```

---

## 监控

### Prometheus 指标

```yaml
metrics:
  enabled: true
  port: 9090
```

### 健康检查

```bash
curl http://localhost:8080/health
```

---

## 下一步

- [运行模式](running-modes.md)
- [配置系统](configuration.md)
