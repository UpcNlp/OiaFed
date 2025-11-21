# TABLE III 批量实验运行指南

## 快速开始

### 1. 测试模式（推荐首次运行）

```bash
cd /home/nlp/ct/projects/MOE-FedCL
PYTHONPATH=. python examples/reproduce_table3_experiments.py --mode test
```

这将运行少量实验来验证系统配置是否正确。

### 2. 运行完整的 TABLE III 实验

```bash
# 顺序执行所有实验
PYTHONPATH=. python examples/reproduce_table3_experiments.py

# 后台运行并保存日志
nohup PYTHONPATH=. python examples/reproduce_table3_experiments.py > /tmp/table3_run.log 2>&1 &
```

**实验规模**：
- 总实验数: ~54个（3个数据集 × 6个算法 × 3个Non-IID设置）
- 每个实验: 100轮联邦学习
- 预计耗时: 数小时到十几小时（取决于硬件）

---

## 批量运行命令详解

### 方法1: 默认运行（推荐）

```bash
PYTHONPATH=. python examples/reproduce_table3_experiments.py
```

**特点**：
- 顺序执行所有实验
- 自动保存配置到日志目录
- 结果记录到 MLflow 和数据库

### 方法2: 使用智能调度器（支持断点续跑）

```bash
PYTHONPATH=. python examples/reproduce_table3_experiments.py \
    --mode all \
    --use-smart-runner \
    --multiprocess \
    --max-concurrent 12 \
    --repetitions 3 \
    --quiet \
    --db-path experiments/table3_tracker.db
```

**后台运行（推荐）**：
```bash
nohup python examples/reproduce_table3_experiments.py \
      --mode all \
      --use-smart-runner \
      --multiprocess \
      --max-concurrent 15 \
      --repetitions 3 \
      --db-path experiments/table3_tracker.db > output.log 2>&1 &
```

**参数说明**：
- `--mode all`: 运行所有实验
- `--use-smart-runner`: 使用智能调度器（支持断点续跑）
- `--multiprocess`: 多进程并发运行
- `--max-concurrent 12`: 最多同时运行12个实验
- `--repetitions 3`: 每个实验重复3次
- `--quiet`: 简洁模式，控制台只显示ERROR和进度信息
- `--db-path`: 指定数据库路径

**关于 `--quiet` 模式**：
- 控制台只显示批量实验进度信息和错误
- 详细训练日志自动保存到 `logs/exp_*/` 目录
- 适用于后台运行和大规模批量实验
- output.log 保持干净，只包含关键信息

### 方法3: 运行特定数据集

```bash
# 只运行 MNIST 数据集
PYTHONPATH=. python examples/reproduce_table3_experiments.py \
    --mode single \
    --dataset MNIST

# 只运行特定 Non-IID 类型
PYTHONPATH=. python examples/reproduce_table3_experiments.py \
    --mode noniid \
    --noniid_type label_skew
```

---

## 实验监控

### 监控方法1: 实时查看日志

```bash
# 查看最新实验的服务器训练日志
tail -f logs/exp_*/train/server.log

# 过滤关键信息
tail -f logs/exp_*/train/server.log | grep -E "(Round|Accuracy|Loss)"

# 查看所有实验目录
ls -lt logs/
```

### 监控方法2: 查看运行日志（后台运行时）

```bash
# 实时查看运行日志
tail -f /tmp/table3_run.log

# 统计完成的实验数量
grep -c "实验完成" /tmp/table3_run.log

# 查看失败的实验
grep "实验失败" /tmp/table3_run.log
```

### 监控方法3: 检查进程状态

```bash
# 查看Python进程
ps aux | grep reproduce_table3_experiments.py

# 查看GPU使用情况
nvidia-smi

# 实时监控GPU（每2秒刷新）
watch -n 2 nvidia-smi
```

### 监控方法4: 查看数据库状态（智能调度器）

```bash
# 查看实验统计
sqlite3 experiments/table3_tracker.db "
SELECT
  status,
  COUNT(*) as count
FROM experiments
GROUP BY status;
"

# 查看最近完成的实验
sqlite3   "
SELECT
  experiment_name,
  status,
  accuracy,
  datetime(start_time, 'unixepoch', 'localtime') as start_time
FROM experiments
ORDER BY start_time DESC
LIMIT 10;
"
```

---

## 查看实验配置

### 新功能：自动配置记录

每个实验运行时，系统会自动保存配置文件到日志目录：

```
logs/exp_20251117-15-30-45/
├── experiment_config.json    ← 完整的实验配置（NEW!）
├── train/
│   ├── server.log
│   └── client_*.log
├── comm/
└── sys/
```

### 查看配置文件

```bash
# 查看最新实验的配置
cat logs/exp_$(ls -t logs/ | grep exp_ | head -1)/experiment_config.json

# 使用 jq 格式化查看
cat logs/exp_$(ls -t logs/ | grep exp_ | head -1)/experiment_config.json | jq '.'

# 查看所有实验的数据集和算法
for dir in logs/exp_*/; do
  echo "=== $dir ==="
  cat $dir/experiment_config.json | jq -r '"\(.dataset.name) - \(.federated_learning.algorithm) - \(.data_partition.noniid_type)"'
done

# 批量提取配置信息
for dir in logs/exp_*/; do
  config="$dir/experiment_config.json"
  if [ -f "$config" ]; then
    dataset=$(jq -r '.dataset.name' "$config")
    algo=$(jq -r '.federated_learning.algorithm' "$config")
    noniid=$(jq -r '.data_partition.noniid_type' "$config")
    alpha=$(jq -r '.data_partition.alpha' "$config")
    echo "$dir: $dataset | $algo | $noniid (α=$alpha)"
  fi
done
```

### 配置文件内容示例

```json
{
  "experiment_id": "exp_20251117-15-30-45",
  "timestamp": "2025-11-17T15:30:45.123456",
  "dataset": {
    "name": "MNIST",
    "data_dir": ""
  },
  "data_partition": {
    "noniid_type": "label_skew",
    "alpha": 0.5,
    "num_clients": 100,
    "samples_per_client": null
  },
  "federated_learning": {
    "algorithm": "FedProx",
    "aggregator": "FedAvgAggregator",
    "num_rounds": 100,
    "clients_per_round": 10,
    "local_epochs": 5
  },
  "model": {
    "name": "MNIST_LeNet",
    "architecture": "MNIST_LeNet",
    "params": {
      "num_classes": 10
    }
  },
  "training": {
    "learning_rate": 0.01,
    "batch_size": 32,
    "optimizer": "SGD",
    "loss_function": "CrossEntropyLoss"
  },
  "communication": {
    "mode": "ProcessAndNetwork",
    "backend": null
  },
  "misc": {
    "seed": null,
    "device": "cuda",
    "early_stopping": true,
    "patience": 5
  }
}
```

---

## 查看实验结果

### 方法1: MLflow UI

```bash
# 启动 MLflow UI
mlflow ui --backend-store-uri experiments/table3_mlruns

# 然后在浏览器访问: http://localhost:5000
```

### 方法2: 查看数据库

```bash
# 查看所有实验结果
sqlite3 experiments/experiment_tracker.db "
SELECT
  experiment_name,
  accuracy,
  loss,
  duration,
  status
FROM experiments
ORDER BY start_time DESC;
"

# 导出到CSV
sqlite3 -header -csv experiments/experiment_tracker.db "
SELECT * FROM experiments;
" > table3_results.csv
```

### 方法3: 查看日志文件

```bash
# 查看特定实验的训练日志
cat logs/exp_20251117-15-30-45/train/server.log

# 提取准确率信息
grep "Accuracy" logs/exp_*/train/server.log
```

---

## 停止和恢复实验

### 停止运行中的实验

```bash
# 找到进程ID
ps aux | grep reproduce_table3_experiments.py

# 优雅停止（让当前实验完成）
kill <PID>

# 强制停止
kill -9 <PID>
```

### 断点续跑（智能调度器）

如果使用了 `--use-smart-runner`，实验会自动保存状态到数据库。重新运行相同命令时，会自动跳过已完成的实验：

```bash
# 相同的命令会自动跳过已完成的实验
PYTHONPATH=. python examples/reproduce_table3_experiments.py \
    --mode all \
    --use-smart-runner \
    --db-path experiments/table3_tracker.db
```

---

## 常见问题

### Q: 如何只运行特定算法？

修改 `examples/reproduce_table3_experiments.py` 中的 `main()` 函数：

```python
experiments = generate_all_experiments(
    datasets=['MNIST'],  # 只运行MNIST
    algorithms=['FedAvg', 'FedProx'],  # 只运行这两个算法
    noniid_types=['label_skew']  # 只运行label_skew
)
```

### Q: 配置文件保存在哪里？

每个实验的配置自动保存在：
```
logs/exp_<时间戳>/experiment_config.json
```

### Q: 如何并发运行多个实验？

使用智能调度器的 `--max-concurrent` 参数：

```bash
PYTHONPATH=. python examples/reproduce_table3_experiments.py \
    --use-smart-runner \
    --multiprocess \
    --max-concurrent 3  # 同时运行3个实验
```

### Q: 如何查看实验进度百分比？

```bash
# 如果使用智能调度器
sqlite3 experiments/table3_tracker.db "
SELECT
  COUNT(CASE WHEN status='completed' THEN 1 END) as completed,
  COUNT(CASE WHEN status='running' THEN 1 END) as running,
  COUNT(CASE WHEN status='failed' THEN 1 END) as failed,
  COUNT(*) as total,
  ROUND(100.0 * COUNT(CASE WHEN status='completed' THEN 1 END) / COUNT(*), 2) as progress_pct
FROM experiments;
"
```

---

## 推荐工作流

### 首次运行

1. **测试系统配置**
   ```bash
   PYTHONPATH=. python examples/reproduce_table3_experiments.py --mode test
   ```

2. **检查配置文件是否生成**
   ```bash
   ls -lh logs/exp_*/experiment_config.json
   cat logs/exp_$(ls -t logs/ | grep exp_ | head -1)/experiment_config.json | jq '.'
   ```

3. **运行完整实验（后台）**
   ```bash
   nohup PYTHONPATH=. python examples/reproduce_table3_experiments.py > /tmp/table3_run.log 2>&1 &
   ```

4. **监控进度**
   ```bash
   tail -f /tmp/table3_run.log
   tail -f logs/exp_*/train/server.log
   ```

### 高级用法（大规模实验）

```bash
# 使用智能调度器 + 多进程 + GPU调度
PYTHONPATH=. python examples/reproduce_table3_experiments.py \
    --mode all \
    --use-smart-runner \
    --multiprocess \
    --max-concurrent 3 \
    --repetitions 3 \
    --enable-gpu-scheduling \
    --db-path experiments/table3_tracker.db

# 在另一个终端监控
watch -n 5 'sqlite3 experiments/table3_tracker.db "SELECT status, COUNT(*) FROM experiments GROUP BY status;"'
```

---

## 日志控制

### 控制台日志抑制

对于批量实验和后台运行，可以通过配置文件和环境变量控制日志输出：

#### 方法1: 配置文件（推荐）

在 `configs/distributed/experiments/table3/server.yaml` 中设置：

```yaml
logging:
  console_enabled: false  # 禁用控制台输出（只输出到文件）
  level: "INFO"           # 控制台日志级别（如果启用）
  file_level: "DEBUG"     # 文件日志级别
```

#### 方法2: 环境变量

```bash
# 抑制组件注册日志（默认已抑制）
export FEDCL_VERBOSE_REGISTRATION=false

# 抑制MLflow详细日志（默认已抑制）
export FEDCL_MLFLOW_VERBOSE=false

# 设置控制台日志级别为ERROR（只显示错误）
export FEDCL_CONSOLE_LOG_LEVEL=ERROR
```

#### 方法3: 使用 --quiet 参数（最简单）

```bash
# 批量实验时使用 --quiet
python examples/reproduce_table3_experiments.py --mode all --quiet
```

这会自动设置 `FEDCL_CONSOLE_LOG_LEVEL=ERROR`，只在控制台显示：
- ✅ 批量实验进度信息
- ✅ 错误消息（ERROR级别）
- ❌ 不显示组件注册日志
- ❌ 不显示训练日志（Epoch, Loss, Accuracy）
- ❌ 不显示MLflow详细日志

所有详细日志仍然保存在文件中：
- 训练日志: `logs/exp_*/train/server.log` 和 `client_*.log`
- 通信日志: `logs/exp_*/comm/`
- 系统日志: `logs/exp_*/sys/`

### 启用详细日志（调试时）

如果需要查看详细的日志输出（例如调试时），可以启用：

```bash
# 启用组件注册日志
export FEDCL_VERBOSE_REGISTRATION=true

# 启用MLflow详细日志
export FEDCL_MLFLOW_VERBOSE=true

# 或者修改 server.yaml
logging:
  console_enabled: true
  level: "DEBUG"
```

---

**最后更新**: 2025-11-17
**自动配置记录功能**: 已启用
**日志控制功能**: 已启用
