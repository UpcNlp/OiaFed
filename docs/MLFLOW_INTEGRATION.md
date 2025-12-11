# MLflow 批量实验集成指南

## 📋 概述

MOE-FedCL 现在支持两种实验记录后端：
- **JSON** (默认): 轻量级，保存到本地JSON文件
- **MLflow**: 强大的实验跟踪和可视化平台

所有实验追踪器都实现统一的 `ExperimentTracker` 接口，支持无缝切换。

## 🚀 快速开始

### 1. 安装 MLflow

```bash
pip install mlflow
```

### 2. 配置 MLflow 服务器（可选）

如果使用带认证的 MLflow 服务器，需要配置 `.env` 文件：

```bash
# 复制配置模板
cp .env.example .env

# 编辑 .env 文件
nano .env
```

`.env` 文件内容：

```bash
# MLflow服务器地址
MLFLOW_TRACKING_URI=http://localhost:5000

# MLflow认证信息（如果服务器启用了认证）
MLFLOW_TRACKING_USERNAME=your_username
MLFLOW_TRACKING_PASSWORD=your_password

# 默认实验名称
MLFLOW_EXPERIMENT_NAME=MOE-FedCL
```

**注意**: `.env` 文件已被 `.gitignore`，不会提交到版本控制。

### 3. 使用方式

#### 方式A：环境变量切换（推荐）

```python
import os
# 在导入 fedcl 之前设置
os.environ['FEDCL_RECORDER_BACKEND'] = 'mlflow'

from fedcl.experiment import Recorder  # 自动使用 MLflow

# 其他代码保持不变
recorder = Recorder.initialize("my_exp", "server", "server_0")
recorder.start_run({"mode": "memory"})
recorder.log_scalar("accuracy", 0.95, step=1)
recorder.finish()
```

#### 方式B：直接导入

```python
from fedcl.experiment import MLflowRecorder

recorder = MLflowRecorder.initialize("my_exp", "server", "server_0")
# ... 使用方式完全相同
```

**MLflow认证**: `MLflowRecorder` 会自动从 `.env` 文件读取认证信息，无需在代码中硬编码。

### 4. 运行批量实验

```bash
# 使用MLflow运行批量实验
python examples/run_batch_experiments_mlflow.py --mode demo

# 查看MLflow UI
mlflow ui --backend-store-uri experiments/mlruns
# 访问: http://localhost:5000
```

## 📊 MLflow UI 功能

启动 MLflow UI后，你可以：

1. **对比实验**: 选中多个实验，点击"Compare"对比结果
2. **可视化曲线**: 查看训练准确率、损失等指标的变化趋势
3. **筛选排序**: 按准确率、损失等指标筛选和排序实验
4. **查看详情**: 查看每个实验的配置参数和运行日志
5. **下载数据**: 导出实验数据和模型文件

## 📁 文件结构

```
MOE-FedCL/
├── fedcl/
│   ├── experiment/
│   │   ├── recorder.py          # JSONRecorder（实现ExperimentTracker）
│   │   ├── mlflow_recorder.py   # MLflowRecorder（实现ExperimentTracker）
│   │   ├── callbacks.py         # 回调函数（支持新接口）
│   │   └── __init__.py          # 自动选择后端
│   │
│   └── integrations/
│       ├── base_tracker.py      # ExperimentTracker抽象接口
│       ├── mlflow_tracker.py    # 通用MLflow追踪器
│       └── __init__.py          # 集成模块入口
│
├── .env                         # MLflow配置（不提交到git）
├── .env.example                 # 配置模板
│
├── examples/
│   ├── run_batch_experiments_mlflow.py  # MLflow批量实验示例
│   └── test_mlflow_integration.py       # 集成测试
│
├── test_mlflow_auth.py          # 认证测试脚本
│
└── experiments/
    ├── results/         # JSON 结果
    └── mlruns/         # MLflow 结果
```

## 🔧 API 参考

### ExperimentTracker 接口

所有追踪器都实现以下接口：

```python
from fedcl.integrations.base_tracker import ExperimentTracker

class ExperimentTracker(ABC):
    """实验追踪抽象接口"""

    @abstractmethod
    def start_run(self, run_name=None, experiment_name=None, tags=None) -> str:
        """开始新run，返回run_id"""

    @abstractmethod
    def end_run(self, status="FINISHED") -> None:
        """结束run"""

    @abstractmethod
    def log_params(self, params: Dict[str, Any]) -> None:
        """记录超参数"""

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """记录指标"""

    @abstractmethod
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """上传文件"""

    @abstractmethod
    def set_tag(self, key: str, value: str) -> None:
        """设置标签"""
```

### MLflowRecorder

实现 `ExperimentTracker` 接口，同时兼容旧API：

```python
# 初始化
recorder = MLflowRecorder.initialize(
    experiment_name="my_exp",    # 实验名称
    role="server",               # server 或 client
    node_id="server_0",          # 节点ID
    base_dir="experiments/mlruns"  # 存储目录
)

# === 新接口（推荐）===
recorder.start_run(tags={'model': 'resnet'})
recorder.log_params({'learning_rate': 0.01, 'batch_size': 32})
recorder.log_metrics({'accuracy': 0.95, 'loss': 0.3}, step=10)
recorder.set_tag('status', 'completed')
recorder.end_run(status="FINISHED")

# === 旧接口（向后兼容）===
recorder.start_run(config_dict)
recorder.log_scalar(name, value, step=round_num)
recorder.log_info(key, value)
recorder.add_artifact(file_path)
recorder.finish(status="COMPLETED")
```

**认证配置**: MLflowRecorder 会自动从环境变量读取认证信息：
- `MLFLOW_TRACKING_URI`: 服务器地址
- `MLFLOW_TRACKING_USERNAME`: 用户名
- `MLFLOW_TRACKING_PASSWORD`: 密码

### BatchExperimentRunner

批量实验运行器保持不变，只需切换 Recorder 后端：

```python
from fedcl.experiment import BatchExperimentRunner

runner = BatchExperimentRunner(
    base_config="configs/distributed/experiments/iid/",
    experiment_variants=[
        {'name': 'exp1_fedavg', 'overrides': {...}},
        {'name': 'exp2_fedprox', 'overrides': {...}},
    ]
)

results = await runner.run_all(parallel=False)
```

## 🎯 使用场景

### JSON 记录器（默认）

适合：
- 快速开发和调试
- 不需要复杂可视化
- 离线环境

### MLflow 记录器

适合：
- 对比多组实验
- 需要可视化分析
- 团队协作
- 超参数搜索

## 💡 最佳实践

### 1. 批量对比实验

```python
# 创建算法对比实验
experiments = create_algorithm_comparison_experiments(
    base_name="mnist_comparison",
    algorithms=['fedavg', 'fedprox', 'scaffold']
)

# MLflow 会自动组织这些实验
# 在 UI 中可以一键对比结果
```

### 2. 网格搜索

```python
# 创建网格搜索
experiments = create_grid_search_experiments(
    base_name="hyperparameter_search",
    param_grid={
        'learning_rate': [0.01, 0.001, 0.0001],
        'batch_size': [32, 64, 128]
    }
)

# MLflow 记录所有组合的结果
# 可以在 UI 中找出最佳配置
```

### 3. 联邦学习场景

```python
# Server 和 Clients 会创建独立的 runs
# 但都属于同一个 experiment
# 可以在 UI 中查看：
# - Server 的聚合指标
# - 每个 Client 的本地训练指标
# - 跨节点的对比分析
```

## 🆚 对比

| 特性 | JSON Recorder | MLflow Recorder |
|------|--------------|-----------------|
| 安装 | 无需额外安装 | 需要 `pip install mlflow` |
| 存储 | 本地 JSON 文件 | MLflow 格式（文件或数据库） |
| 可视化 | 需要自己解析 | 内置 Web UI |
| 实验对比 | 手动对比 | 一键对比 |
| 性能 | 轻量快速 | 稍重但功能强大 |
| 学习曲线 | 简单 | 中等 |

## 🔍 查看结果

### JSON 结果

```bash
# 查看 JSON 文件
cat experiments/results/my_exp/server_server_0/run.json
```

### MLflow 结果

```bash
# 启动 UI
cd /home/nlp/ct/projects/MOE-FedCL
mlflow ui --backend-store-uri experiments/mlruns

# 打开浏览器访问
http://localhost:5000
```

## ⚠️ 注意事项

1. **环境变量设置时机**: 必须在导入 `fedcl.experiment` **之前**设置环境变量
2. **单例模式**: 同一节点的 Recorder 使用单例模式，需要 `Recorder.reset()` 才能创建新实例
3. **并发运行**: MLflow 支持并发运行多个实验，会自动创建不同的 run_id
4. **存储位置**: JSON 默认存储到 `experiments/results/`，MLflow 默认存储到 `experiments/mlruns/`
5. **认证安全**: `.env` 文件包含敏感信息，已被添加到 `.gitignore`，不会提交到版本控制
6. **认证方式**: MLflow 支持 Basic Auth，通过环境变量配置用户名和密码

## 🔐 安全最佳实践

1. **不要硬编码认证信息**: 始终使用 `.env` 文件或环境变量
2. **保护 .env 文件**: 确保 `.env` 在 `.gitignore` 中
3. **使用强密码**: MLflow 服务器密码应该足够复杂
4. **限制访问**: 在生产环境中使用防火墙限制 MLflow 服务器访问

## 🧪 测试 MLflow 认证

运行认证测试脚本：

```bash
# 测试 MLflow 连接和认证
python test_mlflow_auth.py
```

该脚本会：
1. 从 `.env` 加载配置
2. 测试 MLflow 服务器连接
3. 验证认证是否正常
4. 创建测试 run 并记录数据

## 📝 示例

查看完整示例：
- `examples/test_mlflow_integration.py` - 基础功能测试
- `examples/run_batch_experiments_mlflow.py` - 批量实验示例

## 🐛 故障排除

### MLflow 未安装

```
ImportError: No module named 'mlflow'
```

解决：`pip install mlflow`

### 找不到实验

启动 MLflow UI 时指定正确的路径：
```bash
mlflow ui --backend-store-uri experiments/mlruns
```

### 端口已被占用

使用不同端口：
```bash
mlflow ui --port 5001 --backend-store-uri experiments/mlruns
```

## 📚 更多资源

- [MLflow 官方文档](https://mlflow.org/docs/latest/index.html)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [MLflow UI](https://mlflow.org/docs/latest/tracking.html#tracking-ui)
