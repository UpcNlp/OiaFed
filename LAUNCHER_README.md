# FedCL 启动器使用指南

FedCL现在支持多种启动方式，包括命令行启动、Python脚本启动、控制台日志输出、信号处理和后台运行模式。

## 🚀 快速开始

### 1. 命令行启动

```bash
# 基本启动（推荐）
python fedcl_cli.py examples/config_templates/server_client_configs

# 或使用便捷脚本
./start.sh
```

### 2. Python脚本启动

```python
from main import launch_federation

# 简单启动
results = launch_federation("examples/config_templates/server_client_configs")

# 或使用快速启动
from main import quick_start
results = quick_start("examples/config_templates/server_client_configs")
```

## 📋 启动选项

### 命令行参数

```bash
python fedcl_cli.py <config_path> [options]

选项:
  --daemon, -d          后台运行模式
  --log-level LEVEL     日志级别 (DEBUG, INFO, WARNING, ERROR)
  --working-dir DIR     工作目录
  --experiment-id ID    实验ID
  --no-checkpoint       禁用检查点保存
  --quiet, -q           静默模式
```

### 使用示例

```bash
# 1. 基本启动
python fedcl_cli.py examples/config_templates/server_client_configs

# 2. 后台运行
python fedcl_cli.py examples/config_templates/server_client_configs --daemon

# 3. 调试模式
python fedcl_cli.py examples/config_templates/server_client_configs --log-level DEBUG

# 4. 静默模式
python fedcl_cli.py examples/config_templates/server_client_configs --quiet

# 5. 自定义实验ID
python fedcl_cli.py examples/config_templates/server_client_configs --experiment-id my_exp_001

# 6. 禁用检查点
python fedcl_cli.py examples/config_templates/server_client_configs --no-checkpoint
```

## 🎯 功能特性

### ✅ 已实现的功能

1. **命令行启动支持**
   - 完整的CLI界面
   - 丰富的命令行参数
   - 帮助信息和使用示例

2. **Python脚本启动接口**
   - `launch_federation()` 函数
   - `quick_start()` 简化接口
   - 兼容性别名

3. **控制台日志输出**
   - 实时日志显示
   - 彩色日志格式
   - 组件标识（SERVER/CLIENT/FEDERATION）

4. **信号处理和优雅退出**
   - Ctrl+C 优雅退出
   - SIGTERM 终止处理
   - SIGHUP 重新加载（Unix）
   - 自动清理资源

5. **后台运行模式**
   - 守护进程模式
   - PID文件管理
   - 日志文件重定向

6. **检查点和日志自动保存**
   - 默认启用检查点
   - 自动日志管理
   - 可配置保存频率

7. **线程管理**
   - 自动线程清理
   - 超时控制
   - 状态监控

## 📊 日志输出模式

### 分布式模式（服务端+客户端）
当使用配置目录时，控制台会显示：
- 🖥️ 服务端启动和状态
- 👥 客户端注册过程
- 🔄 联邦学习轮次进展
- 📈 聚合和评估结果

### 单配置模式
当使用单个配置文件时，显示相应组件的日志。

### 日志格式
```
HH:mm:ss.SSS | LEVEL    | COMPONENT | MESSAGE
```

示例：
```
10:30:15.123 | INFO     | SERVER    | Starting federation server...
10:30:15.456 | INFO     | CLIENT    | Client client_1 registered
10:30:16.789 | INFO     | FEDERATION| Round 1/10 starting...
```

## 🎮 控制操作

### 中断和停止
- **Ctrl+C**: 优雅退出，清理所有资源
- **SIGTERM**: 终止信号处理
- **SIGHUP**: 重新加载配置（Unix系统）

### 后台模式管理
```bash
# 启动后台模式
python fedcl_cli.py config/ --daemon

# 查看日志
tail -f logs/daemon/stdout.log
tail -f logs/daemon/stderr.log

# 停止后台进程
kill $(cat logs/daemon/fedcl.pid)

# 或使用系统命令
pkill -f fedcl_cli.py
```

## 📁 文件结构

启动后会创建以下目录结构：

```
项目根目录/
├── experiments/          # 实验结果目录
│   └── experiment_ID/
│       ├── config.yaml   # 保存的配置
│       ├── results.json  # 实验结果
│       └── checkpoints/  # 检查点文件
├── logs/                 # 日志目录
│   ├── experiment_logs/  # 实验日志
│   └── daemon/          # 后台模式日志
│       ├── stdout.log
│       ├── stderr.log
│       └── fedcl.pid
└── fedcl_cli.py         # 主启动器
```

## 🔧 配置选项

### 实验配置中的相关选项

```yaml
experiment:
  name: "my_federation"           # 实验名称
  log_level: "INFO"              # 日志级别
  working_dir: "./experiments"    # 工作目录
  log_base_dir: "./logs"         # 日志基础目录
  checkpoint_frequency: 10       # 检查点保存频率
  disable_checkpoint: false      # 是否禁用检查点
```

### 钩子配置（自动启用）

```yaml
hooks:
  checkpoint:
    enabled: true                # 启用检查点
    save_frequency: 10          # 保存频率
    save_dir: "checkpoints"     # 保存目录
    keep_last_n: 5             # 保留最近N个
```

## 🚨 故障排除

### 常见问题

1. **端口占用**
   ```bash
   # 查找占用端口的进程
   lsof -i :8000
   
   # 杀死占用进程
   kill -9 <PID>
   ```

2. **权限问题**
   ```bash
   # 确保启动脚本有执行权限
   chmod +x start.sh
   chmod +x fedcl_cli.py
   ```

3. **依赖问题**
   ```bash
   # 安装依赖
   pip install -r requirements.txt
   
   # 或使用uv
   uv sync
   ```

4. **后台进程无法停止**
   ```bash
   # 强制杀死所有相关进程
   pkill -f fedcl
   
   # 清理PID文件
   rm -f logs/daemon/fedcl.pid
   ```

### 调试模式

启用调试模式获取更详细的信息：

```bash
python fedcl_cli.py config/ --log-level DEBUG
```

## 📚 示例

查看 `demo.py` 文件获取完整的使用示例：

```bash
# 查看功能演示
python demo.py

# 运行脚本启动示例
python demo.py script

# 运行快速启动示例  
python demo.py quick

# 运行后台模式示例
python demo.py daemon
```

## 🎯 最佳实践

1. **开发和测试**: 使用前台模式，便于查看实时日志
2. **生产环境**: 使用后台模式，配合日志监控
3. **调试问题**: 使用DEBUG日志级别
4. **自动化**: 使用Python脚本接口集成到现有系统
5. **资源管理**: 及时清理检查点和日志文件

---

🎉 现在您可以更方便地启动和管理FedCL联邦学习实验了！
