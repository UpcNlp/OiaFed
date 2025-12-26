# 安装指南

本文档介绍如何安装和配置 OiaFed。

---

## 环境要求

| 依赖 | 最低版本 | 推荐版本 |
|------|----------|----------|
| Python | 3.12 | 3.12+ |
| PyTorch | 2.0 | 2.7+ |
| CUDA（可选） | 11.8 | 12.0+ |

---

## 安装方式

### 方式一：从源码安装（推荐）

```bash
# 克隆仓库
git clone https://github.com/oiafed/oiafed.git
cd oiafed

# 使用 uv（推荐）
uv sync

# 或使用 pip
pip install -e .
```

### 方式二：从 PyPI 安装

```bash
pip install oiafed
```

### 方式三：使用 Docker

```bash
# 拉取镜像
docker pull oiafed/oiafed:latest

# 运行容器
docker run -it --gpus all oiafed/oiafed:latest
```

---

## 验证安装

```bash
# 检查版本
python -c "import oiafed; print(oiafed.__version__)"

# 运行简单测试
python -m oiafed.tests.smoke_test
```

---

## GPU 支持

### CUDA 安装

确保安装了匹配的 PyTorch CUDA 版本：

```bash
# 查看 CUDA 版本
nvidia-smi

# 安装对应 PyTorch（示例：CUDA 12.1）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 验证 GPU

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

---

## 可选依赖

### gRPC（分布式模式）

```bash
pip install grpcio grpcio-tools
```

### MLflow（实验追踪）

```bash
pip install mlflow
```

### 开发依赖

```bash
pip install -e ".[dev]"
# 或
uv sync --group dev
```

---

## 常见问题

### Q: `pip install` 失败

检查 Python 版本：
```bash
python --version  # 需要 >= 3.12
```

### Q: CUDA 不可用

1. 确认 NVIDIA 驱动已安装
2. 确认 PyTorch CUDA 版本匹配
3. 重新安装 PyTorch

### Q: gRPC 编译失败

安装编译工具：
```bash
# Ubuntu/Debian
sudo apt-get install build-essential

# macOS
xcode-select --install
```

---

## 下一步

安装完成后，继续阅读 [快速入门](quickstart.md)。
