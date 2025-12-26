# 测试指南

编写和运行 OiaFed 测试。

---

## 测试结构

```
tests/
├── unit/                    # 单元测试
│   ├── test_aggregators.py
│   ├── test_learners.py
│   └── test_registry.py
├── integration/             # 集成测试
│   ├── test_serial_mode.py
│   └── test_communication.py
├── e2e/                     # 端到端测试
│   └── test_full_training.py
├── conftest.py              # pytest fixtures
└── fixtures/                # 测试数据
    └── configs/
```

---

## 运行测试

### 全部测试

```bash
pytest
```

### 特定测试

```bash
# 单个文件
pytest tests/unit/test_aggregators.py

# 单个测试
pytest tests/unit/test_aggregators.py::test_fedavg

# 按标记
pytest -m "not slow"
```

### 覆盖率

```bash
pytest --cov=src --cov-report=html
```

---

## 单元测试

### Aggregator 测试

```python
# tests/unit/test_aggregators.py
import pytest
import torch
from oiafed import ClientUpdate
from oiafed.methods.aggregators import FedAvgAggregator

class TestFedAvgAggregator:
    
    @pytest.fixture
    def aggregator(self):
        return FedAvgAggregator(weighted=True)
    
    @pytest.fixture
    def updates(self):
        return [
            ClientUpdate(
                client_id="c0",
                weights={"layer": torch.tensor([1.0, 2.0])},
                num_samples=100,
            ),
            ClientUpdate(
                client_id="c1",
                weights={"layer": torch.tensor([3.0, 4.0])},
                num_samples=200,
            ),
        ]
    
    def test_aggregate_weighted(self, aggregator, updates):
        result = aggregator.aggregate(updates)
        
        assert "layer" in result
        # (1*100 + 3*200) / 300 = 2.33
        # (2*100 + 4*200) / 300 = 3.33
        expected = torch.tensor([2.333, 3.333])
        assert torch.allclose(result["layer"], expected, atol=0.01)
    
    def test_empty_updates(self, aggregator):
        with pytest.raises(ValueError):
            aggregator.aggregate([])
```

### Learner 测试

```python
# tests/unit/test_learners.py
import pytest
import torch
import torch.nn as nn
from oiafed.methods.learners import DefaultLearner

class TestDefaultLearner:
    
    @pytest.fixture
    def model(self):
        return nn.Linear(10, 2)
    
    @pytest.fixture
    def learner(self, model):
        return DefaultLearner(
            model=model,
            batch_size=32,
            lr=0.01,
        )
    
    @pytest.mark.asyncio
    async def test_train_step(self, learner):
        batch = (torch.randn(32, 10), torch.randint(0, 2, (32,)))
        
        await learner.setup({"device": "cpu"})
        metrics = await learner.train_step(batch, 0)
        
        assert metrics.loss > 0
        assert metrics.num_samples == 32
    
    def test_get_set_weights(self, learner):
        weights = learner.get_weights()
        learner.set_weights(weights)
```

---

## 集成测试

### 通信测试

```python
# tests/integration/test_communication.py
import pytest
from oiafed.cnm import Node, MemoryTransport

@pytest.mark.asyncio
async def test_node_communication():
    # 创建两个节点
    transport = MemoryTransport()
    
    node1 = Node("node1", "trainer", transport)
    node2 = Node("node2", "learner", transport)
    
    # 注册处理器
    @node2.register_handler("echo")
    async def echo(data):
        return {"echo": data["message"]}
    
    # 启动
    await node1.start()
    await node2.start()
    
    # 测试调用
    proxy = node1.get_proxy("node2")
    result = await proxy.call("echo", message="hello")
    
    assert result["echo"] == "hello"
    
    # 停止
    await node1.stop()
    await node2.stop()
```

### Serial 模式测试

```python
# tests/integration/test_serial_mode.py
import pytest
from oiafed import FederationRunner

@pytest.mark.asyncio
async def test_serial_training():
    runner = FederationRunner(
        "tests/fixtures/configs/simple.yaml",
        mode="serial",
        num_clients=3,
    )
    
    result = await runner.run()
    
    assert "round_results" in result
    assert len(result["round_results"]) > 0
```

---

## 端到端测试

```python
# tests/e2e/test_full_training.py
import pytest
from oiafed import FederationRunner

@pytest.mark.slow
@pytest.mark.asyncio
async def test_fedavg_cifar10():
    """完整的 FedAvg + CIFAR-10 测试"""
    runner = FederationRunner(
        "tests/fixtures/configs/fedavg_cifar10.yaml",
        mode="serial",
        num_clients=5,
    )
    
    result = await runner.run()
    
    final_acc = result["final_metrics"]["accuracy"]
    assert final_acc > 0.5  # 至少比随机好
```

---

## Fixtures

### conftest.py

```python
# tests/conftest.py
import pytest
import torch
import torch.nn as nn

@pytest.fixture
def simple_model():
    """简单线性模型"""
    return nn.Linear(10, 2)

@pytest.fixture
def cnn_model():
    """简单 CNN"""
    return nn.Sequential(
        nn.Conv2d(3, 16, 3),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(16 * 30 * 30, 10),
    )

@pytest.fixture
def sample_weights():
    """样本权重"""
    return {
        "weight": torch.randn(2, 10),
        "bias": torch.randn(2),
    }

@pytest.fixture
def sample_updates(sample_weights):
    """样本更新"""
    from oiafed import ClientUpdate
    return [
        ClientUpdate("c0", sample_weights, 100),
        ClientUpdate("c1", sample_weights, 200),
    ]
```

---

## 标记

```python
# 慢速测试
@pytest.mark.slow
def test_full_training(): ...

# 需要 GPU
@pytest.mark.gpu
def test_cuda_training(): ...

# 异步测试
@pytest.mark.asyncio
async def test_async(): ...
```

### 跳过特定测试

```bash
# 跳过慢速测试
pytest -m "not slow"

# 只运行快速测试
pytest -m "not slow and not gpu"
```

---

## Mock

```python
from unittest.mock import Mock, AsyncMock, patch

@pytest.mark.asyncio
async def test_with_mock():
    # Mock 远程调用
    mock_proxy = Mock()
    mock_proxy.call = AsyncMock(return_value={"weights": {}})
    
    trainer = Trainer(...)
    trainer.get_proxy = Mock(return_value=mock_proxy)
    
    result = await trainer.collect_updates(["c0"])
    
    mock_proxy.call.assert_called_once()
```

---

## 配置

### pytest.ini

```ini
[pytest]
asyncio_mode = auto
markers =
    slow: 慢速测试
    gpu: 需要 GPU
testpaths = tests
python_files = test_*.py
python_functions = test_*
```

### pyproject.toml

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

---

## CI 集成

### GitHub Actions

```yaml
# .github/workflows/test.yml
name: Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      
      - name: Install
        run: pip install -e ".[dev]"
      
      - name: Test
        run: pytest --cov=src
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

---

## 下一步

- [代码规范](coding-style.md)
- [贡献指南](../../CONTRIBUTING.md)
