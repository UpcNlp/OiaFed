"""
三种组件使用方式完整演示
examples/builder_usage_demo.py

展示如何使用Builder模式、配置文件和自定义实现
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 80)
print("📚 MOE-FedCL 组件使用方式完整指南")
print("=" * 80)
print()


# ==================== 方式1：配置文件驱动（推荐用于实验）====================

print("=" * 80)
print("🔹 方式1：配置文件驱动（推荐用于批量实验）")
print("=" * 80)
print()

print("✨ 特点：")
print("  ✅ 配置文件管理，易于复现")
print("  ✅ 参数化，支持批量实验")
print("  ✅ 无需修改代码，只需修改配置文件")
print("  ✅ 支持版本控制和团队协作")
print()

print("📝 步骤1：创建配置文件 (configs/experiments/my_experiment.yaml)")
print("-" * 80)
config_example = """
# 实验配置
experiment:
  name: "FedAvg_MNIST"
  description: "FedAvg算法在MNIST数据集上的基准实验"

# 数据集配置
dataset:
  name: "MNIST"
  params:
    root: "./data"
    train: true
    download: true
  partition:
    strategy: "non_iid_label"
    num_clients: 10
    labels_per_client: 2

# 模型配置
model:
  name: "MNIST_CNN"
  params:
    num_classes: 10

# 训练器配置
trainer:
  name: "FedAvgMNIST"
  params:
    max_rounds: 100
    min_clients: 5

# 学习器配置
learner:
  name: "MNISTLearner"
  params:
    learning_rate: 0.01
    batch_size: 32
    local_epochs: 5
"""
print(config_example)

print("📝 步骤2：使用Builder从配置创建组件")
print("-" * 80)
code_example = """
from fedcl.api.builder import build_from_config
from fedcl import FederatedLearning

# 方法1: 从配置文件创建
components = build_from_config("configs/experiments/my_experiment.yaml")

# 方法2: 从字典创建
config_dict = {
    "dataset": {"name": "MNIST", "params": {...}},
    "model": {"name": "MNIST_CNN", "params": {...}},
    ...
}
components = build_from_config(config_dict)

# 使用创建的组件
dataset = components["dataset"]
model = components["model"]
trainer = components["trainer"]
"""
print(code_example)

print("💡 使用场景：")
print("  ✓ 需要运行大量对比实验")
print("  ✓ 多人协作项目，统一配置")
print("  ✓ 论文复现，提供标准配置")
print("  ✓ 超参数搜索和实验管理")
print()

# ==================== 方式2：代码驱动 + Builder（推荐用于快速原型）====================

print("=" * 80)
print("🔹 方式2：代码驱动 + Builder（推荐用于快速原型）")
print("=" * 80)
print()

print("✨ 特点：")
print("  ✅ 代码控制，灵活性高")
print("  ✅ 使用内置组件")
print("  ✅ 类型提示友好，IDE支持好")
print("  ✅ 支持延迟加载")
print()

print("📝 方法1：直接从注册表创建组件")
print("-" * 80)
method1_code = """
from fedcl.api.builder import ComponentBuilder

builder = ComponentBuilder()

# 从注册表名称创建
dataset = builder.build_dataset("MNIST", root="./data", train=True)
model = builder.build_model("MNIST_CNN", num_classes=10)
aggregator = builder.build_aggregator("FedAvg", weighted=True)

# 直接使用
data_stats = dataset.get_statistics()
model_output = model(input_tensor)
"""
print(method1_code)

print("📝 方法2：在Learner中使用Builder（支持延迟加载）")
print("-" * 80)
method2_code = """
from fedcl.api.builder import ComponentBuilder
from fedcl.learner.base_learner import BaseLearner

class MyLearner(BaseLearner):
    def __init__(self, client_id, config):
        super().__init__(client_id, config)
        self.builder = ComponentBuilder()
        self._dataset = None
        self._model = None
        self.config = config

    @property
    def dataset(self):
        # 延迟加载：第一次访问时才创建
        if self._dataset is None:
            self._dataset = self.builder.build_dataset(
                self.config.get("dataset", "MNIST"),
                **self.config.get("dataset_params", {})
            )
        return self._dataset

    @property
    def model(self):
        # 延迟加载：第一次访问时才创建
        if self._model is None:
            self._model = self.builder.build_model(
                self.config.get("model", "MNIST_CNN"),
                **self.config.get("model_params", {})
            )
        return self._model
"""
print(method2_code)

print("💡 使用场景：")
print("  ✓ 快速原型开发")
print("  ✓ 使用内置组件进行实验")
print("  ✓ 需要代码级别的控制")
print("  ✓ 单一任务快速验证")
print()

# ==================== 方式3：完全自定义（推荐用于研究）====================

print("=" * 80)
print("🔹 方式3：完全自定义（推荐用于研究创新）")
print("=" * 80)
print()

print("✨ 特点：")
print("  ✅ 完全控制实现细节")
print("  ✅ 可以实现创新算法")
print("  ✅ 不依赖内置组件")
print("  ✅ 使用装饰器注册后可被复用")
print()

print("📝 步骤1：定义自定义数据集")
print("-" * 80)
dataset_code = """
from fedcl.api import dataset
from fedcl.methods.datasets.base import FederatedDataset

@dataset("CustomDataset", description="我的自定义数据集")
class CustomDataset(FederatedDataset):
    def __init__(self, root: str = "./data", train: bool = True, **kwargs):
        super().__init__(root, train)

        # 完全自定义的数据加载逻辑
        self.data = self._load_my_data()
        self.labels = self._load_my_labels()
        self.num_classes = 10

    def _load_my_data(self):
        # 你的数据加载逻辑
        pass

    def get_statistics(self):
        return {
            'dataset_name': 'CustomDataset',
            'num_samples': len(self.data),
            'num_classes': self.num_classes
        }
"""
print(dataset_code)

print("📝 步骤2：定义自定义模型")
print("-" * 80)
model_code = """
from fedcl.api import model
from fedcl.methods.models.base import FederatedModel
import torch.nn as nn

@model("CustomModel", description="我的自定义模型", task="classification")
class CustomModel(FederatedModel):
    def __init__(self, num_classes: int = 10):
        super().__init__()

        # 你的模型架构
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # ... 更多层
        )
        self.classifier = nn.Linear(64*14*14, num_classes)

        self.set_metadata(
            task_type='classification',
            input_shape=(1, 28, 28),
            output_shape=(num_classes,)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
"""
print(model_code)

print("📝 步骤3：定义自定义Learner")
print("-" * 80)
learner_code = """
from fedcl.api import learner
from fedcl.learner.base_learner import BaseLearner

@learner("CustomLearner", description="我的自定义学习器")
class CustomLearner(BaseLearner):
    def __init__(self, client_id, config):
        super().__init__(client_id, config)

        # 直接实例化自定义组件
        self.dataset = CustomDataset(root="./data")
        self.model = CustomModel(num_classes=10)

        # 自定义优化器和训练逻辑
        self.optimizer = torch.optim.Adam(self.model.parameters())

    async def train(self, params):
        # 完全自定义的训练逻辑
        for epoch in range(params['num_epochs']):
            for batch in self.data_loader:
                # 你的训练代码
                pass

        return TrainingResponse(
            client_id=self.client_id,
            success=True,
            result={"loss": 0.5, "accuracy": 0.85, ...}
        )
"""
print(learner_code)

print("💡 使用场景：")
print("  ✓ 实现创新算法和模型")
print("  ✓ 研究型项目")
print("  ✓ 不需要内置组件")
print("  ✓ 需要完全控制实现细节")
print()

# ==================== 方式对比总结 ====================

print("=" * 80)
print("📊 三种方式对比总结")
print("=" * 80)
print()

comparison = """
┌──────────────┬──────────┬──────────┬──────────┬────────────────────┐
│ 方式         │ 灵活性   │ 易用性   │ 复现性   │ 适用场景           │
├──────────────┼──────────┼──────────┼──────────┼────────────────────┤
│ 配置文件驱动 │ ⭐⭐⭐   │ ⭐⭐⭐⭐⭐ │ ⭐⭐⭐⭐⭐ │ 批量实验、团队协作 │
│ 代码+Builder │ ⭐⭐⭐⭐ │ ⭐⭐⭐⭐   │ ⭐⭐⭐⭐   │ 快速原型、内置组件 │
│ 完全自定义   │ ⭐⭐⭐⭐⭐ │ ⭐⭐⭐     │ ⭐⭐⭐     │ 研究创新、自定义算法│
└──────────────┴──────────┴──────────┴──────────┴────────────────────┘
"""
print(comparison)

print()
print("🎯 推荐策略：")
print("-" * 80)
print("""
1. 新手入门 → 使用方式1（配置文件），快速上手
2. 快速开发 → 使用方式2（Builder），灵活性和便利性兼顾
3. 研究创新 → 使用方式3（自定义），完全控制实现细节
4. 生产部署 → 混合使用，内置组件 + 自定义组件
""")

# ==================== 实际案例 ====================

print("=" * 80)
print("💼 实际案例")
print("=" * 80)
print()

print("📘 案例1：运行经典FedAvg实验")
print("-" * 80)
case1 = """
# 使用配置文件方式
from fedcl.api.builder import build_from_config
from fedcl import FederatedLearning

# 加载预定义的FedAvg配置
config = build_from_config("configs/experiments/fedavg_mnist.yaml")

# 创建联邦学习系统
fl = FederatedLearning.from_config(config)

# 运行训练
result = await fl.run()
"""
print(case1)

print("📗 案例2：快速验证新想法")
print("-" * 80)
case2 = """
# 使用Builder方式
from fedcl.api.builder import ComponentBuilder

builder = ComponentBuilder()

# 使用内置数据集和模型
dataset = builder.build_dataset("MNIST", root="./data")
model = builder.build_model("SimpleCNN", num_classes=10)

# 自定义训练逻辑
class MyExperiment:
    def __init__(self):
        self.dataset = dataset
        self.model = model

    def run(self):
        # 快速验证想法
        pass
"""
print(case2)

print("📙 案例3：实现创新算法")
print("-" * 80)
case3 = """
# 使用完全自定义方式
from fedcl.api import learner, aggregator

# 实现你的创新聚合算法
@aggregator("MyNovelAggregator", description="我的创新聚合算法")
class MyNovelAggregator(BaseAggregator):
    async def aggregate(self, client_results):
        # 你的创新聚合逻辑
        pass

# 实现配套的学习器
@learner("MyNovelLearner", description="配套学习器")
class MyNovelLearner(BaseLearner):
    async def train(self, params):
        # 你的训练逻辑
        pass
"""
print(case3)

# ==================== 最佳实践 ====================

print()
print("=" * 80)
print("💡 最佳实践建议")
print("=" * 80)
print()

best_practices = """
1. **选择合适的方式**
   • 看场景：实验用配置，研究用自定义
   • 看团队：协作用配置，个人用代码
   • 看阶段：原型用Builder，生产用混合

2. **组件复用**
   • 优先使用内置组件（数据集、模型）
   • 自定义算法部分（聚合器、训练器）
   • 保持接口一致性

3. **配置管理**
   • 使用配置文件管理超参数
   • 版本控制配置文件
   • 文档化配置选项

4. **扩展开发**
   • 继承基类
   • 使用装饰器注册
   • 提供元数据和文档
   • 编写单元测试

5. **性能优化**
   • 使用延迟加载减少内存占用
   • 缓存重复使用的组件
   • 注意资源清理

6. **调试技巧**
   • 打印注册表查看可用组件
   • 使用日志追踪组件创建过程
   • 检查配置文件格式
"""
print(best_practices)

print()
print("=" * 80)
print("📚 参考资料")
print("=" * 80)
print("""
• 架构设计文档: docs/architecture_design.md
• Builder API文档: fedcl/api/builder.py
• 配置文件示例: configs/experiments/
• 完整MNIST示例: examples/complete_mnist_demo.py
• 装饰器文档: fedcl/api/decorators.py
""")

print()
print("✅ 演示完成！")
print()
print("🚀 下一步：")
print("  1. 查看 examples/complete_mnist_demo.py 了解真实训练示例")
print("  2. 尝试运行预定义的配置: configs/experiments/fedavg_mnist.yaml")
print("  3. 参考文档创建自己的数据集和模型")
print("  4. 加入社区讨论和贡献代码")
print()
print("=" * 80)
