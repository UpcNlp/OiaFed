"""
测试统一初始化策略
examples/test_unified_init.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# 初始化日志系统
from fedcl.utils.auto_logger import setup_auto_logging
setup_auto_logging()

from fedcl.api import ComponentBuilder
from fedcl.trainer.trainer import BaseTrainer
from fedcl.learner.base_learner import BaseLearner
from fedcl.methods.aggregators.fedavg import FedAvgAggregator
import asyncio

print("=" * 80)
print("测试统一初始化策略")
print("=" * 80)

# 1. 测试Builder的parse_config
print("\n1. 测试Builder.parse_config()")
print("-" * 80)

builder = ComponentBuilder()

# 模拟配置 - 注意：使用注册的名称（小写）
test_config = {
    "training": {
        "aggregator": {
            "name": "fedavg",  # 使用实际注册的名称
            "params": {
                "weighted": True
            }
        }
    }
}

try:
    parsed = builder.parse_config(test_config)
    print(f"✅ 配置解析成功")
    print(f"   解析的组件: {list(parsed.keys())}")
    if 'aggregator' in parsed:
        print(f"   aggregator class: {parsed['aggregator']['class'].__name__}")
        print(f"   aggregator params: {parsed['aggregator']['params']}")
except Exception as e:
    print(f"❌ 配置解析失败: {e}")
    import traceback
    traceback.print_exc()

# 2. 测试Trainer的组件初始化
print("\n2. 测试Trainer的组件延迟加载")
print("-" * 80)

class TestTrainer(BaseTrainer):
    """测试训练器"""

    def _create_default_aggregator(self):
        """提供默认聚合器"""
        print("   → 调用默认创建方法")
        return FedAvgAggregator(weighted=True)

    def _create_default_global_model(self):
        """提供默认模型"""
        print("   → 调用默认创建方法")
        return {"weights": {}}

    async def train_round(self, round_num, client_ids):
        pass

    async def aggregate_models(self, client_results):
        pass

    async def evaluate_global_model(self):
        pass

    def should_stop_training(self, round_num, round_result):
        return False

try:
    # 测试1: 使用配置创建
    print("测试1: 使用配置创建组件")
    parsed_config = builder.parse_config(test_config)
    trainer1 = TestTrainer(config=parsed_config, lazy_init=True)
    print(f"✅ Trainer创建成功 (lazy_init=True)")

    # 访问aggregator触发延迟加载
    print("   访问aggregator（触发延迟加载）...")
    agg = trainer1.aggregator
    print(f"✅ Aggregator加载成功: {type(agg).__name__}")

    # 测试2: 使用默认创建方法
    print("\n测试2: 使用默认创建方法")
    trainer2 = TestTrainer(config={}, lazy_init=True)
    print("   访问global_model（触发默认创建）...")
    model = trainer2.global_model
    print(f"✅ GlobalModel创建成功: {type(model)}")

except Exception as e:
    print(f"❌ Trainer测试失败: {e}")
    import traceback
    traceback.print_exc()

# 3. 测试Learner的组件初始化
print("\n3. 测试Learner的组件延迟加载")
print("-" * 80)

class TestLearner(BaseLearner):
    """测试学习器"""

    def _create_default_dataset(self):
        """提供默认数据集"""
        print("   → 调用默认创建方法")
        return {"data": "test_data"}

    async def train(self, training_params):
        pass

    async def evaluate(self, evaluation_params):
        pass

    async def get_local_model(self):
        pass

    async def set_local_model(self, model_data):
        pass

try:
    print("测试1: 延迟加载dataset")
    learner = TestLearner(client_id="test_client", config={}, lazy_init=True)
    print(f"✅ Learner创建成功")

    # 访问dataset触发延迟加载
    print("   访问dataset（触发延迟加载）...")
    dataset = learner.dataset
    print(f"✅ Dataset加载成功: {dataset}")

except Exception as e:
    print(f"❌ Learner测试失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("✅ 统一初始化策略测试完成！")
print("=" * 80)
