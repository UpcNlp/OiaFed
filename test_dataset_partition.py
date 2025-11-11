"""
测试BaseLearner的自动数据集划分功能
"""
import sys
from pathlib import Path

# 添加项目路径
root = Path(__file__).parent
sys.path.insert(0, str(root))

import asyncio
from typing import Dict, Any

from fedcl.learner.base_learner import BaseLearner
from fedcl.api.builder import ComponentBuilder
from fedcl.api.registry import registry
from fedcl.api.decorators import learner
from fedcl.types import ModelData, TrainingResult, EvaluationResult
from fedcl.utils.auto_logger import get_train_logger

# 导入MNIST数据集（会自动注册）
from fedcl.methods.datasets.mnist import MNISTFederatedDataset

# 简单的测试Learner（使用装饰器注册）
@learner('TestLearner', description='测试学习器', version='1.0')
class TestLearner(BaseLearner):
    """测试学习器"""

    async def train(self, params: Dict[str, Any]) -> TrainingResult:
        return {}

    async def evaluate(self, params: Dict[str, Any]) -> EvaluationResult:
        return {}

    async def get_local_model(self) -> ModelData:
        return {}

    async def set_local_model(self, model_data: ModelData) -> bool:
        return True


def test_auto_partition():
    """测试自动数据划分功能"""
    print("=" * 80)
    print("测试 BaseLearner 自动数据集划分功能")
    print("=" * 80)

    builder = ComponentBuilder()

    # 创建3个客户端，使用不同的划分策略
    strategies = [
        ("iid", {}, "IID均匀划分"),
        ("dirichlet", {"alpha": 0.5}, "Dirichlet Non-IID (alpha=0.5)"),
        ("non_iid_label", {"labels_per_client": 2}, "Label Skew (每客户端2类)")
    ]

    for strategy_name, strategy_params, desc in strategies:
        print(f"\n{'='*80}")
        print(f"测试策略: {desc}")
        print(f"{'='*80}")

        for i in range(3):
            print(f"\n--- Client {i} ---")

            # 配置
            config_dict = {
                "training": {
                    "learner": {
                        "name": "TestLearner",
                        "params": {}
                    },
                    "dataset": {
                        "name": "MNIST",
                        "params": {
                            "root": "./data",
                            "train": True,
                            "download": True
                        },
                        # 数据集划分配置
                        "partition": {
                            "strategy": strategy_name,
                            "num_clients": 3,
                            "seed": 42,
                            "params": strategy_params
                        }
                    }
                }
            }

            # 解析配置
            parsed_config = builder.parse_config(config_dict)

            # 手动添加 partition 配置（ComponentBuilder不会保留这个字段）
            if 'dataset' in parsed_config and 'partition' in config_dict['training']['dataset']:
                parsed_config['dataset']['partition'] = config_dict['training']['dataset']['partition']

            # 创建学习器
            learner = TestLearner(
                client_id=f"client_{i}",
                config=parsed_config,
                lazy_init=False  # 立即初始化
            )

            # 获取数据集（会触发自动划分）
            dataset = learner.dataset

            print(f"  Client ID: {learner.client_id}")
            print(f"  Dataset type: {type(dataset).__name__}")
            print(f"  Samples: {len(dataset)}")

            # 获取类别分布
            if hasattr(dataset, 'indices'):
                # 这是一个Subset
                from collections import Counter
                labels = []
                for idx in dataset.indices[:100]:  # 只取前100个样本查看分布
                    _, label = dataset.dataset[idx]
                    if hasattr(label, 'item'):
                        label = label.item()
                    labels.append(label)
                dist = Counter(labels)
                print(f"  Class distribution (first 100 samples): {dict(sorted(dist.items()))}")

        print()


def test_client_index_extraction():
    """测试client_id解析功能"""
    print("\n" + "=" * 80)
    print("测试 client_id 解析功能")
    print("=" * 80)

    builder = ComponentBuilder()

    test_cases = [
        ("client_0", 0),
        ("client_5", 5),
        ("memory_client_2", 2),
        ("process_client_10_8001", 10),
    ]

    for client_id, expected_idx in test_cases:
        config_dict = {
            "training": {
                "learner": {"name": "TestLearner", "params": {}},
                "dataset": {
                    "name": "MNIST",
                    "params": {"root": "./data", "train": True, "download": False},
                    "partition": {
                        "strategy": "iid",
                        "num_clients": 20,  # 足够大以包含所有测试索引
                        "seed": 42
                    }
                }
            }
        }

        parsed_config = builder.parse_config(config_dict)
        learner = TestLearner(client_id=client_id, config=parsed_config, lazy_init=True)

        try:
            extracted_idx = learner._extract_client_index(client_id, 20)
            status = "✓" if extracted_idx == expected_idx else "✗"
            print(f"  {status} {client_id:30s} → index={extracted_idx} (expected={expected_idx})")
        except Exception as e:
            print(f"  ✗ {client_id:30s} → Error: {e}")


if __name__ == "__main__":
    # 初始化日志系统
    from fedcl.utils.auto_logger import setup_auto_logging
    setup_auto_logging()

    print("\n🚀 测试 BaseLearner 自动数据集划分功能\n")

    # 测试1: client_id解析
    test_client_index_extraction()

    # 测试2: 自动数据划分
    test_auto_partition()

    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)
