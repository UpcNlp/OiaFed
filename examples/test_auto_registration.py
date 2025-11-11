"""
测试自动注册功能
examples/test_auto_registration.py

展示只需导入 fedcl，所有内置组件就会自动注册
"""
import sys
from pathlib import Path

# 添加项目路径
root = Path(__file__).parent.parent
sys.path.insert(0, str(root))

# 只需导入 fedcl 的任何内容，所有内置组件就会自动注册！
# 无需手动 import fedcl.methods
from fedcl.federated_learning import FederatedLearning  # noqa: F401

# 检查注册的组件
from fedcl.api.registry import registry

print("=" * 80)
print("自动注册的组件列表")
print("=" * 80)
print("\n提示：用户只需导入 fedcl 的任何功能，所有内置组件自动注册！")
print("例如：from fedcl.federated_learning import FederatedLearning")
print()

print("\n✓ 已注册的模型:")
for name in registry.models.keys():
    print(f"  - {name}")

print("\n✓ 已注册的学习器:")
for name in registry.learners.keys():
    print(f"  - {name}")

print("\n✓ 已注册的训练器:")
for name in registry.trainers.keys():
    print(f"  - {name}")

print("\n✓ 已注册的聚合器:")
for name in registry.aggregators.keys():
    print(f"  - {name}")

print("\n✓ 已注册的数据集:")
for name in registry.datasets.keys():
    print(f"  - {name}")

print("\n✓ 已注册的评估器:")
for name in registry.evaluators.keys():
    print(f"  - {name}")

print("\n" + "=" * 80)
print("总结")
print("=" * 80)
counts = registry.get_component_count()
print(f"模型: {counts['models']} 个")
print(f"学习器: {counts['learners']} 个")
print(f"训练器: {counts['trainers']} 个")
print(f"聚合器: {counts['aggregators']} 个")
print(f"数据集: {counts['datasets']} 个")
print(f"评估器: {counts['evaluators']} 个")
print(f"\n总共: {sum(counts.values())} 个组件已自动注册！")
print("\n✓ 用户无需任何额外配置，开箱即用！")
