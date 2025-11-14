"""
完全自动化的实验记录示例
examples/demo_auto_experiment_recording.py

展示如何通过配置文件实现零代码的实验记录

使用方法:
    python examples/demo_auto_experiment_recording.py
"""

import asyncio
import sys
from pathlib import Path

# 添加项目路径
root = Path(__file__).parent.parent
sys.path.insert(0, str(root))

from fedcl.federated_learning import FederatedLearning


async def demo_auto_recording():
    """演示：完全自动化的实验记录"""

    print("=" * 80)
    print("演示：完全自动化的实验记录")
    print("=" * 80)
    print()
    print("📋 配置文件中启用了实验记录:")
    print("   experiment:")
    print("     enabled: true")
    print("     name: 'mnist_iid_auto'")
    print()
    print("🚀 用户代码（仅需2行！）:")
    print("   fl = FederatedLearning('configs/...')")
    print("   result = await fl.run()")
    print()
    print("=" * 80)
    print()

    # ===== 用户只需要这2行代码！=====
    fl = FederatedLearning("configs/distributed/experiments/iid/")
    result = await fl.run()
    # ================================

    print()
    print("=" * 80)
    print("✅ 完成！")
    print("=" * 80)
    print()
    print("🎯 实验记录已自动完成:")
    print(f"   准确率: {result.final_accuracy:.4f}")
    print(f"   损失: {result.final_loss:.4f}")
    print(f"   轮数: {result.completed_rounds}")
    print(f"   耗时: {result.total_time:.2f}s")
    print()
    print("📁 结果自动保存到:")
    print("   experiments/results/mnist_iid_auto/")
    print()
    print("💡 无需手动:")
    print("   ✗ 实例化 Recorder")
    print("   ✗ 注册回调函数")
    print("   ✗ 调用 recorder.finish()")
    print("   ✓ 一切都是自动的！")
    print()

    await fl.cleanup()


async def demo_manual_override():
    """演示：通过代码覆盖配置"""

    print("=" * 80)
    print("演示：通过代码参数覆盖配置文件")
    print("=" * 80)
    print()

    fl = FederatedLearning("configs/distributed/experiments/iid/")

    # 可以通过参数覆盖配置文件的设置
    result = await fl.run(exp_config={
        'enabled': True,
        'name': 'manual_override_exp',
        'base_dir': 'experiments/results'
    })

    print()
    print("✅ 实验记录已保存到: experiments/results/manual_override_exp/")
    print()

    await fl.cleanup()


async def demo_disable_recording():
    """演示：禁用实验记录"""

    print("=" * 80)
    print("演示：禁用实验记录")
    print("=" * 80)
    print()

    fl = FederatedLearning("configs/distributed/experiments/iid/")

    # 通过参数禁用实验记录
    result = await fl.run(exp_config={'enabled': False})

    print()
    print("✅ 训练完成，但没有记录实验数据")
    print()

    await fl.cleanup()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='完全自动化的实验记录示例')
    parser.add_argument('--mode',
                       choices=['auto', 'override', 'disable'],
                       default='auto',
                       help='演示模式')

    args = parser.parse_args()

    if args.mode == 'auto':
        asyncio.run(demo_auto_recording())
    elif args.mode == 'override':
        asyncio.run(demo_manual_override())
    else:
        asyncio.run(demo_disable_recording())
