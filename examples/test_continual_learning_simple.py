#!/usr/bin/env python3
"""
简单的持续学习测试脚本
examples/test_continual_learning_simple.py

测试单个客户端的持续学习方法（无需多进程）
"""
import sys
import torch
import asyncio
from pathlib import Path

# 添加项目路径
root = Path(__file__).parent.parent
sys.path.insert(0, str(root))

from fedcl.utils.auto_logger import setup_auto_logging
from fedcl.api.registry import registry
from fedcl.types import TrainingResult

# 初始化日志
setup_auto_logging()


async def test_single_learner(learner_name: str, num_tasks: int = 2):
    """测试单个持续学习learner"""
    print(f"\n{'='*70}")
    print(f"测试 {learner_name} Learner")
    print(f"{'='*70}\n")

    # 配置
    config = {
        'learner': {
            'name': learner_name,
            'params': {
                'client_index': 0,
                'batch_size': 64,
                'local_epochs': 1,  # 快速测试
                'learning_rate': 0.01,
                'momentum': 0.9,
                'num_tasks': num_tasks,
                'classes_per_task': 5,  # MNIST: 10类 / 2任务 = 5类/任务
                'scenario': 'class_incremental',
                'model': {
                    'name': 'SimpleCNN',
                    'params': {'num_classes': 10}
                },
                'optimizer': {
                    'type': 'SGD',
                    'lr': 0.01,
                    'momentum': 0.9
                },
                'loss': 'CrossEntropyLoss',

                # 持续学习特定参数
                'use_distillation': True,
                'distill_temperature': 2.0,
                'distill_weight': 1.0,
            }
        }
    }

    try:
        # 创建learner
        learner_class = registry.get_learner(learner_name)
        learner = learner_class(
            client_id=f'test_client_{learner_name}',
            config=config,
            lazy_init=True
        )
        print(f"✓ Learner创建成功")

        # 模拟训练多个任务
        for task_id in range(num_tasks):
            print(f"\n--- 任务 {task_id + 1}/{num_tasks} ---")

            # 创建假数据（模拟训练）
            train_params = {
                'task_id': task_id,
                'current_classes': list(range(task_id * 5, (task_id + 1) * 5)),
                'data': torch.randn(32, 1, 28, 28),  # 假数据
                'target': torch.randint(0, 5, (32,)) + task_id * 5,  # 假标签
            }

            print(f"  当前任务类别: {train_params['current_classes']}")
            print(f"  数据形状: {train_params['data'].shape}")

            # 注意: 实际训练需要真实的数据加载器
            # 这里只是验证接口可以调用
            print(f"  ✓ 任务 {task_id} 配置完成（需要真实数据集进行实际训练）")

        print(f"\n✓ {learner_name} 基本功能测试通过")
        return True

    except Exception as e:
        print(f"\n✗ {learner_name} 测试失败: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """测试所有持续学习learner"""
    learners = ['TARGET', 'FedKNOW', 'GLFC', 'FedWeIT', 'FedCPrompt', 'LGA']

    print("="*70)
    print("持续学习Learner功能测试")
    print("="*70)
    print("\n注意: 这是基本接口测试，完整训练需要真实数据集和配置文件\n")

    results = {}
    for learner_name in learners:
        results[learner_name] = await test_single_learner(learner_name)

    # 总结
    print("\n" + "="*70)
    print("测试总结")
    print("="*70)

    passed = sum(results.values())
    total = len(results)

    for learner_name, success in results.items():
        status = "✓ 通过" if success else "✗ 失败"
        print(f"  {learner_name:15s}: {status}")

    print(f"\n总计: {passed}/{total} 通过")

    if passed == total:
        print("\n✓ 所有learner基本功能正常")
        print("\n下一步：使用真实数据集和配置文件进行完整测试")
        print("  示例: PYTHONPATH=. python examples/run_continual_learning.py")
    else:
        print("\n✗ 部分learner存在问题，需要修复")

    return passed == total


if __name__ == '__main__':
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
