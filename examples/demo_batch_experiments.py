"""
简单批量实验演示
examples/demo_batch_experiments.py

演示如何批量运行3个简单的实验
"""

import asyncio
import sys
import time
from pathlib import Path

# 添加项目路径
root = Path(__file__).parent.parent
sys.path.insert(0, str(root))

from fedcl.federated_learning import FederatedLearning


async def run_single_experiment(config_path: str, exp_name: str):
    """运行单个实验"""
    print(f"\n{'='*80}")
    print(f"运行实验: {exp_name}")
    print(f"{'='*80}")

    start_time = time.time()

    try:
        # 创建 FederatedLearning 实例
        fl = FederatedLearning(config_path)

        # 初始化
        await fl.initialize()

        # 设置实验记录
        exp_config = {
            'enabled': True,
            'name': exp_name,
            'base_dir': 'experiments/results'
        }
        server_recorder, client_recorders = fl.setup_experiment_recording(exp_config)

        # 运行训练
        result = await fl.run()

        # 保存结果
        if server_recorder and result:
            server_recorder.log_info("final_accuracy", result.final_accuracy)
            server_recorder.log_info("final_loss", result.final_loss)
            server_recorder.log_info("completed_rounds", result.completed_rounds)
            server_recorder.log_info("total_time", result.total_time)
            server_recorder.finish(status="COMPLETED")

        for client_rec in client_recorders:
            client_rec.finish(status="COMPLETED")

        duration = time.time() - start_time

        # 返回结果
        return {
            'name': exp_name,
            'status': 'success',
            'accuracy': result.final_accuracy if result else None,
            'loss': result.final_loss if result else None,
            'rounds': result.completed_rounds if result else None,
            'duration': duration
        }

    except Exception as e:
        print(f"✗ 实验失败: {e}")
        import traceback
        traceback.print_exc()

        return {
            'name': exp_name,
            'status': 'failed',
            'error': str(e),
            'duration': time.time() - start_time
        }
    finally:
        await fl.cleanup()


async def run_batch_experiments_serial(experiments):
    """串行运行批量实验"""
    print("\n" + "="*80)
    print("批量实验运行器 - 串行模式")
    print("="*80)
    print(f"\n将运行 {len(experiments)} 组实验\n")

    results = []

    for i, exp in enumerate(experiments):
        print(f"\n进度: {i+1}/{len(experiments)}")
        result = await run_single_experiment(exp['config'], exp['name'])
        results.append(result)

        # 显示当前结果
        if result['status'] == 'success':
            print(f"✓ 完成: {exp['name']}")
            print(f"  准确率: {result['accuracy']:.4f}")
            print(f"  耗时: {result['duration']:.2f}s")
        else:
            print(f"✗ 失败: {exp['name']}")

    return results


async def run_batch_experiments_parallel(experiments, max_parallel=2):
    """并行运行批量实验"""
    print("\n" + "="*80)
    print(f"批量实验运行器 - 并行模式 (最多{max_parallel}个同时)")
    print("="*80)
    print(f"\n将运行 {len(experiments)} 组实验\n")

    semaphore = asyncio.Semaphore(max_parallel)

    async def run_with_semaphore(exp):
        async with semaphore:
            return await run_single_experiment(exp['config'], exp['name'])

    tasks = [run_with_semaphore(exp) for exp in experiments]
    results = await asyncio.gather(*tasks)

    return list(results)


def print_summary(results):
    """打印结果摘要"""
    print("\n" + "="*80)
    print("实验结果摘要")
    print("="*80)

    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']

    print(f"\n总计: {len(results)} 组实验")
    print(f"成功: {len(successful)} 组")
    print(f"失败: {len(failed)} 组")

    if successful:
        print("\n成功的实验:")
        print(f"{'实验名称':<40} {'准确率':<10} {'轮数':<8} {'耗时(s)':<10}")
        print("-"*80)
        for r in successful:
            print(f"{r['name']:<40} {r['accuracy']:.4f}    {r['rounds']:<8} {r['duration']:.2f}")

        # 找出最佳
        best = max(successful, key=lambda x: x.get('accuracy', 0))
        print(f"\n🏆 最佳结果: {best['name']}")
        print(f"   准确率: {best['accuracy']:.4f}")

    if failed:
        print("\n失败的实验:")
        for r in failed:
            print(f"✗ {r['name']}: {r.get('error', 'Unknown error')}")


async def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='批量实验演示')
    parser.add_argument('--mode', choices=['serial', 'parallel'],
                       default='serial',
                       help='运行模式：串行或并行')
    parser.add_argument('--num', type=int, default=3,
                       help='实验数量')

    args = parser.parse_args()

    # 创建实验列表（这里用相同配置演示，实际应该用不同配置）
    base_config = "configs/distributed/experiments/iid/"
    timestamp = int(time.time())

    experiments = [
        {
            'name': f'batch_demo_run{i+1}_{timestamp}',
            'config': base_config
        }
        for i in range(args.num)
    ]

    # 运行实验
    if args.mode == 'serial':
        results = await run_batch_experiments_serial(experiments)
    else:
        results = await run_batch_experiments_parallel(experiments, max_parallel=2)

    # 打印摘要
    print_summary(results)

    print(f"\n所有实验结果已保存到: experiments/results/")


if __name__ == "__main__":
    asyncio.run(main())
