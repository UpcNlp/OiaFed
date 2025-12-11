#!/usr/bin/env python3
"""
测试新统一日志系统
examples/test_unified_logging.py

测试场景：
1. 本地串行模式（Memory 通信）
2. 三层日志架构（Runtime + Progress + Tracker）
3. 嵌套 Runs（Server 主 run + Client 嵌套 runs）
4. 异常处理和日志收集

使用方法:
    # 测试正常流程
    python examples/test_unified_logging.py

    # 测试异常流程
    python examples/test_unified_logging.py --test-exception
"""
import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件（MLflow 配置）
load_dotenv('.env')

# 添加项目路径
root = Path(__file__).parent.parent
sys.path.insert(0, str(root))

from fedcl.federated_learning import FederatedLearning


async def test_normal_flow():
    """测试正常训练流程"""
    print("=" * 80)
    print("测试场景 1: 正常训练流程")
    print("=" * 80)
    print("配置: configs/test_logging/")
    print("预期结果:")
    print("  - Server 日志: logs/test_unified_logging/run_*/server/runtime.log")
    print("  - Client 日志: logs/test_unified_logging/run_*/clients/client_*/runtime.log")
    print("  - MLflow 追踪: mlruns/ (嵌套 runs)")
    print()

    # 创建 FederatedLearning 实例
    fl = FederatedLearning("configs/test_logging/")

    try:
        # 初始化并运行
        await fl.initialize()
        result = await fl.run()

        # 打印结果
        print("\n" + "=" * 80)
        print("✓ 训练完成")
        print("=" * 80)
        print(f"训练轮数: {result.completed_rounds}/{result.total_rounds}")
        print(f"最终准确率: {result.final_accuracy:.4f}")
        print(f"最终损失: {result.final_loss:.4f}")
        print(f"总耗时: {result.total_time:.2f}s")
        print(f"终止原因: {result.termination_reason}")

        print("\n检查日志文件...")
        import os
        log_base = "logs/test_unified_logging"
        if os.path.exists(log_base):
            # 找到最新的 run 目录
            runs = [d for d in os.listdir(log_base) if d.startswith("run_")]
            if runs:
                latest_run = sorted(runs)[-1]
                run_dir = os.path.join(log_base, latest_run)

                server_log = os.path.join(run_dir, "server", "runtime.log")
                client0_log = os.path.join(run_dir, "clients", "client_0", "runtime.log")
                client1_log = os.path.join(run_dir, "clients", "client_1", "runtime.log")

                print(f"\n日志目录: {run_dir}")
                print(f"  Server 日志: {'✓' if os.path.exists(server_log) else '✗'}")
                print(f"  Client 0 日志: {'✓' if os.path.exists(client0_log) else '✗'}")
                print(f"  Client 1 日志: {'✓' if os.path.exists(client1_log) else '✗'}")

        print("\n检查 MLflow 追踪...")
        if os.path.exists("mlruns"):
            print("  MLflow 数据目录: ✓")
            print("  查看实验: mlflow ui")

        return True

    except Exception as e:
        print(f"\n✗ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # 清理资源
        await fl.cleanup()


async def test_exception_flow():
    """测试异常处理流程"""
    print("=" * 80)
    print("测试场景 2: 异常处理流程")
    print("=" * 80)
    print("预期结果:")
    print("  - 即使训练失败，日志仍会收集")
    print("  - Tracker 状态为 'failed'")
    print()

    # 创建 FederatedLearning 实例
    fl = FederatedLearning("configs/test_logging/")

    try:
        await fl.initialize()

        # 模拟训练中途异常
        print("\n模拟训练异常...")
        raise RuntimeError("Simulated training failure for testing")

    except RuntimeError as e:
        print(f"\n✓ 捕获异常: {e}")
        print("检查日志是否正确保存...")

        import os
        log_base = "logs/test_unified_logging"
        if os.path.exists(log_base):
            runs = [d for d in os.listdir(log_base) if d.startswith("run_")]
            if runs:
                latest_run = sorted(runs)[-1]
                print(f"日志已保存到: {log_base}/{latest_run}")

        return True

    finally:
        await fl.cleanup()


async def main(test_exception: bool = False):
    """主函数"""
    if test_exception:
        success = await test_exception_flow()
    else:
        success = await test_normal_flow()

    if success:
        print("\n" + "=" * 80)
        print("✓ 测试完成")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("✗ 测试失败")
        print("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='测试新统一日志系统')
    parser.add_argument('--test-exception', action='store_true',
                       help='测试异常处理流程')
    args = parser.parse_args()

    asyncio.run(main(test_exception=args.test_exception))
