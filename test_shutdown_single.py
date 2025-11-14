"""
测试单个实验中的 SHUTDOWN 消息发送
"""
import asyncio
import sys

# 设置详细日志
import os
os.environ['FEDCL_LOG_LEVEL'] = 'DEBUG'

from fedcl import FederatedLearning
from fedcl.utils.auto_logger import setup_auto_logging

async def test_shutdown_in_single_experiment():
    """测试在单个实验中 SHUTDOWN 消息是否能够正常发送"""

    print("=" * 80)
    print("测试单个实验中的 SHUTDOWN 消息机制")
    print("=" * 80)

    # 禁用MLflow以避免干扰
    os.environ.pop('FEDCL_RECORDER_BACKEND', None)

    # 初始化FL系统
    fl = FederatedLearning('configs/distributed/experiments/table3/')

    try:
        # 初始化
        print("\n[步骤 1] 初始化 FL 系统...")
        await fl.initialize()
        print("✓ FL 系统初始化完成")

        # 等待客户端注册
        print("\n[步骤 2] 等待客户端注册...")
        await asyncio.sleep(2)

        # 检查客户端是否注册成功
        if fl.server and fl.server.trainer:
            available_clients = fl.server.trainer.get_available_clients()
            print(f"✓ Trainer 可用客户端数: {len(available_clients)}")
            print(f"  客户端列表: {available_clients}")

            if len(available_clients) == 0:
                print("\n❌ 错误：没有可用客户端，无法测试 SHUTDOWN 机制")
                return False

            # 运行1轮训练
            print(f"\n[步骤 3] 运行 1 轮训练...")
            result = await fl.server.trainer.run_training(max_rounds=1)
            if result:
                print(f"✓ 训练完成，准确率: {result.final_accuracy:.4f}")
            else:
                print("✗ 训练失败")
                return False

            # 再次检查可用客户端
            available_clients = fl.server.trainer.get_available_clients()
            print(f"\n[步骤 4] 训练后检查可用客户端...")
            print(f"✓ 可用客户端数: {len(available_clients)}")
            print(f"  客户端列表: {available_clients}")

            # 测试 SHUTDOWN 消息
            print(f"\n[步骤 5] 测试 SHUTDOWN 消息广播...")
            print("  调用 server.stop_server()...")

            # 监控日志：应该看到 "Broadcasting SHUTDOWN" 消息
            await fl.server.stop_server()

            print("\n[步骤 6] 等待客户端接收 SHUTDOWN 消息...")
            await asyncio.sleep(1)

            print("\n✓ stop_server() 调用完成")

        else:
            print("❌ Server 或 Trainer 未初始化")
            return False

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 清理
        print("\n[步骤 7] 清理资源...")
        await fl.cleanup(force_clear_global_state=True)
        print("✓ 清理完成")

    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)
    print("\n请查看日志中的以下内容：")
    print("  1. 'Broadcasting SHUTDOWN to N clients...' - 确认 SHUTDOWN 广播")
    print("  2. '[client_X] Received SHUTDOWN from...' - 确认客户端接收")
    print("  3. '✓ SHUTDOWN broadcast completed' - 确认广播完成")
    print("=" * 80)

    return True

if __name__ == "__main__":
    success = asyncio.run(test_shutdown_in_single_experiment())
    sys.exit(0 if success else 1)
