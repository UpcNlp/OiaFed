"""
使用 FederatedLearning 统一入口类运行联邦学习示例
examples/unified_entry_demo.py

演示如何使用新版 FederatedLearning 类快速启动完整的联邦学习系统

新版特性：
- 基于配置文件的架构
- 每个配置文件必须指定 role（"server" 或 "client"）
- 支持从文件夹加载多个配置
- 自动创建和管理 Server/Client 实例
"""

import asyncio
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fedcl import FederatedLearning


# ============================================
# 示例1: 从配置文件夹加载（推荐方式）
# ============================================
async def example1_from_folder():
    """
    从配置文件夹加载所有节点配置

    要求：
    - 文件夹中至少有1个 server 配置和1个 client 配置
    - 每个配置文件必须指定 role 字段
    """
    print("\n" + "="*60)
    print("示例1: 从配置文件夹加载")
    print("="*60)

    # 假设你有一个配置文件夹，包含：
    # - server.yaml (role: server)
    # - client1.yaml (role: client)
    # - client2.yaml (role: client)
    config_folder = "configs"  # 修改为你的配置文件夹路径

    if not os.path.exists(config_folder):
        print(f"⚠️  配置文件夹不存在: {config_folder}")
        print("跳过此示例")
        return

    try:
        # 创建 FederatedLearning 实例
        fl = FederatedLearning(config_folder)

        # 初始化所有节点
        await fl.initialize()

        # 运行联邦学习训练
        result = await fl.run(max_rounds=5)

        if result:
            print("\n" + "="*60)
            print("训练结果:")
            print(f"  完成轮数: {result.completed_rounds}")
            print(f"  最终准确率: {result.final_accuracy:.4f}")
            print(f"  最终损失: {result.final_loss:.4f}")
            print(f"  总时间: {result.total_time:.2f}秒")
            print("="*60)

        # 清理资源
        await fl.cleanup()

    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

    print("✅ 示例1完成\n")


# ============================================
# 示例2: 从多个配置文件加载
# ============================================
async def example2_from_file_list():
    """
    指定多个配置文件路径
    """
    print("\n" + "="*60)
    print("示例2: 从多个配置文件加载")
    print("="*60)

    config_files = [
        "configs/server.yaml",
        "configs/client1.yaml",
        "configs/client2.yaml",
    ]

    # 检查文件是否存在
    missing_files = [f for f in config_files if not os.path.exists(f)]
    if missing_files:
        print(f"⚠️  以下配置文件不存在:")
        for f in missing_files:
            print(f"    - {f}")
        print("跳过此示例")
        return

    try:
        # 创建 FederatedLearning 实例
        fl = FederatedLearning(config_files)

        # 初始化所有节点
        await fl.initialize()

        # 运行联邦学习训练
        result = await fl.run(max_rounds=3)

        if result:
            print(f"\n✅ 训练完成，最终准确率: {result.final_accuracy:.4f}")

        # 清理资源
        await fl.cleanup()

    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

    print("✅ 示例2完成\n")


# ============================================
# 示例3: 使用上下文管理器（自动清理资源）
# ============================================
async def example3_context_manager():
    """
    使用 async with 自动管理资源生命周期
    """
    print("\n" + "="*60)
    print("示例3: 使用上下文管理器")
    print("="*60)

    config_folder = "configs"

    if not os.path.exists(config_folder):
        print(f"⚠️  配置文件夹不存在: {config_folder}")
        print("跳过此示例")
        return

    try:
        # 使用 async with 自动管理资源
        async with FederatedLearning(config_folder) as fl:
            # 运行训练
            result = await fl.run(max_rounds=3)

            if result:
                print(f"\n✅ 训练完成，最终准确率: {result.final_accuracy:.4f}")

        # 资源会自动清理

    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

    print("✅ 示例3完成\n")


# ============================================
# 示例4: 从单个配置文件加载（单节点）
# ============================================
async def example4_single_node():
    """
    加载单个节点配置（仅启动一个 Server 或 Client）

    适用场景：
    - 分布式部署时，每台机器只运行一个节点
    - 独立启动 Server 或 Client
    """
    print("\n" + "="*60)
    print("示例4: 单节点模式")
    print("="*60)

    config_file = "configs/server.yaml"

    if not os.path.exists(config_file):
        print(f"⚠️  配置文件不存在: {config_file}")
        print("跳过此示例")
        return

    try:
        async with FederatedLearning(config_file) as fl:
            print(f"节点已启动:")
            print(f"  - Servers: {len(fl.servers)}")
            print(f"  - Clients: {len(fl.clients)}")

            # 单节点模式不会自动运行训练
            # 通常用于分布式部署，等待其他节点连接
            print("\n保持运行中（按 Ctrl+C 停止）...")

            # 运行30秒后退出（实际使用时可以持续运行）
            await asyncio.sleep(30)
            print("示例结束")

    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

    print("✅ 示例4完成\n")


# ============================================
# 示例5: 查看系统状态
# ============================================
async def example5_system_status():
    """
    查看系统运行状态
    """
    print("\n" + "="*60)
    print("示例5: 查看系统状态")
    print("="*60)

    config_folder = "configs"

    if not os.path.exists(config_folder):
        print(f"⚠️  配置文件夹不存在: {config_folder}")
        print("跳过此示例")
        return

    try:
        fl = FederatedLearning(config_folder)
        await fl.initialize()

        # 获取系统状态
        status = fl.get_status()
        print(f"\n系统状态:")
        print(f"  节点总数: {status['num_servers'] + status['num_clients']}")
        print(f"    - Servers: {status['num_servers']}")
        print(f"    - Clients: {status['num_clients']}")
        print(f"  已初始化: {status['is_initialized']}")
        print(f"  运行中: {status['is_running']}")

        # 访问第一个 Server（如果有）
        if fl.server:
            server_status = fl.server.get_server_status()
            print(f"\nServer 状态:")
            print(f"  Server ID: {server_status['server_id']}")
            print(f"  模式: {server_status['mode']}")
            print(f"  可用客户端: {server_status['available_clients']}")

        await fl.cleanup()

    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

    print("✅ 示例5完成\n")


# ============================================
# 配置文件示例说明
# ============================================
def print_config_example():
    """打印配置文件示例"""
    print("\n" + "="*60)
    print("配置文件示例")
    print("="*60)

    print("\n📄 server.yaml:")
    print("""
# 服务端配置
role: server          # 必须指定！
mode: memory          # memory/process/network
node_id: demo_server

# Trainer 类
trainer:
  class_path: "examples.demo_trainer.DemoTrainer"

# 全局模型
global_model:
  weights: [0.1, 0.2, 0.3]

# 训练配置
training:
  max_rounds: 10
  min_clients: 2

# 通信配置（可选）
communication:
  heartbeat_interval: 30.0
""")

    print("\n📄 client.yaml:")
    print("""
# 客户端配置
role: client          # 必须指定！
mode: memory
node_id: demo_client_1

# Learner 类
learner:
  class_path: "examples.demo_learner.DemoLearner"

# 客户端配置（可选）
training:
  local_epochs: 5
  batch_size: 32
""")
    print("="*60)


# ============================================
# 主函数
# ============================================
async def main():
    """运行示例"""
    print("="*60)
    print("MOE-FedCL 统一入口使用示例（新版）")
    print("="*60)

    # 打印配置文件格式说明
    print_config_example()

    # 选择要运行的示例
    print("\n可用示例:")
    print("  1. 从配置文件夹加载（推荐）")
    print("  2. 从多个配置文件加载")
    print("  3. 使用上下文管理器")
    print("  4. 单节点模式")
    print("  5. 查看系统状态")

    # 运行示例（取消注释来运行）
    # await example1_from_folder()
    # await example2_from_file_list()
    # await example3_context_manager()
    # await example4_single_node()
    # await example5_system_status()

    print("\n提示: 请取消注释 main() 中的示例代码来运行")
    print("\n" + "="*60)
    print("示例说明完成")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
