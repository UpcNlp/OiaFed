"""
灵活的联邦学习启动脚本 - 支持不同部署模式
examples/run_flexible.py

支持三种使用方式：

1. Memory/Process模式 - 加载文件夹中所有配置，在同一进程创建所有节点:
   python examples/run_flexible.py --config configs/distributed/experiments/iid/

2. Network模式 - 服务器端:
   python examples/run_flexible.py --config configs/distributed/experiments/iid/server.yaml

3. Network模式 - 客户端:
   python examples/run_flexible.py --config configs/distributed/experiments/iid/client_0.yaml
"""
import asyncio
import sys
import argparse
from pathlib import Path

# 添加项目路径
root = Path(__file__).parent.parent
sys.path.insert(0, str(root))

# 导入 FederatedLearning，内置组件会自动注册
from fedcl.federated_learning import FederatedLearning


async def main(config_path: str):
    """主函数"""
    config_path = Path(config_path)

    print("=" * 80)
    print("灵活的联邦学习启动脚本")
    print("=" * 80)

    # 判断配置类型
    if config_path.is_dir():
        print(f"\n模式: Memory/Process (加载文件夹中所有配置)")
        print(f"配置文件夹: {config_path}")
        print(f"将创建: 1个Server + N个Clients (同一进程)")
    else:
        print(f"\n模式: Network (单节点部署)")
        print(f"配置文件: {config_path}")
        print(f"将创建: 单个节点 (Server 或 Client)")

    print("\n支持配置继承 (extends字段)")
    print()

    # 创建FederatedLearning实例（直接传配置路径）
    fl = FederatedLearning(str(config_path))

    try:
        # 初始化并运行
        await fl.initialize()
        result = await fl.run()

        # 打印结果
        print("\n" + "=" * 80)
        print("训练完成")
        print("=" * 80)
        print(f"✓ 训练轮数: {result.completed_rounds}/{result.total_rounds}")
        print(f"✓ 最终准确率: {result.final_accuracy:.4f}")
        print(f"✓ 最终损失: {result.final_loss:.4f}")
        print(f"✓ 总耗时: {result.total_time:.2f}s")
        print(f"✓ 终止原因: {result.termination_reason}")

    except Exception as e:
        print(f"\n[Error] 运行失败: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 清理资源
        await fl.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='灵活的联邦学习启动脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

1. Memory/Process模式 (同一进程运行所有节点):
   python examples/run_flexible.py --config configs/distributed/experiments/iid/

2. Network模式 - 服务器端:
   python examples/run_flexible.py --config configs/distributed/experiments/iid/server.yaml

3. Network模式 - 客户端0:
   python examples/run_flexible.py --config configs/distributed/experiments/iid/client_0.yaml

配置继承:
- 所有配置文件都支持 extends 字段继承基础配置
- 例如: extends: "../../base/server_base.yaml"
        """
    )
    parser.add_argument('--config', type=str, required=True,
                       help='配置文件路径或配置文件夹路径')
    args = parser.parse_args()

    asyncio.run(main(args.config))
