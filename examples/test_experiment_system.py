"""
测试实验管理系统
examples/test_experiment_system.py

使用现有配置测试 Sacred 实验记录功能
"""

import sys
import asyncio
from pathlib import Path

# 添加项目路径
root = Path(__file__).parent.parent
sys.path.insert(0, str(root))

from experiments.sacred_wrapper import SacredRecorder
from experiments.sacred_callbacks import create_sacred_callbacks
from experiments.utils.experiment_id import generate_experiment_id
from experiments.utils.config_loader import load_config_with_inheritance
from fedcl import FederatedLearning


async def test_experiment_with_sacred():
    """
    测试：使用现有配置运行实验并记录到 Sacred
    """
    print("=" * 80)
    print("测试实验管理系统")
    print("=" * 80)

    # 使用现有的配置目录
    config_dir = Path("configs/distributed/experiments/iid")

    if not config_dir.exists():
        print(f"错误：配置目录不存在: {config_dir}")
        print("请确保在项目根目录运行此脚本")
        return

    print(f"\n使用配置目录: {config_dir}")
    print(f"包含文件:")
    for f in sorted(config_dir.glob("*.yaml")):
        print(f"  - {f.name}")

    # 生成实验名称
    exp_name = f"test_sacred_{Path().absolute().stat().st_mtime_ns % 1000000}"
    print(f"\n实验名称: {exp_name}")

    # 创建 FederatedLearning 实例（使用配置目录）
    print(f"\n[1/4] 创建联邦学习系统...")
    fl = FederatedLearning(str(config_dir), auto_setup_logging=True)

    try:
        # 初始化系统
        print(f"\n[2/4] 初始化系统...")
        await fl.initialize()

        # 提取节点信息并初始化 Sacred
        print(f"\n[3/4] 初始化 Sacred 记录器...")

        # 为 Server 初始化 Sacred
        if fl.servers:
            server = fl.servers[0]
            server_config = server.comm_config

            # 初始化 Sacred recorder
            recorder = SacredRecorder.initialize(
                experiment_name=exp_name,
                role="server",
                node_id=server.server_id,
                base_dir="experiments/results"
            )

            # 构建配置字典用于记录
            config_dict = {
                "mode": server_config.mode,
                "role": server_config.role,
                "node_id": server.server_id
            }
            recorder.start_run(config_dict)

            # 创建并注册回调
            callbacks = create_sacred_callbacks(recorder)
            server.trainer.add_callback('after_round', callbacks['round_callback'])
            server.trainer.add_callback('after_evaluation', callbacks['eval_callback'])

            print(f"  ✓ Server Sacred recorder 已初始化")
            print(f"    节点ID: {server.server_id}")
            print(f"    结果将保存到: experiments/results/{exp_name}/server_{server.server_id}/")

        # 为每个 Client 初始化 Sacred
        if fl.clients:
            for i, client in enumerate(fl.clients):
                # 为每个客户端创建独立的 recorder（重置单例）
                SacredRecorder.reset()

                client_recorder = SacredRecorder.initialize(
                    experiment_name=exp_name,
                    role="client",
                    node_id=client.client_id,
                    base_dir="experiments/results"
                )

                client_config = {
                    "mode": client.comm_config.mode,
                    "role": client.comm_config.role,
                    "node_id": client.client_id
                }
                client_recorder.start_run(client_config)

                # 注册客户端回调
                client_callbacks = create_sacred_callbacks(client_recorder)
                if hasattr(client, 'learner') and client.learner:
                    client.learner.add_callback('after_train', client_callbacks['client_train_callback'])

                print(f"  ✓ Client {i} Sacred recorder 已初始化")
                print(f"    节点ID: {client.client_id}")

        # 运行训练
        print(f"\n[4/4] 运行联邦学习训练...")
        print("-" * 80)

        result = await fl.run()

        print("-" * 80)

        if result:
            # 记录最终结果
            if fl.servers:
                recorder.log_info("final_accuracy", result.final_accuracy)
                recorder.log_info("final_loss", result.final_loss)
                recorder.log_info("completed_rounds", result.completed_rounds)
                recorder.log_info("total_time", result.total_time)
                recorder.finish(status="COMPLETED")

            print(f"\n✓ 训练完成!")
            print(f"  最终准确率: {result.final_accuracy:.4f}")
            print(f"  最终损失: {result.final_loss:.4f}")
            print(f"  完成轮次: {result.completed_rounds}/{result.total_rounds}")
            print(f"  总时间: {result.total_time:.2f}s")

            print(f"\n实验结果已保存到:")
            print(f"  experiments/results/{exp_name}/")

            # 列出生成的文件
            result_dir = Path(f"experiments/results/{exp_name}")
            if result_dir.exists():
                print(f"\n生成的结果目录:")
                for node_dir in sorted(result_dir.iterdir()):
                    if node_dir.is_dir():
                        print(f"  - {node_dir.name}/")
                        run_dir = node_dir / "1"
                        if run_dir.exists():
                            for f in sorted(run_dir.glob("*.json")):
                                print(f"    - {f.name}")

        else:
            print(f"\n⚠ 系统运行中（无训练结果）")

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()

        if fl.servers:
            recorder.log_info("error", str(e))
            recorder.finish(status="FAILED")

    finally:
        print(f"\n清理资源...")
        await fl.cleanup()

    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)

    # 测试结果收集工具
    print(f"\n提示：您可以使用以下命令查看结果:")
    print(f"  python experiments/collect_results.py {exp_name}")


if __name__ == "__main__":
    asyncio.run(test_experiment_with_sacred())
