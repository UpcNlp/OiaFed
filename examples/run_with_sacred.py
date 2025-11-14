"""
带 Sacred 记录的联邦学习启动脚本
examples/run_with_sacred.py

在现有运行脚本基础上集成 Sacred 实验记录

使用方法:
    python examples/run_with_sacred.py --config configs/distributed/experiments/iid/ --exp_name test_exp_001
"""
import asyncio
import sys
import argparse
from pathlib import Path

# 添加项目路径
root = Path(__file__).parent.parent
sys.path.insert(0, str(root))

# 先导入实验管理工具
from experiments.sacred_wrapper import SacredRecorder
from experiments.sacred_callbacks import create_sacred_callbacks
from experiments.utils.experiment_id import generate_experiment_id

# 然后导入 FederatedLearning
from fedcl.federated_learning import FederatedLearning


async def main(config_path: str, exp_name: str = None):
    """主函数"""
    config_path = Path(config_path)

    print("=" * 80)
    print("配置驱动的联邦学习训练（带 Sacred 记录）")
    print("=" * 80)
    print(f"\n配置: {config_path}")

    # 判断配置类型
    if config_path.is_dir():
        print("模式: 分布式配置（加载文件夹）")
    else:
        print(f"模式: 单一配置文件")

    # 生成实验名称
    if exp_name is None:
        import time
        exp_name = f"exp_{int(time.time())}"

    print(f"实验名称: {exp_name}")
    print()

    # 创建FederatedLearning实例
    print("[1/3] 创建联邦学习系统...")
    fl = FederatedLearning(str(config_path))

    try:
        # 初始化
        print("[2/3] 初始化系统...")
        await fl.initialize()

        # 集成 Sacred 记录器
        print("[3/3] 设置实验记录...")

        # 为 Server 设置 Sacred
        if fl.servers:
            server = fl.servers[0]

            # 初始化 recorder
            recorder = SacredRecorder.initialize(
                experiment_name=exp_name,
                role="server",
                node_id=server.server_id,
                base_dir="experiments/results"
            )

            config_dict = {
                "mode": server.comm_config.mode,
                "role": server.comm_config.role,
                "node_id": server.server_id
            }
            recorder.start_run(config_dict)

            # 注册回调
            callbacks = create_sacred_callbacks(recorder)
            server.trainer.add_callback('after_round', callbacks['round_callback'])
            server.trainer.add_callback('after_evaluation', callbacks['eval_callback'])

            print(f"  ✓ Server 实验记录已启用")
            print(f"    结果将保存到: experiments/results/{exp_name}/server_{server.server_id}/")

        # 为 Clients 设置 Sacred
        if fl.clients:
            for client in fl.clients:
                # 每个客户端独立的 recorder
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

                # 注册回调
                client_callbacks = create_sacred_callbacks(client_recorder)
                if hasattr(client, 'learner') and client.learner:
                    client.learner.add_callback('after_train', client_callbacks['client_train_callback'])

            print(f"  ✓ {len(fl.clients)} 个 Client 实验记录已启用")

        print("\n" + "-" * 80)
        print("开始训练")
        print("-" * 80 + "\n")

        # 运行训练
        result = await fl.run()

        # 记录最终结果
        if result and fl.servers:
            recorder.log_info("final_accuracy", result.final_accuracy)
            recorder.log_info("final_loss", result.final_loss)
            recorder.log_info("completed_rounds", result.completed_rounds)
            recorder.log_info("total_rounds", result.total_rounds)
            recorder.log_info("total_time", result.total_time)
            recorder.log_info("termination_reason", result.termination_reason)
            recorder.finish(status="COMPLETED")

        # 打印结果
        print("\n" + "=" * 80)
        print("训练完成")
        print("=" * 80)

        if result:
            print(f"✓ 训练轮数: {result.completed_rounds}/{result.total_rounds}")
            print(f"✓ 最终准确率: {result.final_accuracy:.4f}")
            print(f"✓ 最终损失: {result.final_loss:.4f}")
            print(f"✓ 总耗时: {result.total_time:.2f}s")
            print(f"✓ 终止原因: {result.termination_reason}")

            print(f"\n实验结果已保存到: experiments/results/{exp_name}/")
            print(f"\n查看结果：")
            print(f"  python experiments/collect_results.py {exp_name}")

    except Exception as e:
        print(f"\n[Error] 运行失败: {e}")
        import traceback
        traceback.print_exc()

        if fl.servers:
            recorder.log_info("error", str(e))
            recorder.finish(status="FAILED")

    finally:
        # 清理资源
        await fl.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='配置驱动的联邦学习训练（带 Sacred 记录）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

1. 使用配置文件夹:
   python examples/run_with_sacred.py --config configs/distributed/experiments/iid/ --exp_name my_exp

2. 使用单一配置文件:
   python examples/run_with_sacred.py --config configs/some_config.yaml

3. 自动生成实验名称:
   python examples/run_with_sacred.py --config configs/distributed/experiments/iid/

查看结果:
   python experiments/collect_results.py {experiment_name}
        """
    )
    parser.add_argument('--config', type=str, required=True,
                       help='配置文件路径或配置文件夹路径')
    parser.add_argument('--exp_name', type=str, default=None,
                       help='实验名称（可选，默认自动生成）')

    args = parser.parse_args()

    # 检查配置文件/文件夹是否存在
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[Error] 配置路径不存在: {args.config}")
        sys.exit(1)

    asyncio.run(main(args.config, args.exp_name))
