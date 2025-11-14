"""
配置驱动的联邦学习启动脚本（支持实验记录）
examples/run_federated_learning.py

使用方法:
    python examples/run_federated_learning.py --config configs/mnist_true_generic.yaml
    python examples/run_federated_learning.py --config configs/distributed/experiments/iid/
    python examples/run_federated_learning.py --config configs/distributed/experiments/iid/ --sacred --exp_name my_exp
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


async def main(config_path: str, use_sacred: bool = False, exp_name: str = None):
    """主函数"""
    config_path = Path(config_path)

    print("=" * 80)
    print("配置驱动的联邦学习训练" + (" (带实验记录)" if use_sacred else ""))
    print("=" * 80)
    print(f"\n配置: {config_path}")

    # 判断配置类型
    if config_path.is_dir():
        print("模式: 分布式配置（加载文件夹）")
    else:
        print(f"模式: 单一配置文件")

    if use_sacred:
        if exp_name is None:
            import time
            exp_name = f"exp_{int(time.time())}"
        print(f"实验名称: {exp_name}")

    print()

    # 创建FederatedLearning实例（直接传配置路径）
    fl = FederatedLearning(str(config_path))

    try:
        # 初始化并运行
        await fl.initialize()

        # 设置实验记录（如果配置或命令行参数启用）
        server_recorder, client_recorders = None, []
        if use_sacred:
            exp_config = {
                'enabled': True,
                'name': exp_name,
                'base_dir': 'experiments/results'
            }
            server_recorder, client_recorders = fl.setup_experiment_recording(exp_config)

        result = await fl.run()

        # 如果启用实验记录，记录最终结果
        if server_recorder:
            server_recorder.log_info("final_accuracy", result.final_accuracy)
            server_recorder.log_info("final_loss", result.final_loss)
            server_recorder.log_info("completed_rounds", result.completed_rounds)
            server_recorder.log_info("total_time", result.total_time)
            server_recorder.finish(status="COMPLETED")

        for client_rec in client_recorders:
            client_rec.finish(status="COMPLETED")

        # 打印结果
        print("\n" + "=" * 80)
        print("训练完成")
        print("=" * 80)
        print(f"✓ 训练轮数: {result.completed_rounds}/{result.total_rounds}")
        print(f"✓ 最终准确率: {result.final_accuracy:.4f}")
        print(f"✓ 最终损失: {result.final_loss:.4f}")
        print(f"✓ 总耗时: {result.total_time:.2f}s")
        print(f"✓ 终止原因: {result.termination_reason}")

        if use_sacred:
            print(f"\n实验结果已保存到: experiments/results/{exp_name}/")
            print(f"查看结果: python experiments/collect_results.py {exp_name}")

    except Exception as e:
        print(f"\n[Error] 运行失败: {e}")
        import traceback
        traceback.print_exc()

        if server_recorder:
            server_recorder.log_info("error", str(e))
            server_recorder.finish(status="FAILED")
        for client_rec in client_recorders:
            client_rec.finish(status="FAILED")

    finally:
        # 清理资源
        await fl.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='配置驱动的联邦学习训练',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

1. 单一配置文件:
   python examples/run_federated_learning.py --config configs/mnist_true_generic.yaml

2. 分布式配置文件夹:
   python examples/run_federated_learning.py --config configs/distributed/experiments/iid/

3. 启用实验记录:
   python examples/run_federated_learning.py --config configs/distributed/experiments/iid/ --sacred --exp_name my_exp

配置继承:
- 所有配置文件都支持 extends 字段继承基础配置
- 例如: extends: "../../base/server_base.yaml"
        """
    )
    parser.add_argument('--config', type=str, required=True,
                       help='配置文件路径或配置文件夹路径')
    parser.add_argument('--sacred', action='store_true',
                       help='启用实验记录')
    parser.add_argument('--exp_name', type=str, default=None,
                       help='实验名称（启用实验记录时使用）')

    args = parser.parse_args()

    # 检查配置文件/文件夹是否存在
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[Error] 配置路径不存在: {args.config}")
        sys.exit(1)

    asyncio.run(main(args.config, use_sacred=args.sacred, exp_name=args.exp_name))
