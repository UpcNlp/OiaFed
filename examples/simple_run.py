"""
简化的联邦学习启动脚本 - 直接使用配置文件路径
examples/simple_run.py

使用方法:
    # 方式1: 单个配置文件
    python examples/simple_run.py --config configs/full/mnist_experiment.yaml

    # 方式2: 配置文件夹（自动加载所有.yaml文件）
    python examples/simple_run.py --config configs/distributed/

    # 方式3: 多个配置文件
    python examples/simple_run.py --config configs/server.yaml configs/client1.yaml configs/client2.yaml
"""
import asyncio
import sys
import argparse
from pathlib import Path

# 添加项目路径
root = Path(__file__).parent.parent
sys.path.insert(0, str(root))

from fedcl.federated_learning import FederatedLearning

# 导入组件以触发装饰器注册
from fedcl.methods.models.mnist_cnn import MNISTCNNModel
from fedcl.methods.learners.mnist_learner import MNISTLearner
from fedcl.methods.learners.generic import GenericLearner
from fedcl.methods.trainers.fedavg_mnist import FedAvgMNISTTrainer
from fedcl.methods.trainers.generic import GenericTrainer
from fedcl.methods.datasets.mnist import MNISTFederatedDataset


async def main(config_path):
    """主函数"""
    print("=" * 80)
    print("简化的联邦学习启动脚本")
    print("=" * 80)

    # 创建FederatedLearning实例（支持配置继承）
    fl = FederatedLearning(config_path)

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
        print(f"\n[Error] 训练失败: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 清理资源
        await fl.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='简化的联邦学习启动脚本')
    parser.add_argument('--config', type=str, nargs='+', required=True,
                       help='配置文件路径或文件夹路径（支持多个文件）')
    args = parser.parse_args()

    # 如果只有一个配置，直接传字符串；多个则传列表
    config = args.config[0] if len(args.config) == 1 else args.config

    asyncio.run(main(config))
