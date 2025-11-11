"""
直接使用配置文件路径的联邦学习启动脚本
examples/run_fl_direct.py

特点：
1. 直接传配置文件路径给FederatedLearning
2. 支持YAML配置继承
3. 无需手动构建配置对象

使用方法:
    python examples/run_fl_direct.py --config configs/unified/mnist_base.yaml
"""
import asyncio
import sys
import argparse
from pathlib import Path

# 添加项目路径
root = Path(__file__).parent.parent
sys.path.insert(0, str(root))

from fedcl.federated_learning import FederatedLearning
from fedcl.config.loader import ConfigLoader

# 导入组件以触发装饰器注册
from fedcl.methods.models.mnist_cnn import MNISTCNNModel
from fedcl.methods.learners.mnist_learner import MNISTLearner
from fedcl.methods.learners.generic import GenericLearner
from fedcl.methods.trainers.fedavg_mnist import FedAvgMNISTTrainer
from fedcl.methods.trainers.generic import GenericTrainer
from fedcl.methods.datasets.mnist import MNISTFederatedDataset


def convert_unified_config_to_node_configs(config_path: str):
    """
    将统一配置文件转换为节点配置列表

    统一配置格式：一个文件包含server和clients的所有配置
    节点配置格式：每个节点一个(CommunicationConfig, TrainingConfig)元组

    Args:
        config_path: 统一配置文件路径

    Returns:
        List[Tuple[CommunicationConfig, TrainingConfig]]
    """
    from fedcl.config import CommunicationConfig, TrainingConfig

    # 加载统一配置（支持继承）
    config_dict = ConfigLoader.load_with_inheritance(config_path)

    node_configs = []

    # 1. 创建Server配置
    server_comm_config = CommunicationConfig(
        mode=config_dict.get('communication', {}).get('mode', 'memory'),
        role='server',
        node_id=config_dict.get('server', {}).get('node_id', 'server_1')
    )

    server_train_config = TrainingConfig()
    server_train_config.max_rounds = config_dict.get('training', {}).get('max_rounds', 10)
    server_train_config.min_clients = config_dict.get('training', {}).get('min_clients', 2)
    server_train_config.trainer = config_dict.get('server', {}).get('trainer', {})
    # global_model优先从server配置获取，否则从顶层model获取
    server_train_config.global_model = config_dict.get('server', {}).get('global_model') or \
                                       config_dict.get('model', {})

    node_configs.append((server_comm_config, server_train_config))

    # 2. 创建Client配置
    num_clients = config_dict.get('clients', {}).get('num_clients', 3)

    for i in range(num_clients):
        client_comm_config = CommunicationConfig(
            mode=config_dict.get('communication', {}).get('mode', 'memory'),
            role='client',
            node_id=f'client_{i}'
        )

        client_train_config = TrainingConfig()
        client_train_config.learner = config_dict.get('clients', {}).get('learner', {})
        client_train_config.dataset = config_dict.get('clients', {}).get('dataset') or \
                                      config_dict.get('dataset', {})

        node_configs.append((client_comm_config, client_train_config))

    return node_configs


async def main(config_path: str):
    """主函数"""
    print("=" * 80)
    print("直接使用配置文件的联邦学习训练")
    print("=" * 80)
    print(f"\n配置文件: {config_path}")
    print("支持YAML配置继承 (extends字段)")
    print()

    # 转换统一配置为节点配置
    node_configs = convert_unified_config_to_node_configs(config_path)

    # 创建FederatedLearning实例（直接传配置对象列表）
    fl = FederatedLearning(node_configs)

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
    parser = argparse.ArgumentParser(description='直接使用配置文件的联邦学习训练')
    parser.add_argument('--config', type=str, required=True,
                       help='统一配置文件路径（支持YAML继承）')
    args = parser.parse_args()

    asyncio.run(main(args.config))
