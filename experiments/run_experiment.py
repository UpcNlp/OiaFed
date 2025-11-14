"""
联邦学习实验运行脚本
experiments/run_experiment.py

功能：
- 基于配置文件启动联邦学习实验
- 自动记录实验结果到 Sacred
- 支持三种通信模式（Memory/Process/Network）

使用方式：
    # 指定配置文件
    python experiments/run_experiment.py config.yaml

    # 指定配置文件和实验名称
    python experiments/run_experiment.py config.yaml --exp_name my_experiment

    # 使用配置继承
    python experiments/run_experiment.py experiments/configs/experiments/fedavg_mnist.yaml
"""

import sys
import asyncio
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments.sacred_wrapper import SacredRecorder
from experiments.sacred_callbacks import create_sacred_callbacks
from experiments.utils.experiment_id import generate_experiment_id
from experiments.utils.config_loader import load_config_with_inheritance
from fedcl import FederatedLearning


async def run_federated_experiment(config_path: str, exp_name: str = None):
    """
    运行联邦学习实验并使用 Sacred 记录

    Args:
        config_path: 配置文件路径
        exp_name: 自定义实验名称（可选）
    """
    print("=" * 80)
    print("Starting Federated Learning Experiment")
    print("=" * 80)

    # 1. 加载配置（支持继承）
    print(f"\n[1/6] Loading configuration from: {config_path}")
    try:
        config = load_config_with_inheritance(config_path)
        print(f"✓ Configuration loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")
        return

    # 2. 生成实验ID
    if exp_name is None:
        exp_name = generate_experiment_id(config)

    print(f"\n[2/6] Experiment ID: {exp_name}")

    # 3. 提取节点信息
    comm_config = config.get('communication', {})
    role = comm_config.get('role', 'unknown')
    node_id = comm_config.get('node_id', 'unknown')

    print(f"      Role: {role}")
    print(f"      Node ID: {node_id}")
    print(f"      Mode: {comm_config.get('mode', 'unknown')}")

    # 4. 初始化 Sacred 记录器
    print(f"\n[3/6] Initializing Sacred recorder...")
    try:
        recorder = SacredRecorder.initialize(
            experiment_name=exp_name,
            role=role,
            node_id=node_id
        )
        recorder.start_run(config)
        print(f"✓ Sacred recorder initialized")
        print(f"      Results will be saved to: experiments/results/{exp_name}/{role}_{node_id}/")
    except Exception as e:
        print(f"✗ Failed to initialize Sacred: {e}")
        return

    # 5. 注册回调到 Trainer/Learner
    print(f"\n[4/6] Creating federated learning system...")
    try:
        fl = FederatedLearning(config, auto_setup_logging=True)
        await fl.initialize()

        # 创建 Sacred 回调函数
        callbacks = create_sacred_callbacks(recorder)

        # 如果是 Server，注册 Trainer 回调
        if role == 'server' and fl.server and fl.server.trainer:
            trainer = fl.server.trainer
            trainer.add_callback('after_round', callbacks['round_callback'])
            trainer.add_callback('after_evaluation', callbacks['eval_callback'])
            print(f"✓ Server trainer callbacks registered")

        # 注册 Client Learner 回调（如果有）
        if role == 'client' or fl.clients:
            for client in fl.clients:
                if hasattr(client, 'learner') and client.learner:
                    learner = client.learner
                    learner.add_callback('after_train', callbacks['client_train_callback'])
                    learner.add_callback('after_evaluate', callbacks['client_eval_callback'])
            print(f"✓ Client learner callbacks registered")

        print(f"✓ Federated learning system created")

    except Exception as e:
        print(f"✗ Failed to create system: {e}")
        recorder.finish(status="FAILED")
        return

    # 6. 运行训练
    print(f"\n[5/6] Running federated learning...")
    print("-" * 80)

    try:
        result = await fl.run()

        if result:
            # 记录最终结果
            recorder.log_info("final_accuracy", result.final_accuracy)
            recorder.log_info("final_loss", result.final_loss)
            recorder.log_info("completed_rounds", result.completed_rounds)
            recorder.log_info("total_rounds", result.total_rounds)
            recorder.log_info("total_time", result.total_time)
            recorder.log_info("termination_reason", result.termination_reason)

            print("-" * 80)
            print(f"\n✓ Training completed successfully!")
            print(f"      Final Accuracy: {result.final_accuracy:.4f}")
            print(f"      Final Loss: {result.final_loss:.4f}")
            print(f"      Completed Rounds: {result.completed_rounds}/{result.total_rounds}")
            print(f"      Total Time: {result.total_time:.2f}s")

            recorder.finish(status="COMPLETED")
        else:
            print(f"\n✓ System is running (waiting for external control)")
            recorder.finish(status="RUNNING")

    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()

        recorder.log_info("error_message", str(e))
        recorder.finish(status="FAILED")

    finally:
        # 7. 清理资源
        print(f"\n[6/6] Cleaning up...")
        await fl.cleanup()
        print(f"✓ Resources cleaned up")

    print("\n" + "=" * 80)
    print(f"Experiment completed: {exp_name}")
    print(f"Results saved to: experiments/results/{exp_name}/{role}_{node_id}/")
    print("=" * 80)


def main():
    """主函数：解析命令行参数并运行实验"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run federated learning experiment with Sacred logging"
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to configuration file (YAML)"
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Custom experiment name (default: auto-generated)"
    )

    args = parser.parse_args()

    # 检查配置文件是否存在
    if not Path(args.config).exists():
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)

    # 运行实验
    try:
        asyncio.run(run_federated_experiment(args.config, args.exp_name))
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
