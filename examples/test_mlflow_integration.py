"""
测试 MLflow 集成
examples/test_mlflow_integration.py

快速验证 MLflow Recorder 是否正常工作
"""

import sys
import os
from pathlib import Path

# 设置 MLflow 后端
os.environ['FEDCL_RECORDER_BACKEND'] = 'mlflow'

# 添加项目路径
root = Path(__file__).parent.parent
sys.path.insert(0, str(root))

from fedcl.experiment import Recorder

def test_basic_recorder():
    """测试基本的 Recorder 功能"""
    print("=" * 60)
    print("测试 MLflow Recorder 基本功能")
    print("=" * 60)

    # 初始化 recorder
    recorder = Recorder.initialize(
        experiment_name="test_mlflow_integration",
        role="server",
        node_id="server_0"
    )

    # 开始运行
    recorder.start_run({
        "mode": "memory",
        "algorithm": "fedavg",
        "learning_rate": 0.01
    })

    # 记录一些指标
    for round_num in range(1, 6):
        accuracy = 0.5 + round_num * 0.08
        loss = 1.0 - round_num * 0.15

        recorder.log_scalar("server/accuracy", accuracy, step=round_num)
        recorder.log_scalar("server/loss", loss, step=round_num)

        print(f"Round {round_num}: accuracy={accuracy:.4f}, loss={loss:.4f}")

    # 记录最终信息
    recorder.log_info("final_accuracy", 0.95)
    recorder.log_info("final_loss", 0.3)

    # 完成
    recorder.finish(status="COMPLETED")

    print("\n✓ 测试完成！")
    print("\n查看结果:")
    print("  cd /home/nlp/ct/projects/MOE-FedCL")
    print("  mlflow ui --backend-store-uri experiments/mlruns")
    print("  然后访问: http://localhost:5000")


def test_multiple_recorders():
    """测试多个 Recorder (模拟联邦学习场景)"""
    print("\n" + "=" * 60)
    print("测试多个 Recorder (Server + Client)")
    print("=" * 60)

    # Server recorder
    Recorder.reset()
    server_recorder = Recorder.initialize(
        experiment_name="test_federated_scenario",
        role="server",
        node_id="server_0"
    )
    server_recorder.start_run({"mode": "memory"})

    # Client recorders
    client_recorders = []
    for i in range(3):
        Recorder.reset()
        client_rec = Recorder.initialize(
            experiment_name="test_federated_scenario",
            role="client",
            node_id=f"client_{i}"
        )
        client_rec.start_run({"client_id": i, "data_size": 1000 * (i+1)})
        client_recorders.append(client_rec)

    # 模拟训练
    for round_num in range(1, 4):
        # Server metrics
        server_acc = 0.6 + round_num * 0.1
        server_recorder.log_scalar("server/avg_accuracy", server_acc, step=round_num)

        # Client metrics
        for i, client_rec in enumerate(client_recorders):
            client_acc = 0.5 + round_num * 0.1 + i * 0.02
            client_loss = 0.8 - round_num * 0.1
            client_rec.log_scalar(f"client_{i}/accuracy", client_acc, step=round_num)
            client_rec.log_scalar(f"client_{i}/loss", client_loss, step=round_num)

        print(f"Round {round_num}: server_acc={server_acc:.4f}")

    # 完成所有 recorders
    server_recorder.finish()
    for client_rec in client_recorders:
        client_rec.finish()

    print("\n✓ 测试完成！")
    print("现在你可以在 MLflow UI 中对比 server 和 clients 的指标")


if __name__ == "__main__":
    try:
        import mlflow
        print(f"✓ MLflow 版本: {mlflow.__version__}\n")

        # 运行测试
        test_basic_recorder()
        test_multiple_recorders()

        print("\n" + "=" * 60)
        print("所有测试通过！")
        print("=" * 60)

    except ImportError as e:
        print(f"✗ 错误: {e}")
        print("请安装 MLflow: pip install mlflow")
        sys.exit(1)
