"""
测试 MLflow 认证配置
test_mlflow_auth.py

快速验证 MLflow 服务器连接和认证是否正常工作
"""

import os
import sys
from pathlib import Path

# 设置环境（确保从.env加载）
root = Path(__file__).parent
sys.path.insert(0, str(root))

# 加载 .env 文件
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✓ 已从 .env 文件加载环境变量")
except ImportError:
    print("⚠ python-dotenv 未安装，尝试直接使用环境变量")
    print("  安装: pip install python-dotenv")

def test_mlflow_connection():
    """测试 MLflow 服务器连接和认证"""
    print("\n" + "=" * 60)
    print("测试 MLflow 服务器连接和认证")
    print("=" * 60)

    # 显示配置
    tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
    username = os.getenv('MLFLOW_TRACKING_USERNAME')
    password = os.getenv('MLFLOW_TRACKING_PASSWORD')

    print(f"\n配置信息:")
    print(f"  MLFLOW_TRACKING_URI: {tracking_uri}")
    print(f"  MLFLOW_TRACKING_USERNAME: {username}")
    print(f"  MLFLOW_TRACKING_PASSWORD: {'*' * len(password) if password else None}")

    if not tracking_uri:
        print("\n✗ 错误: MLFLOW_TRACKING_URI 未设置")
        return False

    if not username or not password:
        print("\n⚠ 警告: MLflow 认证信息未设置（使用无认证模式）")

    # 尝试连接
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        print(f"\n✓ MLflow 版本: {mlflow.__version__}")

        # 设置 tracking URI
        mlflow.set_tracking_uri(tracking_uri)
        print(f"✓ 已设置 tracking URI: {tracking_uri}")

        # 创建客户端（会触发认证）
        client = MlflowClient()
        print("✓ 已创建 MlflowClient")

        # 尝试列出实验（测试连接和认证）
        print("\n尝试连接到 MLflow 服务器...")
        experiments = client.search_experiments()
        print(f"✓ 成功连接！找到 {len(experiments)} 个实验")

        # 显示现有实验
        if experiments:
            print("\n现有实验:")
            for exp in experiments[:5]:  # 只显示前5个
                print(f"  - {exp.name} (ID: {exp.experiment_id})")

        return True

    except Exception as e:
        print(f"\n✗ 连接失败: {e}")
        print("\n可能的原因:")
        print("  1. MLflow 服务器未启动")
        print("  2. 认证信息不正确")
        print("  3. 网络连接问题")
        return False


def test_mlflow_recorder():
    """测试 MLflowRecorder 是否能正常工作"""
    print("\n" + "=" * 60)
    print("测试 MLflowRecorder")
    print("=" * 60)

    try:
        # 设置使用 MLflow 后端
        os.environ['FEDCL_RECORDER_BACKEND'] = 'mlflow'

        from fedcl.experiment import MLflowRecorder

        print("\n创建 MLflowRecorder 实例...")
        recorder = MLflowRecorder(
            experiment_name="test_auth",
            role="server",
            node_id="test_server"
        )

        print("✓ MLflowRecorder 创建成功")
        print(f"  Run ID: {recorder.run_id}")

        # 记录一些测试数据
        print("\n记录测试数据...")
        recorder.log_params({"test_param": "value"})
        recorder.log_metrics({"test_metric": 0.95}, step=1)
        recorder.set_tag("test_tag", "test_value")

        print("✓ 数据记录成功")

        # 结束 run
        recorder.end_run(status="COMPLETED")
        print("✓ Run 已结束")

        return True

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("MLflow 认证配置测试")
    print("=" * 60)

    # 测试1: 连接测试
    connection_ok = test_mlflow_connection()

    # 测试2: Recorder 测试
    if connection_ok:
        recorder_ok = test_mlflow_recorder()
    else:
        print("\n跳过 Recorder 测试（连接失败）")
        recorder_ok = False

    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"  连接测试: {'✓ 通过' if connection_ok else '✗ 失败'}")
    print(f"  Recorder测试: {'✓ 通过' if recorder_ok else '✗ 失败'}")

    if connection_ok and recorder_ok:
        print("\n✅ 所有测试通过！MLflow 集成配置正确")
        print("\n查看结果:")
        print(f"  访问: {os.getenv('MLFLOW_TRACKING_URI')}")
        sys.exit(0)
    else:
        print("\n❌ 测试失败，请检查配置")
        sys.exit(1)
