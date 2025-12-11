"""
统一日志系统使用示例
examples/demo_unified_logging.py

展示如何使用统一配置管理 Loguru 和 Experiment Loggers
"""

import asyncio
import sys
from pathlib import Path

# 添加项目路径
root = Path(__file__).parent.parent
sys.path.insert(0, str(root))

from fedcl.federated_learning import FederatedLearning


async def demo_unified_logging():
    """演示：统一日志配置"""

    print("=" * 80)
    print("统一日志系统演示")
    print("=" * 80)
    print()
    print("📋 配置文件说明:")
    print("   YAML 文件中包含两个部分:")
    print("   1. logging:    - Loguru 运行时日志配置")
    print("   2. experiment: - Experiment Logger 配置")
    print()
    print("🔗 自动关联:")
    print("   - Loguru 日志文件会自动作为 artifacts 上传到 Experiment Logger")
    print("   - 一次配置，管理所有日志")
    print()
    print("=" * 80)
    print()

    # ===== 只需要这2行代码！=====
    fl = FederatedLearning("configs/examples/unified_logging_example.yaml")
    result = await fl.run()
    # ================================

    print()
    print("=" * 80)
    print("✅ 完成！")
    print("=" * 80)
    print()
    print("📊 查看结果:")
    print()
    print("1. Loguru 运行时日志:")
    print("   ls logs/exp_*/")
    print("   ├── comm/     - 通信日志")
    print("   ├── train/    - 训练日志")
    print("   └── sys/      - 系统日志")
    print()
    print("2. Experiment Logger 结果:")
    print("   ls experiments/results/unified_logging_demo/")
    print("   ├── run_*.json         - JSON 记录（如果启用）")
    print("   └── artifacts/         - 包含收集的 Loguru 日志")
    print()
    print("3. MLflow UI（如果启用 mlflow backend）:")
    print("   mlflow ui --backend-store-uri experiments/mlruns")
    print("   访问: http://localhost:5000")
    print("   可以看到 Loguru 日志文件在 Artifacts 标签下")
    print()
    print("💡 优势:")
    print("   ✓ 一个配置文件管理所有日志")
    print("   ✓ Loguru 日志自动上传到 MLflow/JSON")
    print("   ✓ 方便对比不同实验的完整日志")
    print("   ✓ 调试和结果分析一体化")
    print()

    await fl.cleanup()


async def demo_custom_config():
    """演示：通过代码自定义配置"""

    print("=" * 80)
    print("自定义配置演示")
    print("=" * 80)
    print()

    # 创建配置对象
    from fedcl.config.logging_config import UnifiedLoggingConfig, LoguruConfig, ExperimentLoggerConfig

    config = UnifiedLoggingConfig(
        loguru=LoguruConfig(
            console_enabled=False,  # 关闭控制台输出
            level="WARNING"
        ),
        experiment=ExperimentLoggerConfig(
            enabled=True,
            name="custom_config_demo",
            backends=["json", "mlflow"],
            collect_loguru_logs=True
        )
    )

    print("📝 自定义配置:")
    print(f"   Loguru 控制台: {config.loguru.console_enabled}")
    print(f"   Experiment 后端: {config.experiment.backends}")
    print(f"   收集 Loguru 日志: {config.experiment.collect_loguru_logs}")
    print()

    # 注意：这里只是演示配置对象的创建
    # 实际使用时需要通过 FederatedLearning 的配置文件传递

    print("✅ 配置对象创建成功")
    print()


async def demo_multi_backend():
    """演示：同时使用多个后端"""

    print("=" * 80)
    print("多后端记录演示")
    print("=" * 80)
    print()
    print("📊 启用的后端:")
    print("   - JSONLogger    → experiments/results/")
    print("   - MLflowLogger  → MLflow 数据库")
    print()
    print("💡 每个后端都会自动收集 Loguru 日志文件")
    print()

    # 使用多后端配置的实验
    # （需要先创建对应的配置文件）
    print("✓ 可以在配置文件中指定多个 backends")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='统一日志系统演示')
    parser.add_argument('--mode',
                       choices=['unified', 'custom', 'multi'],
                       default='unified',
                       help='演示模式')

    args = parser.parse_args()

    if args.mode == 'unified':
        asyncio.run(demo_unified_logging())
    elif args.mode == 'custom':
        asyncio.run(demo_custom_config())
    else:
        asyncio.run(demo_multi_backend())
