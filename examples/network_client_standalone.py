"""
Network 模式测试 - 客户端脚本（跨机器版本）
运行在：Linux 服务器 (192.168.31.75 或 192.168.31.166)

这个脚本只启动客户端，连接到远程服务端。
"""

import asyncio
import sys
import os
import argparse
from typing import Dict, Any

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fedcl.utils.auto_logger import setup_auto_logging
from fedcl.types import TrainingResult, EvaluationResult, ModelData
from fedcl.config import load_client_config
from fedcl.federation.client import FederationClient
from fedcl.learner.base_learner import BaseLearner


class SimpleLearner(BaseLearner):
    """简单的测试学习器"""

    def __init__(self, client_id: str, config: Dict[str, Any] = None, logger=None):
        super().__init__(client_id, config or {}, logger)
        self.local_samples = 1000
        self._local_model: ModelData = {"weights": 1.0}
        print(f"[{self.client_id}] 学习器已初始化")

    async def train(self, training_params: Dict[str, Any]) -> TrainingResult:
        """本地训练"""
        global_model = training_params.get("global_model", {})
        round_num = training_params.get("round", 0)

        print(f"\n[{self.client_id}] 执行本地训练 - 第 {round_num} 轮")
        print(f"  全局模型: {global_model}")

        # 模拟训练过程
        await asyncio.sleep(2)

        # 更新本地模型
        current_weights = global_model.get("weights", 1.0)
        self._local_model = {
            "weights": current_weights + 0.1,  # 模拟训练更新
            "client_id": self.client_id,
            "round": round_num
        }

        # 计算训练指标
        loss = 0.5 / (round_num + 1) if round_num > 0 else 0.5
        accuracy = 0.5 + 0.05 * round_num

        print(f"  训练完成: loss={loss:.4f}, acc={accuracy:.4f}")

        return TrainingResult(
            client_id=self.client_id,
            success=True,
            loss=loss,
            accuracy=accuracy,
            samples_count=self.local_samples,
            training_time=2.0,
            model_update=self._local_model
        )

    async def evaluate(self, evaluation_params: Dict[str, Any]) -> EvaluationResult:
        """本地评估"""
        print(f"[{self.client_id}] 执行模型评估")
        await asyncio.sleep(1)

        return EvaluationResult(
            client_id=self.client_id,
            success=True,
            loss=0.3,
            accuracy=0.85,
            samples_count=self.local_samples,
            evaluation_time=1.0
        )

    async def get_local_model(self) -> ModelData:
        """获取本地模型"""
        return self._local_model

    async def set_local_model(self, model_data: ModelData) -> bool:
        """设置本地模型"""
        self._local_model = model_data
        print(f"[{self.client_id}] 接收到全局模型: {model_data}")
        return True


async def main():
    """主函数"""
    setup_auto_logging()
    parser = argparse.ArgumentParser(description="Network 模式测试 - 客户端（跨机器版本）")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="客户端配置文件路径 (例如: configs/network_test/client1.yaml)"
    )
    parser.add_argument(
        "--server-ip",
        type=str,
        default="192.168.31.68",
        help="服务端 IP 地址 (默认: 192.168.31.68)"
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=8000,
        help="服务端端口 (默认: 8000)"
    )

    args = parser.parse_args()

    print("="*60)
    print("Network 模式测试 - 客户端")
    print("="*60)
    print(f"配置文件: {args.config}")
    print(f"服务端地址: {args.server_ip}:{args.server_port}")
    print("="*60)

    try:
        # 加载客户端配置
        client_config = load_client_config(args.config)
        print(f"客户端ID: {client_config.client_id}")
        print(f"配置模式: {client_config.mode}")
        print(f"监听端口: {client_config.transport.port}")
        print()

        # 创建客户端
        client = FederationClient(
            client_config.to_dict(),
            client_id=client_config.client_id
        )

        # 初始化学习器
        await client.initialize_with_learner(
            learner_class=SimpleLearner,
            learner_config={}
        )

        # 启动客户端
        print("启动客户端...")
        await client.start_client()
        print(f"✅ 客户端已启动: {client.client_id}")
        print(f"正在连接到服务端 {args.server_ip}:{args.server_port}...")
        print()

        # 等待注册成功
        await asyncio.sleep(3)

        print("等待服务端训练指令...")
        print("(按 Ctrl+C 停止)\n")

        # 保持运行，等待服务端的训练请求
        while True:
            await asyncio.sleep(10)

    except KeyboardInterrupt:
        print("\n\n停止客户端...")
    except FileNotFoundError as e:
        print(f"\n错误：配置文件不存在: {args.config}")
        print(f"请确保配置文件路径正确")
        print(f"详细错误: {e}")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n清理资源...")
        if 'client' in locals():
            await client.stop_client()


if __name__ == "__main__":
    asyncio.run(main())
