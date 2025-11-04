"""
Network模式 - 独立客户端脚本（新架构）
用于跨机器的联邦学习测试

使用方法:
  python examples/network_client.py --config configs/network_demo/client1.yaml
  python examples/network_client.py --mode class --client-id network_client_1
"""

import asyncio
import sys
import argparse
from pathlib import Path
from typing import Dict, Any

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from fedcl.config import CommunicationConfig, TrainingConfig, ConfigLoader
from fedcl.federation.client import FederationClient
from fedcl.learner.base_learner import BaseLearner
from fedcl.api import learner
from fedcl.utils.auto_logger import setup_auto_logging


# ==================== 定义Learner ====================

@learner('SimpleNetworkLearner',
         description='简单的Network模式学习器',
         version='1.0',
         author='MOE-FedCL')
class SimpleNetworkLearner(BaseLearner):
    """简单的网络模式学习器"""

    def __init__(self, client_id: str, config: Dict[str, Any] = None, lazy_init: bool = True):
        super().__init__(client_id, config, lazy_init)

        # 从配置中获取参数
        params = config.get('learner', {}).get('params', {}) if config else {}
        self.local_epochs = params.get('local_epochs', 1)
        self.learning_rate = params.get('learning_rate', 0.01)
        self.batch_size = params.get('batch_size', 32)
        self.sample_count = params.get('sample_count', 1000)

        # 本地模型
        self._local_model = {"weights": 1.0, "version": 1}

        self.logger.info(f"SimpleNetworkLearner初始化完成 (样本数: {self.sample_count})")

    async def train(self, training_params: Dict[str, Any]) -> Dict[str, Any]:
        """本地训练"""
        round_num = training_params.get("round_number", 0)

        self.logger.info(f"[{self.client_id}] 开始本地训练 - 第 {round_num} 轮")

        # 模拟训练过程
        await asyncio.sleep(2)

        # 更新本地模型
        current_weights = self._local_model.get("weights", 1.0)
        new_weights = current_weights + 0.1  # 模拟训练更新
        self._local_model = {
            "weights": new_weights,
            "client_id": self.client_id,
            "round": round_num,
            "version": self._local_model.get("version", 1) + 1
        }

        # 计算训练指标
        loss = 0.5 / (round_num + 1) if round_num > 0 else 0.5
        accuracy = min(0.5 + 0.05 * round_num, 0.95)

        self.logger.info(f"[{self.client_id}] 训练完成: Loss={loss:.4f}, Acc={accuracy:.4f}")

        return {
            "success": True,
            "loss": loss,
            "accuracy": accuracy,
            "samples": self.sample_count,
            "model_weights": self._local_model,
            "training_time": 2.0
        }

    async def evaluate(self, model_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """本地评估"""
        self.logger.info(f"[{self.client_id}] 执行模型评估")
        await asyncio.sleep(1)

        return {
            "success": True,
            "accuracy": 0.85,
            "loss": 0.3,
            "samples": self.sample_count
        }

    async def get_local_model(self) -> Dict[str, Any]:
        """获取本地模型"""
        return self._local_model

    async def set_local_model(self, model_data: Dict[str, Any]) -> bool:
        """设置本地模型"""
        self._local_model = model_data
        self.logger.info(f"[{self.client_id}] 接收全局模型: weights={model_data.get('weights', 'N/A')}")
        return True

    def get_data_statistics(self) -> Dict[str, Any]:
        """获取数据统计"""
        return {
            "total_samples": self.sample_count,
            "num_classes": 10,
            "feature_dim": 784
        }


# ==================== 主函数 ====================

async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Network模式 - 独立客户端")
    parser.add_argument(
        "--config",
        type=str,
        default="examples/configs/network_demo/client1.yaml",
        help="客户端配置文件路径"
    )
    parser.add_argument(
        "--mode",
        choices=["file", "class"],
        default="file",
        help="配置模式: file=从配置文件, class=使用配置类"
    )
    parser.add_argument(
        "--client-id",
        type=str,
        default="network_client_1",
        help="客户端ID (默认: network_client_1)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="客户端监听地址 (默认: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="客户端端口 (默认: 8001)"
    )
    parser.add_argument(
        "--server-host",
        type=str,
        default="127.0.0.1",
        help="服务器地址 (默认: 127.0.0.1)"
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=8000,
        help="服务器端口 (默认: 8000)"
    )

    args = parser.parse_args()

    # 设置日志
    setup_auto_logging()

    print("\n" + "="*70)
    print("Network模式 - 独立客户端（新架构）")
    print("="*70)

    client = None

    try:
        # 创建配置
        if args.mode == "file":
            print(f"从配置文件加载: {args.config}\n")
            comm_config, train_config = ConfigLoader.load(args.config)
        else:
            print(f"使用配置类创建配置\n")
            comm_config = CommunicationConfig(
                mode="network",
                role="client",
                node_id=args.client_id,
                transport={
                    "type": "websocket",
                    "host": args.host,
                    "port": args.port,
                    "websocket_port": 9502,
                    "timeout": 60.0,
                    "server": {
                        "host": args.server_host,
                        "port": args.server_port
                    }
                }
            )

            train_config = TrainingConfig(
                learner={
                    "name": "SimpleNetworkLearner",
                    "params": {
                        "sample_count": 1000
                    }
                }
            )

        # 创建客户端
        client = FederationClient(comm_config, train_config)

        # 初始化
        print("初始化客户端...")
        await client.initialize()

        # 启动客户端
        print("启动客户端...")
        await client.start_client()

        print(f"\n✅ 客户端已启动")
        print(f"   客户端ID: {client.client_id}")
        print(f"   监听地址: {comm_config.transport.get('host')}:{comm_config.transport.get('port')}")
        print(f"   服务器地址: {args.server_host}:{args.server_port}")
        print(f"   注册状态: {'已注册' if client.is_registered else '未注册'}")

        # 等待注册完成
        if not client.is_registered:
            print("\n等待注册到服务器...")
            await asyncio.sleep(5)
            print(f"   注册状态: {'已注册' if client.is_registered else '未注册'}")

        print("\n等待服务器训练指令...")
        print("(按 Ctrl+C 停止)\n")

        # 保持运行，等待服务器的训练请求
        while True:
            await asyncio.sleep(10)

    except KeyboardInterrupt:
        print("\n\n用户中断，停止客户端...")
    except FileNotFoundError as e:
        print(f"\n❌ 错误：配置文件不存在: {args.config}")
        print(f"   详细错误: {e}")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if client:
            print("\n清理资源...")
            await client.stop_client()


if __name__ == "__main__":
    asyncio.run(main())
