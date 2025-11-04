"""
Network模式 - 独立服务器脚本（新架构）
用于跨机器的联邦学习测试

使用方法:
  python examples/network_server.py --config configs/network_demo/server.yaml
  python examples/network_server.py --mode class
"""

import asyncio
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from fedcl.config import CommunicationConfig, TrainingConfig, ConfigLoader
from fedcl.federation.server import FederationServer
from fedcl.trainer.trainer import BaseTrainer
from fedcl.api import trainer
from fedcl.utils.auto_logger import setup_auto_logging


# ==================== 定义Trainer ====================

@trainer('SimpleNetworkTrainer',
         description='简单的Network模式训练器',
         version='1.0',
         author='MOE-FedCL',
         algorithms=['fedavg'])
class SimpleNetworkTrainer(BaseTrainer):
    """简单的网络模式训练器"""

    def __init__(self, config: Dict[str, Any] = None, lazy_init: bool = True):
        super().__init__(config, lazy_init)

        # 从配置中获取参数
        trainer_params = config.get('trainer', {}).get('params', {}) if config else {}
        self.algorithm = trainer_params.get('algorithm', 'fedavg')
        self.local_epochs = trainer_params.get('local_epochs', 1)
        self.learning_rate = trainer_params.get('learning_rate', 0.01)

        self.logger.info(f"SimpleNetworkTrainer初始化完成 (算法: {self.algorithm})")

    def _create_default_global_model(self):
        """创建默认全局模型"""
        self.logger.info("创建默认全局模型")
        return {
            "model_type": "simple",
            "weights": 1.0,
            "round": 0,
            "version": 1
        }

    async def train_round(self, round_num: int, client_ids: List[str]) -> Dict[str, Any]:
        """执行一轮联邦训练"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Round {round_num} - 选择客户端: {client_ids}")
        self.logger.info(f"{'='*60}")

        # 准备训练参数
        training_params = {
            "round_number": round_num,
            "num_epochs": self.local_epochs,
            "learning_rate": self.learning_rate
        }

        # 并行训练所有客户端
        tasks = []
        for client_id in client_ids:
            if self.is_client_ready(client_id):
                proxy = self._proxy_manager.get_proxy(client_id)
                if proxy:
                    self.logger.info(f"  [{client_id}] 启动训练...")
                    task = proxy.train(training_params)
                    tasks.append((client_id, task))

        # 收集结果
        client_results = {}
        failed_clients = []

        for client_id, task in tasks:
            try:
                result = await task
                if result.success:
                    client_results[client_id] = result
                    self.logger.info(
                        f"  [{client_id}] 训练成功: Loss={result.result['loss']:.4f}, "
                        f"Acc={result.result['accuracy']:.4f}"
                    )
                else:
                    self.logger.error(f"  [{client_id}] 训练失败: {result}")
                    failed_clients.append(client_id)
            except Exception as e:
                self.logger.exception(f"  [{client_id}] 训练异常: {e}")
                failed_clients.append(client_id)

        # 聚合模型
        if client_results:
            aggregated_weights = await self.aggregate_models(client_results)
            if aggregated_weights:
                self.global_model = aggregated_weights

        # 计算平均指标
        if client_results:
            avg_loss = sum(r.result['loss'] for r in client_results.values()) / len(client_results)
            avg_accuracy = sum(r.result['accuracy'] for r in client_results.values()) / len(client_results)
        else:
            avg_loss, avg_accuracy = 0.0, 0.0

        self.logger.info(f"  Round {round_num} 汇总: Loss={avg_loss:.4f}, Acc={avg_accuracy:.4f}")

        return {
            "round": round_num,
            "participants": client_ids,
            "successful_clients": list(client_results.keys()),
            "failed_clients": failed_clients,
            "model_aggregated": bool(client_results),
            "round_metrics": {
                "avg_loss": avg_loss,
                "avg_accuracy": avg_accuracy,
                "successful_count": len(client_results)
            }
        }

    async def aggregate_models(self, client_results: Dict[str, Any]) -> Dict[str, Any]:
        """聚合客户端模型（FedAvg）"""
        self.logger.info("  聚合模型 (FedAvg)...")

        if not client_results:
            return None

        # 计算加权平均
        total_samples = sum(r.result['samples'] for r in client_results.values())

        if total_samples == 0:
            return None

        # 简单加权平均
        weighted_sum = sum(
            r.result['model_weights'].get('weights', 1.0) * r.result['samples']
            for r in client_results.values()
        )

        avg_weights = weighted_sum / total_samples

        # 更新全局模型
        current_round = self.global_model.get('round', 0)
        aggregated_model = {
            "model_type": "simple",
            "weights": avg_weights,
            "round": current_round + 1,
            "num_clients": len(client_results),
            "version": self.global_model.get('version', 1) + 1
        }

        self.logger.info(f"  聚合完成: weights={avg_weights:.4f}, clients={len(client_results)}")

        # 分发全局模型
        await self._distribute_global_model(aggregated_model)

        return aggregated_model

    async def evaluate_global_model(self) -> Dict[str, Any]:
        """评估全局模型"""
        self.logger.info("  评估全局模型...")
        return {
            "accuracy": 0.90,
            "loss": 0.25,
            "samples_count": 10000
        }

    def should_stop_training(self, round_num: int, round_result: Dict[str, Any]) -> bool:
        """判断是否应该停止训练"""
        round_metrics = round_result.get("round_metrics", {})
        avg_accuracy = round_metrics.get("avg_accuracy", 0.0)

        if avg_accuracy >= 0.95:
            self.logger.info(f"  达到目标精度: {avg_accuracy:.4f}")
            return True

        return False

    async def _distribute_global_model(self, global_model: Dict[str, Any]):
        """分发全局模型到所有客户端"""
        model_data = {
            "model_type": global_model.get("model_type", "simple"),
            "parameters": {"weights": global_model}
        }

        tasks = []
        for client_id in self.get_available_clients():
            proxy = self._proxy_manager.get_proxy(client_id)
            if proxy:
                task = proxy.set_model(model_data)
                tasks.append(task)

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            success_count = sum(1 for r in results if not isinstance(r, Exception) and r)
            self.logger.info(f"  全局模型已分发到 {success_count}/{len(tasks)} 个客户端")


# ==================== 主函数 ====================

async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Network模式 - 独立服务器")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/network_demo/server.yaml",
        help="服务器配置文件路径"
    )
    parser.add_argument(
        "--mode",
        choices=["file", "class"],
        default="file",
        help="配置模式: file=从配置文件, class=使用配置类"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="服务器监听地址 (默认: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="服务器端口 (默认: 8000)"
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=5,
        help="训练轮数 (默认: 5)"
    )
    parser.add_argument(
        "--wait",
        type=int,
        default=30,
        help="等待客户端连接的时间（秒） (默认: 30)"
    )

    args = parser.parse_args()

    # 设置日志
    setup_auto_logging()

    print("\n" + "="*70)
    print("Network模式 - 独立服务器（新架构）")
    print("="*70)

    server = None

    try:
        # 创建配置
        if args.mode == "file":
            print(f"从配置文件加载: {args.config}\n")
            comm_config, train_config = ConfigLoader.load(args.config)
        else:
            print(f"使用配置类创建配置\n")
            comm_config = CommunicationConfig(
                mode="network",
                role="server",
                node_id="network_server_main",
                transport={
                    "type": "websocket",
                    "host": args.host,
                    "port": args.port,
                    "websocket_port": 9501,
                    "timeout": 60.0
                }
            )

            train_config = TrainingConfig(
                trainer={
                    "name": "SimpleNetworkTrainer",
                    "params": {
                        "algorithm": "fedavg",
                        "local_epochs": 1,
                        "learning_rate": 0.01
                    }
                }
            )

        # 创建服务器
        server = FederationServer(comm_config, train_config)

        # 初始化
        print("初始化服务器...")
        await server.initialize()

        # 启动服务器
        print("启动服务器...")
        await server.start_server()

        print(f"\n✅ 服务器已启动")
        print(f"   服务器ID: {server.server_id}")
        print(f"   监听地址: {comm_config.transport.get('host')}:{comm_config.transport.get('port')}")
        print(f"\n等待客户端连接（{args.wait}秒）...\n")

        # 等待客户端连接
        await asyncio.sleep(args.wait)

        # 检查已连接的客户端
        available_clients = server.trainer.get_available_clients()
        print(f"\n已注册客户端: {len(available_clients)}")
        for client_id in available_clients:
            print(f"  - {client_id}")

        if len(available_clients) < 2:
            print(f"\n⚠️  警告：期望至少2个客户端，但只有 {len(available_clients)} 个")
            print("继续等待或按 Ctrl+C 取消...\n")
            await asyncio.sleep(args.wait)
            available_clients = server.trainer.get_available_clients()
            print(f"当前已注册客户端: {len(available_clients)}")

        if len(available_clients) == 0:
            print("❌ 错误：没有客户端连接，退出")
            return

        # 开始训练
        print(f"\n{'='*70}")
        print(f"开始联邦学习训练 ({args.rounds} 轮)")
        print(f"{'='*70}\n")

        for round_num in range(1, args.rounds + 1):
            # 选择参与的客户端
            selected_clients = available_clients[:min(2, len(available_clients))]

            # 执行训练轮次
            round_result = await server.trainer.train_round(round_num, selected_clients)

            successful = round_result.get('successful_clients', [])
            print(f"\nRound {round_num} 完成: {len(successful)}/{len(selected_clients)} 客户端成功")

        print(f"\n{'='*70}")
        print("训练完成!")
        print(f"{'='*70}")
        print(f"完成轮数: {args.rounds}")
        print(f"最终全局模型: {server.trainer.global_model}")

        # 保持服务器运行
        print("\n服务器将继续运行...")
        print("按 Ctrl+C 停止\n")
        while True:
            await asyncio.sleep(10)

    except KeyboardInterrupt:
        print("\n\n用户中断，停止服务器...")
    except FileNotFoundError as e:
        print(f"\n❌ 错误：配置文件不存在: {args.config}")
        print(f"   详细错误: {e}")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if server:
            print("\n清理资源...")
            await server.stop_server()


if __name__ == "__main__":
    asyncio.run(main())
