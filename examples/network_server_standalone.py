"""
Network 模式测试 - 服务端脚本（跨机器版本）
运行在：笔记本 (192.168.31.68)

这个脚本只启动服务端，等待远程客户端连接。
"""

import asyncio
import os
import sys
from typing import Dict, Any

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fedcl.utils.auto_logger import setup_auto_logging
from fedcl.types import EvaluationResult, ModelData, RoundResult
from fedcl.config import load_server_config
from fedcl.federation.server import FederationServer
from fedcl.trainer.trainer import BaseTrainer


class SimpleTrainer(BaseTrainer):
    """简单的测试训练器"""

    def __init__(self, global_model=None, training_config=None, logger=None):
        super().__init__(global_model, training_config, logger)
        self.round_count = 0

    async def train_round(self, round_num: int, client_ids: list) -> RoundResult:
        """执行一轮训练"""
        print(f"\n{'='*60}")
        print(f"[Trainer] 第 {round_num} 轮：客户端 {client_ids}")
        print(f"{'='*60}")

        client_results = {}
        successful_clients = []
        failed_clients = []

        # 并发训练所有客户端
        tasks = []
        for cid in client_ids:
            if cid in self.learner_proxies and self.is_client_ready(cid):
                proxy = self.learner_proxies[cid]
                task = proxy.train({
                    "global_model": self.global_model,
                    "epochs": 1,
                    "learning_rate": 0.01,
                    "round": round_num
                })
                tasks.append((cid, task))
            else:
                print(f"  客户端 {cid} 未就绪")
                failed_clients.append(cid)

        # 收集结果
        for cid, task in tasks:
            try:
                result = await task
                client_results[cid] = result
                successful_clients.append(cid)
                print(f"  ✓ {cid}: loss={result.get('loss', 0):.4f}, acc={result.get('accuracy', 0):.4f}")
            except Exception as e:
                print(f"  ✗ {cid} 训练失败: {e}")
                failed_clients.append(cid)

        # 聚合模型
        aggregated_model = await self.aggregate_models(client_results)

        # 更新全局模型
        self.global_model = aggregated_model

        # 计算平均指标
        if client_results:
            avg_loss = sum(r.get("loss", 0) for r in client_results.values()) / len(client_results)
            avg_accuracy = sum(r.get("accuracy", 0) for r in client_results.values()) / len(client_results)
        else:
            avg_loss = 0
            avg_accuracy = 0

        print(f"  聚合结果: avg_loss={avg_loss:.4f}, avg_acc={avg_accuracy:.4f}")

        self.round_count += 1

        return RoundResult(
            participants=client_ids,
            successful_clients=successful_clients,
            failed_clients=failed_clients,
            aggregated_model=aggregated_model,
            round_metrics={
                "avg_loss": avg_loss,
                "avg_accuracy": avg_accuracy
            },
            training_time=1.0
        )

    async def aggregate_models(self, client_results: Dict[str, Any]) -> ModelData:
        """聚合客户端模型"""
        if not client_results:
            return self.global_model

        # 简单平均
        weights_sum = sum(r.get("weights", 1.0) for r in client_results.values())
        avg_weights = weights_sum / len(client_results)

        return {
            "weights": avg_weights,
            "round": self.round_count + 1,
            "num_clients": len(client_results)
        }

    async def evaluate_global_model(self) -> EvaluationResult:
        """评估全局模型"""
        return EvaluationResult(
            client_id="server",
            success=True,
            loss=0.4,
            accuracy=0.90,
            samples_count=10000,
            evaluation_time=0.1
        )

    def should_stop_training(self, round_num: int, round_result: RoundResult) -> bool:
        """判断是否应该停止训练"""
        return False  # 由外部控制轮数


async def main():

    """主函数"""
    setup_auto_logging()
    print("="*60)
    print("Network 模式测试 - 服务端")
    print("="*60)
    print("服务端地址: 192.168.31.68:8000")
    print("等待客户端连接...")
    print("="*60)

    # 初始全局模型
    global_model = {
        "weights": 1.0,
        "round": 0
    }

    try:
        # 加载服务端配置
        server_config = load_server_config("../configs/network_test/server.yaml")
        print(f"配置模式: {server_config.mode}")
        print(f"监听地址: {server_config.transport.host}:{server_config.transport.port}")
        print()

        # 创建服务端
        server = FederationServer(
            server_config.to_dict(),
            server_id=server_config.server_id or "network_server"
        )

        # 初始化训练器
        await server.initialize_with_trainer(
            trainer_class=SimpleTrainer,
            global_model=global_model,
            trainer_config={}
        )

        # 启动服务端
        await server.start_server()
        print(f"✅ 服务端已启动: {server.server_id}")
        print(f"等待客户端注册...\n")

        # 等待客户端连接（这里等待一段时间让客户端连接）
        print("等待客户端连接（30秒）...")
        await asyncio.sleep(30)

        # 检查已连接的客户端
        available_clients = server.trainer.get_available_clients()
        print(f"\n已注册客户端数量: {len(available_clients)}")
        for client_id in available_clients:
            print(f"  - {client_id}")

        if len(available_clients) < 2:
            print(f"\n⚠️  警告：期望2个客户端，但只有 {len(available_clients)} 个已连接")
            print("继续等待或按 Ctrl+C 取消...")
            await asyncio.sleep(30)
            available_clients = server.trainer.get_available_clients()
            print(f"当前已注册客户端: {len(available_clients)}")

        if len(available_clients) == 0:
            print("错误：没有客户端连接，退出")
            return

        # 开始训练
        print(f"\n{'='*60}")
        print("开始联邦学习训练")
        print(f"{'='*60}")

        max_rounds = 3
        for round_num in range(1, max_rounds + 1):
            print(f"\n开始第 {round_num} 轮...")

            # 选择参与的客户端
            selected_clients = available_clients[:2]  # 选择前2个

            # 执行训练轮次
            round_result = await server.trainer.train_round(round_num, selected_clients)

            print(f"第 {round_num} 轮完成: {len(round_result.get('successful_clients', []))} 个客户端成功")

        print(f"\n{'='*60}")
        print("训练完成!")
        print(f"{'='*60}")
        print(f"完成轮数: {max_rounds}")
        print(f"最终模型: {server.trainer.global_model}")

        # 保持服务端运行
        print("\n服务端将继续运行...")
        print("按 Ctrl+C 停止")
        while True:
            await asyncio.sleep(10)

    except KeyboardInterrupt:
        print("\n\n停止服务端...")
    except FileNotFoundError as e:
        print(f"\n错误：配置文件不存在")
        print(f"请确保从项目根目录运行此脚本")
        print(f"详细错误: {e}")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n清理资源...")
        if 'server' in locals():
            await server.stop_server()


if __name__ == "__main__":
    asyncio.run(main())
