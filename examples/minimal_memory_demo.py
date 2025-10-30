# minimal_memory_demo.py
import asyncio
import os
import sys
from typing import Dict, Any

# 添加路径（根据你的项目结构调整）
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fedcl.types import (
    TrainingResult,
    EvaluationResult,
    ModelData,
    RoundResult,
    List
)
# 导入类型
from fedcl.utils.auto_logger import setup_auto_logging
from fedcl.federation.server import FederationServer
from fedcl.federation.client import FederationClient
from fedcl.learner.base_learner import BaseLearner
from fedcl.trainer.base_trainer import BaseTrainer
from fedcl.federation.coordinator import FederationCoordinator
from fedcl.types import FederationConfig
from fedcl.config import (
    ServerConfig, ClientConfig,
    load_server_config, load_client_config,
    create_default_server_config, create_default_client_config
)


# ───────────────────────────────────────
# 1. 客户端 Learner
# ───────────────────────────────────────
class SimpleLearner(BaseLearner):
    def __init__(self, client_id: str, config: Dict[str, Any], logger=None):
        super().__init__(client_id, config, logger)
        self.local_samples = 1000
        self._local_model: ModelData = {"weights": [0.1, 0.2, 0.3]}

    async def train(self, training_params: Dict[str, Any]) -> TrainingResult:
        print(f"[Client {self.client_id}] 开始训练...")
        await asyncio.sleep(0.1)
        return TrainingResult(
            client_id=self.client_id,
            success=True,
            loss=0.5,
            accuracy=0.85,
            samples_count=self.local_samples,
            training_time=0.1,
            model_update=self._local_model  # 或计算 delta
        )

    async def evaluate(self, evaluation_params: Dict[str, Any]) -> EvaluationResult:
        return EvaluationResult(
            client_id=self.client_id,
            success=True,
            loss=0.45,
            accuracy=0.87,
            samples_count=self.local_samples,
            evaluation_time=0.05
        )

    # 用于服务端向客户端“拉取”其本地训练后的模型参数
    async def get_local_model(self) -> ModelData:
        return self._local_model

    # 用于服务端向客户端“推送”新的全局模型，供其下一轮训练使用
    async def set_local_model(self, model_data: ModelData) -> bool:
        self._local_model = model_data
        print(f"[Client {self.client_id}] 接收到全局模型: {model_data}")
        return True



# ───────────────────────────────────────
# 2. 服务端 Trainer
# ───────────────────────────────────────

class SimpleTrainer(BaseTrainer):
    def __init__(self, global_model=None, training_config=None, logger=None):
        super().__init__(global_model, training_config, logger)
        self.round = 0

    async def train_round(self, round_num: int, client_ids: List[str]) -> RoundResult:
        print(f"\n第 {round_num} 轮：客户端 {client_ids}")

        client_results = {}
        successful_clients = []
        failed_clients = []

        # 并发训练
        tasks = []
        for cid in client_ids:
            if cid in self.learner_proxies and self.is_client_ready(cid):
                proxy = self.learner_proxies[cid]
                # 下发全局模型
                task = proxy.train({
                    "global_model": self.global_model,
                    "epochs": 1,
                    "learning_rate": 0.01

                })
                tasks.append((cid, task))
            else:
                failed_clients.append(cid)

        # 收集结果
        for cid, task in tasks:
            try:
                result = await task
                client_results[cid] = result
                successful_clients.append(cid)
            except Exception as e:
                print(f"客户端 {cid} 训练失败: {e}")
                failed_clients.append(cid)

        # 聚合模型
        aggregated_model = await self.aggregate_models(client_results)

        # 更新全局模型
        self.global_model = aggregated_model

        # 构造轮次指标（模拟）
        avg_loss = sum(r.get("loss", 0.5) for r in client_results.values()) / max(len(client_results), 1)
        avg_accuracy = sum(r.get("accuracy", 0.8) for r in client_results.values()) / max(len(client_results), 1)

        return RoundResult(
            participants=client_ids,
            successful_clients=successful_clients,
            failed_clients=failed_clients,
            aggregated_model=aggregated_model,
            round_metrics={
                "avg_loss": avg_loss,
                "avg_accuracy": avg_accuracy
            },
            training_time=0.5  # 模拟时间
        )

    async def aggregate_models(self, client_results: Dict[str, Any]) -> ModelData:
        print("正在聚合模型...")
        await asyncio.sleep(0.1)
        return {"weights": [0.15, 0.25, 0.35]}

    async def evaluate_global_model(self) -> EvaluationResult:
        return EvaluationResult(
            client_id="server",
            success=True,
            loss=0.4,
            accuracy=0.90,
            samples_count=10000,
            evaluation_time=0.1
        )

    def should_stop_training(self, round_num: int, round_result: RoundResult) -> bool:
        return round_num >= 1  # 只跑 1 轮


# ───────────────────────────────────────
# 3. 主函数
# ───────────────────────────────────────
async def main():

    # 初始化日志系统
    setup_auto_logging()
    config = {"mode": "memory", "timeout": 30.0}

    # 构造初始全局模型（ModelData 类型）
    initial_global_model: ModelData = {"weights": [0.1, 0.2, 0.3]}  # 与 SimpleLearner 一致

    # 启动服务端
    server = FederationServer(config)
    # 传入 trainer 类和 global_model
    await server.initialize_with_trainer(
        trainer_class=SimpleTrainer,
        global_model=initial_global_model,
        trainer_config={}  # 可选配置
    )
    await server.start_server()
    print("✅ 服务端启动")


    client1_config = {
        "mode": "process",
        "timeout": 30.0,
        "transport": {
            "specific_config": {
                "port": 0  # 自动分配
            }
        }
    }
    # 启动两个客户端
    client1 = FederationClient.create_client(config, client_id="memory_client_1")
    await client1.initialize_with_learner(SimpleLearner)
    await client1.start_client()
    print("✅ 客户端 1 启动")


    client2_config = {
        "mode": "process",
        "timeout": 30.0,
        "transport": {
            "specific_config": {
                "port": 0  # 自动分配
            }
        }
    }
    client2 = FederationClient.create_client(config, client_id="memory_client_2")
    await client2.initialize_with_learner(SimpleLearner)
    await client2.start_client()
    print("✅ 客户端 2 启动")

    # 使用 FederationCoordinator 启动联邦学习
    coordinator = FederationCoordinator(
        federation_server=server,
        federation_config=FederationConfig(
            max_rounds=1,
            min_clients=2,  # 至少 2 个客户端
            # 可添加其他配置，如收敛阈值等
        )
    )

    # 启动联邦训练（会自动等待客户端、执行轮次、聚合等）
    result = await coordinator.start_federation()
    print(f"✅ 联邦训练完成！最终准确率: {result.final_accuracy:.4f}, 轮数: {result.completed_rounds}")

    # 停止所有组件
    await client1.stop_client()
    await client2.stop_client()
    await server.stop_server()
    print("\n✅ 全部停止")


if __name__ == "__main__":
    asyncio.run(main())