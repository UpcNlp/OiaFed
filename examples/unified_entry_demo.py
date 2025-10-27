"""
使用 FederatedLearning 统一入口类运行联邦学习示例
examples/unified_entry_demo.py

演示如何使用 FederatedLearning 类快速启动完整的联邦学习系统
"""

import asyncio
from typing import Dict, Any
from fedcl import (
    FederatedLearning, run_federated_learning,
    BaseLearner, BaseTrainer,
    FederationConfig
)
from fedcl.types import (
    TrainingResult, EvaluationResult, ModelData, RoundResult, List
)
from fedcl.config import (
    ClientConfig, ServerConfig, TransportLayerConfig
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
            model_update=self._local_model
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

    async def get_local_model(self) -> ModelData:
        return self._local_model

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
        self.global_model = aggregated_model

        # 构造轮次指标
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
            training_time=0.5
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
        return round_num >= 3  # 运行 3 轮


# ============================================
# 示例1: 使用 FederatedLearning 类 + 上下文管理器
# ============================================
async def example1_context_manager():
    """最推荐的使用方式"""
    print("\n" + "="*60)
    print("示例1: 使用 FederatedLearning + 上下文管理器")
    print("="*60)

    # 初始全局模型
    initial_model: ModelData = {"weights": [0.1, 0.2, 0.3]}

    # client_config = ClientConfig(
    #     mode="process",
    #     transport=TransportLayerConfig(port=0)  # 自动分配端口
    # )

    # client_configs = [
    #     ClientConfig(
    #         mode="process",
    #         client_id="alice",  # ✅ 在配置中指定 ID
    #         transport=TransportLayerConfig(port=8001)
    #     ),
    #     ClientConfig(
    #         mode="process",
    #         client_id="bob",
    #         transport=TransportLayerConfig(port=8002)
    #     ),
    #     ClientConfig(
    #         mode="process",
    #         client_id="charlie",
    #         transport=TransportLayerConfig(port=8003)
    #     )
    # ]
    #
    # server_config = ServerConfig(
    #     mode="process",
    #     transport=TransportLayerConfig(port=8000)
    # )

    # 使用上下文管理器自动管理生命周期
    async with FederatedLearning(
        trainer_class=SimpleTrainer,
        learner_class=SimpleLearner,
        global_model=initial_model,
        server_config_path="../configs/server_demo.yaml",
        client_config_path="../configs/clients",
        # server_config=server_config,
        # client_configs=client_configs,
        num_clients=3,
        federation_config=FederationConfig(
            max_rounds=3,
            min_clients=2
        )
    ) as fl:
        # 运行训练
        result = await fl.run(max_rounds=3)

        print("\n" + "="*60)
        print("训练结果:")
        print(f"  完成轮数: {result.completed_rounds}")
        print(f"  最终准确率: {result.final_accuracy:.4f}")
        print(f"  最终损失: {result.final_loss:.4f}")
        print(f"  总时间: {result.total_time:.2f}秒")
        print("="*60)

    # 自动清理资源
    print("✅ 示例1完成\n")


# ============================================
# 示例2: 使用 FederatedLearning 类手动管理
# ============================================
async def example2_manual():
    """手动管理生命周期"""
    print("\n" + "="*60)
    print("示例2: 手动管理 FederatedLearning")
    print("="*60)

    initial_model: ModelData = {"weights": [0.1, 0.2, 0.3]}

    # 创建实例
    fl = FederatedLearning(
        trainer_class=SimpleTrainer,
        learner_class=SimpleLearner,
        global_model=initial_model,
        server_config_path="../configs/server_demo.yaml",
        client_config_path="../configs/client_demo_1.yaml",
        num_clients=2
    )

    try:
        # 初始化
        await fl.initialize()

        # 运行训练
        result = await fl.run(max_rounds=3)

        print(f"\n✅ 训练完成，准确率: {result.final_accuracy:.4f}")

    finally:
        # 清理
        await fl.cleanup()

    print("✅ 示例2完成\n")


# ============================================
# 示例3: 使用便捷函数（一行代码）
# ============================================
async def example3_convenience_function():
    """最简单的使用方式"""
    print("\n" + "="*60)
    print("示例3: 使用便捷函数 run_federated_learning")
    print("="*60)

    initial_model: ModelData = {"weights": [0.1, 0.2, 0.3]}

    # 一行代码运行完整系统
    result = await run_federated_learning(
        trainer_class=SimpleTrainer,
        learner_class=SimpleLearner,
        global_model=initial_model,
        server_config_path="../configs/server_demo.yaml",
        client_config_path="../configs/client_demo_1.yaml",
        num_clients=3,
        max_rounds=3,
        federation_config=FederationConfig(min_clients=2)
    )

    print(f"\n✅ 训练完成，准确率: {result.final_accuracy:.4f}")
    print("✅ 示例3完成\n")


# ============================================
# 示例4: 不使用配置文件（使用默认配置）
# ============================================
async def example4_default_config():
    """使用默认配置，无需配置文件"""
    print("\n" + "="*60)
    print("示例4: 使用默认配置（无配置文件）")
    print("="*60)

    initial_model: ModelData = {"weights": [0.1, 0.2, 0.3]}

    async with FederatedLearning(
        trainer_class=SimpleTrainer,
        learner_class=SimpleLearner,
        global_model=initial_model,
        # 不提供配置文件，使用默认配置
        num_clients=2
    ) as fl:
        result = await fl.run(max_rounds=3)
        print(f"\n✅ 训练完成，准确率: {result.final_accuracy:.4f}")

    print("✅ 示例4完成\n")


# ============================================
# 主函数
# ============================================
async def main():
    """运行所有示例"""
    print("="*60)
    print("MOE-FedCL 统一入口使用示例")
    print("="*60)

    # 运行示例（选择一个运行）
    await example1_context_manager()
    # await example2_manual()
    # await example3_convenience_function()
    # await example4_default_config()

    print("\n" + "="*60)
    print("所有示例运行完成！")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
