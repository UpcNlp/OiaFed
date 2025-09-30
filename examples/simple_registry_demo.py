"""
简化的注册系统演示 - 测试装饰器和自动发现
examples/simple_registry_demo.py
"""

import asyncio
import sys
import os
from typing import Dict, Any, List
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fedcl.federation.server import FederationServer
from fedcl.federation.client import FederationClient
from fedcl.learner.base_learner import BaseLearner
from fedcl.trainer.base_trainer import BaseTrainer, TrainingConfig
from fedcl.types import CommunicationMode, ModelData, TrainingRequest, TrainingResponse
from fedcl.utils.auto_logger import setup_auto_logging

# 导入装饰器注册系统
from fedcl.api import learner, trainer
from fedcl.registry import registry


@learner('DemoMNIST', 
         description='演示用MNIST学习器',
         version='1.0',
         data_type='image',
         model_type='neural_network')
class SimpleMNISTLearner(BaseLearner):
    """简化的MNIST学习器演示类"""
    
    def __init__(self, client_id: str, config: Dict[str, Any] = None, logger=None):
        super().__init__(
            client_id=client_id,
            local_data=None,
            model_config=config.get("model", {}) if config else {},
            training_config=config.get("training", {}) if config else {}
        )
        self.model_data = {
            "model_type": "simple_linear",
            "parameters": np.random.randn(784, 10).tolist(),
            "accuracy": 0.0,
            "loss": 0.0
        }
        print(f"SimpleMNISTLearner {client_id} initialized")
    
    async def train(self, request: TrainingRequest) -> TrainingResponse:
        """训练方法"""
        print(f"Training on client {self.client_id} for round {request.round_number}")
        
        # 模拟训练过程
        await asyncio.sleep(0.1)
        
        # 模拟训练结果
        simulated_accuracy = np.random.uniform(0.7, 0.9)
        simulated_loss = np.random.uniform(0.1, 0.5)
        
        self.model_data["accuracy"] = simulated_accuracy
        self.model_data["loss"] = simulated_loss
        
        return TrainingResponse(
            client_id=self.client_id,
            round_number=request.round_number,
            accuracy=simulated_accuracy,
            loss=simulated_loss,
            samples_used=100,
            model_updates={"gradient_norm": np.random.uniform(0.01, 0.1)}
        )
    
    async def evaluate(self) -> Dict[str, Any]:
        """评估方法"""
        return {
            "accuracy": self.model_data["accuracy"],
            "loss": self.model_data["loss"],
            "samples": 100
        }
    
    async def get_model(self) -> ModelData:
        """获取模型"""
        return {
            "model_type": self.model_data["model_type"],
            "parameters": self.model_data["parameters"],
            "metadata": {"client_id": self.client_id, "timestamp": "2024-01-01"}
        }
    
    async def set_model(self, model: ModelData) -> bool:
        """设置模型"""
        self.model_data["parameters"] = model["parameters"]
        return True
    
    # 动态代理测试方法
    async def custom_method_for_testing(self, data: str, param2: int = 0) -> Dict[str, Any]:
        """自定义测试方法"""
        return {"data": data, "param2": param2, "client_id": self.client_id}
    
    async def get_client_info(self) -> Dict[str, Any]:
        """获取客户端信息"""
        return {"client_id": self.client_id, "status": "active", "model_type": "simple_linear"}
    
    async def compute_gradients(self, loss_fn: str) -> Dict[str, Any]:
        """计算梯度"""
        return {
            "loss_function": loss_fn,
            "gradient_norm": np.random.uniform(0.01, 0.1),
            "client_id": self.client_id
        }
    
    async def get_data_statistics(self) -> Dict[str, Any]:
        """获取数据统计"""
        return {"num_samples": 100, "num_classes": 10, "client_id": self.client_id}
    
    # 实现抽象方法
    async def get_local_model(self) -> ModelData:
        """获取本地模型"""
        return await self.get_model()
    
    async def set_local_model(self, model: ModelData) -> bool:
        """设置本地模型"""
        return await self.set_model(model)


@trainer('DemoFedAvg',
         description='演示用FedAvg训练器',
         version='1.0',
         algorithms=['fedavg', 'weighted_average'])
class FedAvgTrainer(BaseTrainer):
    """简化的FedAvg训练器"""
    
    def __init__(self, global_model: ModelData, training_config=None, logger=None):
        super().__init__(global_model, training_config, logger)
        self.aggregation_strategy = "fedavg"
        self.min_clients = getattr(training_config, 'min_clients', 2) if training_config else 2
        self.max_rounds = getattr(training_config, 'max_rounds', 3) if training_config else 3
        self.current_round = 0
        
        print("FedAvgTrainer initialized with automatic proxy management")
        if global_model:
            print(f"Initial global model loaded: {global_model.get('model_type', 'unknown')}")
    
    async def train_round(self, round_num: int, client_ids: List[str]) -> Dict[str, Any]:
        """执行一轮联邦训练"""
        print(f"\n--- Round {round_num} ---")
        print(f"Selected clients for training: {client_ids}")
        
        # 创建训练请求
        training_request = TrainingRequest(
            round_number=round_num,
            num_epochs=1,
            batch_size=32,
            learning_rate=0.01
        )
        
        # 并行训练所有客户端
        tasks = []
        for client_id in client_ids:
            if self.is_client_ready(client_id):
                task = self._train_client(client_id, training_request)
                tasks.append(task)
        
        # 等待训练完成
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        successful_results = []
        successful_clients = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Client {client_ids[i]} training failed: {result}")
            else:
                successful_results.append(result)
                successful_clients.append(client_ids[i])
        
        if not successful_results:
            print("No successful training results")
            return {
                "round": round_num,
                "successful_clients": [],
                "round_metrics": {"avg_accuracy": 0.0, "avg_loss": float('inf')}
            }
        
        # 计算聚合指标
        avg_accuracy = np.mean([r.accuracy for r in successful_results])
        avg_loss = np.mean([r.loss for r in successful_results])
        
        print(f"Round {round_num} results: avg_accuracy={avg_accuracy:.4f}, avg_loss={avg_loss:.4f}")
        
        # 聚合模型（简化版）
        await self.aggregate_models({r.client_id: r for r in successful_results})
        
        return {
            "round": round_num,
            "successful_clients": successful_clients,
            "round_metrics": {
                "avg_accuracy": avg_accuracy,
                "avg_loss": avg_loss,
                "num_participants": len(successful_results)
            }
        }
    
    async def aggregate_models(self, client_results: Dict[str, Any]) -> Dict[str, Any]:
        """聚合客户端模型（简化版FedAvg）"""
        print("Aggregating models using FedAvg...")
        
        # 获取所有客户端的模型
        models = []
        weights = []
        
        for client_id, result in client_results.items():
            proxy = self._proxy_manager.get_proxy(client_id)
            if proxy:
                model_data = await proxy.get_model()
                models.append(model_data)
                weights.append(result.samples_used)
        
        if not models:
            print("No models to aggregate")
            return None
        
        # 简化的聚合：计算加权平均（仅作演示）
        total_samples = sum(weights)
        print(f"Aggregating {len(models)} models with total {total_samples} samples")
        
        # 在真实实现中，这里会进行实际的模型参数聚合
        # 为演示目的，我们使用第一个模型作为"聚合"结果
        aggregated_model = models[0]
        aggregated_model["metadata"]["aggregation_info"] = {
            "num_models": len(models),
            "total_samples": total_samples,
            "weights": weights
        }
        
        print("Model aggregation completed")
        
        # 将聚合后的模型分发给所有客户端
        await self._distribute_global_model(aggregated_model)
        
        return aggregated_model
    
    async def _train_client(self, client_id: str, request: TrainingRequest) -> TrainingResponse:
        """训练单个客户端"""
        proxy = self._proxy_manager.get_proxy(client_id)
        if proxy is None:
            raise RuntimeError(f"No proxy found for client {client_id}")
        
        return await proxy.train(request)
    
    async def _distribute_global_model(self, global_model: ModelData):
        """分发全局模型"""
        print("Distributing global model to all clients...")
        
        available_clients = self.get_available_clients()
        tasks = []
        
        for client_id in available_clients:
            proxy = self._proxy_manager.get_proxy(client_id)
            if proxy:
                task = proxy.set_model(global_model)
                tasks.append(task)
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            success_count = sum(1 for r in results if r is True)
            print(f"Global model distributed to {success_count}/{len(available_clients)} clients")
        else:
            print("No clients available for model distribution")
    
    async def run_federated_training(self) -> Dict[str, Any]:
        """运行联邦训练流程"""
        print("\n=== Starting Federated Training ===")
        
        # 等待足够的客户端注册
        print(f"Waiting for at least {self.min_clients} clients to register...")
        while len(self.get_available_clients()) < self.min_clients:
            await asyncio.sleep(1.0)
        
        print(f"Found {len(self.get_available_clients())} available clients")
        
        training_results = []
        self.current_round = 0
        
        for round_num in range(1, self.max_rounds + 1):
            self.current_round = round_num
            
            # 获取可用客户端
            available_clients = self.get_available_clients()
            if len(available_clients) < self.min_clients:
                print(f"Not enough clients available ({len(available_clients)} < {self.min_clients})")
                break
            
            # 执行训练轮次
            try:
                round_result = await self.train_round(round_num, available_clients[:self.min_clients])
                training_results.append(round_result)
                
                # 简单的停止条件
                if round_result['round_metrics']['avg_accuracy'] >= 0.85:
                    print("达到目标准确率，停止训练")
                    break
                    
            except Exception as e:
                print(f"Error in round {round_num}: {e}")
                break
        
        training_summary = {
            "completed_rounds": len(training_results),
            "total_rounds": self.max_rounds,
            "round_results": training_results,
            "status": "completed" if len(training_results) > 0 else "failed"
        }
        
        print(f"\n=== Federated Training Completed ===")
        print(f"Completed {len(training_results)}/{self.max_rounds} rounds")
        
        return training_summary
    
    async def evaluate_global_model(self) -> Dict[str, Any]:
        """评估全局模型"""
        print("Evaluating global model...")
        
        available_clients = self.get_available_clients()
        if not available_clients:
            return {"accuracy": 0.0, "loss": float('inf'), "samples_count": 0}
        
        # 并行评估所有客户端
        tasks = []
        for client_id in available_clients:
            if self.is_client_ready(client_id):
                proxy = self._proxy_manager.get_proxy(client_id)
                if proxy:
                    task = proxy.evaluate()
                    tasks.append(task)
        
        if not tasks:
            return {"accuracy": 0.0, "loss": float('inf'), "samples_count": 0}
        
        # 等待评估结果
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        valid_results = []
        total_samples = 0
        for result in results:
            if not isinstance(result, Exception):
                valid_results.append(result)
                total_samples += result.get("samples", 0)
        
        if not valid_results:
            return {"accuracy": 0.0, "loss": float('inf'), "samples_count": 0}
        
        # 计算加权平均
        weighted_accuracy = sum(r["accuracy"] * r.get("samples", 1) for r in valid_results) / total_samples
        weighted_loss = sum(r["loss"] * r.get("samples", 1) for r in valid_results) / total_samples
        
        evaluation_result = {
            "accuracy": weighted_accuracy,
            "loss": weighted_loss,
            "samples_count": total_samples,
            "participants": len(valid_results)
        }
        
        print(f"Global evaluation: Accuracy={weighted_accuracy:.4f}, Loss={weighted_loss:.4f}")
        return evaluation_result
    
    def should_stop_training(self, round_num: int, round_result: Dict[str, Any]) -> bool:
        """判断是否应该停止训练"""
        # 检查最大轮次
        if round_num >= self.max_rounds:
            print(f"Reached maximum rounds ({self.max_rounds})")
            return True
        
        # 检查是否有足够的客户端参与
        successful_clients = len(round_result.get("successful_clients", []))
        if successful_clients < self.min_clients:
            print(f"Not enough successful clients ({successful_clients} < {self.min_clients})")
            return True
        
        # 简单的收敛检查（示例）
        round_metrics = round_result.get("round_metrics", {})
        avg_accuracy = round_metrics.get("avg_accuracy", 0.0)
        
        # 如果准确率达到85%，停止训练
        if avg_accuracy >= 0.85:
            print(f"High accuracy achieved ({avg_accuracy:.4f} >= 0.85)")
            return True
        
        return False


async def create_server(config: Dict[str, Any]) -> FederationServer:
    """创建并启动服务端 - 使用注册表获取训练器"""
    print("Creating FederationServer...")
    print("🔍 使用注册表查找组件...")
    
    # 显示已注册的组件
    components = registry.list_all_components()
    print(f"已注册的训练器: {components['trainers']}")
    
    # 创建服务端
    server = FederationServer(config)
    
    # 从注册表获取训练器类
    print("✅ 从注册表获取训练器: FedAvgTrainer")
    trainer_cls = registry.get_trainer("DemoFedAvg")
    
    # 创建全局模型
    global_model = {
        "model_type": "simple_linear",
        "parameters": np.random.randn(784, 10).tolist(),
        "metadata": {"version": 1.0}
    }
    
    # 初始化trainer
    training_config = TrainingConfig(
        max_rounds=config.get("trainer", {}).get("max_rounds", 3),
        min_clients=config.get("trainer", {}).get("min_clients", 2)
    )
    
    trainer = await server.initialize_with_trainer(
        trainer_class=trainer_cls,
        global_model=global_model,
        trainer_config=training_config
    )
    
    # 启动服务端
    await server.start_server()
    
    return server


async def create_client(client_id: str, config: Dict[str, Any]) -> FederationClient:
    """创建并启动客户端 - 使用注册表获取学习器"""
    print("Creating FederationClient...")
    print("🔍 使用注册表查找学习器...")
    
    # 显示已注册的学习器
    components = registry.list_all_components()
    print(f"已注册的学习器: {components['learners']}")
    
    # 创建客户端
    client = FederationClient(config, client_id)
    
    # 从注册表获取学习器类
    print("✅ 从注册表获取学习器: SimpleMNISTLearner")
    learner_cls = registry.get_learner("DemoMNIST")
    
    # 初始化学习器
    learner = await client.initialize_with_learner(
        learner_class=learner_cls,
        learner_config={}
    )
    
    # 启动客户端
    await client.start_client()
    
    return client


async def run_simple_demo():
    """运行简化演示"""
    print("🚀 MOE-FedCL 注册系统演示")
    print("=" * 50)
    
    # 显示组件注册状态
    print("\n📋 组件注册状态:")
    components = registry.list_all_components()
    print(f"已注册Learner: {components['learners']}")
    print(f"已注册Trainer: {components['trainers']}")
    print()
    
    # 初始化日志系统
    setup_auto_logging()
    
    # 配置
    config = {
        "mode": "memory",
        "timeout": 30.0,
        "trainer": {
            "min_clients": 2,
            "max_rounds": 3
        }
    }
    
    try:
        # 1. 创建服务器
        print("\n1. Starting Server...")
        server = await create_server(config)
        
        # 2. 创建客户端
        print("\n2. Starting Clients...")
        clients = []
        for i in range(3):
            client_id = f"demo_client_{i+1}"
            client = await create_client(client_id, config)
            clients.append(client)
        
        # 3. 等待客户端注册
        print("\n3. Waiting for client registration...")
        await asyncio.sleep(2.0)
        
        # 4. 运行联邦训练
        print("\n4. Running Federated Training...")
        training_results = await server.trainer.run_federated_training()
        print(f"\nTraining Results: {training_results}")
        
        # 5. 测试动态代理调用
        print("\n5. Testing Dynamic Proxy...")
        if clients:
            proxy = server.proxy_manager.get_proxy(clients[0].client_id)
            if proxy:
                # 测试自定义方法
                result = await proxy.custom_method_for_testing("test_data", param2=42)
                print(f"Dynamic call result: {result}")
        
        print("\n✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(run_simple_demo())
