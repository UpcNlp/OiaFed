"""
完整的新架构演示 - 展示自动proxy管理和严格的层次分离
使用装饰器注册系统自动管理组件
examples/complete_new_architecture_demo.py
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
from fedcl.trainer.base_trainer import BaseTrainer
from fedcl.types import CommunicationMode, ModelData, TrainingRequest, TrainingResponse
from fedcl.utils.auto_logger import setup_auto_logging

# 导入装饰器注册系统
from fedcl.api import learner, trainer
from fedcl.registry import registry, ComponentRegistry


@learner('DemoMNIST', 
         description='演示用MNIST学习器',
         version='1.0',
         author='MOE-FedCL Demo',
         dataset='MNIST')
class SimpleMNISTLearner(BaseLearner):
    """简单的MNIST学习器示例 - 使用装饰器注册"""
    
    def __init__(self, client_id: str, config: Dict[str, Any], logger=None):
        super().__init__(client_id, config, logger)
        
        # 模拟MNIST数据统计
        self.data_stats = {
            "total_samples": 1000,
            "num_classes": 10,
            "feature_dim": 784
        }
        
        # 模拟模型参数
        self.model_params = np.random.randn(784, 10) * 0.01
        
        print(f"SimpleMNISTLearner {client_id} initialized")
    
    async def train(self, request: TrainingRequest) -> TrainingResponse:
        """训练方法"""
        print(f"[{self.client_id}] Starting training with {request.num_epochs} epochs")
        
        # 模拟训练过程
        await asyncio.sleep(0.5)  # 模拟训练时间
        
        # 模拟参数更新
        self.model_params += np.random.randn(*self.model_params.shape) * 0.001
        
        # 创建响应
        response = TrainingResponse(
            client_id=self.client_id,
            success=True,
            epochs_completed=request.num_epochs,
            loss=0.5 + np.random.randn() * 0.1,
            accuracy=0.85 + np.random.randn() * 0.05,
            samples_used=self.data_stats["total_samples"],
            training_time=0.5
        )
        
        print(f"[{self.client_id}] Training completed - Loss: {response.loss:.4f}, Accuracy: {response.accuracy:.4f}")
        return response
    
    async def evaluate(self, model_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """评估方法"""
        print(f"[{self.client_id}] Starting evaluation")
        
        # 模拟评估过程
        await asyncio.sleep(0.2)
        
        result = {
            "accuracy": 0.88 + np.random.randn() * 0.03,
            "loss": 0.3 + np.random.randn() * 0.05,
            "samples": self.data_stats["total_samples"]
        }
        
        print(f"[{self.client_id}] Evaluation completed - Accuracy: {result['accuracy']:.4f}")
        return result
    
    async def get_model(self) -> Dict[str, Any]:
        """获取模型数据"""
        return {
            "model_type": "simple_linear",
            "parameters": {"weights": self.model_params.tolist()},
            "metadata": {
                "client_id": self.client_id,
                "model_size": self.model_params.size,
                "data_samples": self.data_stats["total_samples"]
            }
        }
    
    async def set_model(self, model_data: Dict[str, Any]) -> bool:
        """设置模型数据"""
        try:
            if "parameters" in model_data and "weights" in model_data["parameters"]:
                self.model_params = np.array(model_data["parameters"]["weights"])
                print(f"[{self.client_id}] Model updated")
                return True
        except Exception as e:
            print(f"[{self.client_id}] Failed to set model: {e}")
        return False
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """获取数据统计"""
        return self.data_stats.copy()
    
    async def get_local_model(self) -> Dict[str, Any]:
        """获取本地模型参数"""
        return await self.get_model()
    
    async def set_local_model(self, model_data: Dict[str, Any]) -> bool:
        """设置本地模型参数"""
        return await self.set_model(model_data)
    
    # ==================== 动态调用测试方法 ====================
    
    async def custom_method_for_testing(self, param1: str, param2: int = 10) -> Dict[str, Any]:
        """用于测试动态调用的自定义方法"""
        print(f"[{self.client_id}] custom_method_for_testing called with param1={param1}, param2={param2}")
        await asyncio.sleep(0.1)  # 模拟一些处理时间
        
        return {
            "method_name": "custom_method_for_testing",
            "client_id": self.client_id,
            "param1": param1,
            "param2": param2,
            "result": f"Processed {param1} with value {param2}",
            "timestamp": asyncio.get_event_loop().time()
        }
    
    def get_client_info(self) -> Dict[str, Any]:
        """获取客户端信息（同步方法测试）"""
        return {
            "client_id": self.client_id,
            "model_shape": self.model_params.shape,
            "data_stats": self.data_stats,
            "method_type": "synchronous"
        }
    
    async def compute_gradients(self, loss_fn: str = "mse") -> Dict[str, Any]:
        """计算梯度（另一个异步方法测试）"""
        print(f"[{self.client_id}] Computing gradients with {loss_fn} loss function")
        await asyncio.sleep(0.3)
        
        # 模拟梯度计算
        fake_gradients = np.random.randn(*self.model_params.shape) * 0.01
        
        return {
            "gradients": fake_gradients.tolist(),
            "loss_function": loss_fn,
            "gradient_norm": np.linalg.norm(fake_gradients),
            "client_id": self.client_id
        }


@trainer('DemoFedAvg',
         description='演示用联邦平均训练器',
         version='1.0',
         author='MOE-FedCL Demo',
         algorithms=['fedavg', 'weighted_average'])
class FedAvgTrainer(BaseTrainer):
    """联邦平均训练器示例 - 使用装饰器注册"""
    
    def __init__(self, global_model: Dict[str, Any] = None, training_config = None, logger=None):
        # 注意：不再需要传入learner_proxies，会自动管理
        from fedcl.trainer.base_trainer import TrainingConfig
        
        # 处理配置参数
        if isinstance(training_config, dict):
            # 如果传入的是字典，创建TrainingConfig对象
            config_obj = TrainingConfig(
                max_rounds=training_config.get("max_rounds", 5),
                min_clients=training_config.get("min_clients", 2)
            )
        elif isinstance(training_config, TrainingConfig):
            # 如果已经是TrainingConfig对象，直接使用
            config_obj = training_config
        else:
            # 使用默认配置
            config_obj = TrainingConfig()
            
        super().__init__(global_model, config_obj, logger)
        
        self.global_model = global_model
        self.aggregation_strategy = "fedavg"
        self.min_clients = config_obj.min_clients
        self.max_rounds = config_obj.max_rounds
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
            parameters={
                "round_number": round_num,
                "num_epochs": 1,
                "batch_size": 32,
                "learning_rate": 0.01
            }
        )
        
        # 并行向所有选中的客户端发送训练请求
        tasks = []
        for client_id in client_ids:
            if self.is_client_ready(client_id):
                task = self._train_client(client_id, training_request)
                tasks.append(task)
        
        # 等待所有训练完成
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        client_results = {}
        failed_clients = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Client {client_ids[i]} training failed: {result}")
                failed_clients.append(client_ids[i])
            else:
                client_results[client_ids[i]] = result
        
        # 聚合模型
        if client_results:
            aggregated_model = await self.aggregate_models(client_results)
        else:
            aggregated_model = None
        
        # 计算轮次统计
        if client_results:
            avg_loss = np.mean([r.loss for r in client_results.values()])
            avg_accuracy = np.mean([r.accuracy for r in client_results.values()])
        else:
            avg_loss, avg_accuracy = 0.0, 0.0
        
        round_result = {
            "round": round_num,
            "participants": client_ids,
            "successful_clients": list(client_results.keys()),
            "failed_clients": failed_clients,
            "aggregated_model": aggregated_model,
            "round_metrics": {
                "avg_loss": avg_loss,
                "avg_accuracy": avg_accuracy,
                "successful_count": len(client_results)
            }
        }
        
        return round_result
    
    async def test_dynamic_proxy_calls(self) -> Dict[str, Any]:
        """测试动态代理调用功能"""
        print("\n=== Testing Dynamic Proxy Calls ===")
        
        available_clients = self.get_available_clients()
        if not available_clients:
            print("No clients available for dynamic call testing")
            return {"error": "No clients available"}
        
        test_results = {}
        
        for client_id in available_clients[:2]:  # 测试前2个客户端
            print(f"\nTesting dynamic calls for client {client_id}:")
            proxy = self._proxy_manager.get_proxy(client_id)
            
            if proxy is None:
                print(f"No proxy found for client {client_id}")
                continue
            
            client_results = {}
            
            try:
                # 测试1: 调用自定义异步方法
                print("  1. Testing custom_method_for_testing...")
                result1 = await proxy.custom_method_for_testing("test_data", param2=42)
                client_results["custom_method"] = result1
                print(f"     ✓ Success: {result1}")
                
                # 测试2: 调用同步方法
                print("  2. Testing get_client_info...")
                result2 = await proxy.get_client_info()
                client_results["client_info"] = result2
                print(f"     ✓ Success: {result2}")
                
                # 测试3: 调用计算梯度方法
                print("  3. Testing compute_gradients...")
                result3 = await proxy.compute_gradients(loss_fn="cross_entropy")
                client_results["gradients"] = {
                    "loss_function": result3["loss_function"],
                    "gradient_norm": result3["gradient_norm"],
                    "client_id": result3["client_id"]
                }
                print(f"     ✓ Success: Gradient norm = {result3['gradient_norm']:.6f}")
                
                # 测试4: 调用数据统计方法
                print("  4. Testing get_data_statistics...")
                result4 = await proxy.get_data_statistics()
                client_results["data_stats"] = result4
                print(f"     ✓ Success: {result4}")
                
                # 测试5: 尝试调用不存在的方法（测试错误处理）
                print("  5. Testing non_existent_method (should fail)...")
                try:
                    result5 = await proxy.non_existent_method("test_param")
                    client_results["non_existent"] = result5
                    print(f"     ⚠ Unexpected success: {result5}")
                except Exception as e:
                    client_results["non_existent_error"] = str(e)
                    print(f"     ✓ Expected failure: {str(e)}")
                
                test_results[client_id] = client_results
                
            except Exception as e:
                print(f"     ✗ Error during testing: {str(e)}")
                test_results[client_id] = {"error": str(e)}
        
        print(f"\n=== Dynamic Proxy Call Testing Completed ===")
        print(f"Tested {len(test_results)} clients")
        
        return test_results
    
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
                    # 创建评估参数
                    evaluation_params = {
                        "batch_size": 32,
                        "test_data_size": 1000
                    }
                    task = proxy.evaluate(evaluation_params)
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
        
        # 如果准确率达到95%，停止训练
        if avg_accuracy >= 0.95:
            print(f"High accuracy achieved ({avg_accuracy:.4f} >= 0.95)")
            return True
        
        return False
    
    async def _train_client(self, client_id: str, request: TrainingRequest) -> TrainingResponse:
        """训练单个客户端"""
        # 这里通过ProxyManager自动获取正确的proxy
        proxy = self._proxy_manager.get_proxy(client_id)
        if proxy is None:
            raise RuntimeError(f"No proxy found for client {client_id}")
        
        return await proxy.train(request)
    
    async def _distribute_global_model(self, global_model: Dict[str, Any]):
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
                round_result = await self.train_round(round_num, available_clients)
                training_results.append(round_result)
                
                # 检查是否应该停止训练
                if self.should_stop_training(round_num, round_result):
                    break
                    
            except Exception as e:
                print(f"Error in round {round_num}: {e}")
                break
        
        # 最终评估
        final_evaluation = await self.evaluate_global_model()
        
        training_summary = {
            "completed_rounds": len(training_results),
            "total_rounds": self.max_rounds,
            "round_results": training_results,
            "final_evaluation": final_evaluation,
            "status": "completed" if len(training_results) > 0 else "failed"
        }
        
        print(f"\n=== Federated Training Completed ===")
        print(f"Completed {len(training_results)}/{self.max_rounds} rounds")
        print(f"Final accuracy: {final_evaluation.get('accuracy', 0.0):.4f}")
        
        return training_summary


async def create_server(config: Dict[str, Any]) -> FederationServer:
    """创建并启动服务端 - 使用注册表获取训练器"""
    print("Creating FederationServer...")
    print("🔍 使用注册表查找组件...")
    
    # 显示已注册的组件
    components = registry.list_all_components()
    print(f"已注册的训练器: {components['trainers']}")
    
    # 创建FederationServer
    server = FederationServer(config)
    
    # 创建一个初始的全局模型
    global_model = {
        "model_type": "simple_linear",
        "parameters": {"weights": np.random.randn(784, 10).tolist()},
        "metadata": {
            "model_size": 784 * 10,
            "initialization": "random",
            "created_at": "server_startup"
        }
    }
    
    # 从注册表获取训练器类
    trainer_cls = registry.get_trainer('DemoFedAvg')
    print(f"✅ 从注册表获取训练器: {trainer_cls.__name__}")
    
    # 创建TrainingConfig对象
    from fedcl.trainer.base_trainer import TrainingConfig
    training_config = TrainingConfig(
        max_rounds=config.get("trainer", {}).get("max_rounds", 3),
        min_clients=config.get("trainer", {}).get("min_clients", 2)
    )
    
    # 创建并启动训练器
    trainer = await server.initialize_with_trainer(
        trainer_cls, 
        global_model,
        training_config
    )
    
    # 启动服务端
    await server.start_server()
    
    return server


async def create_client(config: Dict[str, Any], client_id: str = None) -> FederationClient:
    """创建并启动客户端 - 使用注册表获取学习器"""
    print(f"Creating FederationClient...")
    print("🔍 使用注册表查找学习器...")
    
    # 显示已注册的学习器
    components = registry.list_all_components()
    print(f"已注册的学习器: {components['learners']}")
    
    # 创建FederationClient
    client = FederationClient(config, client_id)
    
    # 从注册表获取学习器类
    learner_cls = registry.get_learner('DemoMNIST')
    print(f"✅ 从注册表获取学习器: {learner_cls.__name__}")
    
    # 创建并初始化学习器
    learner = await client.initialize_with_learner(
        learner_cls,
        config.get("learner", {})
    )
    
    # 启动客户端
    await client.start_client()
    
    return client


async def run_complete_demo():
    """运行完整演示"""
    print("🚀 MOE-FedCL New Architecture Demo")
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
    base_config = {
        "mode": "memory",  # 使用内存模式简化演示
        "timeout": 30.0,
        "heartbeat_interval": 10.0
    }
    
    server_config = {
        **base_config,
        "trainer": {
            "min_clients": 2,
            "max_rounds": 3
        }
    }
    
    client_config = {
        **base_config,
        "learner": {},
        "stub_config": {
            "registration_retry_attempts": 3,
            "registration_retry_delay": 1.0
        }
    }
    
    # 启动服务端
    print("\n1. Starting Server...")
    server = await create_server(server_config)
    
    # 等待服务端完全启动
    await asyncio.sleep(1.0)
    
    # 启动多个客户端
    print("\n2. Starting Clients...")
    clients = []
    
    for i in range(3):
        client_id = f"demo_client_{i+1}"
        client = await create_client(client_config, client_id)
        clients.append(client)
        
        # 等待客户端注册
        await asyncio.sleep(0.5)
    
    # 等待所有客户端注册完成
    print("\n3. Waiting for client registration...")
    await asyncio.sleep(2.0)
    
    # 查看服务端状态
    server_status = server.get_server_status()
    print(f"\nServer Status: {server_status}")
    
    # 查看客户端状态
    for i, client in enumerate(clients):
        client_status = client.get_client_status()
        print(f"Client {i+1} Status: {client_status}")
    
    # 测试动态代理调用功能
    print("\n4. Testing Dynamic Proxy Calls...")
    try:
        dynamic_test_results = await server.trainer.test_dynamic_proxy_calls()
        print(f"\nDynamic Call Test Results Summary:")
        
        # 检查是否有全局错误
        if "error" in dynamic_test_results and len(dynamic_test_results) == 1:
            print(f"  ❌ {dynamic_test_results['error']}")
        else:
            # 处理每个客户端的结果
            for client_id, results in dynamic_test_results.items():
                if isinstance(results, dict):
                    if "error" in results:
                        print(f"  {client_id}: ❌ {results['error']}")
                    else:
                        successful_calls = sum(1 for k, v in results.items() if k != "non_existent_error" and not isinstance(v, str))
                        print(f"  {client_id}: ✅ {successful_calls} successful dynamic calls")
                else:
                    print(f"  {client_id}: ❌ Unexpected result type: {type(results)}")
    except Exception as e:
        print(f"Dynamic call testing failed: {e}")
        import traceback
        traceback.print_exc()
    
    # 运行联邦训练
    print("\n5. Running Federated Training...")
    try:
        training_results = await server.trainer.run_federated_training()
        print(f"\nTraining Results: {training_results}")
    except Exception as e:
        print(f"Training failed: {e}")
    
    # 清理资源
    print("\n6. Cleaning up...")
    
    # 停止客户端
    for client in clients:
        await client.stop_client()
    
    # 停止服务端
    await server.stop_server()
    
    print("\n✅ Demo completed successfully!")


if __name__ == "__main__":
    # 运行演示
    try:
        asyncio.run(run_complete_demo())
    except KeyboardInterrupt:
        print("\n❌ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
