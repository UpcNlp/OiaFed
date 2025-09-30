#!/usr/bin/env python3
"""
MOE-FedCL 配置驱动模式切换测试Demo
验证Memory模式和Process模式的无缝切换能力
"""

import asyncio
import sys
import yaml
from pathlib import Path
from typing import Dict, Any

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from fedcl.types import CommunicationMode, ModelData
from fedcl.trainer.base_trainer import TrainingConfig
from fedcl.api.decorators import learner, trainer
from fedcl.trainer.base_trainer import BaseTrainer
from fedcl.learner.base_learner import BaseLearner
from fedcl.federation.server import FederationServer
from fedcl.federation.client import FederationClient
from fedcl.factory.factory import ComponentFactory
from fedcl.utils.auto_logger import setup_auto_logging


# 使用装饰器注册组件（与模式无关）
@learner(name="DemoMNIST", version="1.0")
class SimpleMNISTLearner(BaseLearner):
    """简单MNIST学习器 - 支持所有通信模式"""
    
    def __init__(self, client_id: str, config: Dict = None, logger=None, **kwargs):
        """兼容不同调用签名：接受 config 和 logger（由 FederationClient 传入）
        config 可包含 local_data/model_config/training_config 等子项。
        """
        # 兼容性处理：从 config 中提取具体参数
        local_data = None
        model_config = None
        training_config = None
        if isinstance(config, dict):
            local_data = config.get('local_data')
            model_config = config.get('model_config')
            training_config = config.get('training_config')

        super().__init__(client_id, local_data=local_data, model_config=model_config, training_config=training_config)
        # 支持通过 config 指定初始权重
        if isinstance(config, dict) and 'model_weights' in config:
            self.model_weights = config.get('model_weights')
        else:
            self.model_weights = [0.1, 0.2, 0.3]  # 简单的模型权重
        
    async def train(self, training_params: Dict) -> Dict:
        """训练方法"""
        print(f"[{self.client_id}] Training with params: {training_params}")
        # 模拟训练
        await asyncio.sleep(0.1)
        
        # 模拟更新模型权重
        for i in range(len(self.model_weights)):
            self.model_weights[i] += 0.01
            
        # 返回标准训练结果
        return {
            "model_update": {"weights": self.model_weights},
            "loss": 0.1,
            "accuracy": 0.95,
            "samples": 100
        }
    
    async def evaluate(self, evaluation_params: Dict) -> Dict:
        """评估方法"""
        print(f"[{self.client_id}] Evaluating with params: {evaluation_params}")
        await asyncio.sleep(0.05)
        return {
            "accuracy": 0.88,
            "loss": 0.3,
            "samples": 200
        }

    async def get_local_model(self) -> Dict:
        """获取本地模型参数"""
        return {
            "model_id": f"local_model_{self.client_id}",
            "model_data": {"weights": self.model_weights.copy()},
            "metadata": {"client_id": self.client_id}
        }
    
    async def set_local_model(self, model_data: Dict) -> bool:
        """设置本地模型参数"""
        try:
            if "model_data" in model_data and "weights" in model_data["model_data"]:
                self.model_weights = model_data["model_data"]["weights"].copy()
                print(f"[{self.client_id}] Model updated with new weights: {self.model_weights}")
                return True
            return False
        except Exception as e:
            print(f"[{self.client_id}] Failed to set model: {e}")
            return False

    def custom_method_for_testing(self, param1: str, param2: int) -> Dict:
        """自定义测试方法"""
        print(f"[{self.client_id}] custom_method_for_testing called with param1={param1}, param2={param2}")
        return {
            "method_name": "custom_method_for_testing",
            "client_id": self.client_id,
            "param1": param1,
            "param2": param2,
            "result": f"Processed {param1} with value {param2}"
        }
    
    
    async def evaluate_global_model(self) -> dict:
        """评估全局模型"""
        print("🔍 Evaluating global model...")
        
        # 获取可用客户端代理
        available_proxies = await self.get_available_clients()
        
        if not available_proxies:
            return {"accuracy": 0.0, "loss": float('inf'), "participants": 0}
        
        # 选择部分客户端进行评估（这里选择所有可用客户端）
        eval_results = []
        for client_id, proxy in available_proxies.items():
            try:
                result = await proxy.evaluate({"model": self.global_model})
                eval_results.append(result)
                print(f"✅ Client {client_id} evaluation completed")
            except Exception as e:
                print(f"❌ Client {client_id} evaluation failed: {e}")
        
        # 计算全局评估指标
        if eval_results:
            avg_accuracy = sum(r.get("accuracy", 0) for r in eval_results) / len(eval_results)
            avg_loss = sum(r.get("loss", 0) for r in eval_results) / len(eval_results)
            
            return {
                "accuracy": avg_accuracy,
                "loss": avg_loss,
                "participants": len(eval_results)
            }
        
        return {"accuracy": 0.0, "loss": float('inf'), "participants": 0}
    
    def should_stop_training(self, round_num: int, round_result: dict) -> bool:
        """判断是否应该停止训练"""
        # 简单的停止条件：达到最大轮次
        max_rounds = getattr(self.training_config, 'max_rounds', 3)
        should_stop = round_num >= max_rounds
        
        if should_stop:
            print(f"🛑 Training stopped: reached max rounds ({max_rounds})")
        else:
            print(f"▶️ Training continues: round {round_num}/{max_rounds}")
        
        return should_stop


@trainer(name="DemoFedAvg", version="1.0", algorithms=["fedavg", "weighted_average"])
class FedAvgTrainer(BaseTrainer):
    """FedAvg 联邦训练器 - 支持自动代理管理"""
    
    def __init__(self, global_model: ModelData, training_config: TrainingConfig = None):
        super().__init__(global_model, training_config)
        self.current_round = 0  # 初始化当前轮次
        print("FedAvgTrainer initialized with automatic proxy management")
        print(f"Initial global model loaded: {global_model.get('model_id', 'unknown')}")
    
    async def train_round(self, round_num: int, client_ids: list) -> dict:
        """执行一轮联邦训练"""
        self.current_round = round_num  # 更新当前轮次
        print(f"🔄 Round {round_num}: Training with {len(client_ids)} clients")
        
        # 获取可用客户端代理
        all_proxies = self._proxy_manager.get_all_proxies()
        participating_clients = [cid for cid in client_ids if cid in all_proxies]
        
        print(f"Available proxies: {list(all_proxies.keys())}")
        print(f"Participating clients: {participating_clients}")
        
        if not participating_clients:
            return {
                "participants": client_ids,
                "successful_clients": [],
                "failed_clients": client_ids,
                "aggregated_model": self.global_model,
                "round_metrics": {"avg_loss": float('inf'), "avg_accuracy": 0.0},
                "training_time": 0.0
            }
        
        # 并发训练
        training_tasks = []
        client_task_map = {}
        
        for client_id in participating_clients:
            proxy = all_proxies[client_id]
            task = proxy.train({
                "global_model": self.global_model,
                "epochs": 1,
                "learning_rate": 0.01
            })
            training_tasks.append(task)
            client_task_map[task] = client_id
        
        # 收集结果
        client_results = {}
        failed_clients = []
        
        # 等待所有训练任务完成
        if training_tasks:
            results = await asyncio.gather(*training_tasks, return_exceptions=True)
            
            for task, result in zip(training_tasks, results):
                client_id = client_task_map[task]
                if isinstance(result, Exception):
                    print(f"❌ Client {client_id} training failed: {result}")
                    failed_clients.append(client_id)
                else:
                    client_results[client_id] = result
                    print(f"✅ Client {client_id} training completed")
        
        # 输出每个客户端的返回内容
        print("\n=== 客户端训练返回内容 ===")
        for cid, cres in client_results.items():
            print(f"客户端 {cid} 返回: {cres}")
        print("========================\n")
        # 聚合模型
        if client_results:
            aggregated_model = await self.aggregate_models(client_results)
            self.global_model = aggregated_model
        else:
            aggregated_model = self.global_model
        
        # 计算轮次指标
        avg_loss = sum(r.get("loss", 0) for r in client_results.values()) / len(client_results) if client_results else float('inf')
        avg_accuracy = sum(r.get("accuracy", 0) for r in client_results.values()) / len(client_results) if client_results else 0.0
        
        return {
            "participants": client_ids,
            "successful_clients": list(client_results.keys()),
            "failed_clients": failed_clients,
            "aggregated_model": aggregated_model,
            "round_metrics": {"avg_loss": avg_loss, "avg_accuracy": avg_accuracy},
            "training_time": 1.0
        }
    
    async def aggregate_models(self, client_results: dict) -> dict:
        """聚合客户端模型更新"""
        print(f"Aggregating models from {len(client_results)} clients")
        
        # 简单平均聚合
        aggregated_weights = [0.0, 0.0, 0.0]
        total_samples = 0
        
        for client_id, result in client_results.items():
            if "model_update" in result and "weights" in result["model_update"]:
                weights = result["model_update"]["weights"]
                samples = result.get("samples", 1)
                total_samples += samples
                
                for i, w in enumerate(weights):
                    aggregated_weights[i] += w * samples
        
        # 加权平均
        if total_samples > 0:
            aggregated_weights = [w / total_samples for w in aggregated_weights]
        
        return {
            "model_id": f"global_model_round_{self.current_round}",
            "model_data": {"weights": aggregated_weights},
            "metadata": {"total_samples": total_samples, "num_clients": len(client_results)}
        }
    
    async def evaluate_global_model(self) -> dict:
        """评估全局模型"""
        print("🔍 Evaluating global model...")
        
        # 获取可用客户端代理
        available_proxies = await self.get_available_clients()
        
        if not available_proxies:
            return {"accuracy": 0.0, "loss": float('inf'), "participants": 0}
        
        # 选择部分客户端进行评估（这里选择所有可用客户端）
        eval_results = []
        for client_id, proxy in available_proxies.items():
            try:
                result = await proxy.evaluate({"model": self.global_model})
                eval_results.append(result)
                print(f"✅ Client {client_id} evaluation completed")
            except Exception as e:
                print(f"❌ Client {client_id} evaluation failed: {e}")
        
        # 计算全局评估指标
        if eval_results:
            avg_accuracy = sum(r.get("accuracy", 0) for r in eval_results) / len(eval_results)
            avg_loss = sum(r.get("loss", 0) for r in eval_results) / len(eval_results)
            
            return {
                "accuracy": avg_accuracy,
                "loss": avg_loss,
                "participants": len(eval_results)
            }
        
        return {"accuracy": 0.0, "loss": float('inf'), "participants": 0}
    
    def should_stop_training(self, round_num: int, round_result: dict) -> bool:
        """判断是否应该停止训练"""
        # 简单的停止条件：达到最大轮次
        max_rounds = getattr(self.training_config, 'max_rounds', 3)
        should_stop = round_num >= max_rounds
        
        if should_stop:
            print(f"🛑 Training stopped: reached max rounds ({max_rounds})")
        else:
            print(f"▶️ Training continues: round {round_num}/{max_rounds}")
        
        return should_stop


class ConfigDrivenDemo:
    """配置驱动的联邦学习Demo"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        self.factory = ComponentFactory()
        
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def get_communication_mode(self) -> CommunicationMode:
        """从配置获取通信模式"""
        mode_str = self.config.get("federation", {}).get("mode", "memory")
        return CommunicationMode(mode_str)
    
    async def create_server(self) -> FederationServer:
        """根据配置创建服务器"""
        mode = self.get_communication_mode()
        server_config = self.config.get("federation", {}).get("server", {})
        
        # 创建全局模型
        model_config = self.config.get("model", {})
        global_model = {
            "model_id": "initial_global_model",
            "model_data": {"type": model_config.get("type", "simple_linear")},
            "metadata": model_config
        }
        
        # 创建训练配置
        training_config_data = self.config.get("training", {})
        training_config = TrainingConfig(
            max_rounds=training_config_data.get("rounds", 3),
            min_clients=training_config_data.get("client_selection", {}).get("min_clients", 2),
            client_selection_ratio=training_config_data.get("client_selection", {}).get("fraction", 1.0)
        )
        
        print(f"🔧 创建{mode.value.upper()}模式联邦服务器...")
        
        # 根据模式创建服务器
        if mode == CommunicationMode.MEMORY:
            server_id = "memory_server"
        elif mode == CommunicationMode.PROCESS:
            server_id = f"process_server_{server_config.get('port', 8000)}"
        else:  # NETWORK
            server_id = f"network_server_{server_config.get('host', 'localhost')}_{server_config.get('port', 8000)}"
        
        server = FederationServer(
            config={
                "mode": mode.value,
                "server_id": server_id,
                **server_config
            }
        )
        
        # 初始化服务器和训练器
        from fedcl.registry import registry
        trainer_class = registry.get_trainer("DemoFedAvg")
        trainer = await server.initialize_with_trainer(
            trainer_class=trainer_class,
            global_model=global_model,
            trainer_config=training_config  # 传递TrainingConfig对象而不是dict
        )
        
        await server.start_server()
        return server
    
    async def create_clients(self) -> list:
        """根据配置创建客户端"""
        mode = self.get_communication_mode()
        clients_config = self.config.get("federation", {}).get("clients", [])
        clients = []
        
        print(f"🔧 创建{len(clients_config)}个{mode.value.upper()}模式客户端...")
        
        for i, client_config in enumerate(clients_config):
            client_id = client_config.get("id", f"demo_client_{i+1}")
            
            if mode == CommunicationMode.PROCESS:
                # Process模式需要不同的端口，并使用时间戳确保唯一性
                import time
                timestamp = int(time.time() * 1000) % 100000  # 取时间戳后5位
                port = client_config.get("port", 8001 + i)
                full_client_id = f"process_client_{port}_{timestamp}"
            else:
                full_client_id = client_id
            
            # 创建客户端配置
            client_full_config = {
                "mode": mode.value,
                **client_config
            }
            
            # 创建客户端实例
            client = FederationClient(
                config=client_full_config,
                client_id=full_client_id
            )
            
            # 获取学习器类并初始化
            from fedcl.registry import registry
            learner_class = registry.get_learner("DemoMNIST")
            learner = await client.initialize_with_learner(
                learner_class=learner_class,
                learner_config=client_config.get("learner", {})
            )
            
            await client.start_client()
            clients.append(client)
            print(f"✅ 客户端 {full_client_id} 创建并启动成功")
            
        return clients
    
    async def run_demo(self):
        """运行配置驱动的联邦学习Demo"""
        mode = self.get_communication_mode()
        
        # 初始化日志系统
        setup_auto_logging()
        
        print(f"🚀 MOE-FedCL 配置驱动模式切换Demo")
        print(f"==================================================")
        print(f"📋 通信模式: {mode.value.upper()}")
        print(f"📋 配置文件: {self.config_path}")
        print()
        
        try:
            # 1. 创建服务器
            print("1. 创建服务器...")
            server = await self.create_server()
            print("✅ 服务器创建成功")
            
            # 2. 创建客户端
            print("\\n2. 创建客户端...")
            clients = await self.create_clients()
            print(f"✅ 成功创建{len(clients)}个客户端")
            
            # 3. 等待客户端注册
            print("\\n3. 等待客户端注册...")
            await asyncio.sleep(2)
            
            # 4. 检查注册状态
            available_clients = server.trainer.get_available_clients()
            print(f"📊 可用客户端数量: {len(available_clients)}")
            
            if len(available_clients) >= 2:
                # 5. 执行联邦训练
                print("\\n5. 开始联邦训练...")
                training_config = self.config.get("training", {})
                rounds = training_config.get("rounds", 2)
                
                for round_num in range(1, rounds + 1):
                    print(f"\\n--- 第 {round_num} 轮训练 ---")
                    
                    # 选择客户端（这里选择所有可用客户端）
                    selected_clients = available_clients[:min(len(available_clients), 3)]
                    print(f"选中客户端: {selected_clients}")
                    
                    # 执行训练轮
                    try:
                        round_result = await server.trainer.train_round(round_num, selected_clients)
                        print(f"✅ 第{round_num}轮训练完成，准确率: {round_result.get('accuracy', 'N/A')}")
                    except Exception as e:
                        print(f"❌ 第{round_num}轮训练失败: {e}")
                
                print("\\n🎉 联邦训练完成!")
            else:
                print("❌ 可用客户端不足，无法开始训练")
            
            # 6. 清理资源
            print("\\n6. 清理资源...")
            for client in clients:
                try:
                    await client.stop()
                except:
                    pass
            
            try:
                await server.stop()
            except:
                pass
            
            print("✅ 资源清理完成")
            
        except Exception as e:
            print(f"❌ Demo执行失败: {e}")
            import traceback
            traceback.print_exc()


async def main():
    """主函数"""
    if len(sys.argv) != 2:
        print("使用方法: python config_driven_demo.py <config_file>")
        print("示例:")
        print("  python config_driven_demo.py config/memory_demo_config.yaml")
        print("  python config_driven_demo.py config/process_demo_config.yaml")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    if not Path(config_path).exists():
        print(f"❌ 配置文件不存在: {config_path}")
        sys.exit(1)
    
    demo = ConfigDrivenDemo(config_path)
    await demo.run_demo()


if __name__ == "__main__":
    asyncio.run(main())
