"""
FedAvg MNIST 训练器实现
基于手写数字识别的联邦平均算法实现
"""

import asyncio
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import copy

from fedcl.trainer.trainer import BaseTrainer, TrainingConfig
from fedcl.learner.proxy import LearnerProxy
from fedcl.types import ModelData, RoundResult
from fedcl.exceptions import TrainingError


class FedAvgMNISTTrainer(BaseTrainer):
    """基于MNIST的FedAvg训练器实现"""
    
    def __init__(self,
                 trainer_id: str,
                 model_config: Optional[Dict[str, Any]] = None,
                 aggregation_config: Optional[Dict[str, Any]] = None):
        """初始化FedAvg训练器"""
        # 初始化一个空的learner_proxies字典，后续会动态添加
        learner_proxies = {}
        
        # 初始化全局模型
        global_model = self._initialize_global_model(model_config)
        
        # 创建训练配置
        training_config = TrainingConfig(
            max_rounds=10,
            min_clients=2,
            client_selection_ratio=1.0,
            round_timeout=300.0,
            convergence_threshold=0.001,
            patience=5
        )
        
        super().__init__(learner_proxies, global_model, training_config)
        
        self.trainer_id = trainer_id
        self.model_config = model_config or {}
        self.aggregation_config = aggregation_config or {"strategy": "fedavg", "weighted": True}
        
        # FedAvg特定配置
        self.local_epochs = 1  # 本地训练轮数
        self.learning_rate = 0.01  # 学习率
        self.batch_size = 32  # 批次大小
        
        print(f"🚀 FedAvg MNIST Trainer {trainer_id} initialized")
    
    def _initialize_global_model(self, model_config: Optional[Dict[str, Any]] = None) -> ModelData:
        """初始化全局模型（简单的两层神经网络）"""
        # 输入层: 784 (28x28) -> 隐藏层: 128 -> 输出层: 10
        np.random.seed(42)  # 确保可复现性
        
        input_size = model_config.get("input_size", 784) if model_config else 784
        hidden_size = model_config.get("hidden_size", 128) if model_config else 128
        output_size = model_config.get("output_size", 10) if model_config else 10
        
        model = {
            "weights": {
                "W1": np.random.normal(0, 0.1, (input_size, hidden_size)).tolist(),
                "b1": np.zeros(hidden_size).tolist(),
                "W2": np.random.normal(0, 0.1, (hidden_size, output_size)).tolist(),
                "b2": np.zeros(output_size).tolist()
            },
            "model_version": 1,
            "architecture": "simple_nn",
            "input_shape": [input_size],
            "output_shape": [output_size],
            "created_at": datetime.now().isoformat()
        }
        
        print("🧠 初始化全局模型 (784->128->10)")
        return model
    
    def add_learner(self, client_id: str, learner):
        """添加学习器（用于演示）"""
        # 这里简化处理，直接存储学习器引用
        if not hasattr(self, '_direct_learners'):
            self._direct_learners = {}
        self._direct_learners[client_id] = learner
        print(f"📱 添加客户端 {client_id}")
    
    async def get_current_model(self) -> ModelData:
        """获取当前全局模型"""
        return copy.deepcopy(self.global_model)
    
    async def train_round_with_learners(self, 
                                      round_num: int, 
                                      selected_clients: List[str]) -> RoundResult:
        """直接使用学习器进行训练轮次（用于演示）"""
        print(f"\n🔄 开始第 {round_num + 1} 轮训练，参与客户端: {selected_clients}")
        
        if not hasattr(self, '_direct_learners'):
            raise TrainingError("没有可用的学习器")
        
        # 分发全局模型到选中的客户端
        training_results = []
        for client_id in selected_clients:
            if client_id not in self._direct_learners:
                print(f"⚠️  客户端 {client_id} 不存在，跳过")
                continue
            
            learner = self._direct_learners[client_id]
            
            # 发送全局模型
            await learner.set_local_model(self.global_model)
            
            # 执行本地训练
            training_params = {
                "global_model": self.global_model["weights"],
                "epochs": self.local_epochs,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "round_num": round_num
            }
            
            try:
                result = await learner.train(training_params)
                training_results.append(result)
                print(f"   ✅ {client_id} 训练完成")
            except Exception as e:
                print(f"   ❌ {client_id} 训练失败: {e}")
        
        # 聚合模型更新
        if not training_results:
            raise TrainingError("没有成功的训练结果可用于聚合")
        
        aggregation_result = await self.aggregate_updates(training_results)
        
        # 更新全局模型
        self.global_model["weights"] = aggregation_result["aggregated_weights"]
        self.global_model["model_version"] += 1
        
        return {
            "round_num": round_num,
            "participating_clients": selected_clients,
            "training_results": training_results,
            "aggregation_result": aggregation_result,
            "global_model_version": self.global_model["model_version"]
        }
    
    async def aggregate_updates(self, client_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """聚合客户端更新（FedAvg算法）"""
        if not client_results:
            raise TrainingError("没有客户端结果用于聚合")
        
        print(f"🔄 开始FedAvg聚合，客户端数量: {len(client_results)}")
        
        # 计算权重（基于样本数量）
        total_samples = sum(result.get("samples_count", 1) for result in client_results)
        weights = [result.get("samples_count", 1) / total_samples for result in client_results]
        
        # 获取第一个模型作为模板
        first_model = client_results[0]["model_update"]
        aggregated_weights = {}
        
        # 对每个参数进行加权平均
        for param_name in first_model.keys():
            if isinstance(first_model[param_name], list):
                # 转换为numpy数组进行计算
                param_arrays = []
                for i, result in enumerate(client_results):
                    param_array = np.array(result["model_update"][param_name])
                    weighted_param = param_array * weights[i]
                    param_arrays.append(weighted_param)
                
                # 求和得到聚合结果
                aggregated_param = sum(param_arrays)
                aggregated_weights[param_name] = aggregated_param.tolist()
            else:
                # 处理标量参数
                aggregated_weights[param_name] = sum(
                    result["model_update"][param_name] * weights[i]
                    for i, result in enumerate(client_results)
                )
        
        # 计算聚合统计信息
        avg_loss = sum(result.get("loss", 0.0) * weights[i] for i, result in enumerate(client_results))
        avg_accuracy = sum(result.get("accuracy", 0.0) * weights[i] for i, result in enumerate(client_results))
        
        aggregation_result = {
            "aggregated_weights": aggregated_weights,
            "average_loss": avg_loss,
            "average_accuracy": avg_accuracy,
            "participating_clients": len(client_results),
            "total_samples": total_samples,
            "aggregation_method": "fedavg"
        }
        
        print(f"   ✅ 聚合完成: Avg Loss={avg_loss:.4f}, Avg Acc={avg_accuracy:.4f}")
        
        return aggregation_result
    
    # ==================== 抽象方法实现 ====================
    
    async def train_round(self, round_num: int, client_ids: List[str]) -> Dict[str, Any]:
        """执行一轮联邦训练"""
        if hasattr(self, '_direct_learners'):
            return await self.train_round_with_learners(round_num, client_ids)
        else:
            # 使用代理的标准实现
            return await super().train_round(round_num, client_ids)
    
    async def aggregate_models(self, client_results: Dict[str, Any]) -> Dict[str, Any]:
        """聚合客户端模型（FedAvg算法）"""
        if not client_results:
            raise TrainingError("没有客户端结果用于聚合")
        
        # 将字典转换为列表格式，以兼容我们的aggregate_updates方法
        results_list = list(client_results.values())
        aggregation_result = await self.aggregate_updates(results_list)
        
        return aggregation_result["aggregated_weights"]
    
    async def evaluate_global_model(self) -> Dict[str, Any]:
        """评估全局模型"""
        if hasattr(self, '_direct_learners'):
            # 使用所有客户端评估全局模型
            eval_results = []
            
            for client_id, learner in self._direct_learners.items():
                try:
                    result = await learner.evaluate({
                        "model": self.global_model["weights"] if "weights" in self.global_model else self.global_model,
                        "test_data": True
                    })
                    eval_results.append(result)
                except Exception as e:
                    print(f"评估客户端 {client_id} 失败: {e}")
            
            if eval_results:
                total_samples = sum(r.get("samples_count", 1) for r in eval_results)
                weighted_accuracy = sum(r.get("accuracy", 0.0) * r.get("samples_count", 1) for r in eval_results) / total_samples
                weighted_loss = sum(r.get("loss", 0.0) * r.get("samples_count", 1) for r in eval_results) / total_samples
                
                return {
                    "accuracy": weighted_accuracy,
                    "loss": weighted_loss,
                    "samples_count": total_samples,
                    "participants": len(eval_results)
                }
        
        return {"accuracy": 0.0, "loss": float('inf'), "participants": 0}
    
    def should_stop_training(self, round_num: int, round_result: Dict[str, Any]) -> bool:
        """判断是否应该停止训练"""
        # 基于轮数的停止条件
        max_rounds = getattr(self.training_config, 'max_rounds', 10) if self.training_config else 10
        if round_num >= max_rounds - 1:  # round_num是从0开始的
            return True
        
        # 基于收敛的停止条件（可选）
        if "aggregation_result" in round_result:
            avg_accuracy = round_result["aggregation_result"].get("average_accuracy", 0.0)
            if avg_accuracy > 0.95:  # 95%准确率时停止
                print(f"🎯 达到目标准确率 {avg_accuracy:.4f}，提前停止训练")
                return True
        
        return False
    
    async def train_round(self, round_num: int, client_ids: List[str]) -> RoundResult:
        """执行一轮FedAvg训练"""
        print(f"\n🔄 开始第 {round_num} 轮 FedAvg 训练")
        print(f"   参与客户端: {client_ids}")
        
        round_start_time = datetime.now()
        
        # 准备训练参数
        training_params = {
            "global_model": self.global_model["weights"],
            "epochs": self.local_epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "round_num": round_num
        }
        
        # 并行执行客户端训练
        client_results = []
        successful_clients = []
        failed_clients = []
        
        # 创建并发训练任务
        training_tasks = []
        for client_id in client_ids:
            if client_id in self.learner_proxies:
                proxy = self.learner_proxies[client_id]
                task = asyncio.create_task(
                    self._train_client(client_id, proxy, training_params)
                )
                training_tasks.append((client_id, task))
        
        # 等待所有客户端训练完成
        for client_id, task in training_tasks:
            try:
                result = await task
                if result["status"] == "success":
                    client_results.append(result)
                    successful_clients.append(client_id)
                    print(f"   ✅ 客户端 {client_id}: Loss={result['loss']:.4f}, Acc={result['accuracy']:.4f}")
                else:
                    failed_clients.append(client_id)
                    print(f"   ❌ 客户端 {client_id}: {result['error']}")
            except Exception as e:
                failed_clients.append(client_id)
                print(f"   ❌ 客户端 {client_id}: 训练异常 - {str(e)}")
        
        if not successful_clients:
            raise TrainingError(f"Round {round_num}: 没有客户端成功完成训练")
        
        # 执行FedAvg聚合
        print(f"🔗 聚合 {len(successful_clients)} 个客户端的模型更新")
        aggregated_weights = await self._fedavg_aggregate(client_results)
        
        # 更新全局模型
        self.global_model["weights"] = aggregated_weights
        self.global_model["model_version"] += 1
        
        # 计算轮次指标
        avg_loss = np.mean([r["loss"] for r in client_results])
        avg_accuracy = np.mean([r["accuracy"] for r in client_results])
        total_samples = sum([r["samples_count"] for r in client_results])
        
        round_end_time = datetime.now()
        round_duration = (round_end_time - round_start_time).total_seconds()
        
        round_result = {
            "round": round_num,
            "participants": client_ids,
            "successful_clients": successful_clients,
            "failed_clients": failed_clients,
            "aggregated_model": self.global_model,
            "round_metrics": {
                "avg_loss": float(avg_loss),
                "avg_accuracy": float(avg_accuracy),
                "total_samples": int(total_samples),
                "round_duration": round_duration,
                "convergence_metric": float(avg_loss)  # 使用损失作为收敛指标
            },
            "client_results": client_results
        }
        
        print(f"📊 第 {round_num} 轮完成: Loss={avg_loss:.4f}, Acc={avg_accuracy:.4f}, 用时={round_duration:.2f}s")
        
        # 更新训练历史
        self.training_status.round_results.append(round_result)
        
        return round_result
    
    async def _train_client(self, client_id: str, proxy: LearnerProxy, training_params: Dict[str, Any]) -> Dict[str, Any]:
        """训练单个客户端"""
        try:
            # 调用客户端训练
            result = await proxy.train(training_params)
            
            return {
                "client_id": client_id,
                "status": "success",
                "model_update": result.get("model_update", result.get("weights")),
                "loss": result.get("loss", 0.0),
                "accuracy": result.get("accuracy", 0.0),
                "samples_count": result.get("samples_count", result.get("samples", 0)),
                "training_time": result.get("training_time", 0.0)
            }
        except Exception as e:
            return {
                "client_id": client_id,
                "status": "failed",
                "error": str(e),
                "loss": float('inf'),
                "accuracy": 0.0,
                "samples_count": 0,
                "training_time": 0.0
            }
    
    async def _fedavg_aggregate(self, client_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """FedAvg算法：根据样本数加权平均聚合模型"""
        
        if not client_results:
            return self.global_model["weights"]
        
        # 提取模型更新和样本数
        model_updates = []
        sample_counts = []
        
        for result in client_results:
            if result["status"] == "success":
                model_updates.append(result["model_update"])
                sample_counts.append(result["samples_count"])
        
        if not model_updates:
            return self.global_model["weights"]
        
        # 计算权重（基于样本数的加权平均）
        total_samples = sum(sample_counts)
        weights = [count / total_samples for count in sample_counts]
        
        print(f"   样本权重: {[f'{w:.3f}' for w in weights]}")
        
        # 执行加权聚合
        aggregated = {}
        
        # 获取第一个模型的结构作为模板
        first_model = model_updates[0]
        
        for param_name in first_model:
            if isinstance(first_model[param_name], list):
                # 处理权重矩阵
                param_arrays = [np.array(update[param_name]) for update in model_updates]
                
                # 加权平均
                weighted_sum = np.zeros_like(param_arrays[0])
                for weight, param_array in zip(weights, param_arrays):
                    weighted_sum += weight * param_array
                
                aggregated[param_name] = weighted_sum.tolist()
            else:
                # 处理标量参数
                weighted_sum = sum(weight * update[param_name] for weight, update in zip(weights, model_updates))
                aggregated[param_name] = weighted_sum
        
        return aggregated
    
    def select_clients_for_round(self, round_num: int) -> List[str]:
        """选择参与当前轮次的客户端"""
        available_clients = list(self.learner_proxies.keys())
        
        # 计算要选择的客户端数量
        num_clients = len(available_clients)
        num_selected = max(1, int(num_clients * self.training_config.client_selection_ratio))
        
        if self.training_config.client_selection == "all":
            return available_clients
        elif self.training_config.client_selection == "random":
            import random
            return random.sample(available_clients, min(num_selected, num_clients))
        else:
            # 默认选择所有客户端
            return available_clients
    
    async def check_client_readiness(self, client_ids: List[str]) -> Dict[str, bool]:
        """检查客户端就绪状态"""
        readiness = {}
        
        for client_id in client_ids:
            if client_id in self.learner_proxies:
                try:
                    proxy = self.learner_proxies[client_id]
                    # 简单的ping检查
                    await asyncio.wait_for(proxy.ping(), timeout=5.0)
                    readiness[client_id] = True
                except:
                    readiness[client_id] = False
            else:
                readiness[client_id] = False
        
        return readiness
    
    async def initialize(self) -> bool:
        """初始化训练器"""
        print("🔧 初始化 FedAvg MNIST Trainer")
        
        # 检查全局模型
        if not self.global_model:
            self.global_model = self._initialize_global_model()
        
        # 检查客户端连接
        available_clients = []
        for client_id, proxy in self.learner_proxies.items():
            try:
                await asyncio.wait_for(proxy.ping(), timeout=5.0)
                available_clients.append(client_id)
                print(f"   ✅ 客户端 {client_id} 连接正常")
            except:
                print(f"   ❌ 客户端 {client_id} 连接失败")
        
        if len(available_clients) < self.training_config.min_clients:
            print(f"   ❌ 可用客户端数量不足: {len(available_clients)} < {self.training_config.min_clients}")
            return False
        
        print(f"✅ FedAvg Trainer 初始化完成，可用客户端: {len(available_clients)}")
        return True
    
    def should_stop_training(self, round_num: int, round_result: RoundResult) -> bool:
        """判断是否停止训练"""
        
        # 检查最大轮数
        if round_num >= self.training_config.max_rounds:
            print(f"🛑 达到最大训练轮数: {round_num}")
            return True
        
        # 检查收敛条件
        if len(self.training_status.round_results) >= 2:
            current_loss = round_result["round_metrics"]["avg_loss"]
            previous_loss = self.training_status.round_results[-2]["round_metrics"]["avg_loss"]
            
            # 检查损失改善
            loss_improvement = abs(previous_loss - current_loss)
            
            if loss_improvement < self.training_config.convergence_threshold:
                self.training_status.patience_counter += 1
                print(f"🔍 损失改善微小 ({loss_improvement:.6f} < {self.training_config.convergence_threshold})")
                
                if self.training_status.patience_counter >= self.training_config.patience:
                    print(f"🛑 训练收敛: 连续 {self.training_config.patience} 轮无明显改善")
                    return True
            else:
                self.training_status.patience_counter = 0
        
        # 检查准确率阈值（可选）
        current_accuracy = round_result["round_metrics"]["avg_accuracy"]
        if current_accuracy >= 0.99:  # 99%准确率
            print(f"🛑 达到目标准确率: {current_accuracy:.4f}")
            return True
        
        return False
    
    async def evaluate_global_model(self) -> Dict[str, Any]:
        """评估全局模型"""
        print("🔍 评估全局模型")
        
        # 在所有客户端上评估全局模型
        evaluation_params = {
            "model": self.global_model["weights"],
            "test_data": True  # 使用测试数据集
        }
        
        evaluation_results = []
        
        for client_id, proxy in self.learner_proxies.items():
            try:
                result = await proxy.evaluate(evaluation_params)
                evaluation_results.append({
                    "client_id": client_id,
                    "accuracy": result.get("accuracy", 0.0),
                    "loss": result.get("loss", float('inf')),
                    "samples_count": result.get("samples_count", 0)
                })
                print(f"   客户端 {client_id}: Acc={result.get('accuracy', 0):.4f}")
            except Exception as e:
                print(f"   ❌ 客户端 {client_id} 评估失败: {e}")
        
        if not evaluation_results:
            return {
                "accuracy": 0.0,
                "loss": float('inf'),
                "samples_count": 0,
                "message": "所有客户端评估失败"
            }
        
        # 聚合评估结果
        total_samples = sum(r["samples_count"] for r in evaluation_results)
        if total_samples > 0:
            weighted_accuracy = sum(r["accuracy"] * r["samples_count"] for r in evaluation_results) / total_samples
            weighted_loss = sum(r["loss"] * r["samples_count"] for r in evaluation_results) / total_samples
        else:
            weighted_accuracy = np.mean([r["accuracy"] for r in evaluation_results])
            weighted_loss = np.mean([r["loss"] for r in evaluation_results])
        
        global_eval_result = {
            "accuracy": float(weighted_accuracy),
            "loss": float(weighted_loss),
            "samples_count": int(total_samples),
            "client_count": len(evaluation_results),
            "evaluation_time": datetime.now().isoformat()
        }
        
        print(f"🌍 全局模型评估: Acc={weighted_accuracy:.4f}, Loss={weighted_loss:.4f}")
        
        return global_eval_result
