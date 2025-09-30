import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import numpy as np
from typing import Dict, Any, List
from fedcl.fl.simple_abstract_trainer import AbstractFederationTrainer


class SimpleFedAvgTrainer(AbstractFederationTrainer):
    """简单的FedAvg训练器"""
    
    async def train(self) -> Dict[str, Any]:
        """训练实现"""
        num_rounds = self.config.get("num_rounds", 5)
        min_clients = self.config.get("min_clients", 2)
        
        self.logger.info(f"开始联邦训练，共 {num_rounds} 轮，最少 {min_clients} 个客户端")
        
        for round_num in range(num_rounds):
            self.logger.info(f"=== 第 {round_num + 1} 轮训练开始 ===")
            
            # 等待足够的客户端
            try:
                active_clients = await self.wait_for_clients(min_clients)
                self.logger.info(f"参与训练的客户端: {active_clients}")
            except TimeoutError as e:
                self.logger.error(f"客户端数量不足: {e}")
                break
            
            # 批量训练
            self.logger.info("开始客户端本地训练...")
            results = await self.learner_proxy.batch_train(
                active_clients,
                epochs=1,
                round_num=round_num
            )
            
            # 检查训练结果
            successful_clients = []
            for client_id, result in results.items():
                if isinstance(result, dict) and "error" not in result:
                    successful_clients.append(client_id)
                else:
                    self.logger.warning(f"客户端 {client_id} 训练失败: {result}")
            
            if not successful_clients:
                self.logger.error("没有客户端成功完成训练，结束训练")
                break
            
            # 收集权重
            self.logger.info("收集客户端模型权重...")
            weights = await self.learner_proxy.batch_get_weights(successful_clients)
            
            # 简单平均聚合
            try:
                global_weights = self._simple_average(weights)
                self.logger.info("模型权重聚合完成")
            except Exception as e:
                self.logger.error(f"权重聚合失败: {e}")
                # 使用第一个客户端的权重作为全局权重
                global_weights = next(iter(weights.values()))
            
            # 分发全局权重
            self.logger.info("分发全局模型权重...")
            await self.learner_proxy.batch_set_weights(successful_clients, global_weights)
            
            # 记录训练信息
            self.log_training_info(round_num, successful_clients, results)
            
            # 简单的评估（如果客户端支持）
            try:
                eval_results = await self.learner_proxy.batch_evaluate(successful_clients)
                avg_accuracy = self._calculate_average_accuracy(eval_results)
                if avg_accuracy is not None:
                    self.logger.info(f"第 {round_num + 1} 轮平均准确率: {avg_accuracy:.4f}")
            except Exception as e:
                self.logger.debug(f"评估失败（可能客户端不支持评估）: {e}")
        
        return self.get_training_summary()
    
    def _simple_average(self, weights_dict: Dict[str, Any]) -> Dict[str, Any]:
        """简单权重平均"""
        if not weights_dict:
            return {}
        
        # 获取所有客户端的权重
        client_weights = list(weights_dict.values())
        
        # 简化实现：只返回第一个客户端的权重
        # 实际应该做张量平均，但需要PyTorch支持
        first_weights = client_weights[0]
        
        # 如果权重是字典格式且包含torch tensor，可以尝试平均
        if isinstance(first_weights, dict):
            try:
                import torch
                averaged_weights = {}
                
                # 获取所有层的key
                keys = first_weights.keys()
                
                for key in keys:
                    # 收集所有客户端在这一层的权重
                    layer_weights = []
                    for client_weights in client_weights:
                        if key in client_weights:
                            layer_weights.append(client_weights[key])
                    
                    if layer_weights and hasattr(layer_weights[0], 'clone'):
                        # PyTorch tensor平均
                        averaged_weights[key] = sum(layer_weights) / len(layer_weights)
                    else:
                        # 如果不是tensor，使用第一个
                        averaged_weights[key] = layer_weights[0] if layer_weights else first_weights[key]
                
                return averaged_weights
                
            except ImportError:
                self.logger.debug("PyTorch未安装，使用第一个客户端权重")
            except Exception as e:
                self.logger.debug(f"权重平均失败，使用第一个客户端权重: {e}")
        
        return first_weights
    
    def _calculate_average_accuracy(self, eval_results: Dict[str, Any]) -> float:
        """计算平均准确率"""
        accuracies = []
        
        for client_id, result in eval_results.items():
            if isinstance(result, dict) and "accuracy" in result:
                try:
                    accuracies.append(float(result["accuracy"]))
                except (ValueError, TypeError):
                    continue
        
        if accuracies:
            return sum(accuracies) / len(accuracies)
        return None


# 简单的测试函数
async def test_simple_trainer():
    """测试简单训练器"""
    config = {
        "execution_mode": "local",
        "num_rounds": 3,
        "min_clients": 2,
        "heartbeat_timeout": 30.0
    }
    
    trainer = SimpleFedAvgTrainer(config)
    
    # 设置客户端
    client_configs = [
        {"client_id": "client_1", "learner_name": None},
        {"client_id": "client_2", "learner_name": None},
        {"client_id": "client_3", "learner_name": None}
    ]
    
    # 注册客户端
    active_clients = await trainer.setup_clients(client_configs)
    print(f"活跃客户端: {active_clients}")
    
    # 开始训练
    results = await trainer.train()
    print(f"训练完成: {results}")


if __name__ == "__main__":
    asyncio.run(test_simple_trainer())
