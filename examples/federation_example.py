# examples/federation_example.py
"""
联邦学习组件使用示例

展示如何使用FedCL框架的基础联邦组件，包括LocalTrainer、ModelManager和ClientManager。
支持伪联邦和真联邦两种模式。
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from omegaconf import DictConfig
from fedcl.federation import LocalTrainer, ModelManager, ClientManager
from fedcl.core.base_learner import BaseLearner
from fedcl.core.base_aggregator import BaseAggregator
from fedcl.core.execution_context import ExecutionContext
from loguru import logger


class SimpleLearner(BaseLearner):
    """简单学习器实现示例"""
    
    def train_task(self, task_data: DataLoader, task_id: int) -> dict:
        """训练任务实现"""
        return {"task_id": task_id, "status": "completed"}
    
    def evaluate_task(self, task_data: DataLoader, task_id: int) -> dict:
        """评估任务实现"""
        return {"task_id": task_id, "accuracy": 0.95}


class FedAvgAggregator(BaseAggregator):
    """FedAvg聚合器实现示例"""
    
    def aggregate(self, client_updates) -> dict:
        """聚合客户端更新"""
        if not client_updates:
            return {}
            
        # 简单平均聚合
        aggregated_update = {}
        num_clients = len(client_updates)
        
        for param_name in client_updates[0].keys():
            aggregated_update[param_name] = sum(
                update[param_name] for update in client_updates
            ) / num_clients
            
        return aggregated_update
    
    def weight_updates(self, updates) -> list:
        """计算权重"""
        return [1.0 / len(updates)] * len(updates)


class SimpleModel(nn.Module):
    """简单神经网络模型"""
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def create_sample_data(num_samples=100, input_size=10, num_classes=2):
    """创建样本数据"""
    X = torch.randn(num_samples, input_size)
    y = torch.randint(0, num_classes, (num_samples,))
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=16, shuffle=True)


def demonstration_pseudo_federation():
    """演示伪联邦模式"""
    logger.info("=== 伪联邦模式演示 ===")
    
    # 1. 配置
    config = DictConfig({
        "federation_mode": "pseudo",
        "max_clients": 5,
        "selection_strategy": "random",
        "device": "cpu",
        "learning_rate": 0.001,
        "checkpoint_dir": "checkpoints/demo"
    })
    
    # 2. 初始化组件
    logger.info("初始化联邦组件...")
    
    # 客户端管理器
    client_manager = ClientManager(config)
    
    # 学习器和聚合器
    context = ExecutionContext(config, "demo_experiment")
    learner = SimpleLearner(context, config)
    aggregator = FedAvgAggregator(context, config)
    
    # 本地训练器和模型管理器
    local_trainer = LocalTrainer(learner, config)
    model_manager = ModelManager(config, aggregator)
    
    # 3. 注册客户端
    logger.info("注册客户端...")
    clients = []
    for i in range(5):
        client_id = f"client_{i}"
        client_info = {
            "cpu_count": 4,
            "memory_gb": 8,
            "network_bandwidth": 100
        }
        success = client_manager.register_client(client_id, client_info)
        if success:
            clients.append(client_id)
            logger.info(f"客户端 {client_id} 注册成功")
    
    # 4. 初始化全局模型
    logger.info("初始化全局模型...")
    global_model = SimpleModel()
    model_manager.set_global_model(global_model)
    
    # 5. 模拟联邦训练轮次
    logger.info("开始联邦训练...")
    num_rounds = 3
    
    for round_id in range(num_rounds):
        logger.info(f"\n--- 轮次 {round_id + 1} ---")
        
        # 选择客户端
        selected_clients = client_manager.select_clients_for_round(3, round_id)
        logger.info(f"选中客户端: {selected_clients}")
        
        # 广播全局模型
        current_model = model_manager.get_current_model()
        broadcast_results = client_manager.broadcast_to_clients(
            {"model": "global_model_data"}, selected_clients
        )
        logger.info(f"模型广播结果: {broadcast_results}")
        
        # 模拟客户端本地训练
        client_updates = []
        for client_id in selected_clients:
            logger.info(f"客户端 {client_id} 开始本地训练...")
            
            # 创建客户端数据
            client_data = create_sample_data(50)
            
            # 本地训练
            training_metrics = local_trainer.train_epoch(current_model, client_data)
            logger.info(f"客户端 {client_id} 训练完成: {training_metrics}")
            
            # 计算模型更新
            old_params = local_trainer.get_model_parameters(global_model)
            new_params = local_trainer.get_model_parameters(current_model)
            
            # 创建模拟的模型更新
            update = {}
            for name in old_params:
                update[name] = torch.randn_like(old_params[name]) * 0.01
            
            client_updates.append(update)
        
        # 聚合更新
        logger.info("聚合客户端更新...")
        updated_model = model_manager.update_global_model(client_updates)
        
        # 评估全局模型
        test_data = create_sample_data(100)
        eval_metrics = local_trainer.evaluate_model(updated_model, test_data)
        logger.info(f"全局模型评估: {eval_metrics}")
        
        # 保存检查点
        if (round_id + 1) % 2 == 0:
            checkpoint_path = model_manager.save_checkpoint(
                round_id + 1, 
                {"round_metrics": eval_metrics}
            )
            logger.info(f"检查点已保存: {checkpoint_path}")
    
    # 6. 显示统计信息
    logger.info("\n=== 训练统计 ===")
    client_stats = client_manager.get_summary_statistics()
    logger.info(f"客户端管理统计: {client_stats}")
    
    model_stats = model_manager.get_summary_statistics()
    logger.info(f"模型管理统计: {model_stats}")
    
    training_stats = local_trainer.get_training_stats()
    logger.info(f"训练统计: {training_stats}")


def demonstration_client_management():
    """演示客户端管理功能"""
    logger.info("\n=== 客户端管理演示 ===")
    
    config = DictConfig({
        "federation_mode": "pseudo",
        "max_clients": 10,
        "selection_strategy": "capability_based",
        "client_timeout": 60
    })
    
    client_manager = ClientManager(config)
    
    # 注册不同能力的客户端
    clients_info = [
        {"client_id": "powerful_client", "cpu_count": 8, "memory_gb": 16, "network_bandwidth": 1000},
        {"client_id": "medium_client", "cpu_count": 4, "memory_gb": 8, "network_bandwidth": 100},
        {"client_id": "weak_client", "cpu_count": 2, "memory_gb": 4, "network_bandwidth": 50},
        {"client_id": "mobile_client", "cpu_count": 1, "memory_gb": 2, "network_bandwidth": 10}
    ]
    
    for client_info in clients_info:
        client_id = client_info.pop("client_id")
        client_manager.register_client(client_id, client_info)
        logger.info(f"注册客户端: {client_id}")
    
    # 测试不同选择策略
    strategies = ["random", "round_robin", "capability_based"]
    
    for strategy in strategies:
        logger.info(f"\n使用策略: {strategy}")
        client_manager.set_client_selection_strategy(strategy)
        
        for round_id in range(3):
            selected = client_manager.select_clients_for_round(2, round_id)
            logger.info(f"轮次 {round_id}: 选中 {selected}")
    
    # 客户端状态管理
    logger.info("\n客户端状态管理:")
    client_manager.update_client_status("powerful_client", client_manager.clients["powerful_client"].status.__class__.TRAINING)
    
    # 故障处理
    client_manager.handle_client_failure("weak_client", Exception("连接超时"))
    
    # 显示详细信息
    for client_id in ["powerful_client", "weak_client"]:
        info = client_manager.get_client_info(client_id)
        stats = client_manager.get_client_statistics(client_id)
        logger.info(f"客户端 {client_id}: 信息={info['status']}, 统计={stats['failure_count']}")


def main():
    """主函数"""
    logger.info("FedCL联邦组件演示开始")
    
    try:
        # 演示伪联邦模式
        demonstration_pseudo_federation()
        
        # 演示客户端管理
        demonstration_client_management()
        
        logger.info("\n演示完成！")
        
    except Exception as e:
        logger.error(f"演示过程中出现错误: {e}")
        raise


if __name__ == "__main__":
    main()
