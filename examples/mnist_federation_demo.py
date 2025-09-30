"""
MNIST联邦学习端到端演示
使用FedAvg训练器和MNIST学习器进行手写数字识别
"""

import asyncio
import os
import sys
import json
from datetime import datetime
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from examples.fedavg_mnist_trainer import FedAvgMNISTTrainer
from examples.mnist_learner import MNISTLearner
from fedcl.federation.coordinator import FederationCoordinator
from fedcl.config.manager import ConfigManager
from fedcl.types import FederationConfig


async def simple_federation_demo(num_clients=3, num_rounds=2, clients_per_round=2, local_epochs=1):
    """简化的联邦学习演示"""
    print("🎯 简化版 MNIST 联邦学习演示")
    print("="*60)
    
    print(f"📋 实验配置:")
    print(f"   客户端数量: {num_clients}")
    print(f"   训练轮数: {num_rounds}")
    print(f"   每轮参与客户端: {clients_per_round}")
    print(f"   本地训练轮数: {local_epochs}")
    print("="*60)
    
    # 1. 创建训练器（服务器端）
    trainer_config = {
        "model_config": {
            "architecture": "simple_mlp",
            "input_size": 784,
            "hidden_size": 128,
            "output_size": 10,
            "learning_rate": 0.01
        },
        "aggregation_config": {
            "strategy": "fedavg",
            "weighted": True
        }
    }
    
    trainer = FedAvgMNISTTrainer(
        trainer_id="mnist_server",
        model_config=trainer_config["model_config"],
        aggregation_config=trainer_config["aggregation_config"]
    )
    
    # 设置训练参数
    trainer.local_epochs = local_epochs
    
    # 2. 创建学习器（客户端）
    print("\n🏗️  创建客户端学习器")
    
    for i in range(num_clients):
        client_id = f"mnist_client_{i}"
        
        # 每个客户端有不同的训练配置
        training_config = {
            "learning_rate": 0.01 + i * 0.002,  # 微调学习率
            "batch_size": 32,
            "local_epochs": local_epochs
        }
        
        learner = MNISTLearner(
            client_id=client_id,
            local_data=None,  # 将生成合成数据
            model_config=trainer_config["model_config"],
            training_config=training_config
        )
        
        # 添加到训练器
        trainer.add_learner(client_id, learner)
        print(f"   ✅ 创建客户端: {client_id}")
    
    # 3. 执行联邦学习
    print("\n� 开始联邦学习训练")
    print("="*60)
    
    try:
        start_time = datetime.now()
        
        # 执行联邦学习轮次
        all_results = []
        all_clients = [f"mnist_client_{i}" for i in range(num_clients)]
        
        for round_num in range(num_rounds):
            # 随机选择参与客户端
            import random
            selected_clients = random.sample(all_clients, min(clients_per_round, len(all_clients)))
            
            # 执行训练轮次
            round_result = await trainer.train_round_with_learners(round_num, selected_clients)
            all_results.append(round_result)
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        print("\n🎉 联邦学习完成!")
        print("="*60)
        print(f"总训练时间: {total_time:.2f}秒")
        
        # 4. 展示结果
        results = {
            "round_results": all_results,
            "federation_stats": {
                "completed_rounds": num_rounds,
                "total_training_time": total_time,
                "total_participating_clients": num_clients
            }
        }
        await display_results(results, list(trainer._direct_learners.values()), trainer)
        
        # 5. 执行最终评估
        print("\n📊 执行最终模型评估")
        await final_evaluation(trainer, list(trainer._direct_learners.values()))
        
    except Exception as e:
        print(f"❌ 联邦学习执行失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ 演示完成")


async def main():
    """主函数：演示完整的MNIST联邦学习流程"""
    
    print("🚀 启动 MNIST 联邦学习演示")
    
    # 直接运行简化版演示
    await simple_federation_demo(
        num_clients=5,
        num_rounds=3,
        clients_per_round=3,
        local_epochs=2
    )


async def display_results(results: dict, learners: list, trainer):
    """展示训练结果"""
    
    print("\n📈 训练结果汇总")
    print("-" * 50)
    
    if "round_results" in results:
        for round_num, round_result in enumerate(results["round_results"]):
            print(f"\n🔄 第 {round_num + 1} 轮:")
            
            if "training_results" in round_result:
                # 显示客户端训练结果
                for client_result in round_result["training_results"]:
                    client_id = client_result.get("client_id", "unknown")
                    loss = client_result.get("loss", 0.0)
                    accuracy = client_result.get("accuracy", 0.0)
                    samples = client_result.get("samples_count", 0)
                    
                    print(f"   📱 {client_id}: Loss={loss:.4f}, Acc={accuracy:.4f}, Samples={samples}")
            
            if "aggregation_result" in round_result:
                # 显示聚合结果
                agg_result = round_result["aggregation_result"]
                avg_loss = agg_result.get("average_loss", 0.0)
                avg_accuracy = agg_result.get("average_accuracy", 0.0)
                participating_clients = agg_result.get("participating_clients", 0)
                
                print(f"   🎯 聚合结果: Avg Loss={avg_loss:.4f}, Avg Acc={avg_accuracy:.4f}, 参与客户端={participating_clients}")
    
    # 显示整体统计
    if "federation_stats" in results:
        stats = results["federation_stats"]
        print(f"\n📊 联邦统计:")
        print(f"   完成轮数: {stats.get('completed_rounds', 0)}")
        print(f"   总训练时间: {stats.get('total_training_time', 0.0):.2f}秒")
        print(f"   参与客户端总数: {stats.get('total_participating_clients', 0)}")


async def final_evaluation(trainer, learners):
    """执行最终评估"""
    
    print("🔍 全局模型评估:")
    
    # 获取最终的全局模型
    global_model = await trainer.get_current_model()
    
    # 在所有客户端上评估全局模型
    total_accuracy = 0.0
    total_loss = 0.0
    total_samples = 0
    
    print("\n📋 各客户端评估结果:")
    
    for learner in learners:
        try:
            # 使用全局模型进行评估
            eval_result = await learner.evaluate({
                "model": global_model["weights"] if "weights" in global_model else global_model,
                "test_data": True
            })
            
            accuracy = eval_result.get("accuracy", 0.0)
            loss = eval_result.get("loss", 0.0)
            samples = eval_result.get("samples_count", 0)
            
            print(f"   📱 {learner.client_id}: Loss={loss:.4f}, Acc={accuracy:.4f}, Samples={samples}")
            
            # 加权累计
            total_accuracy += accuracy * samples
            total_loss += loss * samples
            total_samples += samples
            
        except Exception as e:
            print(f"   ❌ {learner.client_id} 评估失败: {e}")
    
    # 计算全局平均性能
    if total_samples > 0:
        global_accuracy = total_accuracy / total_samples
        global_loss = total_loss / total_samples
        
        print(f"\n🌟 全局模型性能:")
        print(f"   全局准确率: {global_accuracy:.4f} ({global_accuracy*100:.2f}%)")
        print(f"   全局平均损失: {global_loss:.4f}")
        print(f"   总测试样本: {total_samples}")
    else:
        print("   ⚠️  无法计算全局性能")
    
    # 显示数据分布统计
    print(f"\n📊 数据分布分析:")
    for learner in learners:
        stats = learner.get_data_statistics()
        distribution = stats.get("label_distribution", {})
        preferred = stats.get("preferred_classes", [])
        
        print(f"   📱 {learner.client_id}:")
        print(f"      偏好类别: {preferred}")
        print(f"      数据分布: {distribution}")


async def test_individual_components():
    """测试各个组件的独立功能"""
    print("\n🧪 组件独立测试")
    print("-" * 30)
    
    # 测试MNIST学习器
    print("1️⃣ 测试 MNIST 学习器")
    learner = MNISTLearner("test_client")
    
    # 测试数据生成
    data_stats = learner.get_data_statistics()
    print(f"   数据统计: {data_stats}")
    
    # 测试本地训练
    training_result = await learner.train({
        "epochs": 1,
        "learning_rate": 0.01,
        "batch_size": 32,
        "round_num": 0
    })
    print(f"   训练结果: Loss={training_result['loss']:.4f}, Acc={training_result['accuracy']:.4f}")
    
    # 测试评估
    eval_result = await learner.evaluate({"test_data": True})
    print(f"   评估结果: Loss={eval_result['loss']:.4f}, Acc={eval_result['accuracy']:.4f}")
    
    print("✅ MNIST学习器测试完成")
    
    # 测试FedAvg训练器
    print("\n2️⃣ 测试 FedAvg 训练器")
    
    federation_config = FederationConfig(
        coordinator_id="test_coordinator",
        max_rounds=3,
        min_clients=1,
        client_selection="all"
    )
    
    trainer = FedAvgMNISTTrainer(
        trainer_id="test_trainer",
        model_config={"input_size": 784, "hidden_size": 128, "output_size": 10}
    )
    
    # 测试模型初始化
    model = await trainer.get_current_model()
    print(f"   模型结构: {list(model.keys()) if isinstance(model, dict) else type(model)}")
    
    # 测试聚合功能
    client_updates = [training_result]  # 使用之前的训练结果
    aggregation_result = await trainer.aggregate_updates(client_updates)
    print(f"   聚合结果: {list(aggregation_result.keys())}")
    
    print("✅ FedAvg训练器测试完成")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MNIST联邦学习演示")
    parser.add_argument("--test", action="store_true", help="运行组件测试")
    parser.add_argument("--simple", action="store_true", help="运行简化版演示")
    
    args = parser.parse_args()
    
    if args.test:
        # 运行组件测试
        asyncio.run(test_individual_components())
    elif args.simple:
        # 运行简化版演示（更少轮数和客户端）
        print("🎯 简化版 MNIST 联邦学习演示")
        
        # 修改全局配置为简化版本
        import __main__
        __main__.num_clients = 3
        __main__.num_rounds = 2
        __main__.clients_per_round = 2
        __main__.local_epochs = 1
        
        asyncio.run(main())
    else:
        # 运行完整演示
        asyncio.run(main())
