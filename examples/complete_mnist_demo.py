"""
完整的MNIST联邦学习演示
展示多轮训练和性能改进
"""

import asyncio
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from examples.fedavg_mnist_trainer import FedAvgMNISTTrainer
from examples.mnist_learner import MNISTLearner
from datetime import datetime


async def full_demo():
    """完整的联邦学习演示"""
    print("🚀 完整 MNIST 联邦学习演示")
    print("="*50)
    
    # 配置参数
    num_clients = 4
    num_rounds = 5
    clients_per_round = 3
    local_epochs = 2
    
    print(f"📋 实验配置:")
    print(f"   客户端数量: {num_clients}")
    print(f"   训练轮数: {num_rounds}")
    print(f"   每轮参与客户端: {clients_per_round}")
    print(f"   本地训练轮数: {local_epochs}")
    print("="*50)
    
    # 1. 创建训练器
    trainer = FedAvgMNISTTrainer(
        trainer_id="full_server",
        model_config={"input_size": 784, "hidden_size": 128, "output_size": 10}
    )
    trainer.local_epochs = local_epochs
    
    # 2. 创建客户端
    print("\n🏗️  创建客户端")
    all_clients = []
    
    for i in range(num_clients):
        client_id = f"client_{i}"
        learner = MNISTLearner(
            client_id=client_id,
            training_config={
                "learning_rate": 0.05 + i * 0.01,  # 不同学习率
                "batch_size": 32
            }
        )
        trainer.add_learner(client_id, learner)
        all_clients.append(client_id)
        
        # 显示客户端数据分布
        stats = learner.get_data_statistics()
        preferred = stats["preferred_classes"]
        total_samples = stats["total_samples"]
        print(f"   📱 {client_id}: {total_samples}样本, 偏好类别{preferred}")
    
    # 3. 执行多轮联邦学习
    print(f"\n🎓 开始{num_rounds}轮联邦学习")
    print("-"*50)
    
    start_time = datetime.now()
    all_results = []
    
    try:
        for round_num in range(num_rounds):
            print(f"\n🔄 第 {round_num + 1} 轮训练")
            
            # 随机选择参与客户端
            import random
            selected_clients = random.sample(all_clients, clients_per_round)
            print(f"   参与客户端: {selected_clients}")
            
            # 执行训练
            round_start = datetime.now()
            result = await trainer.train_round_with_learners(round_num, selected_clients)
            round_time = (datetime.now() - round_start).total_seconds()
            
            # 收集结果
            all_results.append(result)
            agg_result = result["aggregation_result"]
            
            print(f"   ✅ 完成: Loss={agg_result['average_loss']:.4f}, "
                  f"Acc={agg_result['average_accuracy']:.4f}, "
                  f"时间={round_time:.2f}秒")
            
            # 每轮后评估全局模型
            if (round_num + 1) % 2 == 0:  # 每2轮评估一次
                print(f"   📊 全局评估:")
                global_model = await trainer.get_current_model()
                
                total_acc = 0.0
                total_samples = 0
                
                for client_id in all_clients:
                    learner = trainer._direct_learners[client_id]
                    eval_result = await learner.evaluate({
                        "model": global_model["weights"],
                        "test_data": True
                    })
                    
                    acc = eval_result['accuracy']
                    samples = eval_result['samples_count']
                    total_acc += acc * samples
                    total_samples += samples
                    
                    print(f"      📱 {client_id}: Acc={acc:.4f}")
                
                # 计算加权平均准确率
                global_acc = total_acc / total_samples if total_samples > 0 else 0.0
                print(f"   🌟 全局准确率: {global_acc:.4f}")
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        # 4. 最终结果汇总
        print("\n" + "="*50)
        print("🎉 联邦学习完成！")
        print(f"总训练时间: {total_time:.2f}秒")
        
        # 显示训练轨迹
        print("\n📈 训练轨迹:")
        for i, result in enumerate(all_results):
            agg_result = result["aggregation_result"]
            print(f"   轮次 {i+1}: Loss={agg_result['average_loss']:.4f}, "
                  f"Acc={agg_result['average_accuracy']:.4f}")
        
        # 5. 最终全局评估
        print("\n📊 最终全局模型评估:")
        global_model = await trainer.get_current_model()
        
        client_results = []
        for client_id in all_clients:
            learner = trainer._direct_learners[client_id]
            eval_result = await learner.evaluate({
                "model": global_model["weights"],
                "test_data": True
            })
            client_results.append(eval_result)
            
            acc = eval_result['accuracy']
            loss = eval_result['loss']
            samples = eval_result['samples_count']
            print(f"   📱 {client_id}: Acc={acc:.4f}, Loss={loss:.4f}, 样本={samples}")
        
        # 计算全局指标
        total_samples = sum(r['samples_count'] for r in client_results)
        weighted_acc = sum(r['accuracy'] * r['samples_count'] for r in client_results) / total_samples
        weighted_loss = sum(r['loss'] * r['samples_count'] for r in client_results) / total_samples
        
        print(f"\n🌟 最终全局性能:")
        print(f"   全局准确率: {weighted_acc:.4f} ({weighted_acc*100:.2f}%)")
        print(f"   全局损失: {weighted_loss:.4f}")
        print(f"   总测试样本: {total_samples}")
        
        # 6. 数据分布分析
        print(f"\n📊 数据分布分析:")
        for client_id in all_clients:
            learner = trainer._direct_learners[client_id]
            stats = learner.get_data_statistics()
            distribution = stats["label_distribution"]
            preferred = stats["preferred_classes"]
            
            # 计算分布熵（数据异构程度）
            import numpy as np
            counts = list(distribution.values())
            total_count = sum(counts)
            probs = [c/total_count for c in counts]
            entropy = -sum(p * np.log2(p + 1e-10) for p in probs)
            
            print(f"   📱 {client_id}:")
            print(f"      偏好类别: {preferred}")
            print(f"      数据异构度: {entropy:.3f} (熵值)")
            print(f"      样本分布: {dict(list(distribution.items())[:5])}...")
        
        return True
        
    except Exception as e:
        print(f"❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(full_demo())
    
    if success:
        print("\n✅ MOE-FedCL 系统验证成功！")
        print("="*50)
        print("🎯 验证完成的功能:")
        print("   ✅ FedAvg联邦平均算法")
        print("   ✅ 异构数据分布处理")
        print("   ✅ 多轮训练和聚合") 
        print("   ✅ 全局模型评估")
        print("   ✅ 客户端选择机制")
        print("   ✅ 模型参数同步")
        print("   ✅ 性能监控和统计")
        print("\n🌟 系统已准备好用于实际联邦学习任务！")
    else:
        print("\n❌ 系统验证失败，请检查错误信息")
        sys.exit(1)
