"""
快速MNIST联邦学习演示
简化版本，用于快速验证系统功能
"""

import asyncio
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from examples.fedavg_mnist_trainer import FedAvgMNISTTrainer
from examples.mnist_learner import MNISTLearner
from datetime import datetime


async def quick_demo():
    """快速演示联邦学习功能"""
    print("⚡ 快速 MNIST 联邦学习演示")
    print("="*40)
    
    # 1. 创建训练器
    trainer = FedAvgMNISTTrainer(
        trainer_id="quick_server",
        model_config={"input_size": 784, "hidden_size": 64, "output_size": 10}  # 减小网络
    )
    trainer.local_epochs = 1  # 只训练1轮
    
    # 2. 创建2个客户端
    print("🏗️  创建客户端")
    clients = []
    for i in range(2):
        client_id = f"client_{i}"
        learner = MNISTLearner(
            client_id=client_id,
            training_config={"learning_rate": 0.1, "batch_size": 64}  # 更大学习率和批次
        )
        trainer.add_learner(client_id, learner)
        clients.append(client_id)
        print(f"   ✅ 客户端 {client_id}")
    
    # 3. 快速联邦学习
    print("\n🚀 开始训练")
    start_time = datetime.now()
    
    try:
        # 只运行1轮训练
        for round_num in range(1):
            print(f"\n🔄 第 {round_num + 1} 轮训练")
            
            # 选择所有客户端
            result = await trainer.train_round_with_learners(round_num, clients)
            
            # 显示结果
            agg_result = result["aggregation_result"]
            print(f"   ✅ 聚合完成: Loss={agg_result['average_loss']:.4f}, Acc={agg_result['average_accuracy']:.4f}")
        
        # 4. 最终评估
        print("\n📊 最终评估")
        global_model = await trainer.get_current_model()
        
        for client_id in clients:
            learner = trainer._direct_learners[client_id]
            eval_result = await learner.evaluate({
                "model": global_model["weights"],
                "test_data": True
            })
            print(f"   📱 {client_id}: Acc={eval_result['accuracy']:.4f}")
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        print(f"\n✅ 演示完成！总时间: {total_time:.2f}秒")
        
        return True
        
    except Exception as e:
        print(f"❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(quick_demo())
    if success:
        print("\n🎉 联邦学习系统验证成功！")
        print("   - FedAvg训练器正常工作")
        print("   - MNIST学习器正常工作") 
        print("   - 模型聚合功能正常")
        print("   - 评估功能正常")
    else:
        print("\n❌ 系统验证失败")
        sys.exit(1)
