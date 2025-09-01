#!/usr/bin/env python3
"""
FedCL 新API使用示例

展示如何使用全新的简洁透明API进行联邦学习，
用户只需专注于算法逻辑，框架自动处理所有分布式细节。
"""

import fedcl


# 步骤1: 定义学习器 - 专注算法逻辑
@fedcl.learner("simple_mnist_learner")
class SimpleMNISTLearner:
    """简单的MNIST学习器示例"""
    
    def __init__(self, config, context):
        """初始化学习器"""
        self.config = config
        self.context = context
        print(f"✅ 学习器初始化: {config.get('learner', 'unknown')}")
    
    def train_task(self, task_data):
        """训练任务 - 专注算法逻辑，框架自动处理分布式细节"""
        print("🔄 正在训练...")
        
        # 用户专注于联邦学习算法
        # 框架自动处理：
        # - 获取全局模型权重
        # - 处理数据分发
        # - 上传模型更新
        # - 与其他客户端通信
        
        # 模拟训练过程
        accuracy = 0.85 + (self.context.get_state("current_round", 0) * 0.02)
        loss = 1.0 - accuracy
        
        print(f"📊 训练完成 - 准确率: {accuracy:.3f}, 损失: {loss:.3f}")
        
        return {
            "accuracy": accuracy,
            "loss": loss,
            "samples": 1000
        }
    
    def evaluate_task(self, task_data):
        """评估任务"""
        print("📈 正在评估...")
        
        # 模拟评估过程
        accuracy = 0.83 + (self.context.get_state("current_round", 0) * 0.01)
        
        print(f"✅ 评估完成 - 准确率: {accuracy:.3f}")
        
        return {"accuracy": accuracy}


# 步骤2: 定义聚合器（可选，有默认实现）
@fedcl.aggregator("simple_weighted_avg")
class SimpleWeightedAvgAggregator:
    """简单的加权平均聚合器"""
    
    def aggregate(self, client_updates):
        """聚合客户端更新 - 专注聚合算法，框架自动处理通信"""
        print("🔀 正在聚合客户端更新...")
        
        # 用户专注于聚合算法逻辑
        # 框架自动处理：
        # - 收集所有客户端更新
        # - 处理网络通信
        # - 分发聚合结果
        
        total_samples = sum(update.get("samples", 0) for update in client_updates)
        
        # 加权平均
        weighted_accuracy = sum(
            update.get("accuracy", 0) * update.get("samples", 0) / total_samples
            for update in client_updates
        )
        
        print(f"📊 聚合完成 - 全局准确率: {weighted_accuracy:.3f}")
        
        return {
            "global_accuracy": weighted_accuracy,
            "total_samples": total_samples
        }


def main():
    """主函数 - 展示极简API使用"""
    print("🚀 FedCL 透明联邦学习API示例")
    print("=" * 50)
    
    # 展示1: 最简单的一行代码启动
    print("\n📚 示例1: 一行代码启动联邦学习")
    try:
        result = fedcl.train(
            learner="simple_mnist_learner",
            dataset="mnist",
            num_clients=3,
            num_rounds=5
        )
        
        print("\n🎉 训练完成!")
        print(f"📊 最终指标: 平均准确率 {result.average_accuracy:.3f}")
        print(f"⏱️  训练用时: {result.training_time:.2f}秒")
        print(f"🔄 训练轮次: {result.total_rounds}")
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
    
    # 展示2: 使用自定义聚合器
    print("\n📚 示例2: 使用自定义聚合器")
    try:
        result = fedcl.train(
            learner="simple_mnist_learner",
            aggregator="simple_weighted_avg",  # 使用自定义聚合器
            dataset="mnist",
            num_clients=2,
            num_rounds=3
        )
        
        print("\n🎉 自定义聚合训练完成!")
        print(f"📊 结果: {result.final_metrics}")
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
    
    # 展示3: 查看注册的组件
    print("\n📚 示例3: 查看已注册的组件")
    components = fedcl.list_components()
    print("📋 已注册的组件:")
    for comp_type, comp_list in components.items():
        print(f"  {comp_type}: {comp_list}")
    
    # 展示4: 获取组件详细信息
    print("\n📚 示例4: 获取组件详细信息")
    info = fedcl.get_component_info("learner", "simple_mnist_learner")
    if info:
        print("ℹ️  学习器信息:")
        print(f"  名称: {info['name']}")
        print(f"  类型: {info['type']}")
        print(f"  类名: {info['class']}")
    
    print("\n🎯 总结:")
    print("✅ 用户只需关心学习器和聚合器的算法逻辑")
    print("✅ 框架自动处理所有分布式细节（权重、梯度、通信等）")
    print("✅ 真联邦和伪联邦对用户完全透明")
    print("✅ 一行代码即可启动复杂的联邦学习任务")


if __name__ == "__main__":
    main()