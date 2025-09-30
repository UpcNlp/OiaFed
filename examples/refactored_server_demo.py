#!/usr/bin/env python3
"""
重构后的服务端架构演示

展示分离后的 FLServerManager 和 抽象TrainerBase 的使用
"""

import asyncio
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fedcl.utils.auto_logger import setup_auto_logging
from fedcl.fl.server import FLServerManager, TrainerBase, SimpleTrainer
from fedcl.comm import MemoryTransport
from fedcl.registry import register_class


class CustomTrainer(TrainerBase):
    """
    自定义训练器示例
    演示如何继承抽象TrainerBase类
    """
    
    def __init__(self, trainer_id: str, server_manager=None, transport=None):
        super().__init__(trainer_id, server_manager, transport)
        self.global_model = {'weights': [1.0, 2.0, 3.0], 'bias': 0.5}
        self.learning_rate = 0.01
    
    async def train_round(self, config=None) -> dict:
        """实现抽象方法：执行一轮训练"""
        self.logger.info("执行自定义训练轮次")
        
        # 获取配置参数
        lr = config.get('learning_rate', self.learning_rate) if config else self.learning_rate
        
        # 1. 调用所有学习器进行本地训练
        train_results = await self.call_all_learners('train', {'learning_rate': lr})
        
        # 2. 过滤有效结果
        valid_results = {k: v for k, v in train_results.items() if v is not None}
        if not valid_results:
            raise ValueError("没有学习器返回有效结果")
        
        # 3. 聚合模型
        models = [result.get('model', {}) for result in valid_results.values()]
        self.global_model = self.aggregate_models(models)
        
        # 4. 广播新模型
        await self.broadcast_to_learners({
            'type': 'model_update', 
            'global_model': self.global_model,
            'learning_rate': lr
        })
        
        return {
            'participating_learners': list(valid_results.keys()),
            'global_model': self.global_model.copy(),
            'learning_rate_used': lr,
            'total_samples': sum(r.get('samples', 0) for r in valid_results.values())
        }
    
    def aggregate_models(self, models):
        """实现抽象方法：聚合模型（加权平均）"""
        if not models:
            return self.global_model
            
        # 简化的加权平均（实际应用中会更复杂）
        aggregated = {'weights': [], 'bias': 0.0}
        
        # 聚合权重
        if models and 'weights' in models[0]:
            weight_dim = len(models[0]['weights'])
            aggregated['weights'] = [0.0] * weight_dim
            
            for model in models:
                if 'weights' in model:
                    for i, w in enumerate(model['weights']):
                        aggregated['weights'][i] += w / len(models)
        
        # 聚合偏置
        bias_values = [m.get('bias', 0.0) for m in models if 'bias' in m]
        if bias_values:
            aggregated['bias'] = sum(bias_values) / len(bias_values)
            
        return aggregated
    
    async def on_training_start(self):
        """训练开始前的钩子"""
        self.logger.info(f"自定义训练器开始训练，初始模型: {self.global_model}")
    
    async def on_round_end(self, round_num: int, result: dict):
        """每轮结束的钩子"""
        samples = result.get('total_samples', 0)
        self.logger.info(f"第 {round_num} 轮完成 - 总样本数: {samples}, 模型: {result['global_model']}")


async def demo_server_manager_standalone():
    """演示独立使用FLServerManager"""
    print("=" * 50)
    print("演示: 独立使用FLServerManager")
    print("=" * 50)
    
    transport = MemoryTransport()
    server_manager = FLServerManager("standalone_server", transport)
    
    # 添加回调函数
    async def on_registration(learner_id, proxy, message):
        print(f"✓ 客户端 {learner_id} 已注册")
    
    def on_disconnection(learner_id, message):
        print(f"✗ 客户端 {learner_id} 已断开连接")
    
    server_manager.add_registration_callback(on_registration)
    server_manager.add_disconnection_callback(on_disconnection)
    
    # 手动添加一些学习器代理
    proxy1 = server_manager.add_learner("client1")
    proxy2 = server_manager.add_learner("client2")
    
    print(f"✓ 已添加 {len(server_manager.learners)} 个学习器代理")
    print(f"✓ 学习器ID: {list(server_manager.learners.keys())}")
    
    # 模拟客户端注册
    await server_manager._handle_client_registration({
        'type': 'client_registration',
        'learner_id': 'client3'
    })
    
    print(f"✓ 注册后学习器数量: {len(server_manager.registered_learners)}")
    print("✓ FLServerManager 独立演示完成")


async def demo_abstract_trainer():
    """演示抽象TrainerBase的使用"""
    print("\n" + "=" * 50)
    print("演示: 抽象TrainerBase - 自定义训练器")
    print("=" * 50)
    
    # 注册自定义训练器
    register_class("CustomTrainer", CustomTrainer)
    
    transport = MemoryTransport()
    
    # 创建自定义训练器
    trainer = CustomTrainer("custom_trainer", transport=transport)
    
    # 手动添加学习器（模拟客户端连接）
    trainer.add_learner("client1")
    trainer.add_learner("client2") 
    trainer.add_learner("client3")
    
    print(f"✓ 自定义训练器创建成功: {trainer.trainer_id}")
    print(f"✓ 学习器数量: {len(trainer.learners)}")
    
    # 模拟训练流程（会失败，因为没有真实客户端）
    try:
        print("⚠ 尝试启动训练流程（预期会失败）...")
        await trainer.start_training(
            rounds=2,
            expected_learner_count=None,  # 不等待，直接用现有的
            config={'learning_rate': 0.05}
        )
    except Exception as e:
        print(f"✗ 训练失败（预期）: {e}")
    
    print("✓ 自定义训练器演示完成")


async def demo_simple_trainer():
    """演示SimpleTrainer的使用"""
    print("\n" + "=" * 50)  
    print("演示: SimpleTrainer - 内置训练器")
    print("=" * 50)
    
    transport = MemoryTransport()
    
    # 创建简单训练器
    trainer = SimpleTrainer("simple_trainer", transport=transport)
    
    # 手动添加学习器
    trainer.add_learner("learner_a")
    trainer.add_learner("learner_b")
    
    print(f"✓ 简单训练器创建成功: {trainer.trainer_id}")
    print(f"✓ 初始模型: {trainer.global_model}")
    print(f"✓ 学习器数量: {len(trainer.learners)}")
    
    # 展示训练器的基础功能
    print(f"✓ 注册的学习器: {trainer.registered_learners}")
    print(f"✓ 服务端管理器ID: {trainer.server_manager.server_id}")
    
    print("✓ SimpleTrainer 演示完成")


async def demo_separation_benefits():
    """演示架构分离的好处"""
    print("\n" + "=" * 50)
    print("演示: 架构分离的好处")
    print("=" * 50)
    
    # 1. 创建共享的服务端管理器
    transport = MemoryTransport()
    shared_manager = FLServerManager("shared_server", transport)
    
    # 添加一些学习器
    shared_manager.add_learner("client_1")
    shared_manager.add_learner("client_2")
    
    # 2. 不同的训练器可以共享同一个管理器
    trainer1 = SimpleTrainer("trainer_1", shared_manager)
    trainer2 = CustomTrainer("trainer_2", shared_manager)
    
    print(f"✓ 两个训练器共享同一个服务端管理器")
    print(f"✓ SimpleTrainer: {trainer1.trainer_id}, 学习器: {len(trainer1.learners)}")
    print(f"✓ CustomTrainer: {trainer2.trainer_id}, 学习器: {len(trainer2.learners)}")
    print(f"✓ 管理器中的学习器: {list(shared_manager.learners.keys())}")
    
    # 3. 展示职责分离
    print("\n架构分离带来的好处:")
    print("- FLServerManager: 专注基础设施管理")
    print("- TrainerBase: 专注联邦学习业务逻辑")
    print("- 可以灵活组合和复用")
    print("- 便于测试和维护")
    
    print("✓ 架构分离演示完成")


async def main():
    """主函数"""
    # 初始化日志系统
    setup_auto_logging()
    
    print("重构后的联邦学习服务端架构演示")
    print("时间:", asyncio.get_event_loop().time())
    
    try:
        # 运行各种演示
        await demo_server_manager_standalone()
        await demo_abstract_trainer()
        await demo_simple_trainer()
        await demo_separation_benefits()
        
        print("\n" + "=" * 50)
        print("架构重构演示完成！")
        print("=" * 50)
        print("新架构优势:")
        print("1. 职责分离：基础设施 vs 业务逻辑")
        print("2. 抽象化：TrainerBase 强制实现核心方法")
        print("3. 可扩展：用户可轻松继承和定制")
        print("4. 可复用：FLServerManager 可被多个训练器共享")
        print("=" * 50)
        
    except KeyboardInterrupt:
        print("\n用户中断演示")
    except Exception as e:
        print(f"\n演示过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
