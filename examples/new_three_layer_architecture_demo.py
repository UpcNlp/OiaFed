#!/usr/bin/env python3
"""
新三层架构演示脚本

演示新的联邦学习架构如何工作：
1. FLCommunicationManager - 通信管理层
2. FLBusinessLogic - 业务逻辑层 
3. FLServerManager - 总管理层

这个脚本展示了：
- 如何直接使用三层架构构建自定义服务端
- 如何通过配置文件创建服务端
- 如何使用不同的业务逻辑（SimpleTrainer vs SimpleBatchTrainer）
- 向后兼容的TrainerBase使用方式
"""

import asyncio
import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fedcl.fl.server import (
    FLCommunicationManager,
    FLBusinessLogic,
    FLServerManager,
    SimpleTrainer,
    SimpleBatchTrainer,
    TrainerBase,
    create_server_from_config,
    create_trainer_from_config
)
from fedcl.fl.client import SimpleLearnerStub, create_client_from_config
from fedcl.comm import MemoryTransport
from fedcl.utils.auto_logger import get_sys_logger


class CustomBusinessLogic(FLBusinessLogic):
    """自定义业务逻辑演示"""
    
    def __init__(self, business_id: str, comm_manager: FLCommunicationManager):
        super().__init__(business_id, comm_manager)
        self.custom_model = {'weights': [0.5, 1.5], 'version': 1}
    
    async def train_round(self, config=None):
        """自定义训练轮次"""
        self.logger.info("执行自定义训练轮次")
        
        # 1. 获取所有学习器的状态
        status_results = await self.call_all_learners('get_status')
        active_learners = [lid for lid, status in status_results.items() 
                          if status and status.get('ready', False)]
        
        if not active_learners:
            return {'error': '没有可用的学习器'}
        
        # 2. 选择性训练
        train_configs = {lid: {'epochs': 1, 'batch_size': 32} 
                        for lid in active_learners}
        train_results = await self.call_learners_selective('train', train_configs)
        
        # 3. 过滤和聚合
        valid_results = {k: v for k, v in train_results.items() if v}
        if valid_results:
            models = [r['model'] for r in valid_results.values()]
            self.custom_model = self.aggregate_models(models)
            self.custom_model['version'] += 1
        
        # 4. 广播更新
        await self.broadcast_to_learners({'model_update': self.custom_model})
        
        return {
            'active_learners': active_learners,
            'trained_learners': list(valid_results.keys()),
            'model_version': self.custom_model['version'],
            'custom_metric': len(valid_results) * 10
        }
    
    def aggregate_models(self, models):
        """自定义模型聚合"""
        if not models:
            return self.custom_model
            
        # 简单的权重平均
        num_models = len(models)
        new_weights = [0.0, 0.0]
        
        for model in models:
            if 'weights' in model:
                for i, weight in enumerate(model['weights']):
                    if i < len(new_weights):
                        new_weights[i] += weight
        
        # 平均化
        new_weights = [w / num_models for w in new_weights]
        
        return {
            'weights': new_weights,
            'version': self.custom_model['version']
        }


async def demo_manual_construction():
    """演示手动构建三层架构"""
    print("\n" + "="*60)
    print("演示1: 手动构建三层架构")
    print("="*60)
    
    # 创建传输层
    transport = MemoryTransport()
    
    # 1. 创建通信管理层
    comm_manager = FLCommunicationManager("manual_server", transport)
    comm_manager.set_auto_create_learners(True)
    
    # 2. 创建业务逻辑层
    business_logic = CustomBusinessLogic("manual_server", comm_manager)
    
    # 3. 创建总管理层
    server_manager = FLServerManager("manual_server", business_logic, comm_manager)
    
    print(f"✅ 已创建三层架构服务端: {server_manager.server_id}")
    print(f"   - 通信管理器: {type(comm_manager).__name__}")
    print(f"   - 业务逻辑: {type(business_logic).__name__}")
    print(f"   - 总管理器: {type(server_manager).__name__}")
    
    # 启动服务端
    await server_manager.start_server()
    
    # 创建几个客户端
    clients = []
    for i in range(3):
        client_id = f"client_{i+1}"
        client = SimpleLearnerStub(client_id, transport)
        clients.append(client)
        
        # 模拟客户端注册
        await transport.send(client_id, "manual_server", {
            'type': 'client_registration',
            'learner_id': client_id,
            'ready': True
        })
    
    print(f"✅ 已创建 {len(clients)} 个客户端并注册")
    
    # 等待注册
    await asyncio.sleep(0.5)
    
    # 执行训练
    print("\n开始训练...")
    results = await server_manager.start_training(rounds=2, expected_learner_count=3)
    
    print(f"✅ 训练完成! 总轮数: {len(results)}")
    for i, result in enumerate(results, 1):
        print(f"   轮次 {i}: 训练学习器 {result.get('trained_learners', [])}, "
              f"模型版本 {result.get('model_version', 'N/A')}")
    
    # 停止服务端
    await server_manager.stop_server()
    print("✅ 服务端已停止")
    
    return server_manager


async def demo_builtin_trainers():
    """演示内置训练器的使用"""
    print("\n" + "="*60)
    print("演示2: 使用内置训练器（SimpleTrainer vs SimpleBatchTrainer）")
    print("="*60)
    
    transport = MemoryTransport()
    
    # 测试SimpleTrainer
    print("\n--- SimpleTrainer ---")
    comm_manager1 = FLCommunicationManager("simple_server", transport)
    simple_trainer = SimpleTrainer("simple_server", comm_manager1)
    server1 = FLServerManager("simple_server", simple_trainer, comm_manager1)
    
    await server1.start_server()
    
    # 创建客户端
    for i in range(2):
        client_id = f"simple_client_{i+1}"
        client = SimpleLearnerStub(client_id, transport)
        await transport.send(client_id, "simple_server", {
            'type': 'client_registration',
            'learner_id': client_id
        })
    
    await asyncio.sleep(0.3)
    results1 = await server1.start_training(rounds=1, expected_learner_count=2)
    print(f"SimpleTrainer 结果: {len(results1)} 轮，参与者 {results1[0].get('participating_learners', [])}")
    await server1.stop_server()
    
    # 测试SimpleBatchTrainer
    print("\n--- SimpleBatchTrainer ---")
    comm_manager2 = FLCommunicationManager("batch_server", transport)
    batch_trainer = SimpleBatchTrainer("batch_server", comm_manager2, batch_size=2)
    server2 = FLServerManager("batch_server", batch_trainer, comm_manager2)
    
    await server2.start_server()
    
    # 创建更多客户端来演示批次训练
    for i in range(4):
        client_id = f"batch_client_{i+1}"
        client = SimpleLearnerStub(client_id, transport)
        await transport.send(client_id, "batch_server", {
            'type': 'client_registration',
            'learner_id': client_id
        })
    
    await asyncio.sleep(0.3)
    results2 = await server2.start_training(rounds=1, expected_learner_count=4)
    print(f"SimpleBatchTrainer 结果: {results2[0].get('total_batches', 'N/A')} 个批次，"
          f"总参与者 {results2[0].get('total_participating_learners', 'N/A')}")
    await server2.stop_server()


async def demo_config_based_creation():
    """演示基于配置的创建"""
    print("\n" + "="*60)
    print("演示3: 基于配置文件创建服务端")
    print("="*60)
    
    # 新架构配置
    new_config = {
        'server': {
            'id': 'config_server_new',
            'business_logic': {
                'type': 'SimpleTrainer',
                'kwargs': {}
            },
            'transport': {
                'type': 'MemoryTransport',
                'kwargs': {}
            },
            'communication': {
                'auto_create_learners': True,
                'max_concurrent': 5
            }
        }
    }
    
    # 旧架构配置（向后兼容）
    old_config = {
        'server': {
            'id': 'config_server_old',
            'trainer': {
                'type': 'SimpleTrainer',
                'kwargs': {}
            },
            'transport': {
                'type': 'MemoryTransport',
                'kwargs': {}
            }
        }
    }
    
    # 创建新架构服务端
    print("创建新架构服务端...")
    new_server = create_server_from_config(new_config)
    print(f"✅ 新架构服务端: {type(new_server).__name__}")
    print(f"   - 业务逻辑类型: {type(new_server.business_logic).__name__}")
    print(f"   - 通信管理器: {type(new_server.comm_manager).__name__}")
    
    # 创建向后兼容训练器
    print("\n创建向后兼容训练器...")
    old_trainer = create_trainer_from_config(old_config)
    print(f"✅ 向后兼容训练器: {type(old_trainer).__name__}")
    print(f"   - 服务端管理器: {type(old_trainer.server_manager).__name__}")
    
    # 简单测试
    await new_server.start_server()
    transport = new_server.comm_manager.transport
    
    # 创建客户端
    for i in range(2):
        client_id = f"config_client_{i+1}"
        await transport.send(client_id, "config_server_new", {
            'type': 'client_registration',
            'learner_id': client_id
        })
    
    await asyncio.sleep(0.2)
    
    print(f"\n✅ 已注册学习器: {list(new_server.registered_learners)}")
    
    await new_server.stop_server()


async def demo_backward_compatibility():
    """演示向后兼容性"""
    print("\n" + "="*60)
    print("演示4: 向后兼容性演示")
    print("="*60)
    
    transport = MemoryTransport()
    
    # 使用TrainerBase（向后兼容包装器）
    class CompatibleTrainer(TrainerBase):
        """向后兼容的训练器"""
        
        def __init__(self, trainer_id: str, **kwargs):
            super().__init__(trainer_id, **kwargs)
            self.custom_data = {'step': 0}
        
        async def train_round(self, config=None):
            self.custom_data['step'] += 1
            
            # 使用熟悉的API
            results = await self.call_all_learners('train', config)
            valid = {k: v for k, v in results.items() if v}
            
            if valid:
                models = [r['model'] for r in valid.values()]
                aggregated = self.aggregate_models(models)
                await self.broadcast_to_learners({'model': aggregated})
            
            return {
                'step': self.custom_data['step'],
                'participants': list(valid.keys()),
                'model': aggregated if valid else None
            }
        
        def aggregate_models(self, models):
            if not models:
                return {'default': True}
            # 简单平均
            result = {}
            for key in models[0].keys():
                if all(isinstance(m.get(key), (int, float)) for m in models):
                    result[key] = sum(m[key] for m in models) / len(models)
                else:
                    result[key] = models[0][key]
            return result
    
    # 创建兼容训练器
    trainer = CompatibleTrainer("compat_trainer", transport=transport)
    print(f"✅ 创建向后兼容训练器: {type(trainer).__name__}")
    print(f"   - 使用新架构: {type(trainer.server_manager).__name__}")
    
    # 创建客户端
    clients = []
    for i in range(2):
        client_id = f"compat_client_{i+1}"
        client = SimpleLearnerStub(client_id, transport)
        clients.append(client)
        
        await transport.send(client_id, "compat_trainer", {
            'type': 'client_registration',
            'learner_id': client_id
        })
    
    await asyncio.sleep(0.2)
    
    # 使用熟悉的API进行训练
    print("\n开始兼容性训练...")
    results = await trainer.start_training(rounds=2, expected_learner_count=2)
    
    print(f"✅ 兼容性训练完成! 步骤: {[r.get('step') for r in results]}")
    for i, result in enumerate(results, 1):
        print(f"   轮次 {i}: 步骤 {result.get('step')}, 参与者 {result.get('participants', [])}")


async def main():
    """主函数"""
    print("新三层联邦学习架构演示")
    print("="*60)
    print("演示架构层次:")
    print("1. FLCommunicationManager - 通信管理层（处理连接、注册、消息路由）")
    print("2. FLBusinessLogic - 业务逻辑层（处理训练、聚合、业务规则）")
    print("3. FLServerManager - 总管理层（协调上述两层，提供统一接口）")
    print("4. TrainerBase - 向后兼容包装器（保持旧API，内部使用新架构）")
    
    try:
        # 演示1：手动构建三层架构
        await demo_manual_construction()
        
        # 演示2：使用内置训练器
        await demo_builtin_trainers()
        
        # 演示3：基于配置创建
        await demo_config_based_creation()
        
        # 演示4：向后兼容性
        await demo_backward_compatibility()
        
        print("\n" + "="*60)
        print("✅ 所有演示完成！")
        print("="*60)
        print("\n总结:")
        print("1. 新架构实现了清晰的职责分离:")
        print("   - 通信管理器：专注通信和连接管理")
        print("   - 业务逻辑：专注训练和模型聚合") 
        print("   - 总管理器：协调两者并提供统一接口")
        print("2. 支持多种业务逻辑实现（SimpleTrainer, SimpleBatchTrainer, 自定义逻辑）")
        print("3. 保持向后兼容性（TrainerBase包装器）")
        print("4. 支持配置文件驱动的创建方式")
        print("5. 业务逻辑与通信基础设施完全解耦")
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
