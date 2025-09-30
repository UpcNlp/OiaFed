#!/usr/bin/env python3
"""
重构后三层架构演示

演示重构后的三层架构：
1. FLCommunicationManager - 通信管理层（负责连接、状态监控、高级通信功能）
2. FLTrainer - 业务逻辑层（负责联邦学习算法和业务规则）
3. FLServer - 总管理层（协调前两层，提供统一接口）

展示功能：
- 通信管理器的状态监控和健康检查
- 高级通信功能（批量调用、重试机制、健康学习器过滤）
- 业务逻辑与基础设施的完全解耦
- 系统级监控和控制
"""

import asyncio
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fedcl.comm import MemoryTransport
from fedcl.fl.server import FLCommunicationManager, FLTrainer, FLServer
from fedcl.fl.client import SimpleLearnerStub


class DemoTrainer(FLTrainer):
    """演示业务逻辑类"""
    
    def __init__(self, business_id: str, comm_manager: FLCommunicationManager):
        super().__init__(business_id, comm_manager)
        self.global_model = {'accuracy': 0.5, 'loss': 1.0}
    
    async def train_round(self, config=None):
        """执行一轮训练"""
        self.logger.info("=== 开始新一轮训练 ===")
        
        # 1. 只对健康的学习器进行训练
        train_results = await self.comm_manager.call_healthy_learners_only('train', config or {})
        
        if not train_results:
            raise ValueError("没有健康的学习器可用于训练")
        
        # 2. 过滤有效结果
        valid_results = {k: v for k, v in train_results.items() if v is not None}
        self.logger.info(f"收到 {len(valid_results)} 个有效训练结果")
        
        # 3. 聚合模型
        models = []
        for learner_id, result in valid_results.items():
            if isinstance(result, dict) and 'model' in result:
                models.append(result['model'])
            else:
                # 模拟结果
                models.append({'accuracy': 0.6 + int(learner_id.split('_')[-1]) * 0.1, 'loss': 0.8})
        
        self.global_model = self.aggregate_models(models)
        
        # 4. 使用重试机制广播模型
        learner_configs = {lid: {'global_model': self.global_model} for lid in valid_results.keys()}
        broadcast_results = await self.comm_manager.batch_call_with_retry(
            'set_model', learner_configs, max_retries=2, retry_delay=0.5
        )
        
        self.logger.info(f"模型广播完成: {len(broadcast_results)}/{len(valid_results)} 成功")
        
        return {
            'participating_learners': list(valid_results.keys()),
            'global_model': self.global_model.copy(),
            'broadcast_success_rate': len(broadcast_results) / len(valid_results) if valid_results else 0
        }
    
    def aggregate_models(self, models):
        """简单模型聚合"""
        if not models:
            return self.global_model
            
        avg_accuracy = sum(m.get('accuracy', 0) for m in models) / len(models)
        avg_loss = sum(m.get('loss', 1.0) for m in models) / len(models)
        
        return {'accuracy': avg_accuracy, 'loss': avg_loss}
    
    async def _on_learner_registered(self, learner_id, proxy, message):
        """学习器注册时的业务回调"""
        await super()._on_learner_registered(learner_id, proxy, message)
        
        # 发送初始模型
        try:
            await proxy.set_model(self.global_model)
            self.logger.info(f"✅ 向 {learner_id} 发送初始模型成功")
        except Exception as e:
            self.logger.warning(f"❌ 向 {learner_id} 发送初始模型失败: {e}")
    
    async def on_training_start(self):
        """训练开始钩子"""
        self.logger.info(f"🚀 联邦学习训练启动！初始模型: {self.global_model}")
    
    async def on_round_end(self, round_num, result):
        """每轮结束钩子"""
        model = result.get('global_model', {})
        participants = len(result.get('participating_learners', []))
        success_rate = result.get('broadcast_success_rate', 0) * 100
        
        self.logger.info(f"📊 第 {round_num} 轮完成 - 参与者: {participants}, "
                        f"准确率: {model.get('accuracy', 0):.3f}, "
                        f"广播成功率: {success_rate:.1f}%")
    
    async def on_training_end(self, results):
        """训练结束钩子"""
        self.logger.info(f"🎉 训练完成！最终模型: {self.global_model}")
        self.logger.info(f"📈 共完成 {len(results)} 轮训练")


async def demo_three_layer_architecture():
    """演示三层架构"""
    print("=" * 60)
    print("🏗️  新三层联邦学习架构演示")
    print("=" * 60)
    
    # 1. 创建传输层
    transport = MemoryTransport()
    
    # 2. 创建通信管理器（第1层：通信管理）
    print("\n📡 步骤1：创建通信管理器")
    comm_manager = FLCommunicationManager("fed_server", transport)
    
    # 启动连接监控
    comm_manager.start_monitoring(interval=5.0)
    print("✅ 通信管理器已创建并启动监控")
    
    # 3. 创建业务逻辑（第2层：业务逻辑）
    print("\n🧠 步骤2：创建业务逻辑处理器")
    trainer = DemoTrainer("demo_business", comm_manager)
    print("✅ 业务逻辑处理器已创建")
    
    # 4. 创建总管理器（第3层：总管理）
    print("\n👑 步骤3：创建服务端总管理器")
    server = FLServer("demo_server", trainer, comm_manager)
    await server.start_server()
    print("✅ 服务端总管理器已启动")
    
    # 5. 模拟客户端连接
    print("\n👥 步骤4：模拟客户端连接")
    clients = []
    for i in range(4):
        client_id = f"client_{i+1}"
        client = SimpleLearnerStub(client_id, transport)
        clients.append(client)
        
        # 模拟客户端注册
        await asyncio.sleep(0.1)
        print(f"📱 客户端 {client_id} 已连接")
    
    # 等待所有客户端注册
    await asyncio.sleep(1)
    
    # 6. 展示通信管理器的状态
    print("\n📊 步骤5：通信状态展示")
    comm_stats = comm_manager.get_communication_stats()
    print(f"  总学习器数: {comm_stats['total_learners']}")
    print(f"  已注册数: {comm_stats['registered_learners']}")
    print(f"  健康学习器数: {comm_stats['healthy_learners']}")
    print(f"  监控状态: {'启用' if comm_stats['monitoring_enabled'] else '禁用'}")
    
    # 展示每个学习器的详细状态
    print("\n🔍 学习器详细状态:")
    learner_status = comm_manager.get_all_learner_status()
    for learner_id, status in learner_status.items():
        print(f"  {learner_id}: {status['status']}, 消息数: {status['message_count']}, "
              f"错误数: {status['error_count']}")
    
    # 7. 模拟一个客户端出现问题
    print("\n⚠️  步骤6：模拟客户端异常")
    problem_client = "client_3"
    if problem_client in comm_manager.learners:
        # 模拟连接问题
        comm_manager.update_learner_activity(problem_client, success=False, error="连接超时")
        comm_manager.update_learner_activity(problem_client, success=False, error="网络错误")
        comm_manager.update_learner_activity(problem_client, success=False, error="响应超时")
        comm_manager.update_learner_activity(problem_client, success=False, error="服务不可用")
        print(f"❌ 客户端 {problem_client} 出现多次错误")
    
    # 检查健康状态变化
    healthy_learners = comm_manager.get_healthy_learners()
    print(f"✅ 当前健康学习器: {healthy_learners} (共 {len(healthy_learners)} 个)")
    
    # 8. 执行联邦学习训练
    print("\n🚀 步骤7：开始联邦学习训练")
    try:
        results = await server.start_training(
            rounds=3,
            expected_learner_count=None,  # 不等待特定数量
            config={'batch_size': 32, 'epochs': 1}
        )
        
        print(f"\n✅ 训练成功完成，共 {len(results)} 轮")
        
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
    
    # 9. 展示最终统计
    print("\n📈 步骤8：最终统计信息")
    system_status = server.get_system_status()
    print(f"  系统运行状态: {'运行中' if system_status['is_running'] else '已停止'}")
    print(f"  训练状态: {'进行中' if system_status['is_training'] else '已完成'}")
    print(f"  完成训练轮数: {system_status['training_rounds_completed']}")
    print(f"  系统运行时长: {system_status['uptime_seconds']:.1f}s")
    
    final_comm_stats = comm_manager.get_communication_stats()
    comm_stats_data = final_comm_stats['communication_stats']
    print(f"  总消息数: {comm_stats_data['total_messages']}")
    print(f"  成功调用数: {comm_stats_data['successful_calls']}")
    print(f"  失败调用数: {comm_stats_data['failed_calls']}")
    print(f"  广播次数: {comm_stats_data['broadcast_count']}")
    
    # 10. 清理资源
    print("\n🧹 步骤9：清理资源")
    comm_manager.stop_monitoring()
    await server.stop_server()
    print("✅ 资源清理完成")
    
    print("\n" + "=" * 60)
    print("✨ 三层架构演示完成！")
    print("=" * 60)
    
    return {
        'comm_manager': comm_manager,
        'trainer': trainer, 
        'server': server,
        'clients': clients
    }


async def demo_advanced_communication_features():
    """演示通信管理器的高级功能"""
    print("\n" + "=" * 60)
    print("🔧 高级通信功能演示")
    print("=" * 60)
    
    # 创建基础设施
    transport = MemoryTransport()
    comm_manager = FLCommunicationManager("advanced_server", transport)
    
    # 添加多个学习器
    print("\n📱 添加学习器:")
    for i in range(5):
        learner_id = f"learner_{i+1}"
        client = SimpleLearnerStub(learner_id, transport)
        comm_manager.add_learner(learner_id, transport)
        print(f"  ✅ {learner_id} 已添加")
    
    # 模拟部分学习器不健康
    print("\n⚠️  模拟部分学习器异常:")
    comm_manager.update_learner_activity("learner_2", success=False, error="网络超时")
    comm_manager.update_learner_activity("learner_2", success=False, error="连接中断") 
    comm_manager.update_learner_activity("learner_2", success=False, error="服务错误")
    comm_manager.update_learner_activity("learner_2", success=False, error="响应超时")
    
    comm_manager.update_learner_activity("learner_4", success=False, error="硬件故障")
    comm_manager.update_learner_activity("learner_4", success=False, error="内存不足")
    comm_manager.update_learner_activity("learner_4", success=False, error="磁盘满了")
    comm_manager.update_learner_activity("learner_4", success=False, error="CPU过载")
    
    print("  ❌ learner_2 和 learner_4 出现多次错误")
    
    healthy_learners = comm_manager.get_healthy_learners()
    print(f"  ✅ 健康学习器: {healthy_learners}")
    
    # 演示选择性调用
    print("\n🎯 选择性调用演示:")
    selective_configs = {
        'learner_1': {'task': 'classification', 'epochs': 5},
        'learner_3': {'task': 'regression', 'epochs': 3},
        'learner_5': {'task': 'clustering', 'epochs': 2}
    }
    
    print("  配置不同的任务参数...")
    for learner_id, config in selective_configs.items():
        print(f"    {learner_id}: {config}")
    
    # 模拟调用（实际中会调用真实方法）
    print("  🔄 执行选择性调用...")
    try:
        # 这里会失败，因为我们没有实现 'start_task' 方法，但展示了功能
        results = await comm_manager.call_learners_selective('ping', 
                                                           {lid: {} for lid in selective_configs.keys()})
        print(f"  ✅ 选择性调用完成: {len(results)} 个响应")
    except Exception as e:
        print(f"  ⚠️  选择性调用演示（预期的方法不存在）: {e}")
    
    # 演示重试机制
    print("\n🔄 重试机制演示:")
    retry_configs = {learner_id: {'data': f'test_data_{i}'} 
                    for i, learner_id in enumerate(healthy_learners, 1)}
    
    print(f"  准备对 {len(retry_configs)} 个健康学习器进行重试调用...")
    try:
        results = await comm_manager.batch_call_with_retry('ping', retry_configs, max_retries=2)
        print(f"  ✅ 重试调用完成: {len(results)} 个成功")
    except Exception as e:
        print(f"  ⚠️  重试演示（预期的方法问题）: {e}")
    
    # 演示广播功能
    print("\n📡 广播功能演示:")
    broadcast_data = {'global_update': 'model_v1.0', 'timestamp': '2025-09-02'}
    
    print("  🔄 广播到所有学习器...")
    try:
        await comm_manager.broadcast_to_learners(broadcast_data, healthy_only=False)
        print("  ✅ 全局广播完成")
    except Exception as e:
        print(f"  ❌ 全局广播失败: {e}")
    
    print("  🔄 仅广播到健康学习器...")  
    try:
        await comm_manager.broadcast_to_learners(broadcast_data, healthy_only=True)
        print("  ✅ 健康学习器广播完成")
    except Exception as e:
        print(f"  ❌ 健康学习器广播失败: {e}")
    
    # 展示最终统计
    print("\n📊 最终通信统计:")
    stats = comm_manager.get_communication_stats()
    comm_data = stats['communication_stats']
    print(f"  总消息数: {comm_data['total_messages']}")
    print(f"  成功调用: {comm_data['successful_calls']}")
    print(f"  失败调用: {comm_data['failed_calls']}")
    print(f"  广播次数: {comm_data['broadcast_count']}")
    print(f"  健康学习器: {stats['healthy_learners']}/{stats['total_learners']}")
    
    print("\n" + "=" * 60)
    print("🔧 高级通信功能演示完成！")
    print("=" * 60)


async def main():
    """主函数"""
    try:
        # 演示1：三层架构
        demo_result = await demo_three_layer_architecture()
        
        await asyncio.sleep(1)
        
        # 演示2：高级通信功能
        await demo_advanced_communication_features()
        
        print("\n🎉 所有演示完成！")
        print("\n💡 重构总结:")
        print("  1️⃣  FLCommunicationManager - 专注通信管理，状态监控，高级通信功能")
        print("  2️⃣  FLTrainer - 专注业务逻辑，算法实现，训练流程")
        print("  3️⃣  FLServer - 总协调者，统一接口，系统控制")
        print("  ✨ 三层架构实现了职责分离，便于维护和扩展！")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
