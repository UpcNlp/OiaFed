# examples/hook_executor_example.py
"""
HookExecutor 使用示例

展示如何使用HookExecutor管理和执行钩子，包括注册、执行、错误处理等功能。
"""

import time
from typing import Any

from fedcl.core.hook_executor import HookExecutor
from fedcl.core.hook import Hook, HookPhase
from fedcl.core.execution_context import ExecutionContext
from fedcl.registry.component_registry import ComponentRegistry
from omegaconf import DictConfig


class MetricsHook(Hook):
    """指标收集钩子示例"""
    
    def __init__(self):
        super().__init__(
            phase=HookPhase.BEFORE_TASK.value,
            priority=10,
            name="MetricsHook"
        )
        self.metrics = []
    
    def execute(self, context: ExecutionContext, **kwargs) -> Any:
        """收集任务开始前的指标"""
        task_id = kwargs.get('task_id', 'unknown')
        metric = {
            'task_id': task_id,
            'timestamp': time.time(),
            'phase': self.phase,
            'hook_name': self.name
        }
        self.metrics.append(metric)
        print(f"📊 MetricsHook: 收集任务 {task_id} 开始指标")
        return metric


class CheckpointHook(Hook):
    """检查点保存钩子示例"""
    
    def __init__(self):
        super().__init__(
            phase=HookPhase.AFTER_TASK.value,
            priority=20,
            name="CheckpointHook"
        )
        self.checkpoints = []
    
    def execute(self, context: ExecutionContext, **kwargs) -> Any:
        """保存任务完成后的检查点"""
        task_id = kwargs.get('task_id', 'unknown')
        results = kwargs.get('results', {})
        
        checkpoint = {
            'task_id': task_id,
            'results': results,
            'timestamp': time.time(),
            'phase': self.phase
        }
        self.checkpoints.append(checkpoint)
        print(f"💾 CheckpointHook: 保存任务 {task_id} 检查点")
        return checkpoint


class ErrorHook(Hook):
    """错误处理钩子示例"""
    
    def __init__(self):
        super().__init__(
            phase=HookPhase.ON_ERROR.value,
            priority=0,  # 最高优先级
            name="ErrorHook"
        )
        self.errors = []
    
    def execute(self, context: ExecutionContext, **kwargs) -> Any:
        """处理系统错误"""
        original_error = kwargs.get('original_error')
        failed_hook = kwargs.get('failed_hook')
        
        error_info = {
            'error_type': type(original_error).__name__,
            'error_message': str(original_error),
            'failed_hook': failed_hook.get_name() if failed_hook else 'unknown',
            'timestamp': time.time()
        }
        self.errors.append(error_info)
        print(f"🚨 ErrorHook: 处理错误 - {error_info['error_message']}")
        return error_info


class FaultyHook(Hook):
    """故意出错的钩子，用于测试错误处理"""
    
    def __init__(self):
        super().__init__(
            phase=HookPhase.BEFORE_TASK.value,
            priority=30,
            name="FaultyHook"
        )
    
    def execute(self, context: ExecutionContext, **kwargs) -> Any:
        """故意抛出异常"""
        print("💥 FaultyHook: 故意抛出异常")
        raise Exception("这是一个测试异常")


def main():
    """主函数 - 演示HookExecutor的使用"""
    print("🚀 HookExecutor 使用示例")
    print("=" * 50)
    
    # 1. 创建组件和配置
    registry = ComponentRegistry()
    config = DictConfig({
        'hook_execution': {
            'error_policy': 'continue',
            'timeout': 10.0,
            'parallel_execution': False,
            'monitoring': {
                'track_execution_time': True,
                'max_execution_time': 5.0
            }
        }
    })
    
    # 2. 创建HookExecutor
    executor = HookExecutor(registry, config)
    
    # 3. 创建执行上下文
    context = ExecutionContext(
        config=config,
        experiment_id='demo_experiment'
    )
    
    # 4. 创建和注册钩子
    print("\n📝 注册钩子...")
    metrics_hook = MetricsHook()
    checkpoint_hook = CheckpointHook()
    error_hook = ErrorHook()
    faulty_hook = FaultyHook()
    
    metrics_id = executor.register_hook(metrics_hook)
    checkpoint_id = executor.register_hook(checkpoint_hook)
    error_id = executor.register_hook(error_hook)
    faulty_id = executor.register_hook(faulty_hook)
    
    print(f"✅ 注册了 {len(executor._hook_instances)} 个钩子")
    
    # 5. 执行before_task阶段的钩子
    print("\n🔄 执行 before_task 阶段钩子...")
    before_results = executor.execute_hooks(
        "before_task", 
        context, 
        task_id="task_001",
        task_data={'type': 'classification', 'samples': 1000}
    )
    print(f"📋 before_task 执行结果: {len(before_results)} 个成功")
    
    # 6. 执行after_task阶段的钩子
    print("\n🔄 执行 after_task 阶段钩子...")
    after_results = executor.execute_hooks(
        "after_task",
        context,
        task_id="task_001",
        results={'accuracy': 0.95, 'loss': 0.15}
    )
    print(f"📋 after_task 执行结果: {len(after_results)} 个成功")
    
    # 7. 获取执行统计
    print("\n📊 执行统计信息:")
    stats = executor.get_execution_stats()
    print(f"  总执行次数: {stats['total_executions']}")
    print(f"  成功次数: {stats['successful_executions']}")
    print(f"  失败次数: {stats['failed_executions']}")
    print(f"  成功率: {stats['success_rate']:.2%}")
    print(f"  平均执行时间: {stats['average_execution_time']:.4f}s")
    
    # 8. 演示钩子启用/禁用
    print("\n🔧 演示钩子启用/禁用...")
    print(f"  禁用前 before_task 钩子数: {len(executor.get_hooks('before_task'))}")
    executor.disable_hook(faulty_id)
    print(f"  禁用 FaultyHook 后 before_task 钩子数: {len(executor._get_enabled_hooks('before_task'))}")
    
    # 9. 再次执行，验证错误钩子不会执行
    print("\n🔄 再次执行 before_task（FaultyHook已禁用）...")
    before_results_2 = executor.execute_hooks(
        "before_task",
        context,
        task_id="task_002"
    )
    print(f"📋 before_task 执行结果: {len(before_results_2)} 个成功")
    
    # 10. 演示错误策略
    print("\n⚙️ 演示错误策略...")
    executor.enable_hook(faulty_id)  # 重新启用错误钩子
    executor.set_error_policy('stop')
    
    print("  设置错误策略为 'stop'，执行 before_task...")
    try:
        executor.execute_hooks("before_task", context, task_id="task_003")
    except Exception as e:
        print(f"  ❌ 执行被停止: {e}")
    
    # 11. 清理演示
    print("\n🧹 清理钩子...")
    executor.clear_hooks("before_task")
    print(f"  清理后 before_task 钩子数: {len(executor.get_hooks('before_task'))}")
    
    # 12. 查看收集的数据
    print("\n📈 收集的数据:")
    print(f"  MetricsHook 收集的指标: {len(metrics_hook.metrics)}")
    print(f"  CheckpointHook 保存的检查点: {len(checkpoint_hook.checkpoints)}")
    print(f"  ErrorHook 处理的错误: {len(error_hook.errors)}")
    
    if metrics_hook.metrics:
        print(f"  最新指标: {metrics_hook.metrics[-1]}")
    
    if checkpoint_hook.checkpoints:
        print(f"  最新检查点: {checkpoint_hook.checkpoints[-1]}")
    
    print("\n✨ 示例完成！")


if __name__ == "__main__":
    main()
