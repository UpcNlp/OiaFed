# examples/hooks_example.py
"""
Hook系统使用示例

演示如何使用MetricsHook和CheckpointHook进行度量记录和检查点保存。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from omegaconf import OmegaConf

from fedcl.core.execution_context import ExecutionContext
from fedcl.core.hook import HookPhase
from fedcl.core.metrics_hook import MetricsHook
from fedcl.core.checkpoint_hook import CheckpointHook


class SimpleModel(nn.Module):
    """简单的示例模型"""
    
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def create_example_hooks(output_dir: Path):
    """创建示例Hook配置"""
    
    # 度量Hook配置
    metrics_config = OmegaConf.create({
        'log_training': True,
        'log_evaluation': True,
        'log_system': True,
        'log_communication': True,
        'training_frequency': 1,
        'evaluation_frequency': 1,
        'system_frequency': 5,
        'communication_frequency': 1,
        'metric_filters': [],
        'excluded_metrics': ['debug_info'],
        'output_format': 'both',
        'output_path': str(output_dir / 'metrics.jsonl')
    })
    
    # 检查点Hook配置
    checkpoint_config = OmegaConf.create({
        'save_frequency': 1,
        'save_model': True,
        'save_optimizer': True,
        'save_scheduler': False,
        'save_experiment_state': True,
        'checkpoint_dir': str(output_dir / 'checkpoints'),
        'naming_pattern': 'checkpoint_round_{round}_epoch_{epoch}',
        'include_timestamp': True,
        'max_checkpoints': 3,
        'compress': False,
        'keep_best_only': False,
        'best_metric': 'accuracy',
        'best_mode': 'max'
    })
    
    # 创建Hook实例
    metrics_hook = MetricsHook(
        phase=HookPhase.AFTER_BATCH.value,
        metrics_config=metrics_config
    )
    
    checkpoint_hook = CheckpointHook(
        phase=HookPhase.AFTER_ROUND.value,
        checkpoint_config=checkpoint_config
    )
    
    return metrics_hook, checkpoint_hook


def simulate_training_loop():
    """模拟训练循环"""
    
    # 创建输出目录
    output_dir = Path('./hooks_example_output')
    output_dir.mkdir(exist_ok=True)
    
    # 创建执行上下文
    config = OmegaConf.create({
        'experiment': {'name': 'hooks_example'},
        'state_storage': {'max_size': 1000},
        'event_system': {'max_queue_size': 100},
        'metrics': {'max_per_name': 1000}
    })
    
    context = ExecutionContext(
        config=config,
        experiment_id='hooks_example_001'
    )
    
    # 创建Hook
    metrics_hook, checkpoint_hook = create_example_hooks(output_dir)
    
    # 创建模型和优化器
    model = SimpleModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    print("开始模拟训练过程...")
    
    # 模拟多轮训练
    for round_num in range(3):
        print(f"\n=== 第 {round_num + 1} 轮 ===")
        
        # 设置当前轮次状态
        context.set_state('current_round', round_num + 1, 'global')
        context.set_state('current_epoch', 0, 'global')
        
        # 模拟epoch训练
        for epoch in range(2):
            context.set_state('current_epoch', epoch + 1, 'global')
            print(f"  Epoch {epoch + 1}")
            
            # 模拟批次训练
            epoch_loss = 0.0
            for batch_idx in range(5):
                context.set_state('current_batch', batch_idx + 1, 'global')
                
                # 模拟训练数据
                batch_size = 32
                data = torch.randn(batch_size, 784)
                targets = torch.randint(0, 10, (batch_size,))
                
                # 前向传播
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, targets)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                # 计算准确率
                _, predicted = torch.max(outputs.data, 1)
                accuracy = (predicted == targets).float().mean().item()
                
                epoch_loss += loss.item()
                
                # 记录训练度量
                training_metrics = {
                    'loss': loss.item(),
                    'accuracy': accuracy,
                    'learning_rate': optimizer.param_groups[0]['lr']
                }
                
                print(f"    Batch {batch_idx + 1}: loss={loss.item():.4f}, acc={accuracy:.4f}")
                
                # 执行度量Hook
                metrics_hook.execute(context, metrics=training_metrics)
            
            # 模拟评估
            avg_loss = epoch_loss / 5
            eval_accuracy = 0.8 + 0.1 * (round_num + epoch * 0.05)  # 模拟逐渐提升的准确率
            
            evaluation_results = {
                'accuracy': eval_accuracy,
                'avg_loss': avg_loss,
                'f1_score': eval_accuracy * 0.95  # 模拟F1分数
            }
            
            print(f"    评估结果: acc={eval_accuracy:.4f}, loss={avg_loss:.4f}")
            
            # 执行评估度量Hook
            metrics_hook_eval = MetricsHook(
                phase=HookPhase.ON_EVALUATION.value,
                metrics_config=metrics_hook.metrics_config
            )
            metrics_hook_eval.execute(context, results=evaluation_results)
        
        # 轮次结束，保存检查点
        print(f"  保存第 {round_num + 1} 轮检查点...")
        checkpoint_hook.execute(
            context,
            model=model,
            optimizer=optimizer,
            metrics={'accuracy': eval_accuracy, 'loss': avg_loss}
        )
    
    # 输出Hook统计信息
    print("\n=== Hook 执行统计 ===")
    
    # 度量Hook统计
    metrics_summary = metrics_hook.get_metric_summary()
    print(f"度量Hook统计:")
    print(f"  总记录次数: {metrics_summary['total_metrics_logged']}")
    print(f"  度量类型: {metrics_summary['metric_types']}")
    
    # 检查点Hook统计
    checkpoint_summary = checkpoint_hook.get_checkpoint_summary()
    print(f"\n检查点Hook统计:")
    print(f"  保存的检查点数量: {checkpoint_summary['total_checkpoints']}")
    print(f"  检查点目录: {checkpoint_summary['checkpoint_dir']}")
    print(f"  最佳检查点: {checkpoint_summary['best_checkpoint']}")
    print(f"  总占用空间: {checkpoint_summary['total_size']} bytes")
    
    # 清理Hook
    metrics_hook.cleanup()
    checkpoint_hook.cleanup()
    
    print(f"\n示例完成！输出文件保存在: {output_dir}")
    print(f"  度量日志: {output_dir / 'metrics.jsonl'}")
    print(f"  检查点: {output_dir / 'checkpoints'}")


def demonstrate_hook_features():
    """演示Hook的高级功能"""
    
    print("\n=== Hook 高级功能演示 ===")
    
    # 创建输出目录
    output_dir = Path('./hooks_advanced_example')
    output_dir.mkdir(exist_ok=True)
    
    # 创建执行上下文
    config = OmegaConf.create({
        'experiment': {'name': 'advanced_hooks'},
        'state_storage': {'max_size': 1000}
    })
    
    context = ExecutionContext(
        config=config,
        experiment_id='advanced_hooks_001'
    )
    
    # 1. 度量过滤示例
    print("\n1. 度量过滤功能")
    filtered_metrics_config = OmegaConf.create({
        'log_training': True,
        'training_frequency': 1,
        'metric_filters': ['loss', 'accuracy'],  # 只记录指定度量
        'excluded_metrics': ['debug_info'],      # 排除调试信息
        'output_format': 'context'
    })
    
    filtered_hook = MetricsHook(
        phase=HookPhase.AFTER_BATCH.value,
        metrics_config=filtered_metrics_config
    )
    
    # 测试过滤
    test_metrics = {
        'loss': 0.5,
        'accuracy': 0.85,
        'learning_rate': 0.001,  # 不在过滤器中，会被过滤
        'debug_info': 'test',    # 在排除列表中，会被过滤
    }
    
    context.set_state('current_batch', 1, 'global')
    filtered_hook.execute(context, metrics=test_metrics)
    
    # 2. 检查点最佳模型保存示例
    print("\n2. 最佳模型检查点功能")
    best_checkpoint_config = OmegaConf.create({
        'save_frequency': 1,
        'save_model': True,
        'save_optimizer': False,
        'save_experiment_state': True,
        'checkpoint_dir': str(output_dir / 'best_checkpoints'),
        'max_checkpoints': 1,
        'keep_best_only': True,
        'best_metric': 'accuracy',
        'best_mode': 'max'
    })
    
    best_checkpoint_hook = CheckpointHook(
        phase=HookPhase.ON_EVALUATION.value,
        checkpoint_config=best_checkpoint_config
    )
    
    # 模拟多次评估，只保留最佳
    model = SimpleModel()
    
    for i, acc in enumerate([0.7, 0.85, 0.8, 0.9, 0.82]):  # 0.9是最高的
        context.set_state('current_round', i + 1, 'global')
        best_checkpoint_hook.execute(
            context,
            model=model,
            metrics={'accuracy': acc}
        )
        print(f"  评估 {i+1}: accuracy={acc}, 最佳={best_checkpoint_hook.best_metric_value}")
    
    # 3. 频率控制示例
    print("\n3. 记录频率控制功能")
    frequency_config = OmegaConf.create({
        'log_training': True,
        'training_frequency': 3,  # 每3个batch记录一次
        'output_format': 'context'
    })
    
    frequency_hook = MetricsHook(
        phase=HookPhase.AFTER_BATCH.value,
        metrics_config=frequency_config
    )
    
    # 测试频率控制
    for batch in range(1, 10):
        context.set_state('current_batch', batch, 'global')
        should_execute = frequency_hook.should_execute(context)
        print(f"  Batch {batch}: 应该执行={should_execute}")
    
    print("\n高级功能演示完成！")


if __name__ == "__main__":
    # 运行基本示例
    simulate_training_loop()
    
    # 运行高级功能演示
    demonstrate_hook_features()
