# examples/training_engine_example.py
"""
TrainingEngine使用示例

展示如何使用TrainingEngine进行训练，包括基本使用、配置管理、钩子集成等。
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from fedcl.core.hook_executor import HookExecutor
from fedcl.core.execution_context import ExecutionContext
from fedcl.core.base_learner import BaseLearner
from fedcl.training.training_engine import TrainingEngine
from fedcl.registry.component_registry import ComponentRegistry


class SimpleModel(nn.Module):
    """简单的神经网络模型"""
    
    def __init__(self, input_size=10, hidden_size=64, output_size=2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SimpleLearner(BaseLearner):
    """简单的学习器实现"""
    
    def __init__(self, context: ExecutionContext, config: DictConfig):
        # 简化初始化，避免父类依赖
        self.context = context
        self.config = config
        self.model = SimpleModel()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device("cpu")
        
    def forward(self, inputs):
        return self.model(inputs)
    
    def compute_loss(self, outputs, targets):
        return self.criterion(outputs, targets)
    
    def backward(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
    
    def optimizer_step(self):
        self.optimizer.step()
    
    def get_model(self):
        return self.model
    
    def get_model_state(self):
        return self.model.state_dict()
    
    def load_model_state(self, state_dict):
        self.model.load_state_dict(state_dict)
    
    def get_optimizer_state(self):
        return self.optimizer.state_dict()
    
    def get_learning_rate(self):
        return self.optimizer.param_groups[0]['lr']
    
    def to(self, device):
        self.model = self.model.to(device)
        self.device = device
        return self
    
    def set_train_mode(self):
        self.model.train()
    
    def set_eval_mode(self):
        self.model.eval()
    
    def compute_accuracy(self, outputs, targets):
        pred = outputs.argmax(dim=1)
        return (pred == targets).float().mean().item()


def create_sample_data(num_samples=1000, input_size=10, num_classes=2):
    """创建样本数据"""
    # 生成随机数据
    X = torch.randn(num_samples, input_size)
    y = torch.randint(0, num_classes, (num_samples,))
    
    # 创建数据集和数据加载器
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    return dataloader


def create_training_config():
    """创建训练配置"""
    config = OmegaConf.create({
        "training": {
            "num_epochs": 5,
            "batch_size": 32,
            "learning_rate": 0.001,
            "early_stopping": {
                "enable": True,
                "patience": 3,
                "min_delta": 0.001
            },
            "optimization": {
                "optimizer": "adam",
                "lr_scheduler": "none",
                "weight_decay": 0.0001
            },
            "gradient": {
                "clip_norm": 1.0,
                "accumulation_steps": 1
            },
            "validation": {
                "interval": 1,
                "metric": "accuracy"
            },
            "checkpointing": {
                "save_interval": 2,
                "save_best": True
            }
        },
        "device": "cpu",
        "seed": 42
    })
    return config


def basic_training_example():
    """基本训练示例"""
    print("=== 基本训练示例 ===")
    
    # 创建配置
    config = create_training_config()
    
    # 创建组件
    registry = ComponentRegistry()
    context = ExecutionContext(config)
    hook_executor = HookExecutor(registry, context)
    learner = SimpleLearner(context, config)
    
    # 创建训练引擎
    training_engine = TrainingEngine(
        hook_executor=hook_executor,
        context=context
    )
    
    # 创建数据
    train_data = create_sample_data(num_samples=1000)
    
    print(f"训练配置: {config.training}")
    print(f"设备: {training_engine.device}")
    
    # 执行训练
    task_id = 1
    try:
        result = training_engine.train_task(task_id, train_data, learner)
        
        print(f"训练完成!")
        print(f"任务ID: {result.task_id}")
        print(f"训练时间: {result.training_time:.2f}s")
        print(f"内存使用: {result.memory_usage:.2f}MB")
        print(f"模型大小: {result.model_size:.2f}MB")
        print(f"训练度量: {result.metrics}")
        
    except Exception as e:
        print(f"训练失败: {str(e)}")


def training_with_validation_example():
    """带验证的训练示例"""
    print("\n=== 带验证的训练示例 ===")
    
    # 创建配置
    config = create_training_config()
    
    # 创建组件
    registry = ComponentRegistry()
    context = ExecutionContext(config)
    hook_executor = HookExecutor(registry, context)
    learner = SimpleLearner(context, config)
    
    # 创建训练引擎
    training_engine = TrainingEngine(
        hook_executor=hook_executor,
        context=context
    )
    
    # 创建训练和验证数据
    train_data = create_sample_data(num_samples=800)
    val_data = create_sample_data(num_samples=200)
    
    print("开始训练循环...")
    
    try:
        # 执行训练循环
        metrics = training_engine.execute_training_loop(learner, train_data, num_epochs=3)
        print(f"训练度量: {metrics}")
        
        # 执行验证
        val_metrics = training_engine.validate_model(learner, val_data)
        print(f"验证度量: {val_metrics}")
        
    except Exception as e:
        print(f"训练失败: {str(e)}")


def training_control_example():
    """训练控制示例"""
    print("\n=== 训练控制示例 ===")
    
    # 创建配置
    config = create_training_config()
    config.training.num_epochs = 10  # 更多轮次用于演示控制
    
    # 创建组件
    registry = ComponentRegistry()
    context = ExecutionContext(config)
    hook_executor = HookExecutor(registry, context)
    learner = SimpleLearner(context, config)
    
    # 创建训练引擎
    training_engine = TrainingEngine(
        hook_executor=hook_executor,
        context=context
    )
    
    # 创建数据
    train_data = create_sample_data(num_samples=1000)
    
    print("开始训练...")
    
    import threading
    import time
    
    def control_training():
        """控制训练的线程函数"""
        time.sleep(2)  # 等待2秒
        print("暂停训练...")
        training_engine.pause_training()
        
        time.sleep(1)  # 暂停1秒
        print("恢复训练...")
        training_engine.resume_training()
        
        time.sleep(2)  # 再等待2秒
        print("停止训练...")
        training_engine.stop_training()
    
    # 启动控制线程
    control_thread = threading.Thread(target=control_training)
    control_thread.start()
    
    try:
        # 执行训练
        metrics = training_engine.execute_training_loop(learner, train_data, num_epochs=10)
        print(f"训练完成，最终度量: {metrics}")
        
        # 获取训练统计
        stats = training_engine.get_training_stats()
        print(f"训练统计: {stats}")
        
    except Exception as e:
        print(f"训练过程中出现异常: {str(e)}")
    
    control_thread.join()


def error_handling_example():
    """错误处理示例"""
    print("\n=== 错误处理示例 ===")
    
    # 创建配置
    config = create_training_config()
    
    # 创建组件
    registry = ComponentRegistry()
    context = ExecutionContext(config)
    hook_executor = HookExecutor(registry, context)
    learner = SimpleLearner(context, config)
    
    # 创建训练引擎
    training_engine = TrainingEngine(
        hook_executor=hook_executor,
        context=context
    )
    
    # 模拟错误情况
    print("模拟内存不足错误...")
    oom_error = RuntimeError("CUDA out of memory")
    result = training_engine.handle_training_error(oom_error, {"learner": learner})
    print(f"OOM错误处理结果: {'可恢复' if result else '不可恢复'}")
    
    print("模拟模型状态错误...")
    # 设置最佳模型状态
    training_engine._best_model_state = learner.get_model_state()
    model_error = RuntimeError("Model state error")
    result = training_engine.handle_training_error(model_error, {"learner": learner})
    print(f"模型状态错误处理结果: {'可恢复' if result else '不可恢复'}")
    
    print("模拟未知错误...")
    unknown_error = ValueError("Unknown error")
    result = training_engine.handle_training_error(unknown_error, {})
    print(f"未知错误处理结果: {'可恢复' if result else '不可恢复'}")
    
    # 显示错误统计
    stats = training_engine.get_training_stats()
    print(f"错误计数: {stats['error_count']}")
    print(f"恢复计数: {stats['recovery_count']}")


def main():
    """主函数"""
    print("TrainingEngine 使用示例")
    print("=" * 50)
    
    # 基本训练示例
    basic_training_example()
    
    # 带验证的训练示例
    training_with_validation_example()
    
    # 训练控制示例
    training_control_example()
    
    # 错误处理示例
    error_handling_example()
    
    print("\n所有示例执行完成!")


if __name__ == "__main__":
    main()
