# examples/training_engine_with_checkpoint.py
"""
TrainingEngine集成CheckpointHook示例

展示如何在训练引擎中使用CheckpointHook来保存客户端模型和训练状态。
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from fedcl.engine.training_engine import RefactoredEnhancedTrainingEngine
from fedcl.core.execution_context import ExecutionContext
from fedcl.core.checkpoint_hook import CheckpointHook
from fedcl.core.hook import HookPhase
from fedcl.federation.state.state_manager import StateManager
from fedcl.federation.state.state_enums import TrainingPhaseState


def create_sample_data(num_samples=100, input_size=784, num_classes=10):
    """创建样本数据"""
    X = torch.randn(num_samples, input_size)
    y = torch.randint(0, num_classes, (num_samples,))
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=16, shuffle=True)


def create_training_config_with_checkpoint():
    """创建包含checkpoint配置的训练配置"""
    config = {
        "training_plan": {
            "total_epochs": 6,
            "execution_strategy": "sequential",
            "phases": [
                {
                    "name": "client_training_phase",
                    "description": "客户端训练阶段",
                    "epochs": [1, 2, 3, 4, 5, 6],
                    "learner": "client_learner_1",
                    "scheduler": "default_scheduler",
                    "priority": 0
                }
            ]
        },
        "learners": {
            "client_learner_1": {
                "class": "default",
                "learning_rate": 0.001,
                "input_size": 784,
                "num_classes": 10,
                "hidden_sizes": [256, 128],
                "optimizer": {
                    "type": "Adam",
                    "lr": 0.001
                }
            }
        },
        "dataloaders": {
            "client_learner_1": "training_data"
        },
        "schedulers": {
            "default_scheduler": {}
        },
        "checkpoint": {
            "save_frequency": 2,  # 每2个epoch保存一次
            "save_model": True,
            "save_optimizer": True,
            "save_scheduler": True,
            "save_experiment_state": True,
            "checkpoint_dir": "checkpoints/client_models",
            "naming_pattern": "client_model_epoch_{epoch}",
            "max_checkpoints": 3,
            "keep_best_only": False
        }
    }
    
    return config


def demonstration_with_checkpoint():
    """演示训练引擎与CheckpointHook集成"""
    print("=== 训练引擎与CheckpointHook集成演示 ===")
    
    # 1. 创建配置
    config = create_training_config_with_checkpoint()
    config_obj = DictConfig(config)
    
    # 2. 创建执行上下文
    context = ExecutionContext(config_obj, "checkpoint_demo")
    
    # 3. 创建状态管理器
    state_manager = StateManager()
    
    # 4. 创建训练引擎
    training_engine = RefactoredEnhancedTrainingEngine(
        context=context,
        config=config,
        control_state_manager=state_manager
    )
    
    # 5. 创建CheckpointHook
    checkpoint_config = OmegaConf.create(config["checkpoint"])
    checkpoint_hook = CheckpointHook(
        phase=HookPhase.AFTER_EPOCH.value,
        checkpoint_config=checkpoint_config,
        priority=100,
        name="client_model_checkpoint"
    )
    
    # 6. 注册CheckpointHook
    training_engine.register_training_hooks([checkpoint_hook])
    print("CheckpointHook已注册")
    
    # 7. 创建训练数据
    training_data = create_sample_data(num_samples=200)
    
    # 8. 初始化训练环境
    print("初始化训练环境...")
    training_engine.initialize_training()
    
    # 9. 手动添加数据加载器（因为我们有真实数据）
    training_engine.dataloaders["client_learner_1"] = training_data
    training_engine.dataloaders["training_data"] = training_data
    
    print(f"训练状态: {training_engine.training_state}")
    print(f"已创建learners: {list(training_engine.learners.keys())}")
    
    # 10. 执行训练计划
    print("开始执行训练计划...")
    try:
        results = training_engine.execute_training_plan()
        
        print("\n=== 训练结果 ===")
        for phase_name, result in results.items():
            print(f"阶段: {phase_name}")
            print(f"  执行的epochs: {result.executed_epochs}")
            print(f"  成功: {result.success}")
            print(f"  执行时间: {result.execution_time:.2f}s")
            if result.metrics:
                final_metrics = result.get_final_metrics()
                print(f"  最终指标: {final_metrics}")
        
        # 11. 显示训练统计
        print("\n=== 训练统计 ===")
        stats = training_engine.get_training_statistics()
        print(f"总阶段数: {stats['total_phases']}")
        print(f"成功阶段数: {stats['完成_phases']}")
        print(f"失败阶段数: {stats['failed_phases']}")
        print(f"训练状态: {stats['training_state']}")
        
        # 12. 检查保存的检查点
        checkpoint_dir = Path(config["checkpoint"]["checkpoint_dir"])
        if checkpoint_dir.exists():
            saved_checkpoints = list(checkpoint_dir.glob("*"))
            print(f"\n=== 保存的检查点 ===")
            print(f"检查点目录: {checkpoint_dir}")
            print(f"保存的检查点数量: {len(saved_checkpoints)}")
            for cp in saved_checkpoints:
                print(f"  - {cp.name}")
        else:
            print("\n检查点目录不存在，可能保存失败")
        
        # 13. 保存训练引擎检查点
        engine_checkpoint_path = "checkpoints/training_engine_final.pt"
        training_engine.save_training_checkpoint(engine_checkpoint_path)
        print(f"训练引擎检查点已保存到: {engine_checkpoint_path}")
        
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 14. 清理训练环境
        training_engine.cleanup_training_environment()
        print("训练环境已清理")


def main():
    """主函数"""
    demonstration_with_checkpoint()


if __name__ == "__main__":
    main()
