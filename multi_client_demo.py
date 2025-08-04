#!/usr/bin/env python3
"""
多客户端训练引擎验证脚本
展示训练引擎处理多个客户端和调度器的能力
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fedcl.core.execution_context import ExecutionContext
from fedcl.federation.state.state_manager import StateManager, TrainingPhaseState
from fedcl.engine.training_engine import RefactoredEnhancedTrainingEngine
from omegaconf import DictConfig

def create_multi_client_config():
    """创建多客户端配置"""
    return {
        "dataloaders": {
            "client_A": {
                "type": "StandardDataLoader",
                "dataset": "MNIST",
                "batch_size": 32,
                "num_samples": 1000,
                "input_size": [1, 28, 28],
                "num_classes": 10
            },
            "client_B": {
                "type": "StandardDataLoader",
                "dataset": "MNIST", 
                "batch_size": 32,
                "num_samples": 800,
                "input_size": [1, 28, 28],
                "num_classes": 10
            },
            "client_C": {
                "type": "StandardDataLoader",
                "dataset": "MNIST",
                "batch_size": 32,
                "num_samples": 1200,
                "input_size": [1, 28, 28],
                "num_classes": 10
            }
        },
        "learners": {
            "client_A": {
                "class": "default",
                "learning_rate": 0.001,
                "optimizer": {"type": "Adam", "lr": 0.001},
                "input_size": 784,
                "num_classes": 10
            },
            "client_B": {
                "class": "default",
                "learning_rate": 0.002,
                "optimizer": {"type": "SGD", "lr": 0.002},
                "input_size": 784,
                "num_classes": 10
            },
            "client_C": {
                "class": "default", 
                "learning_rate": 0.0015,
                "optimizer": {"type": "Adam", "lr": 0.0015},
                "input_size": 784,
                "num_classes": 10
            }
        },
        "schedulers": {
            "round_scheduler": {
                "type": "StandardEpochScheduler",
                "priority": "NORMAL"
            },
            "adaptive_scheduler": {
                "type": "AdaptiveEpochScheduler",
                "priority": "HIGH"
            }
        },
        "training_plan": {
            "total_epochs": 9,
            "execution_strategy": "sequential",
            "phases": [
                {
                    "name": "client_A_round",
                    "description": "Client A training round",
                    "epochs": [1, 2, 3],
                    "learner": "client_A",
                    "scheduler": "round_scheduler",
                    "priority": 1
                },
                {
                    "name": "client_B_round",
                    "description": "Client B training round", 
                    "epochs": [4, 5, 6],
                    "learner": "client_B",
                    "scheduler": "adaptive_scheduler",
                    "priority": 1
                },
                {
                    "name": "client_C_round",
                    "description": "Client C training round",
                    "epochs": [7, 8, 9],
                    "learner": "client_C", 
                    "scheduler": "round_scheduler",
                    "priority": 2
                }
            ]
        }
    }

def main():
    """主验证函数"""
    print("🚀 多客户端训练引擎功能验证")
    print("=" * 50)
    
    try:
        # 创建配置
        config = create_multi_client_config()
        
        # 创建执行上下文和状态管理器
        context = ExecutionContext(DictConfig({}), "multi_client_demo")
        state_manager = StateManager(
            initial_state=TrainingPhaseState.UNINITIALIZED,
            context=context,
            component_id="demo_engine"
        )
        
        # 创建训练引擎
        print("📦 创建多客户端训练引擎...")
        training_engine = RefactoredEnhancedTrainingEngine(
            context=context,
            config=config,
            control_state_manager=state_manager
        )
        print("✅ 训练引擎创建成功")
        
        # 初始化训练
        print("\n🔧 初始化训练环境...")
        training_engine.initialize_training()
        print(f"✅ 训练初始化成功，状态: {training_engine.training_state}")
        
        # 显示创建的组件
        print(f"\n📋 创建的组件:")
        print(f"   Learners: {list(training_engine.learners.keys())}")
        print(f"   DataLoaders: {list(training_engine.dataloaders.keys())}")
        print(f"   Schedulers: {list(training_engine.scheduler_manager.schedulers.keys())}")
        
        # 显示训练计划
        print(f"\n📅 训练计划:")
        for i, phase in enumerate(training_engine.training_plan.phases):
            print(f"   阶段 {i+1}: {phase.name}")
            print(f"      Learner: {phase.learner_id}")
            print(f"      Scheduler: {phase.scheduler_id}")
            print(f"      Epochs: {phase.epochs}")
            print(f"      优先级: {phase.priority}")
        
        # 执行训练计划
        print(f"\n🎯 执行多客户端训练计划...")
        results = training_engine.execute_training_plan()
        
        # 显示结果
        print(f"\n📊 训练结果:")
        print(f"   总阶段数: {len(results)}")
        
        successful_phases = 0
        for phase_name, result in results.items():
            status = "✅" if result.success else "❌"
            print(f"   {status} {phase_name}: {len(result.executed_epochs)} epochs, 耗时 {result.execution_time:.3f}s")
            if result.success:
                successful_phases += 1
        
        print(f"\n🎉 多客户端训练完成!")
        print(f"   成功率: {successful_phases}/{len(results)} ({100*successful_phases/len(results):.1f}%)")
        print(f"   最终状态: {training_engine.training_state}")
        
        # 清理
        training_engine.cleanup_training_environment()
        print("✅ 环境清理完成")
        
        return True
        
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import fedcl  # 确保框架初始化
    success = main()
    sys.exit(0 if success else 1)
