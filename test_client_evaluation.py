#!/usr/bin/env python3
"""
简单的客户端评估配置测试
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from fedcl.engine.training_engine import TrainingEngine
from fedcl.config.config_manager import ConfigManager

def test_client_evaluation_config():
    """测试客户端评估配置读取"""
    config_path = "tests/configs/mnist_real_test/client_1_config.yaml"
    
    print(f"测试配置文件: {config_path}")
    
    # 创建配置管理器
    config_manager = ConfigManager(config_path=config_path)
    
    print("=== 配置管理器测试 ===")
    test_datas = config_manager.get_test_data_configs()
    print(f"test_datas: {list(test_datas.keys()) if test_datas else '空'}")
    
    evaluators = config_manager.get_evaluator_configs()
    print(f"evaluators: {list(evaluators.keys()) if evaluators else '空'}")
    
    evaluation = config_manager.get_evaluation_config()
    print(f"evaluation tasks: {len(evaluation.get('tasks', [])) if evaluation else 0}")
    
    # 创建训练引擎（注意：这可能需要额外的配置）
    print("\n=== 训练引擎测试 ===")
    try:
        # 检查训练引擎的评估初始化方法
        from fedcl.engine.training_engine import RefactoredEnhancedTrainingEngine
        
        # 模拟测试数据初始化
        print("测试训练引擎的评估配置方法...")
        
        # 直接测试配置读取方法
        test_data_configs = config_manager.get_test_data_configs()
        evaluator_configs = config_manager.get_evaluator_configs()
        evaluation_config = config_manager.get_evaluation_config()
        
        print(f"✓ 测试数据配置读取成功: {len(test_data_configs)} 个")
        print(f"✓ 评估器配置读取成功: {len(evaluator_configs)} 个") 
        print(f"✓ 评估任务配置读取成功: {len(evaluation_config.get('tasks', []))} 个任务")
        
        # 检查具体内容
        for name, config in test_data_configs.items():
            print(f"  - 测试数据集 '{name}': {config.get('dataset_config', {}).get('name', 'unknown')}")
            
        for name, config in evaluator_configs.items():
            print(f"  - 评估器 '{name}': {config.get('class', 'unknown')} -> {config.get('test_data', 'unknown')}")
        
        for i, task in enumerate(evaluation_config.get('tasks', [])):
            print(f"  - 评估任务 {i+1}: {task.get('learner', 'unknown')} + {task.get('evaluator', 'unknown')} + {task.get('test_data', 'unknown')}")
        
    except Exception as e:
        print(f"训练引擎测试失败: {e}")

if __name__ == "__main__":
    test_client_evaluation_config()
