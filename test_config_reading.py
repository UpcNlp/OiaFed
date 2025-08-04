#!/usr/bin/env python3
"""
简单的配置读取测试脚本
"""

import sys
import os
import yaml
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.abspath('.'))

from fedcl.config.config_manager import ConfigManager

def test_config_reading():
    """测试配置读取"""
    config_path = "tests/configs/mnist_real_test/client_1_config.yaml"
    
    print(f"测试配置文件: {config_path}")
    
    # 直接读取YAML文件
    with open(config_path, 'r', encoding='utf-8') as f:
        raw_config = yaml.safe_load(f)
    
    print(f"原始配置中的键: {list(raw_config.keys())}")
    
    if 'test_datas' in raw_config:
        print(f"test_datas: {raw_config['test_datas']}")
    else:
        print("原始配置中没有test_datas")
        
    if 'evaluators' in raw_config:
        print(f"evaluators: {raw_config['evaluators']}")
    else:
        print("原始配置中没有evaluators")
        
    if 'evaluation' in raw_config:
        print(f"evaluation: {raw_config['evaluation']}")
    else:
        print("原始配置中没有evaluation")
    
    # 使用ConfigManager读取
    print("\n使用ConfigManager读取:")
    config_manager = ConfigManager(config_path=config_path)
    
    print(f"ConfigManager配置中的键: {list(config_manager.config.keys())}")
    
    test_datas = config_manager.get_test_data_configs()
    print(f"get_test_data_configs(): {test_datas}")
    
    evaluators = config_manager.get_evaluator_configs()
    print(f"get_evaluator_configs(): {evaluators}")
    
    evaluation = config_manager.get_evaluation_config()
    print(f"get_evaluation_config(): {evaluation}")

if __name__ == "__main__":
    test_config_reading()
