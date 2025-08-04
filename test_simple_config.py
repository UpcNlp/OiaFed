#!/usr/bin/env python3
"""
简化的配置测试，不使用验证
"""

import sys
import os
import yaml
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.abspath('.'))

class SimpleConfigManager:
    """简化的配置管理器，不验证"""
    
    def __init__(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
    
    def get_test_data_configs(self):
        """获取测试数据配置"""
        return self.config.get('test_datas', {})
    
    def get_evaluator_configs(self):
        """获取评估器配置"""
        return self.config.get('evaluators', {})
    
    def get_evaluation_config(self):
        """获取评估任务配置"""
        return self.config.get('evaluation', {})

def test_config_reading():
    """测试配置读取"""
    config_path = "tests/configs/mnist_real_test/client_1_config.yaml"
    
    print(f"测试配置文件: {config_path}")
    
    # 使用简化的ConfigManager读取
    config_manager = SimpleConfigManager(config_path)
    
    print(f"配置中的键: {list(config_manager.config.keys())}")
    
    test_datas = config_manager.get_test_data_configs()
    print(f"\nget_test_data_configs(): {test_datas}")
    
    evaluators = config_manager.get_evaluator_configs()
    print(f"\nget_evaluator_configs(): {evaluators}")
    
    evaluation = config_manager.get_evaluation_config()
    print(f"\nget_evaluation_config(): {evaluation}")

if __name__ == "__main__":
    test_config_reading()
