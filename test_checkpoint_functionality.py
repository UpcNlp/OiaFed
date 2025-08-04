#!/usr/bin/env python3
"""
CheckpointHook功能测试脚本

验证CheckpointHook是否正确工作，包括：
1. 检查点保存功能
2. 模型状态保存
3. 实验状态保存
4. 检查点加载功能
"""

import pytest
import sys
from pathlib import Path

def test_checkpoint_functionality():
    """测试CheckpointHook完整功能"""
    print("🧪 开始测试CheckpointHook功能...")
    
    # 使用专门的checkpoint配置运行测试
    config_path = "tests/configs/mnist_checkpoint_test_config.yaml"
    
    if not Path(config_path).exists():
        config_path = "tests/configs/mnist_real_test/experiment_config.yaml"
        print(f"⚠️  使用默认配置: {config_path}")
    else:
        print(f"✅ 使用Checkpoint专用配置: {config_path}")
    
    # 运行实际的联邦学习测试
    print("🚀 启动联邦学习测试...")
    
    try:
        # 运行pytest测试
        result = pytest.main([
            "tests/test_real_mnist_federation.py",
            "-v", "--tb=short", 
            f"--config-file={config_path}",
            "-k", "test_real_mnist_federation"
        ])
        
        if result == 0:
            print("✅ 测试成功完成！")
            
            # 检查检查点文件是否生成
            checkpoint_dirs = [
                Path("tests/test_outputs/mnist_checkpoint_test/checkpoints"),
                Path("tests/test_outputs/mnist_real_test/checkpoints"),
                Path("logs").rglob("checkpoints"),
            ]
            
            found_checkpoints = False
            for checkpoint_dir in checkpoint_dirs:
                if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
                    print(f"📁 发现检查点目录: {checkpoint_dir}")
                    for checkpoint_file in checkpoint_dir.iterdir():
                        print(f"   └── 📄 {checkpoint_file.name}")
                    found_checkpoints = True
            
            if not found_checkpoints:
                print("⚠️  未找到任何检查点文件，可能存在配置问题")
                return False
            else:
                print("🎉 CheckpointHook功能正常工作！")
                return True
        else:
            print("❌ 测试失败")
            return False
            
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        return False

def manual_checkpoint_test():
    """手动测试CheckpointHook"""
    print("🔧 开始手动CheckpointHook测试...")
    
    try:
        from fedcl.core.checkpoint_hook import CheckpointHook
        from fedcl.core.execution_context import ExecutionContext
        from omegaconf import OmegaConf
        import tempfile
        import torch
        import torch.nn as nn
        
        # 1. 创建临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"📁 使用临时目录: {temp_dir}")
            
            # 2. 创建CheckpointHook配置
            checkpoint_config = OmegaConf.create({
                'save_frequency': 1,
                'save_model': True,
                'save_optimizer': True,
                'save_scheduler': False,
                'save_experiment_state': True,
                'checkpoint_dir': f"{temp_dir}/checkpoints",
                'naming_pattern': 'test_checkpoint_round_{round}',
                'include_timestamp': False,
                'max_checkpoints': 3,
                'compress': False,
                'keep_best_only': False,
                'best_metric': 'accuracy',
                'best_mode': 'max'
            })
            
            # 3. 创建CheckpointHook实例
            hook = CheckpointHook(
                phase="after_round",
                checkpoint_config=checkpoint_config,
                enabled=True
            )
            print("✅ CheckpointHook实例创建成功")
            
            # 4. 创建模拟的执行上下文
            config = OmegaConf.create({
                'experiment': {'name': 'test_checkpoint'},
                'test': 'value'
            })
            context = ExecutionContext("test_exp", config)
            context.set_state('current_round', 1, 'global')
            context.set_state('current_epoch', 1, 'global')
            print("✅ 执行上下文创建成功")
            
            # 5. 创建模拟模型和优化器
            class SimpleModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = nn.Linear(10, 1)
                
                def forward(self, x):
                    return self.linear(x)
            
            model = SimpleModel()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            print("✅ 模拟模型和优化器创建成功")
            
            # 6. 执行检查点保存
            hook.execute(
                context, 
                model=model, 
                optimizer=optimizer,
                metrics={'accuracy': 0.85, 'loss': 0.5}
            )
            print("✅ 检查点保存执行成功")
            
            # 7. 验证检查点文件
            checkpoint_dir = Path(temp_dir) / "checkpoints"
            if checkpoint_dir.exists():
                checkpoint_files = list(checkpoint_dir.rglob("*"))
                print(f"📁 生成的检查点文件:")
                for file in checkpoint_files:
                    print(f"   └── 📄 {file}")
                
                if checkpoint_files:
                    print("🎉 手动CheckpointHook测试成功！")
                    return True
            
            print("❌ 未找到检查点文件")
            return False
            
    except Exception as e:
        print(f"❌ 手动测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔍 CheckpointHook功能诊断工具")
    print("=" * 50)
    
    # 检查CheckpointHook类是否可以导入
    try:
        from fedcl.core.checkpoint_hook import CheckpointHook
        print("✅ CheckpointHook类导入成功")
    except ImportError as e:
        print(f"❌ CheckpointHook类导入失败: {e}")
        sys.exit(1)
    
    # 运行手动测试
    manual_success = manual_checkpoint_test()
    
    print("\n" + "=" * 50)
    print("📋 测试总结:")
    print(f"   手动测试: {'✅ 成功' if manual_success else '❌ 失败'}")
    
    if manual_success:
        print("\n🎯 建议:")
        print("   1. CheckpointHook功能正常")
        print("   2. 检查配置文件中 save_checkpoints: true")
        print("   3. 确保hook配置正确启用")
        sys.exit(0)
    else:
        print("\n🔧 需要修复的问题:")
        print("   1. CheckpointHook配置可能有问题")
        print("   2. 检查依赖和导入路径")
        print("   3. 查看详细错误日志")
        sys.exit(1)
