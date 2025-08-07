#!/usr/bin/env python3
"""
FedCL 使用示例

演示如何使用新的命令行和脚本启动功能
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from fedcl.cli import launch_federation, quick_start


def example_script_launch():
    """示例：Python脚本启动"""
    print("=== Python脚本启动示例 ===")
    
    # 方式1：简单启动
    config_path = "examples/config_templates/server_client_configs"
    
    try:
        # 启动联邦学习
        results = launch_federation(
            config=config_path,
            daemon=False,  # 前台运行
            log_level="DEBUG",  # 修改为DEBUG级别以查看详细日志
            enable_checkpoint=True
        )
        
        print(f"实验完成，结果: {results}")
        
    except Exception as e:
        print(f"启动失败: {e}")


def example_quick_start():
    """示例：快速启动"""
    print("=== 快速启动示例 ===")
    
    # 使用我们创建的简化配置
    config_path = "demo_configs"
    
    print("📋 检查点功能已启用:")
    print("   - 服务端: 每轮结束后自动保存")
    print("   - 客户端: 本地训练后自动保存")
    print("   - 保存位置: checkpoints/mnist_demo_*/")
    print()
    
    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            print(f"尝试启动联邦学习 (尝试 {attempt + 1}/{max_retries + 1})")
            
            # 快速启动（最简单的方式）
            results = quick_start(config_path)
            print(f"快速启动完成: {results}")
            
            # 显示生成的检查点文件
            show_checkpoint_summary()
            return results
            
        except Exception as e:
            print(f"第 {attempt + 1} 次尝试失败: {e}")
            if attempt < max_retries:
                print(f"等待5秒后重试...")
                import time
                time.sleep(5)
            else:
                print(f"所有尝试都失败了")
                raise


def show_checkpoint_summary():
    """显示检查点保存摘要"""
    print("\n💾 检查点保存摘要:")
    from pathlib import Path
    
    checkpoint_dirs = [
        ("服务端", "checkpoints/mnist_demo_server"),
        ("客户端1", "checkpoints/mnist_demo_client_1"),
        ("客户端2", "checkpoints/mnist_demo_client_2"), 
        ("客户端3", "checkpoints/mnist_demo_client_3")
    ]
    
    for name, dir_path in checkpoint_dirs:
        path = Path(dir_path)
        if path.exists():
            files = list(path.glob("*.pkl"))
            print(f"   {name}: {len(files)} 个检查点文件")
        else:
            print(f"   {name}: 目录不存在")


def example_with_missing_config():
    """示例：处理缺失配置的情况"""
    print("=== 缺失配置处理示例 ===")
    
    # 使用一个不存在的配置路径
    config_path = "nonexistent_configs"
    
    try:
        print(f"尝试使用不存在的配置路径: {config_path}")
        # 系统会自动创建默认配置
        results = quick_start(config_path)
        print(f"使用默认配置启动成功: {results}")
        print(f"默认配置已创建在: {config_path}/")
        
    except Exception as e:
        print(f"即使使用默认配置也失败了: {e}")


def example_daemon_mode():
    """示例：后台模式"""
    print("=== 后台模式示例 ===")
    
    config_path = "examples/config_templates/server_client_configs"
    
    try:
        # 后台运行
        results = launch_federation(
            config=config_path,
            daemon=True,  # 后台模式
            quiet=True    # 静默模式
        )
        
        print(f"后台启动完成: {results}")
        
    except Exception as e:
        print(f"后台启动失败: {e}")


def show_cli_examples():
    """显示命令行使用示例"""
    print("\n=== 命令行使用示例 ===")
    
    examples = [
        # 基本使用
        "fedcl run examples/config_templates/server_client_configs",
        
        # 单配置文件
        "fedcl run examples/config_templates/server_client_configs/server_config.yaml",
        
        # 后台运行
        "fedcl daemon examples/config_templates/server_client_configs",
        
        # 查看状态
        "fedcl status",
        
        # 查看日志
        "fedcl logs --follow",
        
        # 停止后台进程
        "fedcl stop",
        
        # 清理文件
        "fedcl clean",
        
        # 初始化新项目
        "fedcl init my_project",
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example}")
    
    print("\n中断信号处理:")
    print("- Ctrl+C: 优雅退出")
    print("- SIGTERM: 终止并清理")
    print("- SIGHUP: 重新加载（Unix系统）")
    
    print("\n后台模式:")
    print("- 查看状态: fedcl status")
    print("- 查看日志: fedcl logs --follow")
    print("- 停止进程: fedcl stop")
    
    print("\n安装:")
    print("- 运行安装脚本: ./install.sh")
    print("- 或直接使用: ./bin/fedcl <command>")


def show_features():
    """显示新功能特性"""
    print("\n=== 新功能特性 ===")
    
    features = [
        "✅ 命令行启动支持",
        "✅ Python脚本启动接口", 
        "✅ 控制台实时日志输出",
        "✅ 分布式模式日志显示",
        "✅ 信号处理和优雅退出",
        "✅ 后台运行模式",
        "✅ 检查点自动保存",
        "✅ 线程管理和清理",
        "✅ 日志级别控制",
        "✅ 静默模式支持",
        "🆕 CheckpointHook 自动检查点保存",
        "🆕 分布式检查点管理",
        "🆕 可配置保存策略"
    ]
    
    for feature in features:
        print(f"  {feature}")
    
    print("\n=== 💾 检查点功能特性 ===")
    checkpoint_features = [
        "🔄 自动保存模型参数和训练状态",
        "📊 服务端和客户端分别管理检查点",
        "⚙️ 可配置保存频率和策略",
        "📁 智能文件命名和目录组织",
        "🧹 自动清理过期检查点",
        "🏷️ 支持时间戳和元数据",
        "📈 保存训练统计和评估指标",
        "🔧 支持断点续训（未来版本）"
    ]
    
    for feature in checkpoint_features:
        print(f"  {feature}")
    
    print("\n=== 📋 检查点配置说明 ===")
    print("服务端配置:")
    print("   - 执行阶段: after_round (每轮结束后)")
    print("   - 保存位置: checkpoints/mnist_demo_server/")
    print("   - 最大数量: 5个检查点")
    print("   - 评判指标: accuracy (准确率)")
    
    print("\n客户端配置:")
    print("   - 执行阶段: after_local_training (本地训练后)")
    print("   - 保存位置: checkpoints/mnist_demo_client_*/")
    print("   - 最大数量: 3个检查点")
    print("   - 评判指标: loss (损失函数)")
    
    print("\n=== 日志输出模式 ===")
    print("📊 分布式模式（服务端+客户端）:")
    print("   - 显示联邦学习协调日志")
    print("   - 显示客户端注册和训练过程")
    print("   - 显示聚合和评估结果")
    print("   - 💾 显示检查点保存状态")
    
    print("\n📱 单配置模式:")
    print("   - 显示相应组件的日志")
    print("   - 简化的输出格式")
    print("   - 💾 检查点保存提示")
    
    print("\n🎯 控制台输出特性:")
    print("   - 彩色日志显示")
    print("   - 组件标识（SERVER/CLIENT/FEDERATION）")
    print("   - 时间戳和日志级别")
    print("   - 实验进度指示")
    print("   - 💾 检查点保存进度")


if __name__ == "__main__":
    print("FedCL 新功能演示")
    print("="*50)
    
    # 显示功能特性
    show_features()
    
    # 显示命令行示例
    show_cli_examples()
    
    # 如果有参数，运行相应的示例
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "script":
            example_script_launch()
        elif mode == "quick":
            example_quick_start()
        elif mode == "missing":
            example_with_missing_config()
        elif mode == "daemon":
            example_daemon_mode()
        else:
            print(f"未知模式: {mode}")
            print("可用模式: script, quick, missing, daemon")
    else:
        print("\n使用方法:")
        print("python demo.py [script|quick|missing|daemon]")
        print("\n示例说明:")
        print("- script: 使用现有配置启动")
        print("- quick: 快速启动演示（包含检查点功能）")
        print("- missing: 演示缺失配置的处理")
        print("- daemon: 后台模式演示")
        print("\n💾 检查点功能演示:")
        print("python demo_checkpoint.py config  # 查看检查点配置")
        print("python demo_checkpoint.py run     # 完整检查点演示")
        print("python demo_checkpoint.py check   # 检查检查点文件")
        print("\n或者直接查看上面的示例代码")
