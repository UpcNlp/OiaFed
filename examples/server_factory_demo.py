#!/usr/bin/env python3
"""
服务端配置工厂演示

展示如何使用配置文件创建联邦学习服务端
"""

import asyncio
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fedcl.utils.auto_logger import setup_auto_logging
from fedcl.fl.server import create_server_from_config, create_servers_from_config
from fedcl.fl import SimpleTrainer
from fedcl.registry import register_class

async def demo_single_server():
    """演示从配置文件创建单个服务端"""
    print("=" * 50)
    print("演示: 从配置创建单个服务端")
    print("=" * 50)
    
    # 1. 注册自定义训练器（如果有的话）
    register_class("SimpleTrainer", SimpleTrainer)
    
    # 2. 从配置文件创建服务端
    config_path = project_root / "config" / "server_config.yaml"
    
    try:
        server = create_server_from_config(config_path)
        
        print(f"✓ 服务端创建成功: {server.trainer_id}")
        print(f"✓ 学习器代理数量: {len(server.learners)}")
        print(f"✓ 学习器ID列表: {list(server.learners.keys())}")
        
        # 3. 检查学习器代理状态
        for learner_id, proxy in server.learners.items():
            try:
                status = await proxy.get_status()
                print(f"  - {learner_id}: {status.get('status', 'unknown')}")
            except:
                print(f"  - {learner_id}: 无法获取状态")
                
        print("✓ 单个服务端演示完成")
        
    except Exception as e:
        print(f"✗ 服务端创建失败: {e}")


async def demo_multiple_servers():
    """演示批量创建多个服务端"""
    print("\n" + "=" * 50)
    print("演示: 批量创建多个服务端")
    print("=" * 50)
    
    # 多服务端配置
    multi_config = {
        "servers": {
            "main_server": {
                "id": "main_fed_server",
                "trainer": {
                    "type": "SimpleTrainer",
                    "kwargs": {"max_concurrent": 3}
                },
                "transport": {
                    "type": "MemoryTransport"
                },
                "learners": [
                    {"id": "client1"},
                    {"id": "client2"}
                ]
            },
            "backup_server": {
                "id": "backup_fed_server", 
                "trainer": {
                    "type": "SimpleTrainer",
                    "kwargs": {"max_concurrent": 2}
                },
                "transport": {
                    "type": "MemoryTransport"
                },
                "learners": [
                    {"id": "client3"},
                    {"id": "client4"}
                ]
            }
        }
    }
    
    try:
        servers = create_servers_from_config(multi_config)
        
        print(f"✓ 成功创建 {len(servers)} 个服务端")
        
        for server_name, server in servers.items():
            print(f"  - {server_name}: {server.trainer_id}")
            print(f"    学习器: {list(server.learners.keys())}")
            
        print("✓ 多服务端演示完成")
        
    except Exception as e:
        print(f"✗ 批量创建服务端失败: {e}")


async def demo_custom_config():
    """演示自定义配置"""
    print("\n" + "=" * 50)
    print("演示: 自定义配置创建服务端")
    print("=" * 50)
    
    # 自定义配置字典
    custom_config = {
        "server": {
            "id": "custom_server",
            "trainer": {
                "type": "SimpleTrainer",
                "args": [],
                "kwargs": {
                    "max_concurrent": 8
                }
            },
            "transport": {
                "type": "MemoryTransport",
                "kwargs": {}
            },
            "learners": [
                {"id": "learner_a"},
                {"id": "learner_b"},
                {"id": "learner_c"}
            ]
        }
    }
    
    try:
        server = create_server_from_config(custom_config)
        
        print(f"✓ 自定义服务端创建成功: {server.trainer_id}")
        print(f"✓ 最大并发数: {server.max_concurrent}")
        print(f"✓ 学习器数量: {len(server.learners)}")
        
        # 模拟一轮训练（会失败，因为没有实际的客户端连接）
        try:
            print("⚠ 尝试模拟训练（预期会失败）...")
            result = await server.train_round()
            print(f"✓ 训练结果: {result}")
        except Exception as e:
            print(f"✗ 训练失败（预期）: {e}")
            
        print("✓ 自定义配置演示完成")
        
    except Exception as e:
        print(f"✗ 自定义服务端创建失败: {e}")


async def main():
    """主函数"""
    # 初始化日志系统
    setup_auto_logging()
    
    print("联邦学习服务端配置工厂演示")
    print("时间:", asyncio.get_event_loop().time())
    
    try:
        # 运行各种演示
        await demo_single_server()
        await demo_multiple_servers()
        await demo_custom_config()
        
        print("\n" + "=" * 50)
        print("所有演示完成！")
        print("=" * 50)
        
    except KeyboardInterrupt:
        print("\n用户中断演示")
    except Exception as e:
        print(f"\n演示过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
