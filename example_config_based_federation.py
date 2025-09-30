#!/usr/bin/env python3
"""
基于配置文件的联邦学习示例

展示如何使用 create_server_from_config 和 create_client_from_config 
工厂函数来创建联邦学习系统，支持从YAML/JSON配置文件动态创建trainers和learners。
"""

import asyncio
import yaml
import json
from pathlib import Path
from typing import Dict, Any
from fedcl.fl.server import create_server_from_config
from fedcl.fl.client import create_client_from_config

# 导入FedCL框架 - 会自动注册所有训练器和学习器
import fedcl
from fedcl.fl.server import create_server_from_config
from fedcl.fl.client import create_client_from_config
from fedcl.comm import MemoryTransport
from fedcl.utils.auto_logger import get_sys_logger, setup_auto_logging


def merge_configs(defaults: Dict[str, Any], specific: Dict[str, Any]) -> Dict[str, Any]:
    """
    深度合并配置字典
    
    Args:
        defaults: 默认配置
        specific: 特定配置（会覆盖默认配置）
        
    Returns:
        Dict: 合并后的配置
    """
    import copy
    
    result = copy.deepcopy(defaults)
    
    for key, value in specific.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result

async def run_config_based_federation():
    """运行基于配置文件的联邦学习"""
    logger = get_sys_logger("config_federation")
    
    # 1. 创建共享传输层（模拟分布式环境中的网络）
    transport = MemoryTransport()
    
    # 2. 服务端配置
    server_config = {
        "server": {
            "id": "fed_server",
            "trainer": {
                "type": "SimpleTrainer",  # 使用注册的训练器类型
                "args": [],
                "kwargs": {}
            },
            "auto_start": True
        }
    }
    
    # 3. 客户端配置
    client_configs = []
    for i in range(3):
        client_config = {
            "client": {
                "id": f"client_{i+1}",
                "learner": {
                    "type": "default",  # 使用注册的学习器类型
                    "args": [],
                    "kwargs": {
                        "model_config": {
                            "input_size": 784,
                            "hidden_size": 128,
                            "output_size": 10
                        }
                    }
                },
                "rpc": {
                    "allowed_methods": ["train", "get_model", "set_model", "evaluate", "ping"]
                },
                "auto_start": True
            }
        }
        client_configs.append(client_config)
    
    try:
        # 4. 基于配置创建服务端
        logger.info("创建服务端...")
        server = create_server_from_config(server_config, transport)
        await server.start_server()
        
        # 5. 基于配置创建客户端
        logger.info("创建客户端...")
        clients = []
        for i, config in enumerate(client_configs):
            client = create_client_from_config(config, transport)
            clients.append(client)
            logger.info(f"客户端 {i+1} 创建成功")
        
        # 6. 等待客户端连接
        logger.info("等待客户端连接...")
        await asyncio.sleep(2.0)  # 给客户端一些时间连接
        
        # 7. 开始联邦学习训练
        logger.info("开始联邦学习训练...")
        training_config = {
            "epochs": 5,
            "batch_size": 32,
            "learning_rate": 0.01
        }
        
        results = await server.start_training(
            rounds=3,
            expected_learner_count=len(clients),
            wait_timeout=30.0,
            config=training_config
        )
        
        # 8. 显示结果
        logger.info("训练完成！")
        for i, result in enumerate(results):
            round_num = result['round']
            participants = len(result['participating_learners'])
            metrics = result.get('metrics', {})
            logger.info(f"第 {round_num} 轮: 参与者 {participants}, 指标: {metrics}")
        
        return results
        
    except Exception as e:
        logger.error(f"联邦学习执行失败: {e}")
        raise
    finally:
        # 9. 清理资源
        logger.info("清理资源...")
        try:
            await server.stop_server()
            for client in clients:
                await client.stop()
        except Exception as e:
            logger.warning(f"清理资源时出错: {e}")

async def run_from_yaml_config():
    """运行基于YAML配置的联邦学习演示"""
    logger = get_sys_logger()
    logger.info("开始YAML配置联邦学习演示")
    
    # 创建一个共享的 MemoryTransport 实例
    shared_transport = MemoryTransport("shared_transport")
    
    # 配置文件目录
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    # 服务端配置文件
    server_yaml = config_dir / "server_example.yaml"
    server_config = {
        "server": {
            "id": "yaml_fed_server",
            "transport": {
                "type": "memory",
                "kwargs": {}
            },
            "trainer": {
                "type": "SimpleTrainer",
                "args": [],
                "kwargs": {}
            },
            "auto_start": True
        }
    }
    
    with open(server_yaml, 'w', encoding='utf-8') as f:
        yaml.dump(server_config, f, default_flow_style=False, allow_unicode=True)
    
    # 客户端配置文件
    client_yaml = config_dir / "client_example.yaml"
    client_config = {
        "client": {
            "id": "yaml_client_1",
            "transport": {
                "type": "memory",
                "kwargs": {}
            },
            "learner": {
                "type": "SimpleLearner",
                "args": [],
                "kwargs": {}
            },
            "rpc": {
                "allowed_methods": ["train", "get_model", "set_model", "evaluate", "ping"]
            },
            "auto_start": True
        }
    }
    
    with open(client_yaml, 'w', encoding='utf-8') as f:
        yaml.dump(client_config, f, default_flow_style=False, allow_unicode=True)
    
    try:        
        # 从YAML文件创建服务端和客户端
        logger.info("从YAML配置文件创建服务端和客户端...")
        
        # 读取配置文件
        with open(server_yaml, 'r', encoding='utf-8') as f:
            server_config_dict = yaml.safe_load(f)
        with open(client_yaml, 'r', encoding='utf-8') as f:
            client_config_dict = yaml.safe_load(f)
            
        # 传递共享的transport实例给服务端和客户端
        server = create_server_from_config(server_config_dict, transport=shared_transport)
        client = create_client_from_config(client_config_dict, transport=shared_transport)
        
        await server.start_server()
        await asyncio.sleep(1.0)
        
        # 运行单轮训练作为演示
        logger.info("运行演示训练...")
        results = await server.start_training(
            rounds=2,
            expected_learner_count=1,
            wait_timeout=10.0
        )
        
        logger.info(f"YAML配置训练完成: {len(results)} 轮")
        return results
        
    except Exception as e:
        logger.error(f"YAML配置训练失败: {e}")
        raise
    finally:
        # 清理配置文件
        try:
            server_yaml.unlink(missing_ok=True)
            client_yaml.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(run_from_yaml_config())