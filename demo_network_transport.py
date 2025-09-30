#!/usr/bin/env python3
"""
Network模式传输功能验证
演示网络环境下的NetworkTransport通信
"""

import asyncio
import socket
import time
import sys
from pathlib import Path
import json

# 添加项目路径
root = Path(__file__).parent
sys.path.append(str(root))

from fedcl.transport.network import NetworkTransport
from fedcl.types import TransportConfig
from fedcl.exceptions import TransportError, TimeoutError


def get_local_ip():
    """获取本机IP地址"""
    try:
        # 创建一个UDP socket连接，不实际发送数据
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


async def start_server(server_host: str, server_port: int, client_host: str, client_port: int):
    """启动服务器端"""
    print(f"🖥️  启动服务器: {server_host}:{server_port}")
    
    server_id = f"network_server_{server_host}_{server_port}"
    client_id = f"network_client_{client_host}_{client_port}_12345"
    
    config = TransportConfig(
        type="network",
        timeout=30.0,
        retry_attempts=3,
        specific_config={
            "host": server_host,
            "port": server_port,
            "websocket_port": 8002,  # 避免端口冲突
            "protocol": "tcp",
            "max_connections": 10
        }
    )
    
    transport = NetworkTransport(config)
    
    try:
        # 启动传输层
        await transport.start()
        print(f"✅ 服务器网络传输层启动成功")
        
        # 启动事件监听器
        await transport.start_event_listener(server_id)
        print(f"🎧 服务器事件监听器启动: {server_id}")
        
        # 等待客户端连接
        print(f"⏳ 等待客户端连接...")
        await asyncio.sleep(3)
        
        # 发送训练任务
        training_task = {
            "task_type": "federated_train",
            "round": 1,
            "global_model": {
                "layer1_weights": [0.1, 0.2, 0.3],
                "layer2_weights": [0.4, 0.5, 0.6],
                "layer1_bias": [0.01, 0.02],
                "layer2_bias": [0.03]
            },
            "hyperparams": {
                "learning_rate": 0.01,
                "batch_size": 32,
                "local_epochs": 5
            },
            "client_data_config": {
                "dataset": "CIFAR-10",
                "samples": 1000,
                "classes": [0, 1, 2, 3, 4]
            },
            "timestamp": time.time()
        }
        
        print(f"📤 服务器发送联邦训练任务:")
        print(f"   目标: {server_id} -> {client_id}")
        print(f"   轮次: {training_task['round']}")
        print(f"   数据集: {training_task['client_data_config']['dataset']}")
        
        start_time = time.time()
        result = await transport.send(server_id, client_id, training_task)
        end_time = time.time()
        
        print(f"📨 服务器收到训练结果:")
        print(f"   状态: {result.get('status', 'unknown')}")
        print(f"   准确率: {result.get('accuracy', 'N/A')}")
        print(f"   损失: {result.get('loss', 'N/A')}")
        print(f"   网络延迟: {(end_time - start_time)*1000:.2f}ms")
        
        # 发送模型聚合请求
        aggregation_task = {
            "task_type": "model_aggregation",
            "round": 1,
            "client_weights": result.get("updated_model", {}),
            "aggregation_method": "fedavg",
            "timestamp": time.time()
        }
        
        print(f"📤 服务器发送模型聚合确认:")
        agg_result = await transport.send(server_id, client_id, aggregation_task)
        print(f"📨 聚合确认: {agg_result}")
        
    except Exception as e:
        print(f"❌ 服务器错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await transport.stop()
        print(f"🔌 服务器网络传输层停止")


async def start_client(client_host: str, client_port: int, server_host: str, server_port: int):
    """启动客户端"""
    print(f"👤 启动客户端: {client_host}:{client_port}")
    
    client_id = f"network_client_{client_host}_{client_port}_12345"
    server_id = f"network_server_{server_host}_{server_port}"
    
    config = TransportConfig(
        type="network",
        timeout=30.0,
        retry_attempts=3,
        specific_config={
            "host": client_host,
            "port": client_port,
            "websocket_port": 8003,  # 避免端口冲突
            "protocol": "tcp",
            "connect_to": {
                "host": server_host,
                "port": server_port
            }
        }
    )
    
    transport = NetworkTransport(config)
    
    try:
        # 启动传输层
        await transport.start()
        print(f"✅ 客户端网络传输层启动成功")
        
        # 定义请求处理器
        async def handle_federated_request(source: str, data: dict):
            print(f"📥 客户端收到联邦学习任务: {source} -> {client_id}")
            task_type = data.get('task_type', 'unknown')
            print(f"   任务类型: {task_type}")
            
            if task_type == 'federated_train':
                print(f"🤖 执行联邦训练...")
                print(f"   轮次: {data.get('round', 'N/A')}")
                print(f"   数据集: {data.get('client_data_config', {}).get('dataset', 'N/A')}")
                print(f"   本地样本数: {data.get('client_data_config', {}).get('samples', 'N/A')}")
                
                # 模拟本地训练过程
                await asyncio.sleep(2)  # 模拟训练时间
                
                # 模拟训练结果
                global_model = data.get('global_model', {})
                updated_model = {}
                for key, weights in global_model.items():
                    # 模拟权重更新（添加小的随机变化）
                    if isinstance(weights, list):
                        updated_model[key] = [w + 0.001 * (i + 1) for i, w in enumerate(weights)]
                    else:
                        updated_model[key] = weights
                
                result = {
                    "status": "training_completed",
                    "round": data.get('round', 0),
                    "updated_model": updated_model,
                    "local_metrics": {
                        "accuracy": 0.87,
                        "loss": 0.23,
                        "samples_trained": data.get('client_data_config', {}).get('samples', 0),
                        "epochs_completed": data.get('hyperparams', {}).get('local_epochs', 0)
                    },
                    "training_time": 2.0,
                    "client_id": client_id
                }
                
            elif task_type == 'model_aggregation':
                print(f"📊 处理模型聚合确认...")
                result = {
                    "status": "aggregation_acknowledged",
                    "round": data.get('round', 0),
                    "client_id": client_id,
                    "ready_for_next_round": True
                }
            else:
                result = {
                    "status": "error",
                    "error": f"Unknown task type: {task_type}",
                    "client_id": client_id
                }
            
            print(f"✅ 任务完成，返回结果")
            return result
        
        # 注册处理器并启动监听器  
        transport.register_request_handler(handle_federated_request)
        await transport.start_event_listener(client_id)
        print(f"🎧 客户端事件监听器启动: {client_id}")
        
        # 等待任务处理
        print(f"⏳ 客户端等待联邦学习任务...")
        await asyncio.sleep(15)  # 等待服务器发送任务
        
    except Exception as e:
        print(f"❌ 客户端错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await transport.stop()
        print(f"🔌 客户端网络传输层停止")


async def run_network_demo():
    """运行网络模式演示"""
    print("="*70)
    print("🧪 Network模式传输功能验证")
    print("="*70)
    
    # 配置网络地址
    local_ip = get_local_ip()
    server_host = local_ip
    server_port = 8100
    client_host = local_ip
    client_port = 8101
    
    print(f"📋 网络配置信息:")
    print(f"   本机IP: {local_ip}")
    print(f"   服务器地址: {server_host}:{server_port}")
    print(f"   客户端地址: {client_host}:{client_port}")
    print(f"   通信协议: TCP")
    print()
    
    # 并发启动服务器和客户端
    try:
        await asyncio.gather(
            start_client(client_host, client_port, server_host, server_port),
            start_server(server_host, server_port, client_host, client_port)
        )
        
        print()
        print("="*70)
        print("✅ Network模式验证完成！")
        print("="*70)
        
    except Exception as e:
        print(f"❌ 网络演示失败: {e}")
        import traceback
        traceback.print_exc()


def check_port_available(host: str, port: int) -> bool:
    """检查端口是否可用"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, port))
            return True
    except OSError:
        return False


async def main():
    """主函数"""
    print("🔍 检查网络环境...")
    
    local_ip = get_local_ip()
    server_port = 8100
    client_port = 8101
    
    # 检查端口可用性
    if not check_port_available(local_ip, server_port):
        print(f"❌ 服务器端口 {server_port} 不可用")
        return
    
    if not check_port_available(local_ip, client_port):
        print(f"❌ 客户端端口 {client_port} 不可用")
        return
    
    print(f"✅ 网络环境检查通过")
    print()
    
    await run_network_demo()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 用户中断")
    except Exception as e:
        print(f"❌ 程序错误: {e}")
        import traceback
        traceback.print_exc()
