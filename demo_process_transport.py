#!/usr/bin/env python3
"""
Process模式传输功能验证
演示多进程环境下的ProcessTransport通信
"""

import asyncio
import multiprocessing as mp
import time
import sys
from pathlib import Path
import pickle

# 添加项目路径
root = Path(__file__).parent
sys.path.append(str(root))

from fedcl.transport.process import ProcessTransport
from fedcl.types import TransportConfig
from fedcl.exceptions import TransportError, TimeoutError


# 将处理函数定义在模块级别，以便进程间序列化
async def global_handle_request(source: str, data: dict):
    """全局请求处理函数 - 可以在进程间序列化"""
    print(f"📥 收到任务: {source} -> 处理中")
    task_type = data.get('task_type', 'unknown')
    print(f"   任务类型: {task_type}")
    
    # 模拟任务处理
    if task_type == 'train':
        print(f"🔄 执行训练任务...")
        await asyncio.sleep(1)  # 模拟训练时间
        result = {
            "status": "completed",
            "trained_params": [x * 1.1 for x in data.get('model_params', [])],
            "loss": 0.25,
            "accuracy": 0.89,
            "epochs_completed": data.get('epochs', 0),
            "processing_time": 1.0
        }
    elif task_type == 'evaluate':
        print(f"📊 执行评估任务...")
        await asyncio.sleep(0.5)  # 模拟评估时间
        result = {
            "status": "completed",
            "test_accuracy": 0.92,
            "test_loss": 0.18,
            "samples_processed": data.get('test_data_size', 0)
        }
    else:
        result = {"status": "unknown_task", "error": f"Unknown task type: {task_type}"}
    
    print(f"✅ 任务完成，返回结果")
    return result


async def server_process(server_id: str, client_id: str):
    """服务器进程 - 发送训练任务并接收结果"""
    print(f"🖥️  服务器进程启动: {server_id}")
    
    config = TransportConfig(
        type="process",
        timeout=10.0,
        retry_attempts=3,
        specific_config={
            "max_workers": 2,
            "queue_size": 100
        }
    )
    
    transport = ProcessTransport(config)
    
    try:
        # 启动传输层
        await transport.start()
        print(f"✅ 服务器传输层启动成功")
        
        # 启动事件监听器
        await transport.start_event_listener(server_id)
        print(f"🎧 服务器事件监听器启动: {server_id}")
        
        # 等待客户端准备就绪
        print(f"⏳ 等待客户端 {client_id} 准备就绪...")
        await asyncio.sleep(2)
        
        # 发送训练任务
        training_task = {
            "task_type": "train",
            "model_params": [1.0, 2.0, 3.0, 4.0],
            "epochs": 5,
            "learning_rate": 0.01,
            "timestamp": time.time()
        }
        
        print(f"📤 服务器发送训练任务: {server_id} -> {client_id}")
        print(f"   任务数据: {training_task}")
        
        start_time = time.time()
        result = await transport.send(server_id, client_id, training_task)
        end_time = time.time()
        
        print(f"📨 服务器收到训练结果:")
        print(f"   结果: {result}")
        print(f"   延迟: {(end_time - start_time)*1000:.2f}ms")
        
        # 再发送一个评估任务
        eval_task = {
            "task_type": "evaluate", 
            "test_data_size": 1000,
            "timestamp": time.time()
        }
        
        print(f"📤 服务器发送评估任务: {server_id} -> {client_id}")
        eval_result = await transport.send(server_id, client_id, eval_task)
        print(f"📨 评估结果: {eval_result}")
        
    except Exception as e:
        print(f"❌ 服务器错误: {e}")
    
    finally:
        await transport.stop()
        print(f"🔌 服务器传输层停止")


async def client_process(client_id: str, server_id: str):
    """客户端进程 - 接收任务并返回结果"""
    print(f"👤 客户端进程启动: {client_id}")
    
    config = TransportConfig(
        type="process",
        timeout=10.0,
        retry_attempts=3,
        specific_config={
            "max_workers": 2,
            "queue_size": 100
        }
    )
    
    transport = ProcessTransport(config)
    
    try:
        # 启动传输层
        await transport.start()
        print(f"✅ 客户端传输层启动成功")
        
        # 注册请求处理器 - 使用全局函数
        transport.register_request_handler(client_id, global_handle_request)
        await transport.start_event_listener(client_id)
        print(f"🎧 客户端事件监听器启动: {client_id}")
        
        # 保持运行等待任务
        print(f"⏳ 客户端等待任务...")
        await asyncio.sleep(10)  # 等待足够长的时间处理任务
        
    except Exception as e:
        print(f"❌ 客户端错误: {e}")
    
    finally:
        await transport.stop()
        print(f"🔌 客户端传输层停止")


def run_server(server_id: str, client_id: str):
    """运行服务器进程的包装函数"""
    print(f"🚀 启动服务器进程")
    asyncio.run(server_process(server_id, client_id))


def run_client(client_id: str, server_id: str):
    """运行客户端进程的包装函数"""
    print(f"🚀 启动客户端进程")
    asyncio.run(client_process(client_id, server_id))


async def main():
    """主函数 - 启动多进程通信演示"""
    print("="*60)
    print("🧪 Process模式传输功能验证")
    print("="*60)
    
    # 定义节点ID（根据设计文档格式）
    server_id = "process_server_8000"
    client_id = "process_client_1234_8001"
    
    print(f"📋 配置信息:")
    print(f"   服务器ID: {server_id}")
    print(f"   客户端ID: {client_id}")
    print(f"   通信模式: Process模式 (多进程队列)")
    print()
    
    # 创建进程
    server_proc = mp.Process(target=run_server, args=(server_id, client_id))
    client_proc = mp.Process(target=run_client, args=(client_id, server_id))
    
    try:
        # 先启动客户端，再启动服务器
        print(f"🚀 启动客户端进程...")
        client_proc.start()
        
        await asyncio.sleep(1)  # 等待客户端启动
        
        print(f"🚀 启动服务器进程...")
        server_proc.start()
        
        # 等待进程完成
        server_proc.join(timeout=15)  # 最多等待15秒
        client_proc.join(timeout=5)   # 客户端应该更快结束
        
        print()
        print("="*60)
        if server_proc.exitcode == 0 and client_proc.exitcode == 0:
            print("✅ Process模式验证成功！")
        else:
            print(f"⚠️  进程退出码 - 服务器: {server_proc.exitcode}, 客户端: {client_proc.exitcode}")
        print("="*60)
        
    except Exception as e:
        print(f"❌ 多进程启动失败: {e}")
    
    finally:
        # 确保进程被终止
        if server_proc.is_alive():
            server_proc.terminate()
            server_proc.join()
        if client_proc.is_alive():
            client_proc.terminate()
            client_proc.join()


if __name__ == "__main__":
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 用户中断")
    except Exception as e:
        print(f"❌ 程序错误: {e}")
        import traceback
        traceback.print_exc()
