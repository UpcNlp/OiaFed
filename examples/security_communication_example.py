# examples/security_communication_example.py
"""
安全通信模块使用示例

演示SecurityModule、CommunicationHandler和CommunicationManager的基本使用方法。
"""

import asyncio
import time
import torch
import torch.nn as nn
from omegaconf import DictConfig

from fedcl.communication import (
    SecurityModule,
    CommunicationHandler, 
    CommunicationManager,
    MessageProtocol,
    DataSerializer,
    NetworkInterface
)


class SimpleModel(nn.Module):
    """简单的测试模型"""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
    
    def forward(self, x):
        return self.linear(x)


def security_module_example():
    """SecurityModule使用示例"""
    print("=== SecurityModule 示例 ===")
    
    # 创建安全配置
    security_config = DictConfig({
        'encryption': {
            'algorithm': 'AES-256-GCM',
            'key_rotation_interval': 3600
        },
        'authentication': {
            'method': 'token',
            'token_lifetime': 1800
        },
        'signing': {
            'algorithm': 'HMAC-SHA256',
            'key_size': 256
        }
    })
    
    # 初始化安全模块
    security = SecurityModule(security_config)
    
    # 数据加密解密示例
    test_data = b"Hello, FedCL! This is sensitive data."
    print(f"原始数据: {test_data}")
    
    encrypted_data = security.encrypt_data(test_data)
    print(f"加密数据长度: {len(encrypted_data)} 字节")
    
    decrypted_data = security.decrypt_data(encrypted_data)
    print(f"解密数据: {decrypted_data}")
    print(f"数据完整性: {'✓' if test_data == decrypted_data else '✗'}")
    
    # 数字签名示例
    private_key, public_key = security.generate_key_pair()
    signature = security.sign_data(test_data, private_key)
    is_valid = security.verify_signature(test_data, signature, public_key)
    print(f"数字签名验证: {'✓' if is_valid else '✗'}")
    
    # 客户端认证示例
    client_id = "client_001"
    token = security.generate_client_token(client_id)
    print(f"为客户端 {client_id} 生成令牌: {token[:16]}...")
    
    auth_result = security.authenticate_client(client_id, token)
    print(f"客户端认证: {'✓' if auth_result else '✗'}")
    
    # 数据哈希示例
    hash_value = security.hash_data(test_data)
    print(f"数据哈希: {hash_value[:16]}...")
    
    print()


def communication_handler_example():
    """CommunicationHandler使用示例"""
    print("=== CommunicationHandler 示例 ===")
    
    # 创建组件（在真实环境中这些应该是完整的实现）
    protocol = MessageProtocol()
    serializer = DataSerializer()
    
    security_config = DictConfig({
        'encryption': {'algorithm': 'AES-256-GCM'},
        'authentication': {'method': 'token'}
    })
    security = SecurityModule(security_config)
    
    network_config = DictConfig({
        'host': '0.0.0.0',
        'port': 8080,
        'timeout': 30.0
    })
    network = NetworkInterface(network_config)
    
    # 创建通信处理器
    handler = CommunicationHandler(protocol, serializer, security, network)
    
    # 设置重试策略
    handler.set_retry_policy(max_retries=3, backoff_factor=2.0)
    print("重试策略已设置: 最大重试3次，退避因子2.0")
    
    # 获取通信统计
    stats = handler.get_communication_stats()
    print("初始通信统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("注意: 实际的网络通信需要在真实网络环境中测试")
    print()


def communication_manager_example():
    """CommunicationManager使用示例"""
    print("=== CommunicationManager 示例 ===")
    
    # 创建配置
    comm_config = DictConfig({
        'host': '0.0.0.0',
        'port': 8080,
        'timeouts': {
            'connection': 30.0,
            'send': 10.0,
            'receive': 30.0
        },
        'heartbeat': {
            'interval': 30.0,
            'timeout': 10.0
        }
    })
    
    # 创建通信处理器（模拟）
    from unittest.mock import Mock
    mock_handler = Mock()
    mock_handler.send_message.return_value = True
    mock_handler.broadcast_message.return_value = {"client1": True, "client2": True}
    mock_handler.serializer = Mock()
    mock_handler.serializer.serialize_model.return_value = b"serialized_model_data"
    
    # 创建通信管理器
    manager = CommunicationManager(comm_config, mock_handler)
    
    # 模拟建立客户端连接
    clients = ["client1", "client2", "client3"]
    for i, client in enumerate(clients):
        address = ("127.0.0.1", 8080 + i)
        success = manager.establish_connection(client, address)
        print(f"建立连接 {client}: {'✓' if success else '✗'}")
    
    # 获取活跃客户端
    active_clients = manager.get_active_clients()
    print(f"活跃客户端: {active_clients}")
    
    # 模拟模型广播
    model = SimpleModel()
    print(f"广播模型到 {len(active_clients)} 个客户端...")
    
    try:
        results = manager.broadcast_model(model, active_clients)
        successful_count = sum(1 for success in results.values() if success)
        print(f"模型广播结果: {successful_count}/{len(active_clients)} 成功")
    except Exception as e:
        print(f"模型广播失败: {e}")
    
    # 启动心跳监控
    manager.start_heartbeat_monitor(interval=10.0)
    print("心跳监控已启动")
    
    # 获取客户端统计
    client_stats = manager.get_client_stats()
    print("客户端统计:")
    for client_id, stats in client_stats.items():
        print(f"  {client_id}: {stats['status']}, 地址: {stats['address']}")
    
    # 停止心跳监控
    manager.stop_heartbeat_monitor()
    print("心跳监控已停止")
    
    print()


async def async_communication_example():
    """异步通信示例"""
    print("=== 异步通信示例 ===")
    
    # 创建模拟的通信处理器
    from unittest.mock import Mock, AsyncMock
    from fedcl.communication.message_protocol import Message
    from datetime import datetime
    
    mock_handler = Mock()
    mock_handler.async_send_message = AsyncMock(return_value=True)
    mock_handler.async_receive_message = AsyncMock(return_value=Message(
        message_id="test_msg",
        message_type="response", 
        sender="client1",
        receiver="server",
        timestamp=datetime.now(),
        data={"result": "success"},
        metadata={},
        checksum="abc123"
    ))
    
    # 异步发送消息
    target = "client1"
    message = Message(
        message_id="async_msg",
        message_type="request",
        sender="server", 
        receiver=target,
        timestamp=datetime.now(),
        data={"command": "train"},
        metadata={},
        checksum="def456"
    )
    
    print("发送异步消息...")
    send_result = await mock_handler.async_send_message(target, message)
    print(f"异步发送结果: {'✓' if send_result else '✗'}")
    
    # 异步接收消息
    print("接收异步消息...")
    received_message = await mock_handler.async_receive_message("client1")
    print(f"接收到消息: {received_message.message_type}, 数据: {received_message.data}")
    
    print()


def error_handling_example():
    """错误处理示例"""
    print("=== 错误处理示例 ===")
    
    # 安全模块错误处理
    try:
        security_config = DictConfig({
            'authentication': {'method': 'unsupported_method'}
        })
        security = SecurityModule(security_config)
        security.authenticate_client("client1", "token")
    except Exception as e:
        print(f"安全模块错误: {type(e).__name__}: {e}")
    
    # 通信错误处理
    try:
        # 模拟通信错误
        from fedcl.communication.exceptions import CommunicationError
        raise CommunicationError("模拟的通信错误")
    except Exception as e:
        print(f"通信错误: {type(e).__name__}: {e}")
    
    print()


def main():
    """主函数"""
    print("FedCL 安全通信模块示例")
    print("=" * 50)
    
    # 执行各个示例
    security_module_example()
    communication_handler_example()
    communication_manager_example()
    
    # 执行异步示例
    print("运行异步通信示例...")
    asyncio.run(async_communication_example())
    
    error_handling_example()
    
    print("所有示例执行完成！")


if __name__ == "__main__":
    main()
