# examples/communication_example.py
"""
通信系统使用示例

演示如何使用FedCL通信模块进行消息传递、数据序列化和网络通信。
"""

import torch
import torch.nn as nn
from omegaconf import DictConfig
from datetime import datetime

from fedcl.communication import (
    MessageProtocol, DataSerializer, NetworkInterface,
    Message, ConnectionStatus, ProtocolType
)


class ExampleModel(nn.Module):
    """示例模型"""
    
    def __init__(self, input_size=10, hidden_size=5, output_size=2):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        return self.linear2(x)


def demonstrate_message_protocol():
    """演示消息协议功能"""
    print("=== Message Protocol Demonstration ===")
    
    # 初始化协议
    protocol = MessageProtocol(version="1.0")
    
    # 创建不同类型的消息
    messages = [
        # 心跳消息
        protocol.create_message(
            message_type=MessageProtocol.HEARTBEAT,
            data={"timestamp": datetime.now().isoformat(), "status": "alive"},
            sender="client_001",
            receiver="server"
        ),
        
        # 客户端就绪消息
        protocol.create_message(
            message_type=MessageProtocol.CLIENT_READY,
            data={"capabilities": ["training", "evaluation"], "version": "1.0"},
            sender="client_001",
            receiver="server"
        ),
        
        # 模型更新消息
        protocol.create_message(
            message_type=MessageProtocol.MODEL_UPDATE,
            data={"weights": [1.0, 2.0, 3.0], "gradients": [0.1, 0.2, 0.3]},
            sender="client_001",
            receiver="server",
            metadata={"round": 5, "local_epochs": 3}
        )
    ]
    
    for i, message in enumerate(messages, 1):
        print(f"\\nMessage {i}:")
        print(f"  Type: {message.message_type}")
        print(f"  ID: {message.message_id}")
        print(f"  From: {message.sender} -> To: {message.receiver}")
        print(f"  Valid: {protocol.validate_message(message)}")
        print(f"  Size: {protocol.get_message_size(message)} bytes")
        
        # 测试压缩
        if i == 3:  # 对较大的消息进行压缩
            compressed = protocol.compress_message(message)
            original_size = protocol.get_message_size(message)
            compressed_size = protocol.get_message_size(compressed)
            compression_ratio = compressed_size / original_size
            print(f"  Compressed: {original_size} -> {compressed_size} bytes (ratio: {compression_ratio:.2f})")


def demonstrate_data_serialization():
    """演示数据序列化功能"""
    print("\\n\\n=== Data Serialization Demonstration ===")
    
    # 初始化序列化器
    serializer = DataSerializer(compression_level=6)
    
    # 创建示例模型和数据
    model = ExampleModel()
    tensor = torch.randn(20, 10)
    state_dict = model.state_dict()
    
    print(f"\\nSerializer info: {serializer.get_serializer_info()}")
    
    # 1. 张量序列化
    print(f"\\n1. Tensor Serialization:")
    print(f"   Original tensor shape: {tensor.shape}")
    
    serialized_tensor = serializer.serialize_tensor(tensor)
    print(f"   Serialized size: {len(serialized_tensor)} bytes")
    
    deserialized_tensor = serializer.deserialize_tensor(serialized_tensor)
    print(f"   Deserialized shape: {deserialized_tensor.shape}")
    print(f"   Data integrity: {torch.allclose(tensor, deserialized_tensor)}")
    
    # 2. 模型序列化
    print(f"\\n2. Model Serialization:")
    serialized_model = serializer.serialize_model(model)
    print(f"   Serialized model size: {len(serialized_model)} bytes")
    
    deserialized_model = serializer.deserialize_model(serialized_model, ExampleModel)
    
    # 测试模型等价性
    test_input = torch.randn(1, 10)
    original_output = model(test_input)
    deserialized_output = deserialized_model(test_input)
    print(f"   Model equivalence: {torch.allclose(original_output, deserialized_output)}")
    
    # 3. 状态字典序列化
    print(f"\\n3. State Dict Serialization:")
    print(f"   Original state dict keys: {len(state_dict)}")
    
    serialized_state_dict = serializer.serialize_state_dict(state_dict)
    print(f"   Serialized size: {len(serialized_state_dict)} bytes")
    
    deserialized_state_dict = serializer.deserialize_state_dict(serialized_state_dict)
    print(f"   Deserialized keys: {len(deserialized_state_dict)}")
    
    # 验证参数完整性
    params_match = all(
        torch.allclose(state_dict[key], deserialized_state_dict[key])
        for key in state_dict.keys()
    )
    print(f"   Parameters match: {params_match}")
    
    # 4. 内存使用估算
    print(f"\\n4. Memory Usage Estimation:")
    estimation = serializer.estimate_memory_usage(len(serialized_model))
    print(f"   Input size: {estimation['input_size']} bytes")
    print(f"   Estimated peak memory: {estimation['peak_memory']} bytes")
    print(f"   Estimated steady memory: {estimation['steady_memory']} bytes")


def demonstrate_network_interface():
    """演示网络接口功能"""
    print("\\n\\n=== Network Interface Demonstration ===")
    
    # 初始化网络接口
    config = DictConfig({
        'max_connections': 10,
        'max_workers': 4,
        'use_ssl': False,
        'connection_timeout': 30.0,
        'socket_backlog': 5
    })
    
    network_interface = NetworkInterface(config)
    
    # 获取网络统计信息
    stats = network_interface.get_network_stats()
    print(f"\\nNetwork Interface Statistics:")
    print(f"   Max connections: {stats['connection_pool']['max_connections']}")
    print(f"   Active connections: {stats['connection_pool']['active_connections']}")
    print(f"   SSL enabled: {stats['ssl_enabled']}")
    print(f"   Max workers: {stats['max_workers']}")
    print(f"   Connection timeout: {stats['config']['connection_timeout']}s")
    
    # 关闭网络接口
    network_interface.shutdown()
    print("   Network interface shutdown completed")


def demonstrate_end_to_end_scenario():
    """演示端到端联邦学习通信场景"""
    print("\\n\\n=== End-to-End Federation Scenario ===")
    
    # 初始化组件
    protocol = MessageProtocol()
    serializer = DataSerializer(compression_level=5)
    
    # 模拟联邦学习场景
    print("\\nSimulating federated learning communication...")
    
    # 1. 客户端本地训练完成，准备发送模型更新
    client_model = ExampleModel()
    
    # 序列化模型状态
    serialized_state = serializer.serialize_state_dict(client_model.state_dict())
    
    # 创建模型更新消息
    model_update_data = {
        "state_dict": serialized_state,
        "metrics": {"loss": 0.234, "accuracy": 0.892},
        "training_info": {"epochs": 5, "batch_size": 32, "learning_rate": 0.01}
    }
    
    update_message = protocol.create_message(
        message_type=MessageProtocol.MODEL_UPDATE,
        data=model_update_data,
        sender="client_001",
        receiver="server",
        metadata={"round": 10, "client_type": "mobile"}
    )
    
    print(f"   Created model update message: {update_message.message_id}")
    print(f"   Message size: {protocol.get_message_size(update_message)} bytes")
    
    # 2. 压缩和序列化消息（模拟网络传输）
    compressed_message = protocol.compress_message(update_message)
    serialized_message = protocol.serialize_message(
        message_type=compressed_message.message_type,
        data=compressed_message.data,
        metadata=compressed_message.metadata
    )
    
    print(f"   Serialized message size: {len(serialized_message)} bytes")
    
    # 3. 服务器接收和处理消息
    received_message = protocol.deserialize_message(serialized_message)
    decompressed_message = protocol.compress_message(received_message)
    
    print(f"   Message received and validated: {protocol.validate_message(decompressed_message)}")
    
    # 4. 服务器聚合后发送全局模型
    global_model_data = {
        "model": serializer.serialize_model(client_model),  # 简化：使用同一模型
        "round": 11,
        "participants": ["client_001", "client_002", "client_003"],
        "aggregation_info": {"method": "fedavg", "weights": [0.4, 0.3, 0.3]}
    }
    
    global_message = protocol.create_message(
        message_type=MessageProtocol.GLOBAL_MODEL,
        data=global_model_data,
        sender="server",
        receiver="all_clients",
        metadata={"next_round": 11, "deadline": "2023-01-01T15:00:00"}
    )
    
    print(f"   Created global model message: {global_message.message_id}")
    print(f"   Global model message size: {protocol.get_message_size(global_message)} bytes")
    
    # 5. 客户端接收全局模型
    global_serialized = protocol.serialize_message(
        message_type=global_message.message_type,
        data=global_message.data,
        metadata=global_message.metadata
    )
    
    client_received = protocol.deserialize_message(global_serialized)
    
    # 反序列化全局模型
    global_model = serializer.deserialize_model(
        client_received.data["model"],
        ExampleModel
    )
    
    print(f"   Client received global model successfully")
    print(f"   Next training round: {client_received.metadata['next_round']}")
    
    # 验证模型功能
    test_input = torch.randn(1, 10)
    original_output = client_model(test_input)
    global_output = global_model(test_input)
    
    print(f"   Model functional equivalence: {torch.allclose(original_output, global_output)}")


def main():
    """主函数"""
    print("FedCL Communication System Demonstration")
    print("=" * 50)
    
    try:
        # 演示各个组件功能
        demonstrate_message_protocol()
        demonstrate_data_serialization()
        demonstrate_network_interface()
        demonstrate_end_to_end_scenario()
        
        print("\\n\\n" + "=" * 50)
        print("All demonstrations completed successfully!")
        
    except Exception as e:
        print(f"\\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
