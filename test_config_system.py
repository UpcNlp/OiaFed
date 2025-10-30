"""
简单测试新配置系统
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # 直接导入 config 子模块，避免触发 fedcl.__init__.py
    from fedcl.config.base import BaseConfig
    from fedcl.config.communication import CommunicationConfig
    from fedcl.config.training import TrainingConfig
    from fedcl.config.loader import ConfigLoader, ConfigLoadError

    print("✅ 配置系统导入成功！")
    print()

    # 测试1：创建通信配置
    print("测试1：创建通信配置")
    comm_config = CommunicationConfig(
        mode="network",
        role="server",
        node_id="test_server"
    )
    print(f"  - 模式: {comm_config.mode}")
    print(f"  - 角色: {comm_config.role}")
    print(f"  - 节点ID: {comm_config.node_id}")
    print(f"  - 监听地址: {comm_config.transport['host']}:{comm_config.transport['port']}")
    print()

    # 测试2：创建训练配置
    print("测试2：创建训练配置")
    train_config = TrainingConfig(
        trainer={"name": "FedAvgTrainer", "params": {"max_rounds": 100}},
        aggregator={"name": "FedAvgAggregator"}
    )
    print(f"  - 推断角色: {train_config.infer_role()}")
    print(f"  - 训练器: {train_config.trainer['name']}")
    print(f"  - 聚合器: {train_config.aggregator['name']}")
    print()

    # 测试3：自定义字段
    print("测试3：自定义字段")
    comm_config.set("custom_field", "custom_value")
    print(f"  - 自定义字段: {comm_config.get('custom_field')}")
    print()

    # 测试4：转换为字典
    print("测试4：转换为字典")
    config_dict = comm_config.to_dict()
    print(f"  - 字典键数量: {len(config_dict)}")
    print(f"  - 包含 mode: {'mode' in config_dict}")
    print(f"  - 包含 transport: {'transport' in config_dict}")
    print()

    print("✅ 所有测试通过！")

except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()
