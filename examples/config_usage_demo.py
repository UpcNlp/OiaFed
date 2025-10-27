"""
配置使用示例
examples/config_usage_demo.py

演示如何使用新的配置类来初始化服务端和客户端
"""

import asyncio
from fedcl.config import (
    ServerConfig, ClientConfig,
    load_server_config, load_client_config,
    create_default_server_config, create_default_client_config
)
from fedcl.federation.server import FederationServer
from fedcl.federation.client import FederationClient
from fedcl.trainer.base_trainer import BaseTrainer
from fedcl.learner.base_learner import BaseLearner


# ============================================
# 示例1: 使用YAML配置文件
# ============================================

async def example1_yaml_config():
    """从YAML文件加载配置"""
    print("=" * 60)
    print("示例1: 使用YAML配置文件")
    print("=" * 60)

    # 1. 加载服务端配置
    server_config = load_server_config("configs/server_demo.yaml")
    print(f"服务端配置: {server_config.mode} 模式, 端口: {server_config.transport.port}")

    # 2. 加载客户端配置
    client_config = load_client_config("configs/client_demo_1.yaml")
    print(f"客户端配置: {client_config.mode} 模式")

    # 3. 创建服务端
    server = FederationServer(server_config.to_dict())
    print(f"服务端已创建: {server.server_id}")

    # 4. 创建客户端
    client = FederationClient(client_config.to_dict())
    print(f"客户端已创建: {client.client_id}")


# ============================================
# 示例2: 使用代码创建配置
# ============================================

async def example2_code_config():
    """通过代码创建配置"""
    print("\n" + "=" * 60)
    print("示例2: 使用代码创建配置")
    print("=" * 60)

    # 1. 创建默认服务端配置
    server_config = create_default_server_config(mode="process", port=8000)
    print(f"服务端配置: {server_config.mode} 模式")

    # 2. 创建默认客户端配置
    client_config = create_default_client_config(
        mode="process",
        client_host="127.0.0.1",
        client_port=0  # 自动分配
    )
    print(f"客户端配置: {client_config.mode} 模式")


# ============================================
# 示例3: 自定义配置
# ============================================

async def example3_custom_config():
    """自定义配置参数"""
    print("\n" + "=" * 60)
    print("示例3: 自定义配置参数")
    print("=" * 60)

    # 1. 创建服务端配置并自定义参数
    from fedcl.config import TransportLayerConfig, FederationLayerConfig

    server_config = ServerConfig(
        mode="process",
        transport=TransportLayerConfig(
            port=9000,  # 自定义端口
            timeout=60.0,  # 更长的超时时间
            retry_attempts=5
        ),
        federation=FederationLayerConfig(
            max_rounds=200,  # 更多训练轮数
            min_clients=5
        )
    )
    print(f"自定义服务端: 端口={server_config.transport.port}, 最大轮数={server_config.federation.max_rounds}")

    # 2. 创建客户端配置
    from fedcl.config import StubLayerConfig

    client_config = ClientConfig(
        mode="process",
        stub=StubLayerConfig(
            auto_register=True,
            registration_retry_attempts=5
        )
    )
    print(f"自定义客户端: 重试次数={client_config.stub.registration_retry_attempts}")


# ============================================
# 示例4: 完整的服务端启动流程
# ============================================

class MyTrainer(BaseTrainer):
    """示例训练器"""

    async def aggregate_models(self, client_models):
        """聚合模型"""
        print("聚合客户端模型...")
        return {}

    async def train_round(self, round_num: int):
        """执行一轮训练"""
        print(f"执行第 {round_num} 轮训练")
        return {}


async def example4_server_startup():
    """完整的服务端启动流程"""
    print("\n" + "=" * 60)
    print("示例4: 完整的服务端启动流程")
    print("=" * 60)

    # 1. 加载配置
    server_config = load_server_config("configs/server_demo.yaml")

    # 2. 创建服务端实例
    server = FederationServer(server_config.to_dict())

    # 3. 初始化训练器
    global_model = {}  # 初始全局模型
    trainer = await server.initialize_with_trainer(
        trainer_class=MyTrainer,
        global_model=global_model,
        trainer_config={}
    )

    # 4. 启动服务端
    await server.start_server()
    print(f"✅ 服务端已启动: {server.server_id}")

    # 5. 等待一段时间后停止
    await asyncio.sleep(2)
    await server.stop_server()
    print("✅ 服务端已停止")


# ============================================
# 示例5: 完整的客户端启动流程
# ============================================

class MyLearner(BaseLearner):
    """示例学习器"""

    async def train(self, global_model, **kwargs):
        """本地训练"""
        print("执行本地训练...")
        return {"loss": 0.5, "accuracy": 0.85}

    async def evaluate(self, model, **kwargs):
        """本地评估"""
        print("执行本地评估...")
        return {"test_loss": 0.3, "test_accuracy": 0.90}


async def example5_client_startup():
    """完整的客户端启动流程"""
    print("\n" + "=" * 60)
    print("示例5: 完整的客户端启动流程")
    print("=" * 60)

    # 1. 加载配置
    client_config = load_client_config("configs/client_demo_1.yaml")

    # 2. 创建客户端实例
    client = FederationClient(client_config.to_dict())

    # 3. 初始化学习器
    learner = await client.initialize_with_learner(
        learner_class=MyLearner,
        learner_config={}
    )

    # 4. 启动客户端（会自动注册到服务端）
    await client.start_client()
    print(f"✅ 客户端已启动: {client.client_id}")

    # 5. 等待一段时间后停止
    await asyncio.sleep(2)
    await client.stop_client()
    print("✅ 客户端已停止")


# ============================================
# 示例6: Process模式多客户端
# ============================================

async def example6_multi_clients():
    """Process模式启动多个客户端"""
    print("\n" + "=" * 60)
    print("示例6: Process模式多客户端")
    print("=" * 60)

    # 加载基础配置
    base_config = load_client_config("configs/client_demo_1.yaml")

    # 创建多个客户端
    clients = []
    for i in range(3):
        # 为每个客户端创建独立配置
        config = ClientConfig.from_dict(base_config.to_dict())
        config.client_id = f"client_{i+1}"

        client = FederationClient(config.to_dict(), client_id=config.client_id)
        clients.append(client)
        print(f"创建客户端: {client.client_id}")

    print(f"✅ 共创建 {len(clients)} 个客户端")


# ============================================
# 主函数
# ============================================

async def main():
    """运行所有示例"""

    # 示例1-3: 配置创建方式
    await example1_yaml_config()
    await example2_code_config()
    await example3_custom_config()

    # 示例6: 多客户端
    await example6_multi_clients()

    # 注意: 示例4和5需要实际的网络通信，这里只演示配置
    # 完整的启动流程请参考 examples/mnist_process_demo.py


if __name__ == "__main__":
    # 导入必要的配置类
    from fedcl.config import TransportLayerConfig, StubLayerConfig, FederationLayerConfig

    asyncio.run(main())

    print("\n" + "=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)
