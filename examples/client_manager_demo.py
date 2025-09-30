"""
ClientManager 使用演示

演示如何使用ClientManager来管理不同类型的学习器：
1. 继承BaseFLNode的学习器
2. 自定义学习器类
"""

import asyncio
from fedcl.comm.memory_transport import MemoryTransport
from fedcl.fl.client import ClientManager, create_fl_client, create_custom_client
from fedcl.fl.server import SimpleLearnerStub
from fedcl.utils.auto_logger import setup_auto_logging


class CustomLearner:
    """自定义学习器示例（不继承BaseFLNode）"""
    
    def __init__(self, name: str):
        self.name = name
        self.model = {"weights": [1.0, 2.0, 3.0]}
        self.training_rounds = 0
    
    def get_status(self):
        """获取状态"""
        return {
            "name": self.name,
            "model_size": len(self.model["weights"]),
            "training_rounds": self.training_rounds
        }
    
    async def train(self, data=None):
        """训练方法"""
        self.training_rounds += 1
        
        # 模拟训练过程
        await asyncio.sleep(0.1)
        
        # 更新模型
        self.model["weights"] = [w * 1.1 for w in self.model["weights"]]
        
        return {
            "round": self.training_rounds,
            "model": self.model.copy(),
            "loss": 0.1 / self.training_rounds,
            "accuracy": min(0.99, 0.5 + self.training_rounds * 0.1)
        }
    
    def get_model(self):
        """获取模型"""
        return self.model.copy()
    
    def ping(self):
        """心跳检测"""
        return f"pong from {self.name}"


async def demo_client_manager():
    """演示ClientManager的使用"""
    
    # 初始化日志系统
    setup_auto_logging()
    
    # 创建传输层
    transport = MemoryTransport()
    
    print("=== 演示1: 使用继承BaseFLNode的学习器 ===")
    
    # 方式1: 直接创建ClientManager
    learner1 = SimpleLearnerStub("learner1", transport)
    client1 = ClientManager(
        client_id="client1",
        learner_object=learner1,
        transport=transport,
        allowed_methods=["train", "get_model", "get_status", "ping"],
        auto_start=False  # 手动启动以便观察过程
    )
    
    print(f"客户端1状态: {client1.get_status()}")
    
    # 手动启动
    await client1.start()
    print(f"启动后客户端1状态: {client1.get_status()}")
    
    # 通过ClientManager调用学习器方法
    result1 = await client1.train()
    print(f"客户端1训练结果: {result1}")
    
    print("\n=== 演示2: 使用工厂函数创建（BaseFLNode） ===")
    
    # 方式2: 使用工厂函数
    client2 = create_fl_client(
        client_id="client2",
        learner_class=SimpleLearnerStub,
        transport=transport,
        learner_args=("learner2", transport),
        allowed_methods=["train", "get_model", "get_status", "ping"]
    )
    
    await asyncio.sleep(0.1)  # 等待自动启动完成
    print(f"客户端2状态: {await client2.get_status_async()}")
    
    result2 = await client2.train()
    print(f"客户端2训练结果: {result2}")
    
    print("\n=== 演示3: 使用自定义学习器 ===")
    
    # 方式3: 自定义学习器
    custom_learner = CustomLearner("custom_learner_3")
    client3 = create_custom_client(
        client_id="client3",
        learner_object=custom_learner,
        transport=transport,
        allowed_methods=["train", "get_model", "get_status", "ping"]
    )
    
    await asyncio.sleep(0.1)  # 等待自动启动完成
    print(f"客户端3状态: {await client3.get_status_async()}")
    
    result3 = await client3.train()
    print(f"客户端3训练结果: {result3}")
    
    # 演示直接访问学习器对象
    direct_result = await client3.get_learner().train()
    print(f"直接调用学习器训练结果: {direct_result}")
    
    print("\n=== 演示4: 批量操作 ===")
    
    clients = [client1, client2, client3]
    
    # 并行训练所有客户端
    train_tasks = [client.train() for client in clients]
    batch_results = await asyncio.gather(*train_tasks)
    
    for i, result in enumerate(batch_results):
        print(f"客户端{i+1}批量训练结果: {result}")
    
    # 获取所有客户端状态
    status_tasks = [client.get_status_async() for client in clients]
    batch_status = await asyncio.gather(*status_tasks)
    
    for i, status in enumerate(batch_status):
        print(f"客户端{i+1}最终状态: {status}")
    
    print("\n=== 清理资源 ===")
    
    # 停止所有客户端
    for client in clients:
        await client.stop()
    
    print("演示完成！")


if __name__ == "__main__":
    asyncio.run(demo_client_manager())
