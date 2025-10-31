"""
完整的MNIST联邦学习演示 - 真实训练版本
使用新架构实现真实的MNIST联邦学习训练
examples/complete_mnist_demo.py
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from fedcl.learner.base_learner import BaseLearner
from fedcl.trainer.trainer import BaseTrainer, TrainingConfig
from fedcl.types import TrainingRequest, TrainingResponse
from fedcl import FederatedLearning

# 导入新实现的数据集和模型管理
from fedcl.api.decorators import dataset, model
from fedcl.api.registry import registry
from fedcl.methods.datasets.base import FederatedDataset
from fedcl.methods.models.base import FederatedModel

# 导入装饰器
from fedcl.api import learner, trainer
from fedcl.utils.auto_logger import get_train_logger

# ==================== 1. 注册真实的MNIST数据集 ====================

@dataset(
    name='MNIST',
    description='MNIST手写数字数据集',
    dataset_type='image_classification',
    num_classes=10
)
class MNISTFederatedDataset(FederatedDataset):
    """MNIST联邦数据集实现"""

    def __init__(self, root: str = './data', train: bool = True, download: bool = True):
        super().__init__(root, train, download)

        # 数据转换
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # 加载MNIST数据集
        self.dataset = torchvision.datasets.MNIST(
            root=root,
            train=train,
            download=download,
            transform=transform
        )

        # 设置属性
        self.num_classes = 10
        self.input_shape = (1, 28, 28)

    def get_statistics(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        return {
            'dataset_name': 'MNIST',
            'num_samples': len(self.dataset),
            'num_classes': self.num_classes,
            'input_shape': self.input_shape,
            'train': self.train,
        }

# ==================== 2. 注册真实的CNN模型 ====================

@model(
    name='MNIST_CNN',
    description='MNIST CNN分类模型',
    task='classification',
    input_shape=(1, 28, 28),
    output_shape=(10,)
)
class MNISTCNNModel(FederatedModel):
    """MNIST CNN模型"""

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # 设置元数据
        self.set_metadata(
            task_type='classification',
            input_shape=(1, 28, 28),
            output_shape=(num_classes,)
        )

        # 定义网络结构
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, num_classes)

        # 损失函数
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = x.view(-1, 64 * 14 * 14)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

    def get_weights_as_dict(self) -> Dict[str, torch.Tensor]:
        """获取模型权重"""
        return {k: v.cpu().clone() for k, v in self.state_dict().items()}

    def set_weights_from_dict(self, weights: Dict[str, torch.Tensor], strict: bool = True):
        """设置模型权重"""
        self.load_state_dict(weights)

    def get_param_count(self) -> int:
        """获取参数数量"""
        return sum(p.numel() for p in self.parameters())

# ==================== 3. 实现真实的Learner ====================

@learner('MNISTLearner',
         description='MNIST数据集学习器',
         version='1.0',
         author='MOE-FedCL',
         dataset='MNIST')
class MNISTLearner(BaseLearner):
    """MNIST学习器 - 实现真实的训练"""

    def __init__(self, client_id: str, config: Dict[str, Any], logger=None):
        super().__init__(client_id, config, logger)
        self.logger = get_train_logger(client_id)

        # 训练配置
        self.learning_rate = config.get('learning_rate', 0.01)
        self.batch_size = config.get('batch_size', 32)
        self.local_epochs = config.get('local_epochs', 1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载数据集
        self._load_dataset()

        # 创建模型
        self.model = MNISTCNNModel(num_classes=10).to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()

        print(f"✅ MNISTLearner {client_id} initialized with {len(self.train_dataset)} samples")

    def _load_dataset(self):
        """加载数据集"""
        # 从注册表获取MNIST数据集
        mnist_dataset_cls = registry.get_dataset('MNIST')
        mnist_dataset = mnist_dataset_cls(root='./data', train=True, download=True)

        # 获取客户端数据划分
        num_clients = int(self.client_id.split('_')[1]) + 3  # 假设总共3个客户端
        client_datasets = mnist_dataset.partition(
            num_clients=num_clients,
            strategy='non_iid_label',
            labels_per_client=2
        )

        # 获取当前客户端的数据
        client_idx = int(self.client_id.split('_')[1])
        if client_idx in client_datasets:
            self.train_dataset = client_datasets[client_idx]
        else:
            # 如果没有划分，使用部分数据
            total_size = len(mnist_dataset.dataset)
            start = (client_idx * total_size) // num_clients
            end = ((client_idx + 1) * total_size) // num_clients
            indices = list(range(start, end))
            self.train_dataset = Subset(mnist_dataset.dataset, indices)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

    async def train(self, params: Dict[str, Any]) -> TrainingResponse:
        """训练方法"""
        num_epochs = params.get("num_epochs", self.local_epochs)
        round_number = params.get("round_number", 1)

        self.logger.info(f"  [{self.client_id}] Round {round_number}, Training {num_epochs} epochs...")

        # 设置模型为训练模式
        self.model.train()

        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0

            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                epoch_correct += pred.eq(target.view_as(pred)).sum().item()
                epoch_samples += data.size(0)

            avg_epoch_loss = epoch_loss / epoch_samples
            epoch_accuracy = epoch_correct / epoch_samples
            self.logger.info(f"    [{self.client_id}] Epoch {epoch+1}: Loss={avg_epoch_loss:.4f}, Acc={epoch_accuracy:.4f}")

            total_loss += epoch_loss
            correct_predictions += epoch_correct
            total_samples += epoch_samples

        # 计算平均值
        avg_loss = total_loss / total_samples
        accuracy = correct_predictions / total_samples

        # 获取模型权重（直接返回torch.Tensor，底层会自动转换）
        model_weights = self.model.get_weights_as_dict()

        # 创建训练响应
        response = TrainingResponse(
            request_id="",  # 会被stub填充
            client_id=self.client_id,
            success=True,
            result={
                "epochs_completed": num_epochs,
                "loss": avg_loss,
                "accuracy": accuracy,
                "samples_used": total_samples,
                "model_weights": model_weights  # 框架会自动序列化tensor
            },
            execution_time=0.0
        )

        self.logger.info(f"  [{self.client_id}] Round {round_number} completed: Loss={avg_loss:.4f}, Acc={accuracy:.4f}")
        return response

    async def evaluate(self, model_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """评估方法"""
        self.model.eval()

        # 如果提供了模型权重，先更新模型
        if model_data and "model_weights" in model_data:
            weights = model_data["model_weights"]
            # 将numpy数组转换为torch tensor
            torch_weights = {}
            for k, v in weights.items():
                if isinstance(v, np.ndarray):
                    torch_weights[k] = torch.from_numpy(v)
                else:
                    torch_weights[k] = v
            self.model.set_weights_from_dict(torch_weights)

        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.train_loader:  # 使用训练数据作为评估数据
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += data.size(0)

        return {
            "accuracy": correct / total,
            "loss": test_loss / total,
            "samples": total
        }

    async def get_model(self) -> Dict[str, Any]:
        """获取模型数据

        直接返回torch.Tensor，框架会自动序列化
        """
        return {
            "model_type": "mnist_cnn",
            "parameters": {"weights": self.model.get_weights_as_dict()},
            "metadata": {
                "client_id": self.client_id,
                "samples": len(self.train_dataset),
                "param_count": self.model.get_param_count()
            }
        }

    async def set_model(self, model_data: Dict[str, Any]) -> bool:
        """设置模型数据

        接受torch.Tensor或numpy数组，自动转换
        """
        try:
            if "parameters" in model_data and "weights" in model_data["parameters"]:
                weights = model_data["parameters"]["weights"]
                # 智能转换：支持numpy数组、torch.Tensor和dict
                torch_weights = {}
                for k, v in weights.items():
                    if isinstance(v, np.ndarray):
                        torch_weights[k] = torch.from_numpy(v)
                    elif torch.is_tensor(v):
                        torch_weights[k] = v
                    else:
                        torch_weights[k] = v
                self.model.set_weights_from_dict(torch_weights)
                return True
        except Exception as e:
            self.logger.exception(f"  [{self.client_id}] Failed to set model: {e}")
        return False

    def get_data_statistics(self) -> Dict[str, Any]:
        """获取数据统计"""
        return {
            "total_samples": len(self.train_dataset),
            "num_classes": 10,
            "feature_dim": 784,
            "input_shape": (1, 28, 28)
        }

    async def get_local_model(self) -> Dict[str, Any]:
        return await self.get_model()

    async def set_local_model(self, model_data: Dict[str, Any]) -> bool:
        return await self.set_model(model_data)

# ==================== 4. 实现真实的Trainer ====================

@trainer('FedAvgMNIST',
         description='MNIST联邦平均训练器',
         version='1.0',
         author='MOE-FedCL',
         algorithms=['fedavg'])
class FedAvgMNISTTrainer(BaseTrainer):
    """联邦平均训练器 - 实现真实的模型聚合"""

    def __init__(self, global_model: Dict[str, Any] = None, training_config=None, logger=None):
        # 创建全局模型对象（在调用父类初始化之前）
        self.global_model_obj = MNISTCNNModel(num_classes=10)

        # 处理配置参数
        if isinstance(training_config, dict):
            config_obj = TrainingConfig(
                max_rounds=training_config.get("max_rounds", 5),
                min_clients=training_config.get("min_clients", 2),
                client_selection_ratio=training_config.get("client_selection_ratio", 1.0)
            )
        elif isinstance(training_config, TrainingConfig):
            config_obj = training_config
        else:
            config_obj = TrainingConfig()

        # 调用父类初始化（传入None作为global_model，因为我们使用自己的模型对象）
        super().__init__(None, config_obj, logger)
        self.logger = get_train_logger("FedAvgMNISTTrainer")

        # 确保 self.global_model 指向我们的模型对象
        self.global_model = self.global_model_obj

        # 如果提供了初始权重，设置到模型中
        if global_model and "parameters" in global_model:
            weights = global_model["parameters"].get("weights", {})
            torch_weights = {}
            for k, v in weights.items():
                if isinstance(v, np.ndarray):
                    torch_weights[k] = torch.from_numpy(v)
                else:
                    torch_weights[k] = v
            self.global_model.set_weights_from_dict(torch_weights)

        self.logger.info(f"FedAvgMNISTTrainer initialized with {self.global_model.get_param_count():,} parameters")

    async def train_round(self, round_num: int, client_ids: List[str]) -> Dict[str, Any]:
        """执行一轮联邦训练"""
        self.logger.info(f"\n--- Round {round_num} ---")
        self.logger.info(f"  Selected clients: {client_ids}")

        # 创建训练请求
        training_params = {
            "round_number": round_num,
            "num_epochs": 1,
            "batch_size": 32,
            "learning_rate": 0.01
        }

        # 并行训练所有客户端
        tasks = []
        for client_id in client_ids:
            if self.is_client_ready(client_id):
                proxy = self._proxy_manager.get_proxy(client_id)
                if proxy:
                    self.logger.info(f"  [{client_id}] Starting training...")
                    task = proxy.train(training_params)
                    tasks.append((client_id, task))

        # 收集结果
        client_results = {}
        failed_clients = []

        for client_id, task in tasks:
            try:
                result = await task
                if result.success:
                    client_results[client_id] = result
                    self.logger.info(f"  [{client_id}] Training succeeded: Loss={result.result['loss']:.4f}, Acc={result.result['accuracy']:.4f}")
                else:
                    self.logger.error(f"  [{client_id}] Training failed: {result}")
                    failed_clients.append(client_id)
            except Exception as e:
                self.logger.exception(f"  [{client_id}] Training failed: {e}")
                failed_clients.append(client_id)

        # 聚合模型
        aggregated_weights = None
        if client_results:
            aggregated_weights = await self.aggregate_models(client_results)
            if aggregated_weights:
                self.global_model.set_weights_from_dict(aggregated_weights)

        # 计算轮次指标
        if client_results:
            avg_loss = np.mean([r.result['loss'] for r in client_results.values()])
            avg_accuracy = np.mean([r.result['accuracy'] for r in client_results.values()])
        else:
            avg_loss, avg_accuracy = 0.0, 0.0

        self.logger.info(f"  Round {round_num} summary: Loss={avg_loss:.4f}, Acc={avg_accuracy:.4f}")

        # 返回结果 - 注意不要包含可能被父类误用的字段名
        return {
            "round": round_num,
            "participants": client_ids,
            "successful_clients": list(client_results.keys()),
            "failed_clients": failed_clients,
            # 不使用 "aggregated_model" 这个键名，避免父类误用
            "model_aggregated": aggregated_weights is not None,
            "round_metrics": {
                "avg_loss": avg_loss,
                "avg_accuracy": avg_accuracy,
                "successful_count": len(client_results)
            }
        }

    async def aggregate_models(self, client_results: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """聚合客户端模型（FedAvg）

        支持接收torch.Tensor或numpy数组，自动转换
        """
        self.logger.info("  Aggregating models using FedAvg...")

        if not client_results:
            return None

        # 获取所有客户端的模型权重
        client_weights = []
        client_samples = []

        for client_id, result in client_results.items():
            if "model_weights" in result.result:
                # 智能转换：支持numpy数组和torch.Tensor
                torch_weights = {}
                for k, v in result.result["model_weights"].items():
                    if isinstance(v, np.ndarray):
                        torch_weights[k] = torch.from_numpy(v)
                    elif torch.is_tensor(v):
                        torch_weights[k] = v
                    else:
                        # 如果既不是numpy也不是tensor，尝试转换
                        torch_weights[k] = torch.tensor(v)
                client_weights.append(torch_weights)
                client_samples.append(result.result['samples_used'])

        if not client_weights:
            return None

        # 计算加权平均
        total_samples = sum(client_samples)
        aggregated_weights = {}

        # 获取第一个模型的键
        first_model_keys = client_weights[0].keys()

        for key in first_model_keys:
            # 加权平均每个参数
            weighted_sum = torch.zeros_like(client_weights[0][key])
            for weights, samples in zip(client_weights, client_samples):
                weighted_sum += weights[key] * samples
            aggregated_weights[key] = weighted_sum / total_samples

        # 分发全局模型（直接传递tensor，框架会自动序列化）
        await self._distribute_global_model(aggregated_weights)

        return aggregated_weights

    async def evaluate_global_model(self) -> Dict[str, Any]:
        """评估全局模型"""
        self.logger.info("  Evaluating global model...")

        available_clients = self.get_available_clients()
        if not available_clients:
            return {"accuracy": 0.0, "loss": float('inf'), "samples_count": 0}

        # 准备全局模型数据（直接传递torch.Tensor，框架会自动序列化）
        global_model_data = {
            "model_weights": self.global_model.get_weights_as_dict()
        }

        # 并行评估
        tasks = []
        for client_id in available_clients:
            proxy = self._proxy_manager.get_proxy(client_id)
            if proxy:
                task = proxy.evaluate(global_model_data)
                tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理结果 - 注意：proxy.evaluate() 返回的是 TrainingResponse 对象
        valid_results = []
        for r in results:
            if not isinstance(r, Exception):
                # 检查是否是 TrainingResponse 对象
                if hasattr(r, 'result') and hasattr(r, 'success'):
                    # 是 TrainingResponse 对象，提取 result 字段
                    if r.success and r.result:
                        valid_results.append(r.result)
                elif isinstance(r, dict):
                    # 是字典，直接使用
                    valid_results.append(r)

        if not valid_results:
            return {"accuracy": 0.0, "loss": float('inf'), "samples_count": 0}

        total_samples = sum(r.get("samples", 0) for r in valid_results)
        if total_samples == 0:
            return {"accuracy": 0.0, "loss": float('inf'), "samples_count": 0}

        weighted_accuracy = sum(r["accuracy"] * r.get("samples", 1) for r in valid_results) / total_samples
        weighted_loss = sum(r["loss"] * r.get("samples", 1) for r in valid_results) / total_samples

        return {
            "accuracy": weighted_accuracy,
            "loss": weighted_loss,
            "samples_count": total_samples
        }

    def should_stop_training(self, round_num: int, round_result: Dict[str, Any]) -> bool:
        """判断是否应该停止训练"""
        # 检查最大轮次
        if round_num >= self.training_config.max_rounds:
            return True

        # 检查准确率收敛
        round_metrics = round_result.get("round_metrics", {})
        avg_accuracy = round_metrics.get("avg_accuracy", 0.0)

        if avg_accuracy >= 0.98:
            self.logger.info(f"  High accuracy achieved: {avg_accuracy:.4f}")
            return True

        return False

    async def _distribute_global_model(self, global_weights: Dict[str, torch.Tensor]):
        """分发全局模型到所有客户端

        注意：直接传递torch.Tensor，框架会自动序列化
        """
        global_model_data = {
            "model_type": "mnist_cnn",
            "parameters": {"weights": global_weights}
        }

        tasks = []
        for client_id in self.get_available_clients():
            proxy = self._proxy_manager.get_proxy(client_id)
            if proxy:
                task = proxy.set_model(global_model_data)
                tasks.append(task)

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            success_count = sum(1 for r in results if not isinstance(r, Exception) and r)
            self.logger.info(f"  Global model distributed to {success_count}/{len(tasks)} clients")

# ==================== 5. 主演示程序 ====================

async def demo_real_mnist_training():
    """
    真实MNIST联邦学习演示
    """
    print("=" * 80)
    print("🚀 MNIST联邦学习真实训练演示")
    print("=" * 80)

    # 🔧 清理Memory模式的共享状态
    from fedcl.communication.memory_manager import MemoryCommunicationManager
    from fedcl.transport.memory import MemoryTransport
    print("\n🧹 清理Memory模式共享状态...")
    MemoryCommunicationManager.clear_global_state()
    MemoryTransport.clear_global_state()
    print("✅ 共享状态已清理\n")

    # 显示已注册的组件
    print("📋 已注册组件:")
    print(f"  Datasets: {list(registry.datasets.keys())}")
    print(f"  Models: {list(registry.models.keys())}")
    print(f"  Trainers: {list(registry.trainers.keys())}")
    print(f"  Learners: {list(registry.learners.keys())}")

    from fedcl.config import CommunicationConfig, TrainingConfig

    # 创建服务器配置
    server_comm_config = CommunicationConfig(
        mode="process",
        role="server",
        node_id="server_1"
    )

    server_train_config = TrainingConfig()
    server_train_config.trainer = {
        "name": "FedAvgMNIST",
        "max_rounds": 5,
        "min_clients": 2,
        "client_selection_ratio": 1.0
    }
    server_train_config.aggregator = {"name": None}
    server_train_config.model = {"name": "MNIST_CNN"}

    # 创建客户端配置
    client_configs = []
    for i in range(3):
        client_comm_config = CommunicationConfig(
            mode="process",
            role="client",
            node_id=f"client_{i}"
        )

        client_train_config = TrainingConfig()
        client_train_config.learner = {
            "name": "MNISTLearner",
            "learning_rate": 0.01,
            "batch_size": 32,
            "local_epochs": 1
        }
        client_train_config.dataset = {"name": "MNIST"}

        client_configs.append((client_comm_config, client_train_config))

    # 合并所有配置
    all_configs = [(server_comm_config, server_train_config)] + client_configs

    # 创建FederatedLearning
    fl = FederatedLearning(all_configs)

    try:
        # 初始化系统
        print("\n🔧 初始化联邦学习系统...")
        await fl.initialize()
        print("✅ 系统初始化完成")

        # 运行训练
        print("\n" + "=" * 80)
        print("🏋️ 开始真实MNIST训练...")
        print("=" * 80)

        result = await fl.run(max_rounds=5)

        # 显示结果
        print("\n" + "=" * 80)
        print("🎉 训练完成！")
        print("=" * 80)
        print(f"\n📊 训练结果:")
        print(f"  完成轮数: {result.completed_rounds}/{result.total_rounds}")
        print(f"  终止原因: {result.termination_reason}")
        print(f"  最终准确率: {result.final_accuracy:.4f}")
        print(f"  最终损失: {result.final_loss:.4f}")
        print(f"  总时间: {result.total_time:.2f}秒")

        # 显示训练轨迹
        print(f"\n📈 训练轨迹:")
        for i, round_result in enumerate(result.training_history):
            metrics = round_result.get("round_metrics", {})
            print(f"  Round {i+1}: Loss={metrics.get('avg_loss', 0):.4f}, "
                  f"Acc={metrics.get('avg_accuracy', 0):.4f}, "
                  f"Clients={metrics.get('successful_count', 0)}")

    finally:
        # 清理资源
        await fl.cleanup()

    print("\n✅ 演示完成!")

# ==================== 6. 程序入口 ====================

if __name__ == "__main__":
    print("=" * 80)
    print("MOE-FedCL 真实MNIST联邦学习演示")
    print("=" * 80)
    print("\n特性:")
    print("  ✅ 真实MNIST数据集加载和划分")
    print("  ✅ 真实CNN模型训练")
    print("  ✅ FedAvg聚合算法")
    print("  ✅ 装饰器注册组件")
    print("  ✅ 配置文件驱动")
    print("  ✅ 异步训练和评估")
    print()

    # 运行演示
    try:
        asyncio.run(demo_real_mnist_training())
    except KeyboardInterrupt:
        print("\n❌ 被用户中断")
    except Exception as e:
        print(f"\n❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()