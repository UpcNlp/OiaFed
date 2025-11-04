"""
完整的MNIST联邦学习演示 - 真实训练版本（统一初始化策略）
使用新架构实现真实的MNIST联邦学习训练
examples/complete_mnist_demo.py

使用统一初始化策略：
- 所有组件（Dataset, Model, Aggregator）在Trainer/Learner内部初始化
- 支持延迟加载（lazy_init=True）
- ComponentBuilder.parse_config() 解析配置，返回类引用和参数
- 配置格式：training: {trainer: {name: ..., params: ...}}
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Subset

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from fedcl.learner.base_learner import BaseLearner
from fedcl.trainer.trainer import BaseTrainer
from fedcl.types import TrainingResponse
from fedcl import FederatedLearning

# 导入新实现的数据集和模型管理
from fedcl.api.decorators import dataset, model
from fedcl.api.registry import registry
from fedcl.methods.datasets.base import FederatedDataset
from fedcl.methods.models.base import FederatedModel

# 导入装饰器
from fedcl.api import learner, trainer

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
    """MNIST学习器 - 实现真实的训练（使用统一初始化策略）"""

    def __init__(self, client_id: str, config: Dict[str, Any] = None, lazy_init: bool = True):
        """初始化MNIST学习器

        Args:
            client_id: 客户端ID
            config: 配置字典（由ComponentBuilder.parse_config()生成）
            lazy_init: 是否延迟初始化组件
        """
        super().__init__(client_id, config, lazy_init)

        # 提取训练参数（从config.learner.params）
        if not hasattr(self, 'learning_rate'):
            self.learning_rate = 0.01
        if not hasattr(self, 'batch_size'):
            self.batch_size = 32
        if not hasattr(self, 'local_epochs'):
            self.local_epochs = 1

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 组件占位符（延迟加载）
        self._model = None
        self._optimizer = None
        self._criterion = None
        self._train_loader = None

        self.logger.info(f"MNISTLearner {client_id} 初始化完成 (lazy_init={lazy_init})")

    def _create_default_dataset(self):
        """创建默认数据集（按需加载）"""
        self.logger.info(f"Client {self.client_id}: 加载MNIST数据集...")
        return self._load_dataset()

    def _load_dataset(self):
        """加载数据集"""
        # 从注册表获取MNIST数据集
        mnist_dataset_cls = registry.get_dataset('MNIST')
        mnist_dataset = mnist_dataset_cls(root='./data', train=True, download=True)

        # 获取底层的 PyTorch Dataset
        base_dataset = mnist_dataset.dataset  # torchvision.datasets.MNIST

        # 简单的IID划分（手动划分）
        num_clients = 3
        client_idx = int(self.client_id.split('_')[1])

        # 计算每个客户端的数据范围
        total_size = len(base_dataset)
        samples_per_client = total_size // num_clients
        start_idx = client_idx * samples_per_client
        end_idx = start_idx + samples_per_client if client_idx < num_clients - 1 else total_size

        # 创建索引列表
        indices = list(range(start_idx, end_idx))

        # 创建 Subset
        train_dataset = Subset(base_dataset, indices)

        self.logger.info(f"Client {self.client_id}: 数据集加载完成，样本数={len(train_dataset)}")
        return train_dataset

    @property
    def model(self):
        """延迟加载模型"""
        if self._model is None:
            self._model = MNISTCNNModel(num_classes=10).to(self.device)
            self.logger.debug(f"Client {self.client_id}: 模型创建完成")
        return self._model

    @property
    def optimizer(self):
        """延迟加载优化器"""
        if self._optimizer is None:
            self._optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
            self.logger.debug(f"Client {self.client_id}: 优化器创建完成")
        return self._optimizer

    @property
    def criterion(self):
        """延迟加载损失函数"""
        if self._criterion is None:
            self._criterion = nn.CrossEntropyLoss()
        return self._criterion

    @property
    def train_loader(self):
        """延迟加载数据加载器"""
        if self._train_loader is None:
            # 触发数据集加载
            dataset = self.dataset

            # 打印数据集类型以验证
            dataset_type = type(dataset).__name__
            self.logger.info(f"Client {self.client_id}: 检测到数据集类型 = {dataset_type}")

            # 检查是否是 FederatedDataset（需要进一步处理）
            if hasattr(dataset, 'dataset'):
                # 这是一个 FederatedDataset 包装器（如 MNISTFederatedDataset），需要获取实际的数据
                self.logger.info(f"Client {self.client_id}: 使用 {dataset_type}，从中提取底层数据集")

                # 获取底层的 PyTorch Dataset
                base_dataset = dataset.dataset  # torchvision.datasets.MNIST
                self.logger.debug(f"Client {self.client_id}: 底层数据集类型 = {type(base_dataset).__name__}")

                # 简单的IID划分
                num_clients = 3
                client_idx = int(self.client_id.split('_')[1])

                total_size = len(base_dataset)
                samples_per_client = total_size // num_clients
                start_idx = client_idx * samples_per_client
                end_idx = start_idx + samples_per_client if client_idx < num_clients - 1 else total_size

                indices = list(range(start_idx, end_idx))
                actual_dataset = Subset(base_dataset, indices)

                self.logger.info(f"Client {self.client_id}: 从 {dataset_type} 加载数据集，样本数={len(actual_dataset)}")
            else:
                # 已经是标准的 PyTorch Dataset（从 _create_default_dataset 返回）
                self.logger.info(f"Client {self.client_id}: 使用标准 PyTorch Dataset")
                actual_dataset = dataset

            self._train_loader = DataLoader(
                actual_dataset,
                batch_size=self.batch_size,
                shuffle=True
            )
            self.logger.debug(f"Client {self.client_id}: 数据加载器创建完成")
        return self._train_loader

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
        # 获取数据集（触发延迟加载）
        dataset = self.dataset
        return {
            "model_type": "mnist_cnn",
            "parameters": {"weights": self.model.get_weights_as_dict()},
            "metadata": {
                "client_id": self.client_id,
                "samples": len(dataset),
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
        # 获取数据集（触发延迟加载）
        dataset = self.dataset
        return {
            "total_samples": len(dataset),
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
    """联邦平均训练器 - 实现真实的模型聚合（使用统一初始化策略）"""

    def __init__(self, config: Dict[str, Any] = None, lazy_init: bool = True, logger=None):
        """初始化FedAvgMNIST训练器

        Args:
            config: 配置字典（由ComponentBuilder.parse_config()生成）
            lazy_init: 是否延迟初始化组件
            logger: 日志记录器
        """
        super().__init__(config, lazy_init, logger)

        # 提取训练参数（从config.trainer.params）
        if not hasattr(self, 'local_epochs'):
            self.local_epochs = 1
        if not hasattr(self, 'learning_rate'):
            self.learning_rate = 0.01
        if not hasattr(self, 'batch_size'):
            self.batch_size = 32

        # 组件占位符（延迟加载）
        self._global_model_obj = None
        self._test_loader = None  # 服务器端测试数据加载器

        self.logger.info("FedAvgMNISTTrainer初始化完成")

    @property
    def test_loader(self):
        """延迟加载服务器端测试数据集"""
        if self._test_loader is None:
            self.logger.info("加载服务器端MNIST测试数据集...")
            # 加载完整的MNIST测试集（10000个样本）
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
            self._test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
            self.logger.info(f"服务器端测试集加载完成: {len(test_dataset)} 个样本")
        return self._test_loader

    def _create_default_global_model(self):
        """创建默认全局模型"""
        self.logger.info("创建默认MNIST CNN全局模型")
        model = MNISTCNNModel(num_classes=10)
        return {
            "model_type": "mnist_cnn",
            "parameters": {"weights": model.get_weights_as_dict()},
            "model_obj": model  # 保存模型对象以便后续使用
        }

    @property
    def global_model_obj(self):
        """延迟加载全局模型对象"""
        if self._global_model_obj is None:
            # 触发全局模型加载
            global_model_data = self.global_model
            if isinstance(global_model_data, dict) and "model_obj" in global_model_data:
                self._global_model_obj = global_model_data["model_obj"]
            else:
                # 如果没有模型对象，创建新的
                self._global_model_obj = MNISTCNNModel(num_classes=10)
                if isinstance(global_model_data, dict) and "parameters" in global_model_data:
                    weights = global_model_data["parameters"].get("weights", {})
                    torch_weights = {}
                    for k, v in weights.items():
                        if isinstance(v, np.ndarray):
                            torch_weights[k] = torch.from_numpy(v)
                        else:
                            torch_weights[k] = v
                    self._global_model_obj.set_weights_from_dict(torch_weights)
            self.logger.debug(f"全局模型对象创建完成，参数数量: {self._global_model_obj.get_param_count():,}")
        return self._global_model_obj

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
                self.global_model_obj.set_weights_from_dict(aggregated_weights)

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
        """评估全局模型（服务器端评估）

        在服务器端使用独立的测试集直接评估全局模型，
        而不是通过客户端评估，这样更快速、可靠。
        """
        self.logger.info("  在服务器端评估全局模型...")

        # 获取全局模型和测试数据
        model = self.global_model_obj
        test_loader = self.test_loader

        # 设置为评估模式
        model.eval()

        test_loss = 0.0
        correct = 0
        total = 0

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += data.size(0)

        accuracy = correct / total if total > 0 else 0.0
        avg_loss = test_loss / total if total > 0 else float('inf')

        self.logger.info(f"  服务器端评估结果: Acc={accuracy:.4f}, Loss={avg_loss:.4f}, Samples={total}")

        return {
            "accuracy": accuracy,
            "loss": avg_loss,
            "samples_count": total
        }

    def should_stop_training(self, round_num: int, round_result: Dict[str, Any]) -> bool:
        """判断是否应该停止训练"""

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
    真实MNIST联邦学习演示（使用统一初始化策略）
    """
    print("=" * 80)
    print("[Start] MNIST联邦学习真实训练演示（统一初始化策略）")
    print("=" * 80)

    # 清理Memory模式的共享状态
    from fedcl.communication.memory_manager import MemoryCommunicationManager
    from fedcl.transport.memory import MemoryTransport
    print("\n[Cleanup] 清理Memory模式共享状态...")
    MemoryCommunicationManager.clear_global_state()
    MemoryTransport.clear_global_state()
    print("[OK] 共享状态已清理\n")

    # 显示已注册的组件
    print("[Components] 已注册组件:")
    print(f"  Datasets: {list(registry.datasets.keys())}")
    print(f"  Models: {list(registry.models.keys())}")
    print(f"  Trainers: {list(registry.trainers.keys())}")
    print(f"  Learners: {list(registry.learners.keys())}")

    from fedcl.config import CommunicationConfig, TrainingConfig
    from fedcl.api import ComponentBuilder

    # 使用ComponentBuilder解析配置
    builder = ComponentBuilder()

    # 创建服务器配置（新格式）
    server_config_dict = {
        "training": {
            "trainer": {
                "name": "FedAvgMNIST",
                "params": {
                    "max_rounds": 5,
                    "min_clients": 2,
                    "client_selection_ratio": 1.0,
                    "local_epochs": 1,
                    "learning_rate": 0.01,
                    "batch_size": 32
                }
            },
            "global_model": {
                "name": "MNIST_CNN",
                "params": {
                    "num_classes": 10
                }
            }
        }
    }

    # 解析服务器配置
    server_parsed_config = builder.parse_config(server_config_dict)

    server_comm_config = CommunicationConfig(
        mode="process",
        role="server",
        node_id="server_1"
    )

    server_train_config = TrainingConfig()
    # 设置旧格式的配置（用于BusinessInitializer）
    server_train_config.trainer = {
        "name": "FedAvgMNIST",
        "max_rounds": 5,
        "min_clients": 2,
        "client_selection_ratio": 1.0
    }
    # 传递解析后的配置（用于Trainer的统一初始化）
    server_train_config.parsed_config = server_parsed_config
    server_train_config.max_rounds = 5
    server_train_config.min_clients = 2
    # 设置model配置以避免BusinessInitializer出错
    server_train_config.model = {"name": "MNIST_CNN"}

    # 创建客户端配置（新格式）
    client_configs = []
    for i in range(3):
        client_config_dict = {
            "training": {
                "learner": {
                    "name": "MNISTLearner",
                    "params": {
                        "learning_rate": 0.01,
                        "batch_size": 32,
                        "local_epochs": 1
                    }
                },
                "dataset": {
                    # 使用注册表中的 MNIST 数据集，对应 MNISTFederatedDataset 类（第49行定义）
                    "name": "MNIST",  # 这会创建 MNISTFederatedDataset 实例
                    "params": {
                        "root": "./data",
                        "train": True,
                        "download": True
                    }
                }
            }
        }

        # 解析客户端配置
        client_parsed_config = builder.parse_config(client_config_dict)

        client_comm_config = CommunicationConfig(
            mode="process",
            role="client",
            node_id=f"client_{i}"
        )

        client_train_config = TrainingConfig()
        # 设置旧格式的配置（用于BusinessInitializer）
        client_train_config.learner = {
            "name": "MNISTLearner",
            "learning_rate": 0.01,
            "batch_size": 32,
            "local_epochs": 1
        }
        # 设置 dataset 配置，会通过注册表创建 MNISTFederatedDataset
        client_train_config.dataset = {"name": "MNIST"}
        # 传递解析后的配置（用于Learner的统一初始化）
        client_train_config.parsed_config = client_parsed_config

        client_configs.append((client_comm_config, client_train_config))

    # 合并所有配置
    all_configs = [(server_comm_config, server_train_config)] + client_configs

    # 创建FederatedLearning
    fl = FederatedLearning(all_configs)

    try:
        # 初始化系统
        print("\n[Init] 初始化联邦学习系统...")
        await fl.initialize()
        print("[OK] 系统初始化完成")

        # 运行训练
        print("\n" + "=" * 80)
        print("[Training] 开始真实MNIST训练...")
        print("=" * 80)

        result = await fl.run(max_rounds=5)

        # 显示结果
        print("\n" + "=" * 80)
        print("[Done] 训练完成！")
        print("=" * 80)
        print(f"\n[Results] 训练结果:")
        print(f"  完成轮数: {result.completed_rounds}/{result.total_rounds}")
        print(f"  终止原因: {result.termination_reason}")
        print(f"  最终准确率: {result.final_accuracy:.4f}")
        print(f"  最终损失: {result.final_loss:.4f}")
        print(f"  总时间: {result.total_time:.2f}秒")

        # 显示训练轨迹
        print(f"\n[History] 训练轨迹:")
        for i, round_result in enumerate(result.training_history):
            metrics = round_result.get("round_metrics", {})
            print(f"  Round {i+1}: Loss={metrics.get('avg_loss', 0):.4f}, "
                  f"Acc={metrics.get('avg_accuracy', 0):.4f}, "
                  f"Clients={metrics.get('successful_count', 0)}")

    finally:
        # 清理资源
        await fl.cleanup()

    print("\n[OK] 演示完成!")

# ==================== 6. 程序入口 ====================

if __name__ == "__main__":
    print("=" * 80)
    print("MOE-FedCL 真实MNIST联邦学习演示")
    print("=" * 80)
    print("\n特性:")
    print("  [OK] 真实MNIST数据集加载和划分")
    print("  [OK] 真实CNN模型训练")
    print("  [OK] FedAvg聚合算法")
    print("  [OK] 装饰器注册组件")
    print("  [OK] 配置文件驱动")
    print("  [OK] 异步训练和评估")
    print()

    # 运行演示
    try:
        asyncio.run(demo_real_mnist_training())
    except KeyboardInterrupt:
        print("\n[X] 被用户中断")
    except Exception as e:
        print(f"\n[Error] 演示失败: {e}")
        import traceback
        traceback.print_exc()