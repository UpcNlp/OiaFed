"""
Network模式 - 独立客户端脚本（真实MNIST训练版本）
用于跨机器的联邦学习测试

使用方法:
  # 客户端1
  python examples/network_client.py

  # 客户端2（需要修改脚本中的配置文件路径）
  python examples/network_client.py

配置文件:
  - examples/configs/network_demo/client1.yaml
  - examples/configs/network_demo/client2.yaml
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from fedcl.config import ConfigLoader
from fedcl.federation.client import FederationClient
from fedcl.learner.base_learner import BaseLearner
from fedcl.types import TrainingResponse
from fedcl.utils.auto_logger import setup_auto_logging

# 导入新实现的数据集和模型管理
from fedcl.api.decorators import dataset, model
from fedcl.api.registry import registry
from fedcl.methods.datasets.base import FederatedDataset
from fedcl.methods.models.base import FederatedModel
from fedcl.api import learner


# ==================== 1. 注册真实的MNIST数据集 ====================

@dataset(
    name='MNIST',
    description='MNIST手写数字数据集',
    dataset_type='image_classification',
    num_classes=10
)
class MNISTFederatedDataset(FederatedDataset):
    """MNIST联邦数据集实现"""

    def __init__(self, root: str = 'examples/data', train: bool = True, download: bool = True):
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
        self.dropout1(x)
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
        mnist_dataset = mnist_dataset_cls(root='examples/data', train=True, download=True)

        # 获取底层的 PyTorch Dataset
        base_dataset = mnist_dataset.dataset  # torchvision.datasets.MNIST

        # 简单的IID划分（手动划分）
        # 从 client_id 中提取索引（如 "network_client_1" -> 1）
        try:
            client_idx = int(self.client_id.split('_')[-1])
        except:
            client_idx = 0

        # 假设最多3个客户端
        num_clients = 3

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
                try:
                    client_idx = int(self.client_id.split('_')[-1])
                except:
                    client_idx = 0

                num_clients = 3
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


# ==================== 主函数 ====================

async def main():
    """主函数"""
    # 设置日志
    setup_auto_logging()

    print("\n" + "="*70)
    print("Network模式 - MNIST联邦学习客户端（真实训练版本）")
    print("="*70)

    # 配置文件路径（默认使用 client1.yaml）
    # 如果要运行客户端2，请修改为 client2.yaml
    config_file = "examples/configs/network_demo/client1.yaml"

    client = None

    try:
        print(f"\n从配置文件加载: {config_file}")
        comm_config, train_config = ConfigLoader.load(config_file)

        print(f"客户端ID: {comm_config.node_id}")
        print(f"服务器地址: {comm_config.transport.get('server', {}).get('host')}:{comm_config.transport.get('server', {}).get('port')}")

        # 创建客户端
        client = FederationClient(comm_config, train_config)

        # 初始化
        print("\n初始化客户端...")
        await client.initialize()

        # 启动客户端
        print("启动客户端...")
        await client.start_client()

        print(f"\n[OK] 客户端已启动并等待服务器训练指令")

        # 等待注册完成
        if not client.is_registered:
            print("\n等待注册到服务器...")
            await asyncio.sleep(5)
            print(f"   注册状态: {'已注册' if client.is_registered else '未注册'}")

        print("\n[INFO] 客户端正在运行...")
        print("(按 Ctrl+C 停止)\n")

        # 保持运行，等待服务器的训练请求
        while True:
            await asyncio.sleep(10)

    except KeyboardInterrupt:
        print("\n\n[INFO] 用户中断，停止客户端...")
    except FileNotFoundError as e:
        print(f"\n[Error] 配置文件不存在: {config_file}")
        print(f"   详细错误: {e}")
        print("\n请确保配置文件路径正确，或修改脚本中的 config_file 变量")
    except Exception as e:
        print(f"\n[Error] {e}")
        import traceback
        traceback.print_exc()
    finally:
        if client:
            print("\n清理资源...")
            await client.stop_client()


if __name__ == "__main__":
    asyncio.run(main())
