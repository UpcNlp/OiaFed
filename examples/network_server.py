"""
Network模式 - 独立服务器脚本（真实MNIST训练版本）
用于跨机器的联邦学习测试

使用方法:
  python examples/network_server.py

配置文件:
  - examples/configs/network_demo/server.yaml
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from fedcl.config import ConfigLoader
from fedcl.federation.server import FederationServer
from fedcl.trainer.trainer import BaseTrainer
from fedcl.api import trainer
from fedcl.utils.auto_logger import setup_auto_logging

# 导入新实现的数据集和模型管理
from fedcl.api.decorators import dataset, model
from fedcl.methods.datasets.base import FederatedDataset
from fedcl.methods.models.base import FederatedModel


# ==================== 1. 注册真实的MNIST数据集 ====================

@dataset(
    name='MNIST',
    description='MNIST手写数字数据集',
    dataset_type='image_classification',
    num_classes=10
)
class MNISTFederatedDataset(FederatedDataset):
    """MNIST联邦数据集实现"""

    def __init__(self, root: str = '.examples/data', train: bool = True, download: bool = True):
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


# ==================== 3. 定义Trainer ====================

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
            test_dataset = datasets.MNIST('examples/data', train=False, download=True, transform=transform)
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


# ==================== 主函数 ====================

async def main():
    """主函数"""
    # 设置日志
    setup_auto_logging()

    print("\n" + "="*70)
    print("Network模式 - MNIST联邦学习服务器（真实训练版本）")
    print("="*70)

    # 配置文件路径
    config_file = "examples/configs/network_demo/server.yaml"

    # 训练参数
    wait_time = 30  # 等待客户端连接的时间（秒）

    server = None

    try:
        print(f"\n从配置文件加载: {config_file}")
        comm_config, train_config = ConfigLoader.load(config_file)

        print(f"服务器ID: {comm_config.node_id}")
        print(f"监听地址: {comm_config.transport.get('host')}:{comm_config.transport.get('port')}")

        # 创建服务器
        server = FederationServer(comm_config, train_config)

        # 初始化
        print("\n初始化服务器...")
        await server.initialize()

        # 启动服务器
        print("启动服务器...")
        await server.start_server()

        print(f"\n[OK] 服务器已启动")
        print(f"\n等待客户端连接（{wait_time}秒）...\n")

        # 等待客户端连接
        await asyncio.sleep(wait_time)

        # 检查已连接的客户端
        available_clients = server.trainer.get_available_clients()
        print(f"\n已注册客户端: {len(available_clients)}")
        for client_id in available_clients:
            print(f"  - {client_id}")

        min_clients = server.trainer.min_clients
        if len(available_clients) < min_clients:
            print(f"\n[Warning] 期望至少{min_clients}个客户端，但只有 {len(available_clients)} 个")
            print(f"继续等待{wait_time}秒或按 Ctrl+C 取消...\n")
            await asyncio.sleep(wait_time)
            available_clients = server.trainer.get_available_clients()
            print(f"当前已注册客户端: {len(available_clients)}")

        if len(available_clients) == 0:
            print("[Error] 没有客户端连接，退出")
            return

        # 开始训练 - 使用 Trainer 的 run_training 方法
        max_rounds = server.trainer.max_rounds  # 从 trainer 中读取配置
        print(f"\n{'='*70}")
        print(f"开始联邦学习训练 (最多 {max_rounds} 轮)")
        print(f"{'='*70}\n")

        result = await server.trainer.run_training(max_rounds)

        # 显示训练结果
        print(f"\n{'='*70}")
        print("训练完成!")
        print(f"{'='*70}")
        print(f"实际完成轮数: {result.completed_rounds}/{max_rounds}")
        print(f"停止原因: {result.termination_reason}")
        print(f"\n最终全局模型评估:")
        print(f"  准确率: {result.final_accuracy:.4f}")
        print(f"  损失: {result.final_loss:.4f}")
        print(f"  总时间: {result.total_time:.2f}秒")

        # 保持服务器运行
        print("\n服务器将继续运行...")
        print("按 Ctrl+C 停止\n")
        while True:
            await asyncio.sleep(10)

    except KeyboardInterrupt:
        print("\n\n[INFO] 用户中断，停止服务器...")
    except FileNotFoundError as e:
        print(f"\n[Error] 配置文件不存在: {config_file}")
        print(f"   详细错误: {e}")
        print("\n请确保配置文件路径正确")
    except Exception as e:
        print(f"\n[Error] {e}")
        import traceback
        traceback.print_exc()
    finally:
        if server:
            print("\n清理资源...")
            await server.stop_server()


if __name__ == "__main__":
    asyncio.run(main())
