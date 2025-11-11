"""
FedAvg MNIST 训练器
fedcl/methods/trainers/fedavg_mnist.py
"""
import asyncio
from typing import Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from fedcl.api.decorators import trainer
from fedcl.trainer.trainer import BaseTrainer


@trainer('FedAvgMNIST',
         description='MNIST联邦平均训练器',
         version='1.0',
         author='MOE-FedCL',
         algorithms=['fedavg'])
class FedAvgMNISTTrainer(BaseTrainer):
    """联邦平均训练器 - 实现真实的模型聚合（使用统一初始化策略）

    特性：
    - FedAvg 聚合算法
    - 服务器端全局模型评估
    - 自动模型分发
    """

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
            test_dataset = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)
            self._test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
            self.logger.info(f"服务器端测试集加载完成: {len(test_dataset)} 个样本")
        return self._test_loader

    def _create_default_global_model(self):
        """创建默认全局模型"""
        self.logger.info("创建默认MNIST CNN全局模型")
        # 从注册表获取模型类
        from fedcl.api.registry import registry
        model_class = registry.get_model('MNIST_CNN')
        model = model_class(num_classes=10)
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
            # 从注册表获取模型类
            from fedcl.api.registry import registry
            model_class = registry.get_model('MNIST_CNN')

            if isinstance(global_model_data, dict) and "model_obj" in global_model_data:
                self._global_model_obj = global_model_data["model_obj"]
            else:
                # 如果没有模型对象，创建新的
                self._global_model_obj = model_class(num_classes=10)
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
