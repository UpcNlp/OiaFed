"""
通用联邦训练器 - 配置驱动的FedAvg实现
fedcl/methods/trainers/generic.py

完全通过配置文件驱动，无需为每个数据集编写专门的Trainer。
"""
import asyncio
from typing import Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from fedcl.api.decorators import trainer
from fedcl.trainer.trainer import BaseTrainer
from fedcl.methods.models.base import (
    get_weights_as_dict,
    set_weights_from_dict,
    get_param_count
)


@trainer('Generic',
         description='通用联邦训练器 - FedAvg算法',
         version='1.0',
         author='MOE-FedCL',
         algorithms=['fedavg'])
class GenericTrainer(BaseTrainer):
    """通用联邦训练器 - 实现FedAvg算法

    配置示例:
    {
        "trainer": {
            "name": "Generic",
            "params": {
                "local_epochs": 1,
                "learning_rate": 0.01,
                "batch_size": 32,
                "test_dataset": {  # 可选：服务器端评估数据集
                    "name": "MNIST",
                    "params": {"root": "./data", "train": false}
                }
            }
        },
        "global_model": {
            "name": "MNIST_CNN",
            "params": {"num_classes": 10}
        }
    }
    """

    def __init__(self, config: Dict[str, Any] = None, lazy_init: bool = True, logger=None):
        super().__init__(config, lazy_init, logger)

        # 提取训练参数
        if not hasattr(self, 'local_epochs'):
            self.local_epochs = 1
        if not hasattr(self, 'learning_rate'):
            self.learning_rate = 0.01
        if not hasattr(self, 'batch_size'):
            self.batch_size = 32

        # 测试数据集配置（可选）
        trainer_params = (self.config or {}).get('trainer', {}).get('params', {})
        self.test_dataset_config = trainer_params.get('test_dataset')

        # 组件占位符
        self._global_model_obj = None
        self._test_loader = None

        self.logger.info("GenericTrainer初始化完成")

    @property
    def test_loader(self):
        """延迟加载服务器端测试数据集"""
        if self._test_loader is None and self.test_dataset_config:
            try:
                from fedcl.api.registry import registry
                dataset_name = self.test_dataset_config.get('name')
                dataset_params = self.test_dataset_config.get('params', {})

                self.logger.info(f"加载服务器端测试数据集: {dataset_name}")
                dataset_class = registry.get_dataset(dataset_name)
                test_dataset = dataset_class(**dataset_params)

                self._test_loader = DataLoader(
                    test_dataset,
                    batch_size=self.test_dataset_config.get('batch_size', 1000),
                    shuffle=False
                )
                self.logger.info(f"服务器端测试集加载完成: {len(test_dataset)} 个样本")
            except Exception as e:
                self.logger.warning(f"无法加载测试数据集: {e}")
                self._test_loader = None

        return self._test_loader

    def _find_global_model_config(self):
        """智能查找global_model配置"""
        if not hasattr(self, 'config') or not self.config:
            self.logger.warning("Trainer没有config属性或config为空")
            return None

        # 方式1: 直接在config中
        if 'global_model' in self.config:
            self.logger.debug(f"找到global_model配置: config['global_model']")
            return self.config['global_model']

        # 方式2: 在training子项中
        if 'training' in self.config and 'global_model' in self.config['training']:
            self.logger.debug(f"找到global_model配置: config['training']['global_model']")
            return self.config['training']['global_model']

        # 方式3: 在trainer的params中（不推荐但支持）
        if 'trainer' in self.config and 'params' in self.config['trainer']:
            params = self.config['trainer']['params']
            if 'global_model' in params:
                self.logger.debug(f"找到global_model配置: config['trainer']['params']['global_model']")
                return params['global_model']

        self.logger.warning(f"未找到global_model配置，config keys: {list(self.config.keys())}")
        return None

    def _create_default_global_model(self):
        """创建默认全局模型"""
        global_model_config = self._find_global_model_config()

        self.logger.debug(f"_create_default_global_model: global_model_config = {global_model_config}")

        if not global_model_config:
            raise ValueError(
                "必须在配置中指定global_model。"
                "可以在config['global_model']或config['training']['global_model']中指定。"
            )

        model_name = global_model_config.get('name')
        if not model_name:
            raise ValueError(f"必须在global_model中指定name，当前配置: {global_model_config}")

        model_params = global_model_config.get('params', {})

        self.logger.info(f"创建全局模型: {model_name}")
        from fedcl.api.registry import registry
        model_class = registry.get_model(model_name)
        model = model_class(**model_params)

        return {
            "model_type": model_name,
            "parameters": {"weights": get_weights_as_dict(model)},
            "model_obj": model
        }

    @property
    def global_model_obj(self):
        """延迟加载全局模型对象"""
        if self._global_model_obj is None:
            global_model_data = self.global_model
            from fedcl.api.registry import registry

            global_model_config = self._find_global_model_config()
            if not global_model_config:
                raise ValueError("无法找到global_model配置")

            model_name = global_model_config.get('name')
            model_params = global_model_config.get('params', {})
            model_class = registry.get_model(model_name)

            if isinstance(global_model_data, dict) and "model_obj" in global_model_data:
                self._global_model_obj = global_model_data["model_obj"]
            else:
                self._global_model_obj = model_class(**model_params)
                if isinstance(global_model_data, dict) and "parameters" in global_model_data:
                    weights = global_model_data["parameters"].get("weights", {})
                    torch_weights = {}
                    for k, v in weights.items():
                        if isinstance(v, np.ndarray):
                            torch_weights[k] = torch.from_numpy(v)
                        else:
                            torch_weights[k] = v
                    set_weights_from_dict(self._global_model_obj, torch_weights)

            self.logger.debug(f"全局模型对象创建完成，参数数量: {get_param_count(self._global_model_obj):,}")
        return self._global_model_obj

    async def train_round(self, round_num: int, client_ids: List[str]) -> Dict[str, Any]:
        """执行一轮联邦训练"""
        self.logger.info(f"\n--- Round {round_num} ---")
        self.logger.info(f"  Selected clients: {client_ids}")

        # 创建训练请求
        training_params = {
            "round_number": round_num,
            "num_epochs": self.local_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate
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
                    self.logger.info(
                        f"  [{client_id}] Training succeeded: "
                        f"Loss={result.result['loss']:.4f}, Acc={result.result['accuracy']:.4f}"
                    )
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
                set_weights_from_dict(self.global_model_obj, aggregated_weights)

        # 计算轮次指标
        if client_results:
            avg_loss = np.mean([r.result['loss'] for r in client_results.values()])
            avg_accuracy = np.mean([r.result['accuracy'] for r in client_results.values()])
        else:
            avg_loss, avg_accuracy = 0.0, 0.0

        self.logger.info(f"  Round {round_num} summary: Loss={avg_loss:.4f}, Acc={avg_accuracy:.4f}")

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
        """聚合客户端模型（FedAvg算法）"""
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

        # 分发全局模型
        await self._distribute_global_model(aggregated_weights)

        return aggregated_weights

    async def evaluate_global_model(self) -> Dict[str, Any]:
        """评估全局模型（服务器端评估）"""
        if not self.test_loader:
            self.logger.warning("未配置测试数据集，跳过服务器端评估")
            return {}  # 返回空字典，表示没有评估结果

        self.logger.info("  在服务器端评估全局模型...")

        model = self.global_model_obj
        test_loader = self.test_loader

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
        round_metrics = round_result.get("round_metrics", {})
        avg_accuracy = round_metrics.get("avg_accuracy", 0.0)

        # 高准确率提前停止
        if avg_accuracy >= 0.98:
            self.logger.info(f"  High accuracy achieved: {avg_accuracy:.4f}")
            return True

        return False

    async def _distribute_global_model(self, global_weights: Dict[str, torch.Tensor]):
        """分发全局模型到所有客户端"""
        # 尝试从config中获取模型名称
        global_model_config = self._find_global_model_config()
        model_name = global_model_config.get('name', 'unknown') if global_model_config else 'unknown'

        global_model_data = {
            "model_type": model_name,
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
