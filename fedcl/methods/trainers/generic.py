"""
通用联邦训练器 - 配置驱动的FedAvg实现
fedcl/methods/trainers/generic.py

完全通过配置文件驱动，无需为每个数据集编写专门的Trainer。
"""
import asyncio
from typing import Dict, Any, List, Optional
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

    def __init__(self, config: Dict[str, Any] = None, lazy_init: bool = True, logger=None, server_id: Optional[str] = None):
        super().__init__(config, lazy_init, logger, server_id)

        # 提取训练参数
        if not hasattr(self, 'local_epochs'):
            self.local_epochs = 1
        if not hasattr(self, 'learning_rate'):
            self.learning_rate = 0.01
        if not hasattr(self, 'batch_size'):
            self.batch_size = 32

        # 学习率衰减配置
        trainer_params = (self.config or {}).get('trainer', {}).get('params', {})
        self.lr_decay_enabled = trainer_params.get('lr_decay', False)
        self.lr_decay_step = trainer_params.get('lr_decay_step', 10)  # 每10轮衰减
        self.lr_decay_rate = trainer_params.get('lr_decay_rate', 0.95)  # 衰减率0.95
        self.current_lr = self.learning_rate  # 当前学习率

        # 早停（Early Stopping）配置
        self.early_stopping_enabled = trainer_params.get('early_stopping', True)  # 默认启用早停
        self.patience = trainer_params.get('patience', 5)  # 默认连续5轮没有提升就停止
        self.min_delta = trainer_params.get('min_delta', 0.001)  # 最小提升阈值0.1%
        self.best_accuracy = 0.0  # 历史最佳准确率
        self.patience_counter = 0  # 当前计数器

        # 测试数据集配置（可选）
        self.test_dataset_config = trainer_params.get('test_dataset')

        # 组件占位符
        self._global_model_obj = None
        self._test_loader = None

        self.logger.info("GenericTrainer初始化完成")
        if self.lr_decay_enabled:
            self.logger.info(f"学习率衰减已启用: 每{self.lr_decay_step}轮衰减{self.lr_decay_rate}")

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

                # 透明访问：registry 返回的工厂类在实例化时直接返回 PyTorch Dataset
                # Trainer 无需感知 FederatedDataset 包装器的存在
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

        # 方式4: 在config['model']中（新的标准位置）
        if 'model' in self.config:
            self.logger.info(f"找到global_model配置: config['model'] = {self.config['model']}")
            return self.config['model']

        self.logger.warning(f"未找到global_model配置，config keys: {list(self.config.keys())}")
        return None

    def _create_default_global_model(self):
        """创建默认全局模型"""
        global_model_config = self._find_global_model_config()

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

    def _create_default_aggregator(self):
        """创建默认聚合器（FedAvg）"""
        self.logger.info("未在配置中指定aggregator，使用默认聚合器: FedAvg")

        from fedcl.api.registry import registry
        aggregator_class = registry.get_aggregator('fedavg')

        # 使用默认的FedAvg参数
        aggregator_params = {
            'weighted': True,  # 默认使用加权聚合
            'config': {}
        }

        return aggregator_class(**aggregator_params)

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

        # 学习率衰减
        if self.lr_decay_enabled and round_num > 1:
            if (round_num - 1) % self.lr_decay_step == 0:
                self.current_lr *= self.lr_decay_rate
                self.logger.info(f"  学习率衰减: {self.current_lr:.6f}")

        self.logger.info(f"  Selected clients: {client_ids}")
        self.logger.info(f"  Current learning rate: {self.current_lr:.6f}")

        # 创建训练请求
        training_params = {
            "round_number": round_num,
            "num_epochs": self.local_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.current_lr  # 使用当前学习率
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
        """聚合客户端模型（使用配置的聚合器）"""
        if not client_results:
            return None

        # 准备聚合器所需的数据格式
        client_updates = []
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

                # 构建聚合器期望的格式
                client_update = {
                    "client_id": client_id,
                    "model_weights": torch_weights,
                    "num_samples": result.result.get('samples_used', 1),
                }

                # 传递额外的数据（如prototypes）
                if "prototypes" in result.result:
                    client_update["prototypes"] = result.result["prototypes"]

                client_updates.append(client_update)

        if not client_updates:
            return None

        # 使用aggregator进行聚合
        self.logger.info(f"  Aggregating {len(client_updates)} clients using aggregator...")
        aggregation_result = self.aggregator.aggregate(client_updates)

        # 提取聚合后的权重
        if isinstance(aggregation_result, dict) and "aggregated_weights" in aggregation_result:
            aggregated_weights = aggregation_result["aggregated_weights"]
        else:
            # 兼容直接返回权重的情况
            aggregated_weights = aggregation_result

        # 分发全局模型（包括可能的prototypes）
        await self._distribute_global_model(aggregated_weights, aggregation_result)

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
        """判断是否应该停止训练（基于Early Stopping）"""
        if not self.early_stopping_enabled:
            return False

        round_metrics = round_result.get("round_metrics", {})
        current_accuracy = round_metrics.get("avg_accuracy", 0.0)

        # 检查是否有改进
        if current_accuracy > self.best_accuracy + self.min_delta:
            # 有改进：更新最佳准确率，重置计数器
            self.best_accuracy = current_accuracy
            self.patience_counter = 0
            self.logger.info(f"  准确率提升至: {current_accuracy:.4f} (best={self.best_accuracy:.4f})")
        else:
            # 无改进：增加计数器
            self.patience_counter += 1
            self.logger.info(
                f"  准确率未提升: {current_accuracy:.4f} (best={self.best_accuracy:.4f}), "
                f"patience={self.patience_counter}/{self.patience}"
            )

            # 检查是否达到耐心上限
            if self.patience_counter >= self.patience:
                self.logger.info(
                    f"  Early stopping triggered: {self.patience} rounds without improvement"
                )
                return True

        return False

    async def _distribute_global_model(self, global_weights: Dict[str, torch.Tensor],
                                      aggregation_result: Dict[str, Any] = None):
        """分发全局模型到所有客户端"""
        # 尝试从config中获取模型名称
        global_model_config = self._find_global_model_config()
        model_name = global_model_config.get('name', 'unknown') if global_model_config else 'unknown'

        global_model_data = {
            "model_type": model_name,
            "parameters": {"weights": global_weights}
        }

        # 如果aggregation_result包含prototypes（如FedProto），则一并分发
        if aggregation_result and isinstance(aggregation_result, dict):
            if "global_prototypes" in aggregation_result:
                global_model_data["prototypes"] = aggregation_result["global_prototypes"]
                self.logger.info(
                    f"  Including {len(aggregation_result['global_prototypes'])} "
                    f"global prototypes in model distribution"
                )

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
