"""
FedAvg标准训练器 - 基于统一初始化策略
fedcl/methods/trainers/fedavg.py

实现McMahan et al.的FedAvg算法
"""

from typing import Dict, Any, List
import asyncio
from loguru import logger

from ...trainer.trainer import BaseTrainer
from ...types import ModelData, RoundResult, EvaluationResult
from ...api import trainer


@trainer("fedavg", description="FedAvg标准训练器", algorithm="FedAvg")
class FedAvgTrainer(BaseTrainer):
    """
    FedAvg标准训练器

    实现McMahan et al. (2017) 的联邦平均算法

    使用统一初始化策略：
    - aggregator: 从配置自动创建或使用默认FedAvg
    - global_model: 从配置自动创建或使用默认实现
    """

    def __init__(self, config: Dict[str, Any] = None, lazy_init: bool = True):
        """
        初始化FedAvg训练器

        Args:
            config: 配置字典，包含组件类引用和参数
            lazy_init: 是否延迟初始化组件
        """
        super().__init__(config, lazy_init)

        # FedAvg特定参数（从config.trainer.params提取）
        if not hasattr(self, 'convergence_threshold'):
            self.convergence_threshold = 0.001
        if not hasattr(self, 'patience'):
            self.patience = 5

        logger.info("FedAvgTrainer初始化完成")

    def _create_default_aggregator(self):
        """提供默认FedAvg聚合器"""
        from ..aggregators.fedavg import FedAvgAggregator
        logger.info("使用默认FedAvg聚合器")
        return FedAvgAggregator(weighted=True)

    def _create_default_global_model(self):
        """提供默认全局模型（简单示例）"""
        import numpy as np
        logger.info("使用默认全局模型（简单NN）")
        return {
            "weights": {
                "W1": np.random.normal(0, 0.1, (784, 128)).tolist(),
                "b1": np.zeros(128).tolist(),
                "W2": np.random.normal(0, 0.1, (128, 10)).tolist(),
                "b2": np.zeros(10).tolist()
            },
            "model_version": 1,
            "architecture": "simple_nn"
        }

    async def train_round(self, round_num: int, client_ids: List[str]) -> RoundResult:
        """
        执行一轮FedAvg训练

        Args:
            round_num: 当前轮次编号
            client_ids: 参与训练的客户端ID列表

        Returns:
            RoundResult: 轮次训练结果
        """
        logger.info(f"开始第 {round_num} 轮训练，参与客户端: {client_ids}")

        # 1. 并发发送训练任务到所有客户端
        training_tasks = []
        for client_id in client_ids:
            if client_id in self.learner_proxies:
                proxy = self.learner_proxies[client_id]

                # 构建训练参数
                training_params = {
                    "global_model": self.global_model,
                    "round_num": round_num,
                    "epochs": getattr(self, 'local_epochs', 5),
                    "learning_rate": getattr(self, 'learning_rate', 0.01),
                    "batch_size": getattr(self, 'batch_size', 32)
                }

                # 异步调用客户端训练
                task = proxy.train(training_params)
                training_tasks.append((client_id, task))

        # 2. 等待所有训练完成并收集结果
        client_results = {}
        failed_clients = []

        for client_id, task in training_tasks:
            try:
                result = await task
                client_results[client_id] = result
                logger.debug(f"客户端 {client_id} 训练完成")
            except Exception as e:
                logger.error(f"客户端 {client_id} 训练失败: {e}")
                failed_clients.append(client_id)

        if not client_results:
            raise RuntimeError(f"第 {round_num} 轮：所有客户端训练都失败")

        # 3. 聚合模型
        aggregated_model = await self.aggregate_models(client_results)

        # 4. 更新全局模型
        self.global_model = aggregated_model

        # 5. 计算轮次指标
        avg_loss = sum(r.get("loss", 0) for r in client_results.values()) / len(client_results)
        avg_accuracy = sum(r.get("accuracy", 0) for r in client_results.values()) / len(client_results)

        return {
            "round_num": round_num,
            "participants": client_ids,
            "successful_clients": list(client_results.keys()),
            "failed_clients": failed_clients,
            "aggregated_model": aggregated_model,
            "round_metrics": {
                "avg_loss": avg_loss,
                "avg_accuracy": avg_accuracy,
                "num_participants": len(client_results)
            }
        }

    async def aggregate_models(self, client_results: Dict[str, Any]) -> ModelData:
        """
        聚合客户端模型

        Args:
            client_results: 客户端训练结果 {client_id: training_result}

        Returns:
            ModelData: 聚合后的全局模型
        """
        logger.debug(f"聚合 {len(client_results)} 个客户端模型")

        # 使用聚合器（会触发延迟加载）
        aggregator = self.aggregator

        # 准备聚合数据
        client_data_list = []
        for client_id, result in client_results.items():
            client_data = {
                "client_id": client_id,
                "model_update": result.get("model_update") or result.get("weights"),
                "num_samples": result.get("samples_count") or result.get("samples", 1),
                "metrics": {
                    "loss": result.get("loss", 0),
                    "accuracy": result.get("accuracy", 0)
                }
            }
            client_data_list.append(client_data)

        # 执行聚合
        aggregation_result = await aggregator.aggregate(
            client_results=client_data_list,
            global_model=self.global_model
        )

        return aggregation_result.get("aggregated_model") or aggregation_result

    async def evaluate_global_model(self) -> EvaluationResult:
        """
        评估全局模型

        Returns:
            EvaluationResult: 评估结果
        """
        logger.info("评估全局模型")

        # 选择评估客户端（默认使用所有可用客户端）
        available_clients = self._proxy_manager.get_available_clients()

        if not available_clients:
            logger.warning("没有可用客户端进行评估")
            return {
                "accuracy": 0.0,
                "loss": float('inf'),
                "participants": 0
            }

        # 并发评估
        eval_tasks = []
        for client_id in available_clients:
            proxy = self.learner_proxies[client_id]
            task = proxy.evaluate({"model": self.global_model})
            eval_tasks.append((client_id, task))

        # 收集评估结果
        eval_results = []
        for client_id, task in eval_tasks:
            try:
                result = await task
                eval_results.append(result)
            except Exception as e:
                logger.error(f"客户端 {client_id} 评估失败: {e}")

        if not eval_results:
            return {
                "accuracy": 0.0,
                "loss": float('inf'),
                "participants": 0
            }

        # 计算平均指标
        avg_accuracy = sum(r.get("accuracy", 0) for r in eval_results) / len(eval_results)
        avg_loss = sum(r.get("loss", 0) for r in eval_results) / len(eval_results)

        return {
            "accuracy": avg_accuracy,
            "loss": avg_loss,
            "participants": len(eval_results),
            "metrics": {
                "avg_accuracy": avg_accuracy,
                "avg_loss": avg_loss
            }
        }

    def should_stop_training(self, round_num: int, round_result: RoundResult) -> bool:
        """
        判断是否应该停止训练

        Args:
            round_num: 当前轮次
            round_result: 轮次结果

        Returns:
            bool: 是否应该停止训练
        """
        # 检查是否达到最大轮数
        if round_num >= self.training_config.max_rounds:
            logger.info(f"达到最大轮数 {self.training_config.max_rounds}，停止训练")
            return True

        # 检查收敛性（基于准确率）
        current_accuracy = round_result.get("round_metrics", {}).get("avg_accuracy", 0)

        if abs(current_accuracy - self.training_status.best_accuracy) < self.convergence_threshold:
            self.training_status.patience_counter += 1
            if self.training_status.patience_counter >= self.patience:
                logger.info(f"模型收敛（patience={self.patience}），停止训练")
                return True
        else:
            # 重置patience计数器
            self.training_status.patience_counter = 0
            if current_accuracy > self.training_status.best_accuracy:
                self.training_status.best_accuracy = current_accuracy

        return False
