"""
标准学习器 - 基于统一初始化策略
fedcl/methods/learners/standard.py

适用于常规联邦学习场景的标准客户端学习器
"""

from typing import Dict, Any
import asyncio
from datetime import datetime
import copy
from loguru import logger

from ...learner.base_learner import BaseLearner
from ...types import TrainingResult, EvaluationResult, ModelData
from ...api import learner


@learner("standard", description="标准联邦学习客户端")
class StandardLearner(BaseLearner):
    """
    标准联邦学习客户端学习器

    支持常规的联邦学习场景：
    - 接收全局模型
    - 本地训练
    - 返回模型更新
    - 本地评估

    使用统一初始化策略：
    - dataset: 从配置自动创建或使用默认实现
    - local_model: 从配置自动创建或使用默认实现
    """

    def __init__(self, client_id: str, config: Dict[str, Any] = None, lazy_init: bool = True):
        """
        初始化标准学习器

        Args:
            client_id: 客户端ID
            config: 配置字典
            lazy_init: 是否延迟初始化组件
        """
        super().__init__(client_id, config, lazy_init)

        # 提取训练参数（从learner.params）
        if not hasattr(self, 'learning_rate'):
            self.learning_rate = 0.01
        if not hasattr(self, 'batch_size'):
            self.batch_size = 32

        logger.info(f"StandardLearner {client_id} 初始化完成")

    def _create_default_dataset(self):
        """提供默认数据集（合成数据）"""
        import numpy as np

        logger.info(f"Client {self.client_id}: 使用合成MNIST数据集")

        # 生成合成数据（简化版）
        np.random.seed(hash(self.client_id) % 2**32)

        # 每个客户端200-400个样本
        num_samples = np.random.randint(200, 400)

        data = {
            "X_train": np.random.randn(num_samples, 784).astype(np.float32),
            "y_train": np.random.randint(0, 10, num_samples).astype(np.int32),
            "X_test": np.random.randn(50, 784).astype(np.float32),
            "y_test": np.random.randint(0, 10, 50).astype(np.int32),
            "num_classes": 10,
            "num_samples": num_samples
        }

        return data

    def _create_default_local_model(self):
        """提供默认本地模型"""
        import numpy as np

        logger.info(f"Client {self.client_id}: 使用默认简单NN模型")

        # 简单的两层神经网络权重
        np.random.seed(hash(self.client_id) % 2**32)

        model = {
            "W1": np.random.normal(0, 0.1, (784, 128)).tolist(),
            "b1": np.zeros(128).tolist(),
            "W2": np.random.normal(0, 0.1, (128, 10)).tolist(),
            "b2": np.zeros(10).tolist()
        }

        return model

    async def train(self, training_params: Dict[str, Any]) -> TrainingResult:
        """
        执行本地训练

        Args:
            training_params: 训练参数
                - global_model: 全局模型参数
                - epochs: 本地训练轮数
                - learning_rate: 学习率
                - batch_size: 批次大小
                - round_num: 轮次编号

        Returns:
            TrainingResult: 训练结果
        """
        async with self._lock:
            logger.info(f"Client {self.client_id} 开始本地训练")

            # 解析训练参数
            global_model = training_params.get("global_model", {})
            epochs = training_params.get("epochs", 5)
            learning_rate = training_params.get("learning_rate", self.learning_rate)
            batch_size = training_params.get("batch_size", self.batch_size)
            round_num = training_params.get("round_num", 0)

            start_time = datetime.now()

            try:
                # 1. 获取数据集（触发延迟加载）
                dataset = self.dataset

                # 2. 初始化或更新本地模型
                if global_model:
                    # 从全局模型初始化
                    if "weights" in global_model:
                        self._local_model = copy.deepcopy(global_model["weights"])
                    else:
                        self._local_model = copy.deepcopy(global_model)
                elif self._local_model is None:
                    # 使用默认模型
                    self._local_model = self._create_default_local_model()

                # 3. 执行本地训练
                training_loss, training_accuracy = await self._local_training(
                    dataset, epochs, learning_rate, batch_size
                )

                # 4. 准备返回结果
                end_time = datetime.now()
                training_time = (end_time - start_time).total_seconds()

                # 更新统计信息
                self.training_count += 1
                self.last_training_time = end_time

                result = {
                    "client_id": self.client_id,
                    "model_update": copy.deepcopy(self._local_model),
                    "weights": copy.deepcopy(self._local_model),  # 兼容性
                    "loss": float(training_loss),
                    "accuracy": float(training_accuracy),
                    "samples_count": len(dataset.get("X_train", [])),
                    "samples": len(dataset.get("X_train", [])),  # 兼容性
                    "training_time": training_time,
                    "epochs_completed": epochs,
                    "round_num": round_num,
                    "success": True
                }

                logger.info(
                    f"Client {self.client_id} 训练完成: "
                    f"Loss={training_loss:.4f}, Acc={training_accuracy:.4f}"
                )

                return result

            except Exception as e:
                error_msg = f"Client {self.client_id} 训练失败: {str(e)}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

    async def _local_training(
        self,
        dataset: Dict[str, Any],
        epochs: int,
        learning_rate: float,
        batch_size: int
    ):
        """
        执行本地SGD训练

        Args:
            dataset: 数据集
            epochs: 训练轮数
            learning_rate: 学习率
            batch_size: 批次大小

        Returns:
            (avg_loss, avg_accuracy): 平均损失和准确率
        """
        import numpy as np

        X_train = dataset.get("X_train")
        y_train = dataset.get("y_train")

        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0

        for epoch in range(epochs):
            # 随机打乱数据
            indices = np.random.permutation(len(X_train))

            epoch_loss = 0.0
            epoch_accuracy = 0.0
            epoch_batches = 0

            # 批次训练
            for i in range(0, len(X_train), batch_size):
                batch_indices = indices[i:i + batch_size]
                X_batch = X_train[batch_indices]
                y_batch = y_train[batch_indices]

                # 前向传播
                predictions, loss = self._forward_pass(X_batch, y_batch)

                # 反向传播
                await self._backward_pass(X_batch, y_batch, predictions, learning_rate)

                # 计算准确率
                batch_accuracy = np.mean(np.argmax(predictions, axis=1) == y_batch)

                epoch_loss += loss
                epoch_accuracy += batch_accuracy
                epoch_batches += 1

                # 让出控制权（模拟异步）
                if epoch_batches % 5 == 0:
                    await asyncio.sleep(0.001)

            # 计算epoch平均值
            if epoch_batches > 0:
                epoch_loss /= epoch_batches
                epoch_accuracy /= epoch_batches

                total_loss += epoch_loss
                total_accuracy += epoch_accuracy
                num_batches += 1

        # 返回平均损失和准确率
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_accuracy = total_accuracy / num_batches if num_batches > 0 else 0.0

        return avg_loss, avg_accuracy

    def _forward_pass(self, X, y):
        """前向传播（简单NN）"""
        import numpy as np

        # 第一层：784 -> 128
        z1 = np.dot(X, np.array(self._local_model["W1"])) + np.array(self._local_model["b1"])
        a1 = np.maximum(0, z1)  # ReLU

        # 第二层：128 -> 10
        z2 = np.dot(a1, np.array(self._local_model["W2"])) + np.array(self._local_model["b2"])

        # Softmax
        exp_z2 = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
        a2 = exp_z2 / np.sum(exp_z2, axis=1, keepdims=True)

        # 交叉熵损失
        m = X.shape[0]
        a2_clipped = np.clip(a2, 1e-15, 1 - 1e-15)
        loss = -np.mean(np.log(a2_clipped[range(m), y]))

        # 保存中间结果
        self._forward_cache = {
            "X": X, "z1": z1, "a1": a1, "z2": z2, "a2": a2, "y": y
        }

        return a2, loss

    async def _backward_pass(self, X, y, predictions, learning_rate):
        """反向传播（简单NN）"""
        import numpy as np

        m = X.shape[0]

        # 从缓存获取中间结果
        a1 = self._forward_cache["a1"]
        a2 = predictions

        # 输出层梯度
        y_one_hot = np.eye(10)[y]
        dz2 = a2 - y_one_hot
        dW2 = (1/m) * np.dot(a1.T, dz2)
        db2 = (1/m) * np.sum(dz2, axis=0)

        # 隐藏层梯度
        da1 = np.dot(dz2, np.array(self._local_model["W2"]).T)
        dz1 = da1 * (self._forward_cache["z1"] > 0)  # ReLU导数
        dW1 = (1/m) * np.dot(X.T, dz1)
        db1 = (1/m) * np.sum(dz1, axis=0)

        # 更新参数
        W1 = np.array(self._local_model["W1"])
        b1 = np.array(self._local_model["b1"])
        W2 = np.array(self._local_model["W2"])
        b2 = np.array(self._local_model["b2"])

        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

        # 更新模型
        self._local_model["W1"] = W1.tolist()
        self._local_model["b1"] = b1.tolist()
        self._local_model["W2"] = W2.tolist()
        self._local_model["b2"] = b2.tolist()

        await asyncio.sleep(0.0001)  # 模拟异步

    async def evaluate(self, evaluation_params: Dict[str, Any]) -> EvaluationResult:
        """
        执行本地评估

        Args:
            evaluation_params: 评估参数
                - model: 要评估的模型（可选）
                - test_data: 是否使用测试数据

        Returns:
            EvaluationResult: 评估结果
        """
        async with self._lock:
            logger.info(f"Client {self.client_id} 开始评估")

            # 解析评估参数
            model = evaluation_params.get("model")
            use_test_data = evaluation_params.get("test_data", True)

            start_time = datetime.now()

            try:
                # 获取数据集
                dataset = self.dataset

                # 选择评估数据
                if use_test_data and "X_test" in dataset:
                    X_eval = dataset["X_test"]
                    y_eval = dataset["y_test"]
                else:
                    X_eval = dataset["X_train"]
                    y_eval = dataset["y_train"]

                # 临时设置模型
                original_model = self._local_model
                if model:
                    if "weights" in model:
                        self._local_model = copy.deepcopy(model["weights"])
                    else:
                        self._local_model = copy.deepcopy(model)

                # 执行评估
                import numpy as np
                predictions, loss = self._forward_pass(X_eval, y_eval)
                accuracy = np.mean(np.argmax(predictions, axis=1) == y_eval)

                # 恢复原模型
                self._local_model = original_model

                end_time = datetime.now()
                evaluation_time = (end_time - start_time).total_seconds()

                # 更新统计
                self.evaluation_count += 1
                self.last_evaluation_time = end_time

                result = {
                    "client_id": self.client_id,
                    "accuracy": float(accuracy),
                    "loss": float(loss),
                    "samples_count": len(X_eval),
                    "evaluation_time": evaluation_time
                }

                logger.info(
                    f"Client {self.client_id} 评估完成: "
                    f"Loss={loss:.4f}, Acc={accuracy:.4f}"
                )

                return result

            except Exception as e:
                error_msg = f"Client {self.client_id} 评估失败: {str(e)}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

    async def get_local_model(self) -> ModelData:
        """获取本地模型参数"""
        if self._local_model is None:
            self._local_model = self._create_default_local_model()

        return {
            "weights": copy.deepcopy(self._local_model),
            "model_version": getattr(self, '_model_version', 1),
            "client_id": self.client_id,
            "last_updated": datetime.now().isoformat()
        }

    async def set_local_model(self, model_data: ModelData) -> bool:
        """设置本地模型参数"""
        try:
            if "weights" in model_data:
                self._local_model = copy.deepcopy(model_data["weights"])
            else:
                self._local_model = copy.deepcopy(model_data)

            self._model_version = model_data.get("model_version", 1)

            logger.debug(f"Client {self.client_id} 更新本地模型")
            return True

        except Exception as e:
            logger.error(f"Client {self.client_id} 模型更新失败: {e}")
            return False
