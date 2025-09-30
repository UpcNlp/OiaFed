"""
MNIST 学习器实现
基于手写数字识别的客户端学习器
"""

import asyncio
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import copy

from fedcl.learner.base_learner import BaseLearner
from fedcl.types import TrainingResult, EvaluationResult, ModelData
from fedcl.exceptions import TrainingError


class MNISTLearner(BaseLearner):
    """基于MNIST的客户端学习器实现"""
    
    def __init__(self,
                 client_id: str,
                 local_data: Dict[str, Any] = None,
                 model_config: Dict[str, Any] = None,
                 training_config: Dict[str, Any] = None):
        """初始化MNIST学习器"""
        super().__init__(client_id, local_data, model_config, training_config)
        
        # 训练配置
        self.learning_rate = training_config.get("learning_rate", 0.01) if training_config else 0.01
        self.batch_size = training_config.get("batch_size", 32) if training_config else 32
        
        # 生成或加载本地数据
        if local_data is None:
            self.local_data = self._generate_synthetic_data()
        else:
            self.local_data = local_data
        
        # 本地模型（初始为空）
        self._local_model = None
        
        print(f"📱 MNIST 学习器 {client_id} 初始化完成")
        print(f"   数据集大小: {len(self.local_data['X_train'])} 训练样本, {len(self.local_data['X_test'])} 测试样本")
        print(f"   数据分布: {self._get_label_distribution()}")
    
    def _generate_synthetic_data(self) -> Dict[str, Any]:
        """生成合成MNIST数据（模拟真实场景）"""
        np.random.seed(hash(self.client_id) % 2**32)  # 基于客户端ID的随机种子
        
        # 模拟不同客户端的数据异构性
        try:
            # 尝试从客户端ID中提取数字
            if '_' in self.client_id:
                client_num = int(self.client_id.split('_')[-1])
            else:
                # 如果没有下划线，使用哈希值
                client_num = abs(hash(self.client_id)) % 100
        except ValueError:
            # 如果解析失败，使用哈希值
            client_num = abs(hash(self.client_id)) % 100
        
        # 不同客户端偏向不同的数字类别（Non-IID分布）
        preferred_classes = [(client_num * 3 + i) % 10 for i in range(3)]  # 每个客户端偏向3个数字
        other_classes = [i for i in range(10) if i not in preferred_classes]
        
        # 生成训练数据
        train_samples = np.random.randint(200, 400)  # 减少样本量：200-400个训练样本
        test_samples = np.random.randint(50, 100)     # 减少样本量：50-100个测试样本
        
        # 生成偏向特定类别的数据
        X_train = []
        y_train = []
        X_test = []
        y_test = []
        
        # 训练数据：70%来自偏好类别，30%来自其他类别
        for i in range(train_samples):
            if np.random.random() < 0.7:  # 70%概率选择偏好类别
                label = np.random.choice(preferred_classes)
            else:  # 30%概率选择其他类别
                label = np.random.choice(other_classes)
            
            # 生成该类别的合成图像特征（28x28=784维）
            image_features = self._generate_digit_features(label)
            X_train.append(image_features)
            y_train.append(label)
        
        # 测试数据：相对均衡分布
        for i in range(test_samples):
            label = np.random.randint(0, 10)
            image_features = self._generate_digit_features(label)
            X_test.append(image_features)
            y_test.append(label)
        
        return {
            "X_train": np.array(X_train, dtype=np.float32),
            "y_train": np.array(y_train, dtype=np.int32),
            "X_test": np.array(X_test, dtype=np.float32),
            "y_test": np.array(y_test, dtype=np.int32),
            "num_classes": 10,
            "input_shape": [784],
            "preferred_classes": preferred_classes
        }
    
    def _generate_digit_features(self, digit: int) -> np.ndarray:
        """为指定数字生成合成特征向量"""
        # 为不同数字生成不同的特征模式
        base_pattern = np.random.normal(digit * 0.1, 0.5, 784)  # 基础模式
        
        # 添加一些数字特定的特征
        if digit == 0:  # 圆形特征
            base_pattern[100:150] += 1.0
            base_pattern[600:650] += 1.0
        elif digit == 1:  # 垂直线特征
            base_pattern[::10] += 0.8
        elif digit == 2:  # 曲线特征
            base_pattern[200:300] += 0.6
        # ... 可以添加更多数字特定特征
        
        # 添加噪声
        noise = np.random.normal(0, 0.1, 784)
        features = base_pattern + noise
        
        # 归一化到[0,1]
        features = (features - features.min()) / (features.max() - features.min() + 1e-8)
        
        return features.astype(np.float32)
    
    def _get_label_distribution(self) -> Dict[int, int]:
        """获取标签分布统计"""
        unique, counts = np.unique(self.local_data["y_train"], return_counts=True)
        return {int(label): int(count) for label, count in zip(unique, counts)}
    
    async def train(self, training_params: Dict[str, Any]) -> TrainingResult:
        """执行本地训练"""
        async with self._lock:
            print(f"🔄 客户端 {self.client_id} 开始本地训练")
            
            # 解析训练参数
            global_model = training_params.get("global_model", {})
            epochs = training_params.get("epochs", 1)
            learning_rate = training_params.get("learning_rate", self.learning_rate)
            batch_size = training_params.get("batch_size", self.batch_size)
            round_num = training_params.get("round_num", 0)
            
            start_time = datetime.now()
            
            try:
                # 初始化或更新本地模型
                if global_model:
                    self._local_model = copy.deepcopy(global_model)
                elif self._local_model is None:
                    # 初始化本地模型
                    self._local_model = self._initialize_local_model()
                
                # 执行本地SGD训练
                training_loss, training_accuracy = await self._local_sgd_training(
                    epochs, learning_rate, batch_size
                )
                
                # 计算模型更新（发送给服务器的是更新后的完整模型）
                model_update = copy.deepcopy(self._local_model)
                
                end_time = datetime.now()
                training_time = (end_time - start_time).total_seconds()
                
                # 更新统计信息
                self.training_count += 1
                self.last_training_time = end_time
                
                result = {
                    "model_update": model_update,
                    "weights": model_update,  # 兼容性
                    "loss": float(training_loss),
                    "accuracy": float(training_accuracy),
                    "samples_count": len(self.local_data["X_train"]),
                    "samples": len(self.local_data["X_train"]),  # 兼容性
                    "training_time": training_time,
                    "epochs_completed": epochs,
                    "client_id": self.client_id,
                    "round_num": round_num
                }
                
                print(f"   ✅ 训练完成: Loss={training_loss:.4f}, Acc={training_accuracy:.4f}")
                
                return result
                
            except Exception as e:
                error_msg = f"客户端 {self.client_id} 训练失败: {str(e)}"
                print(f"   ❌ {error_msg}")
                raise TrainingError(error_msg)
    
    async def _local_sgd_training(self, epochs: int, learning_rate: float, batch_size: int) -> Tuple[float, float]:
        """执行本地SGD训练"""
        
        X_train = self.local_data["X_train"]
        y_train = self.local_data["y_train"]
        
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        for epoch in range(epochs):
            # 随机打乱训练数据
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
                
                # 反向传播和参数更新
                await self._backward_pass(X_batch, y_batch, predictions, learning_rate)
                
                # 计算准确率
                batch_accuracy = np.mean(np.argmax(predictions, axis=1) == y_batch)
                
                epoch_loss += loss
                epoch_accuracy += batch_accuracy
                epoch_batches += 1
                
                # 模拟异步训练（让出控制权）
                if epoch_batches % 5 == 0:  # 减少频率
                    await asyncio.sleep(0.001)
            
            # 计算平均指标
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
    
    def _forward_pass(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """前向传播"""
        # 第一层：784 -> 128
        z1 = np.dot(X, np.array(self._local_model["W1"])) + np.array(self._local_model["b1"])
        a1 = self._relu(z1)
        
        # 第二层：128 -> 10
        z2 = np.dot(a1, np.array(self._local_model["W2"])) + np.array(self._local_model["b2"])
        a2 = self._softmax(z2)
        
        # 计算交叉熵损失
        loss = self._cross_entropy_loss(a2, y)
        
        # 保存中间结果用于反向传播
        self._forward_cache = {
            "X": X, "z1": z1, "a1": a1, "z2": z2, "a2": a2, "y": y
        }
        
        return a2, loss
    
    async def _backward_pass(self, X: np.ndarray, y: np.ndarray, predictions: np.ndarray, learning_rate: float):
        """反向传播和参数更新"""
        m = X.shape[0]  # 批次大小
        
        # 从缓存获取前向传播的中间结果
        a1 = self._forward_cache["a1"]
        a2 = predictions
        
        # 计算梯度
        # 输出层梯度
        y_one_hot = np.eye(10)[y]
        dz2 = a2 - y_one_hot
        dW2 = (1/m) * np.dot(a1.T, dz2)
        db2 = (1/m) * np.sum(dz2, axis=0)
        
        # 隐藏层梯度
        da1 = np.dot(dz2, np.array(self._local_model["W2"]).T)
        dz1 = da1 * self._relu_derivative(self._forward_cache["z1"])
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
        
        # 更新模型参数
        self._local_model["W1"] = W1.tolist()
        self._local_model["b1"] = b1.tolist()
        self._local_model["W2"] = W2.tolist()
        self._local_model["b2"] = b2.tolist()
        
        # 模拟异步计算
        await asyncio.sleep(0.0001)  # 减少睡眠时间
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU激活函数"""
        return np.maximum(0, x)
    
    def _relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """ReLU导数"""
        return (x > 0).astype(np.float32)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax激活函数"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def _cross_entropy_loss(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """交叉熵损失函数"""
        m = predictions.shape[0]
        # 避免log(0)
        predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
        
        # 计算交叉熵
        log_likelihood = -np.log(predictions[range(m), labels])
        loss = np.mean(log_likelihood)
        
        return float(loss)
    
    async def evaluate(self, evaluation_params: Dict[str, Any]) -> EvaluationResult:
        """执行本地评估"""
        async with self._lock:
            print(f"🔍 客户端 {self.client_id} 开始评估")
            
            # 解析评估参数
            model = evaluation_params.get("model", self._local_model)
            use_test_data = evaluation_params.get("test_data", True)
            
            start_time = datetime.now()
            
            try:
                # 选择评估数据集
                if use_test_data:
                    X_eval = self.local_data["X_test"]
                    y_eval = self.local_data["y_test"]
                else:
                    X_eval = self.local_data["X_train"]
                    y_eval = self.local_data["y_train"]
                
                # 临时设置模型用于评估
                original_model = self._local_model
                if model:
                    self._local_model = copy.deepcopy(model)
                
                # 执行评估
                predictions, loss = self._forward_pass(X_eval, y_eval)
                accuracy = np.mean(np.argmax(predictions, axis=1) == y_eval)
                
                # 恢复原模型
                self._local_model = original_model
                
                end_time = datetime.now()
                evaluation_time = (end_time - start_time).total_seconds()
                
                # 更新统计信息
                self.evaluation_count += 1
                self.last_evaluation_time = end_time
                
                result = {
                    "accuracy": float(accuracy),
                    "loss": float(loss),
                    "samples_count": len(X_eval),
                    "evaluation_time": evaluation_time,
                    "client_id": self.client_id
                }
                
                print(f"   ✅ 评估完成: Loss={loss:.4f}, Acc={accuracy:.4f}")
                
                return result
                
            except Exception as e:
                error_msg = f"客户端 {self.client_id} 评估失败: {str(e)}"
                print(f"   ❌ {error_msg}")
                raise TrainingError(error_msg)
    
    async def get_local_model(self) -> ModelData:
        """获取本地模型参数"""
        if self._local_model is None:
            self._local_model = self._initialize_local_model()
        
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
            
            print(f"📥 客户端 {self.client_id} 更新本地模型")
            return True
            
        except Exception as e:
            print(f"❌ 客户端 {self.client_id} 模型更新失败: {e}")
            return False
    
    def _initialize_local_model(self) -> Dict[str, Any]:
        """初始化本地模型"""
        # 使用客户端特定的随机种子
        np.random.seed(hash(self.client_id) % 2**32)
        
        model = {
            "W1": np.random.normal(0, 0.1, (784, 128)).tolist(),
            "b1": np.zeros(128).tolist(),
            "W2": np.random.normal(0, 0.1, (128, 10)).tolist(),
            "b2": np.zeros(10).tolist()
        }
        
        return model
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """获取数据统计信息"""
        return {
            "total_samples": len(self.local_data["X_train"]),
            "test_samples": len(self.local_data["X_test"]),
            "num_classes": self.local_data["num_classes"],
            "input_shape": self.local_data["input_shape"],
            "label_distribution": self._get_label_distribution(),
            "preferred_classes": self.local_data["preferred_classes"],
            "data_type": "synthetic_mnist",
            "available": True
        }
